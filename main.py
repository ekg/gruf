import os
import sys
import math
import torch
import random
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI

from lightning_min_lm import LightningMinLM
from data_module import Enwik8DataModule

# Set float32 matmul precision to address warnings
torch.set_float32_matmul_precision('high')

# Functions for text generation
def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1, keepdim = True):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim, keepdim = keepdim)

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(-1, ind, val)
    return probs

def base_decoding(
    net,
    prompt: torch.Tensor,
    seq_len: int,
    temperature = 1.,
    filter_thres = 0.9,
):
    prompt_seq_len, out = prompt.shape[-1], prompt.clone()
    sample_num_times = max(0, seq_len - prompt_seq_len)

    prev_hiddens = None

    for _ in range(sample_num_times):
        # Handle both cases: when forward returns a tuple and when it returns just logits
        output = net(out, prev_hiddens=prev_hiddens)
        
        if isinstance(output, tuple):
            logits, next_prev_hiddens = output
        else:
            logits = output
            next_prev_hiddens = None
            
        logits = logits[:, -1]

        if hasattr(net, 'model') and hasattr(net.model, 'can_cache') and net.model.can_cache:
            prev_hiddens = next_prev_hiddens

        logits = top_k(logits, thres = filter_thres)
        sample = gumbel_sample(logits, temperature = temperature, dim = -1)

        out = torch.cat((out, sample), dim = -1)

    return out[..., prompt_seq_len:]

# define text generation function to be used as a callback
def generate_text_during_training(trainer, pl_module, batch_idx, prime_length=128, generate_length=512, generate_every=500):
    """Text generation function that can be used as a callback during training."""
    if trainer.global_rank == 0 and (batch_idx + 1) % generate_every == 0:
        pl_module.eval()
        
        # Get validation dataset from datamodule
        val_dataset = trainer.datamodule.val_dataset if hasattr(trainer.datamodule, 'val_dataset') else None
        
        if val_dataset is None:
            print("No validation dataset available for text generation", flush=True)
            return
        
        try:
            # Get a sample from validation set
            inp = random.choice(val_dataset)[:prime_length]
            inp = inp.to(pl_module.device)
            
            prime = decode_tokens(inp)
            print(f"\n\n===== GENERATION AT STEP {batch_idx+1} =====")
            print(f"INPUT: {prime}")
            
            prompt = inp[None, ...]
            
            # Generate text
            sampled = base_decoding(
                pl_module, 
                prompt, 
                generate_length,
                temperature=1.0,
                filter_thres=0.9
            )
            
            base_decode_output = decode_tokens(sampled[0])
            print(f"\nOUTPUT: {base_decode_output}")
            print("=" * 50)
        except Exception as e:
            print(f"Error during text generation: {str(e)}")
        
        pl_module.train()

if __name__ == "__main__":
    import gzip
    import numpy as np
    from torch.utils.data import DataLoader, Dataset
    from pytorch_lightning.callbacks import LambdaCallback
    
    # TextSamplerDataset - identical to train.py
    class TextSamplerDataset(Dataset):
        def __init__(self, data, seq_len):
            super().__init__()
            self.data = data
            self.seq_len = seq_len

        def __len__(self):
            return self.data.size(0) // self.seq_len

        def __getitem__(self, index):
            rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
            full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
            return full_seq.cuda()
    
    # cycle function - identical to train.py
    def cycle(loader):
        while True:
            for data in loader:
                yield data
    
    # Create a simple lambda callback for text generation
    text_gen_callback = LambdaCallback(
        on_train_batch_end=lambda trainer, pl_module, outputs, batch, batch_idx, unused=0: 
            generate_text_during_training(
                trainer, 
                pl_module, 
                batch_idx,
                prime_length=128,
                generate_length=512,
                generate_every=500
            )
    )
    
    # If using CLI, load data first to pass validation dataset to callback
    if len(sys.argv) > 1 and sys.argv[1] == "fit":
        # Load data the same way as train.py
        with gzip.open("./data/enwik8.gz") as file:
            data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
            np_train, np_valid = np.split(data, [int(90e6)])
            data_val = torch.from_numpy(np_valid)
        
        val_dataset = TextSamplerDataset(data_val, 512)
        
        # Use a custom data module that loads data identically to train.py
        class EnwikTextSamplerDataModule(pl.LightningDataModule):
            def __init__(self, batch_size=4, seq_len=512, data_path="./data/enwik8.gz"):
                super().__init__()
                self.batch_size = batch_size
                self.seq_len = seq_len
                self.data_path = data_path
                
                # Load the data immediately so it's available for callbacks
                with gzip.open(self.data_path) as file:
                    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
                    np_train, np_valid = np.split(data, [int(90e6)])
                    self.data_train = torch.from_numpy(np_train)
                    self.data_val = torch.from_numpy(np_valid)
                
                self.train_dataset = TextSamplerDataset(self.data_train, self.seq_len)
                self.val_dataset = TextSamplerDataset(self.data_val, self.seq_len)
            
            def train_dataloader(self):
                train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size)
                return CycledDataLoader(cycle(train_loader))
            
            def val_dataloader(self):
                val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size)
                return CycledDataLoader(cycle(val_loader))
        
        # Custom dataloader that returns cycled data
        class CycledDataLoader:
            def __init__(self, cycled_iterator):
                self.cycled_iterator = cycled_iterator
            
            def __iter__(self):
                return self
            
            def __next__(self):
                return next(self.cycled_iterator)
    
        # Make sure to register the checkpoint callback as class_path for CLI compatibility
        # LightningCLI handles argument parsing and instantiation
        cli = LightningCLI(
            LightningMinLM,
            EnwikTextSamplerDataModule,
            seed_everything_default=42,
            run=True,  # Run fit by default
            trainer_defaults={
                "callbacks": [
                    # Model checkpoint callback
                    {"class_path": "pytorch_lightning.callbacks.ModelCheckpoint",
                     "init_args": {
                         "monitor": "val_loss",
                         "filename": "minlm-{epoch:02d}-{val_loss:.2f}",
                         "save_top_k": 3,
                         "mode": "min"
                     }},
                    # Use the lambda callback directly
                    text_gen_callback
                ],
                "accumulate_grad_batches": 4  # Match GRAD_ACCUM_EVERY from train.py
            }
        )
    else:
        print("Please use the 'fit' command to run training.")
