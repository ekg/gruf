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

# Text generation callback
class TextGenerationCallback(pl.Callback):
    def __init__(self, prime_length=128, generate_length=512, generate_every=500):
        super().__init__()
        self.val_dataset = None
        self.prime_length = prime_length
        self.generate_length = generate_length
        self.generate_every = generate_every
    
    def setup(self, trainer, pl_module, stage=None):
        # Access val_dataset from the datamodule when it's available
        if hasattr(trainer.datamodule, 'val_dataset'):
            self.val_dataset = trainer.datamodule.val_dataset
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Only generate text on rank 0
        if trainer.global_rank == 0 and (batch_idx + 1) % self.generate_every == 0:
            pl_module.eval()
            
            if self.val_dataset is None:
                print("No validation dataset available for text generation", flush=True)
                return
            
            try:
                # Get a sample from validation set
                inp = random.choice(self.val_dataset)[:self.prime_length]
                inp = inp.to(pl_module.device)
                
                prime = decode_tokens(inp)
                print(f"\n\n===== GENERATION AT STEP {batch_idx+1} =====")
                print(f"INPUT: {prime}")
                
                prompt = inp[None, ...]
                
                # Generate text
                sampled = base_decoding(
                    pl_module, 
                    prompt, 
                    self.generate_length,
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
    # LightningCLI handles argument parsing and instantiation
    cli = LightningCLI(
        LightningMinLM,
        Enwik8DataModule,
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
                # Text generation callback
                {"class_path": __name__ + ".TextGenerationCallback",
                 "init_args": {
                     "prime_length": 128,
                     "generate_length": 512,
                     "generate_every": 500
                 }}
            ]
        }
    )
