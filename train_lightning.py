import math
import torch
import time
import os
import gzip
import numpy as np
import random
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
import sys
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, Dataset

# Set float32 matmul precision to address the warning
torch.set_float32_matmul_precision('high')

from lightning_min_lm import LightningMinLM

# Constants (matching the original training script)
NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRAD_ACCUM_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY = 100
PRIME_LENGTH = 128
GENERATE_EVERY = 500
GENERATE_LENGTH = 512
SEQ_LEN = 512

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
        logits, next_prev_hiddens = net(out, prev_hiddens=prev_hiddens)
        logits = logits[:, -1]

        if net.model.can_cache:
            prev_hiddens = next_prev_hiddens

        logits = top_k(logits, thres = filter_thres)
        sample = gumbel_sample(logits, temperature = temperature, dim = -1)

        out = torch.cat((out, sample), dim = -1)

    return out[..., prompt_seq_len:]

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

# Custom progress tracking callback
class TrainingMetricsCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.tokens_processed = 0
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.start_time is None:
            self.start_time = time.time()
            self.tokens_processed = 0
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        if trainer.global_rank == 0:  # Only print from rank 0
            # Count tokens processed
            batch_size = batch.size(0)
            seq_len = batch.size(1) - 1  # -1 because we're using the last token as the target
            self.tokens_processed += batch_size * seq_len
            
            # Calculate tokens per second
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                tokens_per_sec = self.tokens_processed / elapsed
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs
                
                print(f"training loss: {loss.item():.3f} | {tokens_per_sec:.2f} tokens/s", end="\r")

# Text generation callback - simplified to match train.py pattern
class TextGenerationCallback(pl.Callback):
    def __init__(self, val_dataset, prime_length=128, generate_length=512, generate_every=500):
        super().__init__()
        self.val_dataset = val_dataset
        self.prime_length = prime_length
        self.generate_length = generate_length
        self.generate_every = generate_every
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        # Only generate text on rank 0
        if trainer.global_rank == 0 and (batch_idx + 1) % self.generate_every == 0:
            pl_module.eval()
            
            try:
                # Get a sample from validation set
                inp = random.choice(self.val_dataset)[:self.prime_length]
                inp = inp.cuda()
                
                prime = decode_tokens(inp)
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
            except Exception as e:
                print(f"Error during text generation: {str(e)}")
            
            pl_module.train()

if __name__ == "__main__":
    # Load and prepare data - exactly as in train.py
    with gzip.open("./data/enwik8.gz") as file:
        data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
        np_train, np_valid = np.split(data, [int(90e6)])
        data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)

    # Create datasets and dataloaders - exactly as in train.py
    train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
    val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Cycle the loaders - exactly as in train.py
    train_loader_iter = cycle(train_loader)
    val_loader_iter = cycle(val_loader)
    
    # Set up model
    model = LightningMinLM(
        num_tokens=256,
        dim=512,
        depth=6,
        ff_mult=4,
        learning_rate=LEARNING_RATE,
        use_lstm=False  # set to True for minLSTM
    )
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="minlm-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min"
    )
    
    text_gen_callback = TextGenerationCallback(
        val_dataset=val_dataset,
        prime_length=PRIME_LENGTH,
        generate_length=GENERATE_LENGTH,
        generate_every=GENERATE_EVERY
    )
    
    metrics_callback = TrainingMetricsCallback()
    
    progress_bar = TQDMProgressBar(refresh_rate=20)  # Update every 20 steps
    
    # Create a CSV logger
    logger = CSVLogger("logs", name="min_lm_training")
    
    # Set up trainer
    trainer = Trainer(
        max_steps=NUM_BATCHES,
        accumulate_grad_batches=GRAD_ACCUM_EVERY,  # Match grad accumulation from train.py
        accelerator="gpu",
        devices="auto",  # use all available GPUs
        strategy="ddp" if torch.cuda.device_count() > 1 else None,
        gradient_clip_val=0.5,
        callbacks=[checkpoint_callback, text_gen_callback, metrics_callback, progress_bar],
        val_check_interval=VALIDATE_EVERY,
        logger=logger
    )
    
    # Create a custom dataloader that returns the cycled data
    # This makes the Lightning version use exactly the same data pattern as train.py
    class CycledDataLoader:
        def __init__(self, cycled_iterator):
            self.cycled_iterator = cycled_iterator
        
        def __iter__(self):
            return self
        
        def __next__(self):
            return next(self.cycled_iterator)
    
    # Train model with the cycled data loaders
    trainer.fit(
        model, 
        train_dataloaders=CycledDataLoader(train_loader_iter),
        val_dataloaders=CycledDataLoader(val_loader_iter)
    )
