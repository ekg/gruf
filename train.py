import math
import gzip
import random
import tqdm
import numpy as np
import time
import os

import torch
from torch.optim import Adam
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from minGRU_pytorch.minLM import minLM

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRAD_ACCUM_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY = 100
PRIME_LENGTH = 128
GENERATE_EVERY = 500
GENERATE_LENGTH = 512
SEQ_LEN = 512

# helpers

def exists(v):
    return v is not None

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

# sampling helpers

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
    prompt: Tensor,
    seq_len: int,
    temperature = 1.,
    filter_thres = 0.9,
):
    prompt_seq_len, out = prompt.shape[-1], prompt.clone()
    sample_num_times = max(0, seq_len - prompt_seq_len)

    prev_hiddens = None

    for _ in range(sample_num_times):
        logits, next_prev_hiddens = net(out, return_prev_hiddens = True, prev_hiddens = prev_hiddens)
        logits = logits[:, -1]

        if net.can_cache:
            prev_hiddens = next_prev_hiddens

        logits = top_k(logits, thres = filter_thres)
        sample = gumbel_sample(logits, temperature = temperature, dim = -1)

        out = torch.cat((out, sample), dim = -1)

    return out[..., prompt_seq_len:]

# the min{GRU|LSTM} char language model

model = minLM(
    num_tokens = 256,
    dim = 512,
    depth = 6,
    use_lstm = False # set to True for minLSTM
).cuda()

# prepare enwik8 data

with gzip.open("./data/enwik8.gz") as file:
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    np_train, np_valid = np.split(data, [int(90e6)])
    data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len, shuffle_indices=False):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        # Total number of valid starting positions
        self.num_valid_starts = self.data.size(0) - self.seq_len - 1
        
        # Create indices for all valid starting positions
        self.indices = torch.arange(self.num_valid_starts)
        if shuffle_indices:
            # Create a shuffled version of indices but keep original ordering available
            self.shuffled_indices = torch.randperm(self.num_valid_starts)
        else:
            self.shuffled_indices = None

    def __len__(self):
        return self.num_valid_starts

    def __getitem__(self, index):
        # Use shuffled indices if enabled, otherwise use sequential access
        if self.shuffled_indices is not None:
            start_idx = self.shuffled_indices[index]
        else:
            start_idx = index
            
        # Get sequence starting at the determined position
        full_seq = self.data[start_idx : start_idx + self.seq_len + 1].long()
        return full_seq.cuda()
    
    def reshuffle(self):
        """Reshuffle indices at the beginning of each epoch"""
        if self.shuffled_indices is not None:
            self.shuffled_indices = torch.randperm(self.num_valid_starts)

# Calculate optimal number of workers
num_workers = min(31, os.cpu_count() or 4)
print(f"Using {num_workers} dataloader workers")

train_dataset = TextSamplerDataset(data_train, SEQ_LEN, shuffle_indices=True)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN, shuffle_indices=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=num_workers, pin_memory=True, shuffle=False)

# optimizer

optim = Adam(model.parameters(), lr = LEARNING_RATE)

train_loader = cycle(train_loader)
val_loader = cycle(val_loader)

# Track total tokens processed and time for global token/s calculation
total_tokens_processed = 0
global_start_time = time.time()

# training
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval = 10.0, desc = "training"):
    model.train()
    batch_start_time = time.time()

    for _ in range(GRAD_ACCUM_EVERY):
        data = next(train_loader)
        loss = model(data, return_loss = True)
        (loss / GRAD_ACCUM_EVERY).backward()
    
    # Update total tokens processed
    tokens_in_batch = BATCH_SIZE * GRAD_ACCUM_EVERY * SEQ_LEN
    total_tokens_processed += tokens_in_batch
    
    # Calculate tokens per second (both for this batch and overall)
    batch_elapsed = time.time() - batch_start_time
    global_elapsed = time.time() - global_start_time
    
    batch_tokens_per_sec = tokens_in_batch / batch_elapsed if batch_elapsed > 0 else 0
    global_tokens_per_sec = total_tokens_processed / global_elapsed if global_elapsed > 0 else 0
    
    print(f"Batch {i} | Loss: {loss.item():.4f} | {global_tokens_per_sec:.2f} tokens/s", end="\r")

    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

    optim.step()
    optim.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            valid_data = next(val_loader)
            loss = model(valid_data, return_loss = True)
            print(f"\nvalidation loss: {loss.item():.4f}")

    if i % GENERATE_EVERY == 0:
        model.eval()

        inp = random.choice(val_dataset)[:PRIME_LENGTH]
        inp = inp.cuda()

        prime = decode_tokens(inp)
        print(f"INPUT: {prime}")

        prompt = inp[None, ...]

        sampled = base_decoding(model, prompt, GENERATE_LENGTH)

        base_decode_output = decode_tokens(sampled[0])

        print(f"\nOUTPUT: {base_decode_output}")
