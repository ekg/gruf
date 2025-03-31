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
from config import MODEL_CONFIG, TRAINING_CONFIG, calculate_model_size, get_parameter_count_str

# Use constants from config (with defaults for backward compatibility if needed)
NUM_BATCHES = TRAINING_CONFIG["num_batches"]
BATCH_SIZE = TRAINING_CONFIG["batch_size"]
GRAD_ACCUM_EVERY = TRAINING_CONFIG.get("grad_accum_every", 4)  # Default for train.py
LEARNING_RATE = TRAINING_CONFIG["learning_rate"]
VALIDATE_EVERY = TRAINING_CONFIG.get("validate_every", 100)    # Default for train.py
PRIME_LENGTH = TRAINING_CONFIG["prime_length"]
GENERATE_EVERY = TRAINING_CONFIG.get("generate_every", 500)    # Default for train.py
GENERATE_LENGTH = TRAINING_CONFIG["generate_length"]
SEQ_LEN = TRAINING_CONFIG.get("seq_len", 512)                  # Default for train.py

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
    num_tokens=MODEL_CONFIG["num_tokens"],
    dim=MODEL_CONFIG["dim"],
    depth=MODEL_CONFIG["depth"],
    ff_mult=MODEL_CONFIG["ff_mult"],
    expansion=MODEL_CONFIG["expansion"],
    conv_kernel_size=MODEL_CONFIG["conv_kernel_size"],
    use_lstm=MODEL_CONFIG["use_lstm"],
    enable_conv=MODEL_CONFIG["enable_conv"],
    dropout=MODEL_CONFIG["dropout"]
).cuda()

print(f"Created model with {get_parameter_count_str(MODEL_CONFIG)} parameters")

# prepare enwik8 data

with gzip.open("./data/enwik8.gz") as file:
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    np_train, np_valid = np.split(data, [int(90e6)])
    data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        # Define dataset length such that one epoch covers the full data
        # Each sample is seq_len tokens, so we need data_size/seq_len samples to cover all
        self.samples_per_epoch = max(1, self.data.size(0) // self.seq_len)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
        # Random sampling from anywhere in the data
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq.cuda()

# Calculate optimal number of workers
num_workers = min(31, os.cpu_count() or 4)
print(f"Using {num_workers} dataloader workers")

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
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
