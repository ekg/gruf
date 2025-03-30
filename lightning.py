import os
import gzip
import random
import numpy as np
import math
import time
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger

# Set environment variables
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_SOCKET_IFNAME"] = "eno1"  # Use the specific interface shown in your logs

# Import the minLM model
from minGRU_pytorch.minLM import minLM

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
        logits, next_prev_hiddens = net(out, return_prev_hiddens = True, prev_hiddens = prev_hiddens)
        logits = logits[:, -1]

        if net.can_cache:
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
        return full_seq  # Let Lightning handle device placement

# LightningMinLM model
class LightningMinLM(pl.LightningModule):
    def __init__(
        self,
        num_tokens=256,
        dim=512,
        depth=6,
        ff_mult=4,
        expansion=1.5,
        conv_kernel_size=3,
        learning_rate=1e-4,
        use_lstm=False
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = minLM(
            num_tokens=num_tokens,
            dim=dim,
            depth=depth,
            ff_mult=ff_mult,
            expansion=expansion,
            conv_kernel_size=conv_kernel_size,
            use_lstm=use_lstm
        )
        # For tracking tokens per second
        self.total_tokens_processed = 0
        self.start_time = time.time()
    
    def forward(self, x, prev_hiddens=None):
        return self.model(x, return_loss=False, return_prev_hiddens=True, prev_hiddens=prev_hiddens)
    
    def training_step(self, batch, batch_idx, hiddens=None):
        loss = self.model(batch, return_loss=True)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        
        # Update tokens processed count
        tokens_in_batch = batch.numel()
        self.total_tokens_processed += tokens_in_batch
        
        # Only display on rank 0 to avoid duplicate output
        if self.global_rank == 0:
            elapsed = time.time() - self.start_time
            tokens_per_sec = self.total_tokens_processed / elapsed if elapsed > 0 else 0
            print(f"Batch {batch_idx} | Loss: {loss.item():.4f} | {tokens_per_sec:.2f} tokens/s", end="\r")
        
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        loss = self.model(batch, return_loss=True)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        
        if self.global_rank == 0 and batch_idx == 0:
            print(f"\nvalidation loss: {loss.item():.4f}")
            
        return {"val_loss": loss}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

# Text generation callback
class TextGenerationCallback(pl.Callback):
    def __init__(self, val_dataset, prime_length=128, generate_length=512, generate_every=500):
        super().__init__()
        self.val_dataset = val_dataset
        self.prime_length = prime_length
        self.generate_length = generate_length
        self.generate_every = generate_every
    
    def on_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Only generate text on rank 0 and only during training
        if (trainer.global_rank == 0 and 
            trainer.state.stage == "train" and 
            (trainer.global_step + 1) % self.generate_every == 0):
            
            # Store original mode and switch to eval
            was_training = pl_module.training
            pl_module.eval()
            
            try:
                # Get a sample from validation set
                inp = random.choice(self.val_dataset)[:self.prime_length]
                inp = inp.cuda()
                
                prime = decode_tokens(inp)
                print(f"\n--- SAMPLE GENERATION AT STEP {trainer.global_step} ---")
                print(f"INPUT: {prime}")
                
                prompt = inp[None, ...]
                
                # Generate text
                sampled = base_decoding(
                    pl_module.model, 
                    prompt, 
                    self.generate_length,
                    temperature=1.0,
                    filter_thres=0.9
                )
                
                base_decode_output = decode_tokens(sampled[0])
                print(f"\nOUTPUT: {base_decode_output}")
                print(f"--- END SAMPLE GENERATION ---\n")
                
            except Exception as e:
                print(f"Error during text generation: {str(e)}")
            finally:
                # Restore the model to its original training state
                if was_training:
                    pl_module.train()

def main():
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('medium')

    print(f"CUDA AVAILABLE: {torch.cuda.is_available()}")
    print(f"GPU COUNT: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Load and prepare data
    print("Loading data from enwik8.gz...")
    with gzip.open("./data/enwik8.gz") as file:
        data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
        np_train, np_valid = np.split(data, [int(90e6)])
        data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)
    print(f"Data loaded - Train: {data_train.shape}, Val: {data_val.shape}")

    # Create datasets and dataloaders
    print("Creating datasets and dataloaders...")
    train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
    val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
    
    # Set up model
    print("Creating model...")
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
    
    # Create logger
    csv_logger = CSVLogger(
        save_dir="logs",
        name="min_lm_training",
        flush_logs_every_n_steps=10
    )
    
    # Create a DDPStrategy with the gloo backend
    ddp_strategy = DDPStrategy(
        process_group_backend="gloo",
        find_unused_parameters=False,
        static_graph=False  # Setting to False to avoid DDP autograd hooks issue
    ) if torch.cuda.device_count() > 1 else "auto"

    # Create trainer
    trainer = pl.Trainer(
        max_steps=NUM_BATCHES,
        accumulate_grad_batches=GRAD_ACCUM_EVERY,
        accelerator="gpu",
        devices="auto",
        strategy=ddp_strategy,
        gradient_clip_val=0.5,
        callbacks=[checkpoint_callback, text_gen_callback],
        val_check_interval=VALIDATE_EVERY,
        logger=csv_logger,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        limit_val_batches=4,
    )

    print(f"Starting training with {torch.cuda.device_count()} GPUs")
    print(f"Config: bs={BATCH_SIZE}, grad_accum={GRAD_ACCUM_EVERY}, lr={LEARNING_RATE}, seq_len={SEQ_LEN}")
    
    # Start training
    print("Starting training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("Training completed.")

if __name__ == "__main__":
    main()
