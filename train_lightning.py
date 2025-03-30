import math
import torch
import time
import os
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.utilities import rank_zero_only
import sys
from pytorch_lightning.loggers import CSVLogger

# Set float32 matmul precision to address the warning
torch.set_float32_matmul_precision('high')

# Force more aggressive logging
os.environ["PYTORCH_LIGHTNING_RANK_ZERO_ONLY"] = "0"  # To ensure logging from all ranks

from lightning_min_lm import LightningMinLM
from data_module import Enwik8DataModule

# Constants (matching the original training script)
NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY = 100
PRIME_LENGTH = 128
GENERATE_EVERY = 500
GENERATE_LENGTH = 512
SEQ_LEN = 512
TRUNCATED_BPTT_STEPS = 128  # For recurrent training

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

# Debug initialization - print when each process starts
def print_debug_info():
    global_rank = int(os.environ.get("GLOBAL_RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # This will print from every process
    print(f"STARTING: Process rank {global_rank}/{world_size} (local: {local_rank}) - PID: {os.getpid()}", 
          flush=True, file=sys.stderr)

print_debug_info()

# Custom progress tracking callback
class TrainingMetricsCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.last_batch_idx = 0
        self.tokens_processed = 0
        print("TrainingMetricsCallback initialized", flush=True, file=sys.stderr)
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Not using rank_zero_only to get logging from all processes
        global_rank = trainer.global_rank
        if self.start_time is None or batch_idx < self.last_batch_idx:
            self.start_time = time.time()
            self.tokens_processed = 0
            print(f"[Rank {global_rank}] Starting batch timing", flush=True, file=sys.stderr)
        self.last_batch_idx = batch_idx
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_rank = trainer.global_rank
        # Count tokens processed
        batch_size = batch.size(0)
        seq_len = batch.size(1) - 1  # -1 because we're using the last token as the target
        self.tokens_processed += batch_size * seq_len
        
        # Calculate tokens per second
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            tokens_per_sec = self.tokens_processed / elapsed
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            
            # Print for all ranks to debug
            msg = f"[Rank {global_rank}] Step {batch_idx}: loss: {loss.item():.3f} | {tokens_per_sec:.2f} tokens/s"
            print(msg, flush=True, file=sys.stderr)

# Text generation callback
class TextGenerationCallback(pl.Callback):
    def __init__(self, val_dataset, prime_length=128, generate_length=512, generate_every=500):
        super().__init__()
        self.val_dataset = val_dataset
        self.prime_length = prime_length
        self.generate_length = generate_length
        self.generate_every = generate_every
        print("TextGenerationCallback initialized", flush=True, file=sys.stderr)
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Only generate text on rank 0
        if trainer.global_rank == 0 and (batch_idx + 1) % self.generate_every == 0:
            print(f"\n\nStarting text generation at step {batch_idx+1}", flush=True, file=sys.stderr)
            pl_module.eval()
            
            try:
                # Get a sample from validation set
                inp = random.choice(self.val_dataset)[:self.prime_length]
                inp = inp.cuda()
                
                prime = decode_tokens(inp)
                print(f"\n\n===== GENERATION AT STEP {batch_idx+1} =====", flush=True)
                print(f"INPUT: {prime}", flush=True)
                
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
                print(f"\nOUTPUT: {base_decode_output}", flush=True)
                print("=" * 50, flush=True)
            except Exception as e:
                print(f"Error during text generation: {str(e)}", flush=True, file=sys.stderr)
            
            pl_module.train()

if __name__ == "__main__":
    import random
    
    # Set up data module
    data_module = Enwik8DataModule(
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN
    )
    data_module.setup()
    
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
        val_dataset=data_module.val_dataset,
        prime_length=PRIME_LENGTH,
        generate_length=GENERATE_LENGTH,
        generate_every=GENERATE_EVERY
    )
    
    metrics_callback = TrainingMetricsCallback()
    
    progress_bar = TQDMProgressBar(refresh_rate=20)  # Update every 20 steps
    
    # Create a CSV logger
    logger = CSVLogger("logs", name="min_lm_training")
    
    # Print trainer setup info
    gpu_count = torch.cuda.device_count()
    print(f"Setting up trainer with {gpu_count} GPUs...", flush=True, file=sys.stderr)
    
    # Set up trainer
    trainer = Trainer(
        max_steps=NUM_BATCHES,  # to match the original script
        accelerator="gpu",
        devices="auto",  # use all available GPUs
        strategy="ddp" if torch.cuda.device_count() > 1 else None,
        gradient_clip_val=0.5,
        callbacks=[checkpoint_callback, text_gen_callback, metrics_callback, progress_bar],
        val_check_interval=VALIDATE_EVERY,
        enable_progress_bar=True,
        log_every_n_steps=1,  # Log every step for more visibility
        logger=logger
    )
    
    print("Trainer setup complete, starting training...", flush=True, file=sys.stderr)
    
    # Train model
    print("Starting trainer.fit()...", flush=True, file=sys.stderr)
    trainer.fit(model, data_module)
    print("Training complete!", flush=True, file=sys.stderr)
