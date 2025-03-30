import math
import torch
import time
import os
import gzip
import numpy as np
import random
import logging
import sys
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset

# Set environment variables for NCCL
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_SOCKET_IFNAME"] = "eno1"  # Use the specific interface shown in your logs

# Force all print statements to flush immediately
import builtins
_original_print = builtins.print
builtins.print = lambda *args, **kwargs: _original_print(*args, **(dict(kwargs, flush=True)))

# Set up extremely verbose logging
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG for maximum verbosity
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log", mode='w'),  # Overwrite existing log
        logging.StreamHandler(sys.stdout)  # Direct to stdout
    ]
)
logger = logging.getLogger("min_lm_training")
logger.setLevel(logging.DEBUG)
logger.info("========== LOGGING INITIALIZED ==========")

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
# Simpler progress bar that follows the pattern from test_lightning.py
class CustomProgressBar(TQDMProgressBar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Progress bar initialized")
        
    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        # Add tokens/sec if available
        if hasattr(trainer, 'logged_metrics'):
            if 'tokens_per_second' in trainer.logged_metrics:
                items["tokens/s"] = f"{trainer.logged_metrics['tokens_per_second']:.2f}"
        return items

class TrainingMetricsCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.tokens_processed = 0
        self.logger = logging.getLogger("min_lm_training.metrics")
        print("TrainingMetricsCallback initialized")
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if batch_idx % 10 == 0:  # Reduce logging frequency
            self.logger.debug(f"Rank {trainer.global_rank} starting batch {batch_idx}")
        
        if self.start_time is None:
            self.start_time = time.time()
            self.tokens_processed = 0
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        # Count tokens processed
        batch_size = batch.size(0)
        seq_len = batch.size(1) - 1  # -1 because we're using the last token as the target
        self.tokens_processed += batch_size * seq_len

        # Calculate tokens per second
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            tokens_per_sec = self.tokens_processed / elapsed
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
    
            # Log through Lightning
            self.log('train_loss', loss, prog_bar=True, sync_dist=True)
            self.log('tokens_per_second', tokens_per_sec, prog_bar=True, sync_dist=True)
    
            # Simplified logging with rank information like in test_lightning.py
            print(f"Rank {trainer.global_rank} | Batch {batch_idx} | Loss: {loss.item():.4f} | Tokens/s: {tokens_per_sec:.2f}")
    
            # Log to file with sufficient detail but not overwhelming
            self.logger.info(f"Rank {trainer.global_rank} | Step {trainer.global_step} | Batch {batch_idx} | Loss: {loss.item():.4f} | {tokens_per_sec:.2f} tokens/s")

# Text generation callback - simplified to match train.py pattern
class TextGenerationCallback(pl.Callback):
    def __init__(self, val_dataset, prime_length=128, generate_length=512, generate_every=500):
        super().__init__()
        self.val_dataset = val_dataset
        self.prime_length = prime_length
        self.generate_length = generate_length
        self.generate_every = generate_every
        self.logger = logging.getLogger("min_lm_training.text_gen")
        print("TextGenerationCallback initialized")
    
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
                self.logger.info(f"\n--- SAMPLE GENERATION AT STEP {trainer.global_step} ---")
                self.logger.info(f"INPUT: {prime}")
                
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
                self.logger.info(f"OUTPUT: {base_decode_output}")
                self.logger.info(f"--- END SAMPLE GENERATION ---")
                
                # Write generated text to CSV logs too
                if hasattr(trainer.logger, 'experiment'):
                    if isinstance(trainer.logger, TensorBoardLogger):
                        trainer.logger.experiment.add_text(
                            f"generated_text_step_{trainer.global_step}", 
                            f"Input: {prime}\n\nOutput: {base_decode_output}",
                            trainer.global_step
                        )
                    
                # Also save the generated text to a separate file for easy viewing
                with open(f"generated_text_step_{trainer.global_step}.txt", "w") as f:
                    f.write(f"INPUT: {prime}\n\nOUTPUT: {base_decode_output}")
            except Exception as e:
                self.logger.error(f"Error during text generation: {str(e)}")
            finally:
                # Restore the model to its original training state
                if was_training:
                    pl_module.train()

if __name__ == "__main__":
    # Set precision
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('medium')
    
    # Basic diagnostic output following test_lightning.py style
    print(f"CUDA AVAILABLE: {torch.cuda.is_available()}")
    print(f"GPU COUNT: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Load and prepare data - with simplified logging
    print("Loading data from enwik8.gz...")
    with gzip.open("./data/enwik8.gz") as file:
        data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
        np_train, np_valid = np.split(data, [int(90e6)])
        data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)
    print(f"Data loaded - Train: {data_train.shape}, Val: {data_val.shape}")

    # Create datasets and dataloaders - simplified
    print("Creating datasets and dataloaders...")
    train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
    val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)
    
    # Cycle the loaders
    train_loader_iter = cycle(train_loader)
    val_loader_iter = cycle(val_loader)
    
    # Test basic loader functionality (quick sanity check)
    test_batch = next(train_loader_iter)
    print(f"Data loader check - batch shape: {test_batch.shape}")
    
    # Set up model - simplified like in test_lightning.py
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
    
    metrics_callback = TrainingMetricsCallback()
    
    progress_bar = CustomProgressBar(refresh_rate=20)  # Update every 20 steps
    
    # Create multi-logger setup for better real-time visibility
    # Use CSVLogger as the main logger, which works well for CLI environments
    csv_logger = CSVLogger(
        save_dir="logs",
        name="min_lm_training",
        flush_logs_every_n_steps=10  # Write to disk more frequently
    )
    
    # Keep TensorBoard logger as well if available, but we don't rely on it
    tb_logger = TensorBoardLogger("logs", name="min_lm_training")
    
    # Use CSV logger as our primary logger
    primary_logger = csv_logger
    
    # Set up trainer with gloo backend for distributed training
    print("Setting up PyTorch Lightning Trainer with gloo backend...")
    
    # Create a DDPStrategy with the gloo backend, as in test_lightning.py
    ddp_strategy = DDPStrategy(process_group_backend="gloo") if torch.cuda.device_count() > 1 else "auto"
    
    trainer = Trainer(
        max_steps=NUM_BATCHES,
        accumulate_grad_batches=GRAD_ACCUM_EVERY,
        accelerator="gpu",
        devices="auto",
        strategy=ddp_strategy,
        gradient_clip_val=0.5,
        callbacks=[checkpoint_callback, text_gen_callback, metrics_callback, progress_bar],
        val_check_interval=VALIDATE_EVERY,
        logger=primary_logger,
        log_every_n_steps=10,  # Less frequent logging
        num_sanity_val_steps=0,  # Skip validation sanity checks like in test_lightning.py
    )
    print("Trainer created successfully")
    
    # Simplified configuration logging
    print(f"Starting training with {torch.cuda.device_count()} GPUs")
    print(f"Config: bs={BATCH_SIZE}, grad_accum={GRAD_ACCUM_EVERY}, lr={LEARNING_RATE}, seq_len={SEQ_LEN}")
    
    # Create a custom dataloader that returns the cycled data
    # This makes the Lightning version use exactly the same data pattern as train.py
    class CycledDataLoader:
        def __init__(self, cycled_iterator):
            self.cycled_iterator = cycled_iterator
        
        def __iter__(self):
            return self
        
        def __next__(self):
            return next(self.cycled_iterator)
    
    # Training start - similar to test_lightning.py
    print("Starting training...")
    trainer.fit(
        model, 
        train_dataloaders=CycledDataLoader(train_loader_iter),
        val_dataloaders=CycledDataLoader(val_loader_iter)
    )
    print("Training completed.")
