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
class CustomProgressBar(TQDMProgressBar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("CustomProgressBar initialized")
        
    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        # Add more visible metrics
        if hasattr(trainer, 'logged_metrics'):
            items["PROGRESS"] = f"STEP {trainer.global_step}"
            if 'tokens_per_second' in trainer.logged_metrics:
                items["TOKENS/S"] = f"{trainer.logged_metrics['tokens_per_second']:.2f}"
            if 'batch_loss' in trainer.logged_metrics:
                items["LOSS"] = f"{trainer.logged_metrics['batch_loss']:.5f}"
        return items
    
    def on_train_batch_end(self, *args, **kwargs):
        # Force progress bar to print on every batch
        print("\n")  # Add extra newline for visibility
        print("PROGRESS BAR UPDATE")
        super().on_train_batch_end(*args, **kwargs)

class TrainingMetricsCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.start_time = None
        self.tokens_processed = 0
        self.logger = logging.getLogger("min_lm_training.metrics")
        print("TrainingMetricsCallback initialized")
        self.logger.info("TrainingMetricsCallback initialized")
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        print(f"\n========== STARTING BATCH {batch_idx} ==========")
        self.logger.debug(f"Starting batch {batch_idx}, global_step={trainer.global_step}")
        
        if self.start_time is None:
            self.start_time = time.time()
            self.tokens_processed = 0
            print("Timer started")
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        # Always log on all processes to help debug
        # Remove the global_rank check to log from all processes
        
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
            trainer.logger.log_metrics({
                "tokens_per_second": tokens_per_sec,
                "batch_loss": loss.item()
            }, step=trainer.global_step)
            
            # Extreme logging - print EVERY batch with clear markers
            print(f"======== STEP {trainer.global_step} ========")
            print(f"BATCH {batch_idx} METRICS:")
            print(f"LOSS: {loss.item():.5f}")
            print(f"TOKENS/SEC: {tokens_per_sec:.2f}")
            print(f"ELAPSED TIME: {elapsed:.2f}s")
            print(f"BATCH SHAPE: {batch.shape}")
            print(f"GPU MEM ALLOCATED: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
            print(f"GPU MEM RESERVED: {torch.cuda.memory_reserved() / 1024**2:.2f}MB")
            print(f"================================")
            
            # Also log to file with extreme detail
            self.logger.info(f"STEP {trainer.global_step} | BATCH {batch_idx} | LOSS: {loss.item():.5f} | {tokens_per_sec:.2f} tokens/s | TIME: {elapsed:.2f}s")

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
        self.logger.info("TextGenerationCallback initialized")
    
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
    # Initial diagnostic output
    print("========== PROGRAM STARTED ==========")
    print(f"CUDA AVAILABLE: {torch.cuda.is_available()}")
    print(f"GPU COUNT: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"PYTORCH VERSION: {torch.__version__}")
    print(f"PYTORCH LIGHTNING VERSION: {pl.__version__}")
    print("====================================")
    
    # Load and prepare data - exactly as in train.py
    print("Loading data from enwik8.gz...")
    try:
        with gzip.open("./data/enwik8.gz") as file:
            print("File opened successfully, reading data...")
            data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
            np_train, np_valid = np.split(data, [int(90e6)])
            data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)
            print("========== DATA LOADED ==========")
            print(f"TRAIN DATA SHAPE: {data_train.shape}")
            print(f"VAL DATA SHAPE: {data_val.shape}")
            print("================================")
    except Exception as e:
        print(f"ERROR LOADING DATA: {str(e)}")
        raise

    # Create datasets and dataloaders - exactly as in train.py
    print("Creating datasets and dataloaders...")
    train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
    val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # Cycle the loaders - exactly as in train.py
    print("Setting up cycled data loaders...")
    train_loader_iter = cycle(train_loader)
    val_loader_iter = cycle(val_loader)
    
    # Test data loaders
    print("========== TESTING DATA LOADERS ==========")
    try:
        print("Testing train loader...")
        test_batch = next(train_loader_iter)
        print(f"SUCCESS - Train batch shape: {test_batch.shape}")
        
        print("Testing val loader...")
        test_batch = next(val_loader_iter)
        print(f"SUCCESS - Val batch shape: {test_batch.shape}")
    except Exception as e:
        print(f"ERROR IN DATA LOADERS: {str(e)}")
        print("This could be why you're not seeing output!")
        raise
    print("=======================================")
    
    # Set up model
    print("========== CREATING MODEL ==========")
    try:
        model = LightningMinLM(
            num_tokens=256,
            dim=512,
            depth=6,
            ff_mult=4,
            learning_rate=LEARNING_RATE,
            use_lstm=False  # set to True for minLSTM
        )
        print("MODEL CREATED:")
        print(model)
        print("==================================")
    except Exception as e:
        print(f"ERROR CREATING MODEL: {str(e)}")
        raise
    
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
    
    # Set up trainer with improved distributed settings
    print("Setting up PyTorch Lightning Trainer...")
    try:
        trainer = Trainer(
            max_steps=NUM_BATCHES,
            accumulate_grad_batches=GRAD_ACCUM_EVERY,  # Match grad accumulation from train.py
            accelerator="gpu",
            devices="auto",  # use all available GPUs
            strategy=DDPStrategy(find_unused_parameters=False) if torch.cuda.device_count() > 1 else "auto",
            gradient_clip_val=0.5,
            callbacks=[checkpoint_callback, text_gen_callback, metrics_callback, progress_bar],
            val_check_interval=VALIDATE_EVERY,
            logger=primary_logger,
            log_every_n_steps=1,  # Add explicit logging frequency
            enable_progress_bar=True,
            enable_model_summary=True,
            enable_checkpointing=True
        )
        print("Trainer created successfully")
    except Exception as e:
        print(f"ERROR CREATING TRAINER: {str(e)}")
        raise
    
    # Log the training configuration
    logger.info(f"Starting training with {torch.cuda.device_count()} GPUs")
    logger.info(f"Batch size: {BATCH_SIZE}, Grad accumulation: {GRAD_ACCUM_EVERY}")
    logger.info(f"Learning rate: {LEARNING_RATE}, Sequence length: {SEQ_LEN}")
    
    print("======== TRAINING CONFIGURATION ========")
    print(f"BATCH SIZE: {BATCH_SIZE}")
    print(f"GRAD ACCUMULATION: {GRAD_ACCUM_EVERY}")
    print(f"EFFECTIVE BATCH SIZE: {BATCH_SIZE * GRAD_ACCUM_EVERY}")
    print(f"LEARNING RATE: {LEARNING_RATE}")
    print(f"SEQUENCE LENGTH: {SEQ_LEN}")
    print(f"NUM BATCHES: {NUM_BATCHES}")
    print(f"VALIDATE EVERY: {VALIDATE_EVERY}")
    print(f"GENERATE EVERY: {GENERATE_EVERY}")
    print("=======================================")
    
    # Create a custom dataloader that returns the cycled data
    # This makes the Lightning version use exactly the same data pattern as train.py
    class CycledDataLoader:
        def __init__(self, cycled_iterator):
            self.cycled_iterator = cycled_iterator
        
        def __iter__(self):
            return self
        
        def __next__(self):
            return next(self.cycled_iterator)
    
    # Pre-training sanity check
    print("========== PRE-TRAINING SANITY CHECK ==========")
    print("About to call trainer.fit()")
    print("If you don't see ANY output after this, the training loop may not be starting")
    print("=============================================")
    
    # Train model with the cycled data loaders
    try:
        print("STARTING TRAINING NOW!")
        trainer.fit(
            model, 
            train_dataloaders=CycledDataLoader(train_loader_iter),
            val_dataloaders=CycledDataLoader(val_loader_iter)
        )
        print("TRAINING COMPLETED SUCCESSFULLY!")
    except Exception as e:
        print(f"ERROR DURING TRAINING: {str(e)}")
        import traceback
        traceback.print_exc()
