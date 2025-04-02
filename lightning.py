import os
os.environ["NCCL_P2P_DISABLE"] = "1"

import gzip
import random
import numpy as np
import math
import time
import argparse
import re
import pytorch_lightning as pl
import torch
import mmap
import datetime
import csv
import json
from torch import nn
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
try:
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
    DEEPSPEED_OPTIMIZERS_AVAILABLE = True
except ImportError:
    DEEPSPEED_OPTIMIZERS_AVAILABLE = False

# Set environment variables
os.environ["NCCL_DEBUG"] = "INFO"

# Import the minLM model and configuration
from minGRU_pytorch.minLM import minLM
from config import MODEL_CONFIG, TRAINING_CONFIG, calculate_model_size, get_parameter_count_str

# Load configuration constants
NUM_BATCHES = TRAINING_CONFIG["num_batches"]
BATCH_SIZE = TRAINING_CONFIG["batch_size"]
GRAD_ACCUM_EVERY = TRAINING_CONFIG["grad_accum_every"]
LEARNING_RATE = TRAINING_CONFIG["learning_rate"]
VALIDATE_EVERY = TRAINING_CONFIG["validate_every"]
PRIME_LENGTH = TRAINING_CONFIG["prime_length"]
GENERATE_EVERY = TRAINING_CONFIG["generate_every"]
GENERATE_LENGTH = TRAINING_CONFIG["generate_length"]
SEQ_LEN = TRAINING_CONFIG["seq_len"]

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
        # Define dataset length such that one epoch covers the full data
        # Each sample is seq_len tokens, so we need data_size/seq_len samples to cover all
        self.samples_per_epoch = max(1, self.data.size(0) // self.seq_len)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
        # Random sampling from anywhere in the data
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
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
        use_lstm=False,
        enable_conv=False,
        dropout=0.0
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
            use_lstm=use_lstm,
            enable_conv=enable_conv,
            dropout=dropout
        )
        # For tracking tokens per second
        self.total_tokens_processed = 0
        self.start_time = None  # Will be set on first training step
        self.register_buffer('global_tokens', torch.tensor(0, dtype=torch.long))
        
    def forward(self, x, prev_hiddens=None):
        return self.model(x, return_loss=False, return_prev_hiddens=True, prev_hiddens=prev_hiddens)
    
    def training_step(self, batch, batch_idx, hiddens=None):
        # Initialize start_time on first training step
        if self.start_time is None:
            self.start_time = time.time()
            if self.global_rank == 0:
                print(f"Starting tokens/s timing at step {batch_idx}")
        
        # Debug optimizer type and parameters every 100 steps
        if batch_idx % 100 == 0 and self.global_rank == 0:
            if hasattr(self.trainer, 'optimizers'):
                opt = self.trainer.optimizers[0] if self.trainer.optimizers else None
                print(f"Step {batch_idx} - Optimizer type: {type(opt).__name__}")
                
                # Check if params are on the expected devices
                if opt is not None:
                    param_devices = set()
                    for param_group in opt.param_groups:
                        for param in param_group['params']:
                            if param.device not in param_devices:
                                param_devices.add(param.device)
                    print(f"Parameter devices: {param_devices}")
                    
        loss = self.model(batch, return_loss=True)
        # Log train_loss for display in progress bar (on_step=True) but use a simpler name
        self.log('train_loss', loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=False)
        # Also log for epoch aggregation without showing in progress bar
        self.log('train_loss_epoch', loss, prog_bar=False, sync_dist=True, on_step=False, on_epoch=True)
        
        # Update tokens processed count - local to this process
        tokens_in_batch = batch.numel()
        self.total_tokens_processed += tokens_in_batch
        
        # Track tokens across all processes
        if self.trainer.world_size > 1:
            # Convert to tensor for all_reduce
            batch_tokens = torch.tensor(tokens_in_batch, device=self.device)
            # Sum across all processes
            torch.distributed.all_reduce(batch_tokens, op=torch.distributed.ReduceOp.SUM)
            # Update global counter
            self.global_tokens += batch_tokens
            global_tokens_processed = self.global_tokens.item()
        else:
            self.global_tokens += tokens_in_batch
            global_tokens_processed = self.global_tokens.item()
        
        # Only display on rank 0 to avoid duplicate output
        if self.global_rank == 0:
            elapsed = time.time() - self.start_time
            global_tokens_per_sec = global_tokens_processed / elapsed if elapsed > 0 else 0
            
            # Just log the raw value - the progress bar will format it
            self.log('toks/s', global_tokens_per_sec / 1000, prog_bar=True)  # Convert to thousands for cleaner display
        
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        loss = self.model(batch, return_loss=True)
        # Calculate bits per byte (bpb)
        bpb = loss / math.log(2)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log('bpb', bpb, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)  # bits per byte
        return {"val_loss": loss, "bpb": bpb}
    
    def configure_optimizers(self):
        # Check if using DeepSpeed strategy
        using_deepspeed = isinstance(self.trainer.strategy, DeepSpeedStrategy) if self.trainer is not None else False
        
        if using_deepspeed and DEEPSPEED_OPTIMIZERS_AVAILABLE:
            # Try to get DeepSpeed config information safely
            zero_stage = 0
            offload_optimizer = False
            
            try:
                # Different DeepSpeed versions may have different attribute structures
                if hasattr(self.trainer.strategy, 'config'):
                    ds_config = self.trainer.strategy.config
                    if isinstance(ds_config, dict):
                        # Config is a dictionary
                        zero_stage = ds_config.get("zero_optimization", {}).get("stage", 0)
                        offload_optimizer = ds_config.get("zero_optimization", {}).get("offload_optimizer", False)
                    else:
                        # Config might be an object with attributes
                        zero_stage = getattr(ds_config, "zero_stage", 0)
                        offload_optimizer = getattr(ds_config, "offload_optimizer", False)
            except (AttributeError, KeyError) as e:
                print(f"Warning: Error accessing DeepSpeed config: {e}")
                
            # When using ZeRO-3 with CPU offloading, use DeepSpeedCPUAdam
            if zero_stage == 3 and offload_optimizer:
                print("Using DeepSpeedCPUAdam optimizer for ZeRO-3 with CPU offloading")
                return DeepSpeedCPUAdam(self.parameters(), lr=self.learning_rate)
            else:
                # Otherwise use DeepSpeed's FusedAdam for better performance
                print("Using DeepSpeed FusedAdam optimizer")
                return FusedAdam(self.parameters(), lr=self.learning_rate)
        else:
            # For regular training or when DeepSpeed optimizers aren't available
            print("Using standard PyTorch Adam optimizer")
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

# Custom progress bar that formats token/s more cleanly
class TokensPerSecFormatter(TQDMProgressBar):
    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        # Format tokens/s if present
        if 'toks/s' in items:
            # Will be in thousands already from our conversion above
            items['toks/s'] = f"{items['toks/s']:.2f}k/s"
        return items

# Training metrics logger callback
class MetricsLoggerCallback(pl.Callback):
    def __init__(self, log_path):
        super().__init__()
        self.log_path = log_path
        self.start_time = None
        # Track the last validation loss for reporting during training steps
        self.last_val_loss = None
        self.last_val_bpb = None
        # Don't create the file here - it will be created before the trainer is instantiated
        
    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Log every 10 steps but only on rank 0 to avoid duplicates in distributed training
        if trainer.is_global_zero and (trainer.global_step % 10 == 0):
            self._log_metrics(trainer, pl_module)
    
    def on_validation_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            self._log_metrics(trainer, pl_module, is_validation=True)
    
    def _log_metrics(self, trainer, pl_module, is_validation=False):
        # Only log metrics on the main process
        if not trainer.is_global_zero:
            return
            
        elapsed = time.time() - self.start_time
        global_tokens = pl_module.global_tokens.item()
        tokens_per_sec = global_tokens / elapsed if elapsed > 0 else 0
        
        # Get the loss values from callback metrics
        train_loss = trainer.callback_metrics.get('train_loss', torch.tensor(0.0)).item()
        
        # Get validation loss - if present, update our stored value
        val_loss = trainer.callback_metrics.get('val_loss')
        if val_loss is not None and is_validation:
            val_loss_value = val_loss.item()
            self.last_val_loss = val_loss_value
        else:
            # Use last known value or NA if we've never seen one
            val_loss_value = self.last_val_loss if self.last_val_loss is not None else "NA"
        
        # Get bits per byte if available - similar approach
        val_bpb = trainer.callback_metrics.get('bpb')  # bits per byte
        if val_bpb is not None and is_validation:
            val_bpb_value = val_bpb.item()
            self.last_val_bpb = val_bpb_value
        else:
            # Use last known value or NA if we've never seen one
            val_bpb_value = self.last_val_bpb if self.last_val_bpb is not None else "NA"
        
        try:
            with open(self.log_path, 'a') as f:
                # Write tab-separated values directly
                values = [
                    str(trainer.global_step),
                    str(trainer.current_epoch),
                    f"{elapsed:.2f}",
                    str(global_tokens),
                    f"{tokens_per_sec:.2f}",
                    f"{train_loss:.6f}",
                    str(val_loss_value),
                    str(val_bpb_value),
                    str(pl_module.learning_rate),
                    str(BATCH_SIZE),
                    str(GRAD_ACCUM_EVERY),
                    str(SEQ_LEN)
                ]
                f.write('\t'.join(values) + '\n')
        except (FileNotFoundError, PermissionError):
            # Silently fail if we can't write to the file
            # This can happen during distributed training
            pass



def parse_gpu_ids(gpu_spec):
    """
    Parse a GPU specification string into a list of GPU ids.
    Examples: 
      "0,1,2,3" -> [0, 1, 2, 3]
      "0-3" -> [0, 1, 2, 3]
      "0,2-4,7" -> [0, 2, 3, 4, 7]
    """
    if not gpu_spec:
        return None
        
    gpu_ids = []
    parts = gpu_spec.split(',')
    
    for part in parts:
        if '-' in part:
            # Handle range like "0-3"
            start, end = map(int, part.split('-'))
            gpu_ids.extend(range(start, end + 1))
        else:
            # Handle single number
            gpu_ids.append(int(part))
            
    return sorted(list(set(gpu_ids)))  # Remove duplicates and sort

def parse_size_with_suffix(size_str):
    """
    Parse a string with optional k, m, g suffix into a number.
    Examples:
      "1k" -> 1024
      "100k" -> 102400 (100*1024)
      "2m" -> 2097152 (2*1024*1024)
      "3g" -> 3221225472 (3*1024*1024*1024)
      "42" -> 42 (no suffix, unchanged)
    """
    if not isinstance(size_str, str):
        return size_str
        
    pattern = r'^(\d+(?:\.\d+)?)([kmg])?$'
    match = re.match(pattern, size_str.lower())
    if not match:
        try:
            return float(size_str)
        except ValueError:
            raise ValueError(f"Invalid size format: {size_str}")
            
    value, suffix = match.groups()
    value = float(value)
    
    if suffix == 'k':
        return value * 1024
    elif suffix == 'm':
        return value * 1024 * 1024
    elif suffix == 'g':
        return value * 1024 * 1024 * 1024
    else:
        return value

def create_deepspeed_config(zero_stage, bf16, offload_optimizer, offload_parameters, learning_rate):
    """Create DeepSpeed configuration based on user options"""
    # Calculate world size for correct train_batch_size
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    config = {
        "train_micro_batch_size_per_gpu": BATCH_SIZE,
        "steps_per_print": 100,
        "zero_optimization": {
            "stage": zero_stage,
            "contiguous_gradients": True,
            "overlap_comm": True
        },
        "fp16": {
            "enabled": not bf16,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "bf16": {
            "enabled": bf16
        },
        # Remove optimizer config - we'll handle this in configure_optimizers()
        "zero_allow_untested_optimizer": True,
        "wall_clock_breakdown": False
    }
    
    # Add CPU offloading if requested (for ZeRO-2 and ZeRO-3)
    if zero_stage >= 2 and offload_optimizer:
        config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True
        }
        
    # Parameter offloading only works with ZeRO-3
    if zero_stage == 3 and offload_parameters:
        config["zero_optimization"]["offload_param"] = {
            "device": "cpu",
            "pin_memory": True
        }
        
    return config

def round_to_multiple(n, multiple=32):
    """Round a number to the nearest multiple of a given value."""
    return multiple * round(n / multiple)

def solve_for_dimension(target_params, depth, vocab_size=256, ff_mult=4, expansion=1.5):
    """
    Solve for the dimension that will give approximately the target parameter count
    given the other model parameters.
    """
    from math import sqrt
    
    # Simplified model parameter count formula
    # params ≈ 2 * dim * vocab_size + depth * (4 * dim * dim * expansion + 2 * dim * dim * ff_mult)
    
    # Rearranging to solve for dim:
    # params = 2 * dim * vocab_size + depth * dim * dim * (4 * expansion + 2 * ff_mult)
    # params = 2 * dim * vocab_size + depth * dim * dim * factor
    # params = 2 * dim * vocab_size + depth * factor * dim^2
    
    factor = 4 * expansion + 2 * ff_mult
    
    # This is a quadratic equation of the form: a*dim^2 + b*dim - target_params = 0
    a = depth * factor
    b = 2 * vocab_size
    c = -target_params
    
    # Quadratic formula: dim = (-b + sqrt(b^2 - 4*a*c)) / (2*a)
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        raise ValueError("No solution exists for the given target parameter count")
    
    dim = (-b + sqrt(discriminant)) / (2*a)
    return round_to_multiple(dim)

def solve_for_depth(target_params, dim, vocab_size=256, ff_mult=4, expansion=1.5):
    """
    Solve for the depth that will give approximately the target parameter count
    given the other model parameters.
    """
    # params ≈ 2 * dim * vocab_size + depth * (4 * dim * dim * expansion + 2 * dim * dim * ff_mult)
    
    # Rearranging to solve for depth:
    # params = 2 * dim * vocab_size + depth * dim * dim * (4 * expansion + 2 * ff_mult)
    # params - 2 * dim * vocab_size = depth * dim * dim * factor
    # depth = (params - 2 * dim * vocab_size) / (dim * dim * factor)
    
    embed_params = 2 * dim * vocab_size
    factor = 4 * expansion + 2 * ff_mult
    layer_params = dim * dim * factor
    
    depth = (target_params - embed_params) / layer_params
    return max(1, round(depth))  # Ensure at least 1 layer

# Global run timestamp to ensure consistent directory naming
RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def main():
    # Make variables global
    global SEQ_LEN
    global BATCH_SIZE
    global GRAD_ACCUM_EVERY
    global LEARNING_RATE
    global NUM_BATCHES
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a minLM model with PyTorch Lightning")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to the training data file (e.g., 'data/enwik8.gz')")
    parser.add_argument("--gpus", type=str, default=None, 
                        help="Comma-separated list or range of GPU IDs to use (e.g., '0,1,2' or '0-2' or '0,2-4')")
    
    # Model architecture arguments
    parser.add_argument("--dim", type=str, default=None,
                        help="Model hidden dimension (default: 512, will be rounded to multiple of 32). Can use k/m/g suffix.")
    parser.add_argument("--depth", type=int, default=None,
                        help="Number of model layers (default: 6).")
    parser.add_argument("--params", type=str, default=None,
                        help="Target parameter count (e.g., 15m for 15M params). Can use k/m/g suffix.")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=None,
                        help=f"Batch size per GPU (default: {TRAINING_CONFIG['batch_size']})")
    parser.add_argument("--grad_accum", type=int, default=None,
                        help=f"Gradient accumulation steps (default: {TRAINING_CONFIG['grad_accum_every']})")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help=f"Learning rate (default: {TRAINING_CONFIG['learning_rate']})")
    parser.add_argument("--seq_len", type=str, default=None,
                        help=f"Sequence length for training (default: {TRAINING_CONFIG['seq_len']}). Can use k/m/g suffix (e.g., '1k'=1024, '8k'=8192).")
    parser.add_argument("--steps", type=str, default=None,
                        help=f"Total training steps (default: {TRAINING_CONFIG['num_batches']}). Can use k/m/g suffix (e.g., '1k'=1024, '100k'=100,000).")
    parser.add_argument("--output", type=str, default=None,
                        help="Directory to save checkpoints (default: auto-generated name with params and timestamp)")
    parser.add_argument("--use-f32", dest="use_bf16", action="store_false", default=True,
                        help="Use FP32 precision instead of BF16 (default: BF16)")
                        
    # DeepSpeed arguments
    parser.add_argument("--deepspeed", action="store_true",
                        help="Enable DeepSpeed for training (default: False)")
    parser.add_argument("--zero_stage", type=int, default=2, choices=[0, 1, 2, 3],
                        help="ZeRO optimization stage (0-3, higher = more memory efficient but slower)")
    parser.add_argument("--offload_optimizer", action="store_true",
                        help="Offload optimizer states to CPU (reduces GPU memory, but slower)")
    parser.add_argument("--offload_parameters", action="store_true",
                        help="Offload parameters to CPU (for ZeRO-3, reduces GPU memory but slower)")
    parser.add_argument("--deepspeed_config", type=str, default=None,
                        help="Path to DeepSpeed JSON config file (overrides other DeepSpeed args)")
    
    args = parser.parse_args()
    
    # Parse GPU IDs
    gpu_ids = parse_gpu_ids(args.gpus)
    
    pl.seed_everything(42)
    torch.set_float32_matmul_precision('medium')

    print(f"CUDA AVAILABLE: {torch.cuda.is_available()}")
    print(f"GPU COUNT: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            
    if gpu_ids:
        print(f"Using GPUs: {gpu_ids}")
    
    # Helper function to detect if a file is gzipped
    def is_gzip_file(filepath):
        with open(filepath, 'rb') as test_f:
            return test_f.read(2) == b'\x1f\x8b'
    
    # Load and prepare data
    print(f"Loading data from {args.data}...")
    
    if is_gzip_file(args.data):
        print("Detected gzip format, loading into memory...")
        with gzip.open(args.data) as file:
            data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
            np_train, np_valid = np.split(data, [int(90e6)])
            data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)
    else:
        print("Detected raw format, using memory mapping...")
        # Get file size
        file_size = os.path.getsize(args.data)
        # Map the file into memory
        with open(args.data, 'r+b') as f:
            mm = mmap.mmap(f.fileno(), 0)
            # Create a numpy array using the memory map
            data = np.frombuffer(mm, dtype=np.uint8, count=min(int(95e6), file_size))
            # Split data (but don't copy it)
            train_size = min(int(90e6), len(data))
            np_train, np_valid = data[:train_size], data[train_size:min(int(95e6), len(data))]
            # Convert to PyTorch tensors
            data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)
    
    print(f"Data loaded - Train: {data_train.shape}, Val: {data_val.shape}")

    # Create datasets and dataloaders
    print(f"Creating datasets and dataloaders with sequence length: {SEQ_LEN}...")
    train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
    val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
    
    # Calculate optimal number of workers (typically CPU count)
    num_workers = min(31, os.cpu_count() or 4)
    print(f"Using {num_workers} dataloader workers")
    
    # Set shuffle=True for training but False for validation as recommended
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=num_workers, 
        pin_memory=True, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=num_workers, 
        pin_memory=True, 
        shuffle=False  # Disable shuffling for validation as recommended
    )
    
    # Parse numerical arguments with potential suffixes
    dim_value = parse_size_with_suffix(args.dim) if args.dim is not None else None
    depth_value = args.depth  # Already an int, no parsing needed
    params_value = parse_size_with_suffix(args.params) if args.params is not None else None
    seq_len_value = int(parse_size_with_suffix(args.seq_len)) if args.seq_len is not None else SEQ_LEN
    batch_size_value = args.batch_size if args.batch_size is not None else BATCH_SIZE
    grad_accum_value = args.grad_accum if args.grad_accum is not None else GRAD_ACCUM_EVERY
    learning_rate_value = args.learning_rate if args.learning_rate is not None else LEARNING_RATE
    
    # Get user-requested total steps
    total_requested_steps = int(parse_size_with_suffix(args.steps)) if args.steps is not None else NUM_BATCHES
    
    # Override config values with command line arguments if provided
    SEQ_LEN = seq_len_value
    BATCH_SIZE = batch_size_value
    GRAD_ACCUM_EVERY = grad_accum_value
    LEARNING_RATE = learning_rate_value
    
    # Get world size for distributed training
    world_size = torch.cuda.device_count() if torch.cuda.is_available() and gpu_ids else 1
    
    # Adjust NUM_BATCHES for distributed training - each GPU will do this many steps
    # So the total steps across all GPUs will equal the requested total
    NUM_BATCHES = total_requested_steps // world_size
    if NUM_BATCHES == 0:
        NUM_BATCHES = 1  # Ensure at least 1 step per GPU
    
    # Configure model architecture based on command line arguments
    if params_value is not None:
        # Get target parameter count
        target_params = params_value
        
        if dim_value is not None and depth_value is None:
            # If dimension is specified but not depth, solve for depth
            dim = round_to_multiple(dim_value)
            depth = solve_for_depth(
                target_params, 
                dim, 
                MODEL_CONFIG["num_tokens"], 
                MODEL_CONFIG["ff_mult"], 
                MODEL_CONFIG["expansion"]
            )
            print(f"Target params: {args.params}M, Dimension: {dim}, Calculated depth: {depth}")
        elif dim_value is None and depth_value is not None:
            # If depth is specified but not dimension, solve for dimension
            depth = depth_value
            dim = solve_for_dimension(
                target_params, 
                depth, 
                MODEL_CONFIG["num_tokens"], 
                MODEL_CONFIG["ff_mult"], 
                MODEL_CONFIG["expansion"]
            )
            print(f"Target params: {args.params}M, Calculated dimension: {dim}, Depth: {depth}")
        else:
            # If neither is specified or both are specified, adjust dimension
            if dim_value is not None and depth_value is not None:
                print(f"Warning: Both dimension and depth specified with target params. Ignoring target params.")
                dim = round_to_multiple(dim_value)
                depth = depth_value
            else:
                # Scale both depth and dimension according to scaling laws
                # Use a logarithmic scaling for depth to get more balanced architecture
                # Depth scales approximately with the cube root of parameter count
                # Start with depth=6 for 15M params, then scale appropriately
                
                base_params = 15 * 1024 * 1024  # 15M params as the reference point
                base_depth = 6  # Reference depth for 15M params
                
                # Calculate a balanced depth based on parameter count scaling
                if target_params >= base_params:
                    # Scale up from the base configuration
                    scaling_factor = (target_params / base_params) ** (1/3)  # Cube root scaling
                    depth = max(base_depth, round(base_depth * scaling_factor))
                else:
                    # Scale down from the base configuration, but more conservatively
                    scaling_factor = (target_params / base_params) ** (1/4)  # Fourth root scaling for small models
                    depth = max(2, round(base_depth * scaling_factor))  # Minimum depth of 2
                
                # Now solve for dimension with the calculated depth
                dim = solve_for_dimension(
                    target_params, 
                    depth, 
                    MODEL_CONFIG["num_tokens"], 
                    MODEL_CONFIG["ff_mult"], 
                    MODEL_CONFIG["expansion"]
                )
                print(f"Target params: {args.params}, Calculated balanced scaling - Dimension: {dim}, Depth: {depth}")
    else:
        # No target params specified, use explicit values or defaults
        dim = round_to_multiple(dim_value) if dim_value is not None else MODEL_CONFIG["dim"]
        depth = depth_value if depth_value is not None else MODEL_CONFIG["depth"]
        
    # Update model config with the calculated values
    MODEL_CONFIG["dim"] = dim
    MODEL_CONFIG["depth"] = depth
    
    # Set up model
    print(f"Creating model with dimension={dim}, depth={depth}...")
    model = LightningMinLM(
        num_tokens=MODEL_CONFIG["num_tokens"],
        dim=MODEL_CONFIG["dim"],
        depth=MODEL_CONFIG["depth"],
        ff_mult=MODEL_CONFIG["ff_mult"],
        expansion=MODEL_CONFIG["expansion"],
        conv_kernel_size=MODEL_CONFIG["conv_kernel_size"],
        learning_rate=LEARNING_RATE,
        use_lstm=MODEL_CONFIG["use_lstm"],
        enable_conv=MODEL_CONFIG["enable_conv"],
        dropout=MODEL_CONFIG["dropout"]
    )
    
    # Print model parameter count to verify size
    expected_params = calculate_model_size(MODEL_CONFIG)
    actual_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Information:")
    print(f"Dimension: {MODEL_CONFIG['dim']}")
    print(f"Depth: {MODEL_CONFIG['depth']}")
    print(f"Expected parameters: {expected_params:,}")
    print(f"Actual parameters: {actual_params:,}")
    
    # Verify the model configuration
    first_layer_shape = model.model.layers[0][2].to_hidden_and_gate.weight.shape
    print(f"First layer weight shape: {first_layer_shape}")
    
    # Check if we're using distributed training
    using_distributed = torch.cuda.device_count() > 1 and gpu_ids is not None and len(gpu_ids) > 1
    
    # Determine if this is the main process - for Lightning, we're the main process on the first GPU or in single GPU mode
    # We don't manually initialize the process group - Lightning will handle this
    is_main_process = (not using_distributed) or (gpu_ids is None) or (0 in gpu_ids and gpu_ids.index(0) == 0)

    # Only the main process generates the directory name
    if is_main_process:
        if args.output:
            checkpoint_dir = args.output
        else:
            # Generate a unique name based on parameters and timestamp
            params_str = f"{actual_params/1000000:.1f}M" if actual_params >= 1000000 else f"{actual_params/1000:.1f}K"
            checkpoint_dir = f"gruf_{params_str}_{RUN_TIMESTAMP}"
        
        # Create the directory
        print(f"Creating checkpoint directory: {checkpoint_dir}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model configuration for easy reloading
        config = {
            **MODEL_CONFIG, 
            **{
                "learning_rate": LEARNING_RATE, 
                "seq_len": SEQ_LEN, 
                "batch_size": BATCH_SIZE,
                "num_batches": NUM_BATCHES,
                "total_steps": total_requested_steps,
                "use_bf16": args.use_bf16
            }
        }
        
        # Add DeepSpeed configuration if enabled
        if args.deepspeed:
            config["deepspeed_enabled"] = True
            config["zero_stage"] = args.zero_stage
            config["offload_optimizer"] = args.offload_optimizer
            config["offload_parameters"] = args.offload_parameters
        with open(os.path.join(checkpoint_dir, "model_config.json"), "w") as f:
            json.dump(config, f, indent=2)
            
        # Create the metrics TSV file
        metrics_log_path = os.path.join(checkpoint_dir, "training_metrics.tsv")
        with open(metrics_log_path, 'w') as f:
            # Write tab-separated header
            header = [
                "step", "epoch", "time", "tokens_processed", 
                "tokens_per_sec", "train_loss", "val_loss", "bpb",
                "learning_rate", "batch_size", "grad_accum", "seq_len"
            ]
            f.write('\t'.join(header) + '\n')
    else:
        # Non-main processes start with an empty directory name
        checkpoint_dir = ""
    
    # In Lightning, each process will create its own copy of the directory
    # The main process creates it first, others will see it already exists
    
    # Set up callbacks - all processes use the same directory
    # Best models based on validation loss
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="val_loss",
        filename="minlm-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min"
    )

    # Periodic backups every 1000 steps
    backup_checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="minlm-backup-step-{step}",
        every_n_train_steps=1000,
        save_top_k=2,
        save_last=True,  # Also save the latest model as 'last.ckpt'
        monitor="step"    # Monitor the step count for save_top_k
    )
    
    progress_bar = TokensPerSecFormatter()
    
    
    # Create logger
    csv_logger = CSVLogger(
        save_dir="logs",
        name="min_lm_training",
        flush_logs_every_n_steps=10,
        version=RUN_TIMESTAMP  # Use the global timestamp
    )
    
    # Setup the appropriate distributed training strategy (DeepSpeed or DDP)
    if args.deepspeed:
        print(f"Using DeepSpeed with ZeRO Stage-{args.zero_stage}")
        if args.offload_optimizer:
            print(f"Offloading optimizer states to CPU")
        if args.offload_parameters and args.zero_stage == 3:
            print(f"Offloading parameters to CPU")
        
        # Use a JSON config file if provided
        if args.deepspeed_config and os.path.exists(args.deepspeed_config):
            print(f"Using DeepSpeed config from: {args.deepspeed_config}")
            strategy = DeepSpeedStrategy(config=args.deepspeed_config, zero_allow_untested_optimizer=True)
        else:
            # Create DeepSpeed config from arguments
            ds_config = create_deepspeed_config(
                args.zero_stage, 
                args.use_bf16, 
                args.offload_optimizer,
                args.offload_parameters,
                LEARNING_RATE
            )
            # Let our configure_optimizers handle the optimizer setup
            strategy = DeepSpeedStrategy(
                config=ds_config,
                zero_allow_untested_optimizer=True
            )
    else:
        # Regular DDP strategy with NCCL backend for better GPU performance
        strategy = DDPStrategy(
            process_group_backend="nccl",
            find_unused_parameters=False,
            static_graph=False  # Setting to False to avoid DDP autograd hooks issue
        ) if torch.cuda.device_count() > 1 else "auto"

    # Calculate number of epochs needed to reach NUM_BATCHES
    total_samples = len(train_dataset)
    steps_per_epoch_global = total_samples // BATCH_SIZE
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    steps_per_epoch_per_gpu = steps_per_epoch_global // world_size
    if steps_per_epoch_per_gpu == 0:
        steps_per_epoch_per_gpu = 1  # Avoid division by zero
    max_epochs = math.ceil(NUM_BATCHES / steps_per_epoch_per_gpu)
    
    # Calculate tokens per epoch
    tokens_per_sample = SEQ_LEN
    tokens_per_epoch_per_gpu = steps_per_epoch_per_gpu * BATCH_SIZE * tokens_per_sample
    tokens_per_epoch_total = tokens_per_epoch_per_gpu * world_size
    effective_batch_size = BATCH_SIZE * world_size * GRAD_ACCUM_EVERY
    
    print(f"\n--- Training Configuration ---")
    print(f"Total dataset size: {data_train.shape[0]:,} characters")
    print(f"Sequence length: {SEQ_LEN} tokens")
    print(f"Total samples: {total_samples:,} (dataset size / sequence length)")
    print(f"Running on {world_size} GPUs")
    print(f"Batch size per GPU: {BATCH_SIZE}")
    print(f"Global batch size: {BATCH_SIZE * world_size}")
    print(f"Gradient accumulation: {GRAD_ACCUM_EVERY}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Precision: {'BF16' if args.use_bf16 else 'FP32'}")
    print(f"Steps per epoch per GPU: {steps_per_epoch_per_gpu}")
    print(f"Steps per epoch total: {steps_per_epoch_global}")
    print(f"Tokens per epoch per GPU: {tokens_per_epoch_per_gpu:,}")
    print(f"Tokens per epoch total: {tokens_per_epoch_total:,}")
    print(f"Training for {max_epochs} epochs to reach approximately {NUM_BATCHES} steps")
    print(f"-----------------------------\n")

    # Initialize metrics logger with the path that should now exist
    # All processes now have the same checkpoint directory name
    metrics_log_path = os.path.join(checkpoint_dir, "training_metrics.tsv")
    print(f"Using checkpoint directory: {checkpoint_dir}")
    
    metrics_logger = MetricsLoggerCallback(metrics_log_path)
    
    # Create trainer with precision settings based on args
    precision = "bf16-mixed" if args.use_bf16 else 32
    
    trainer = pl.Trainer(
        max_steps=NUM_BATCHES,  # Each GPU will do this many steps
        accumulate_grad_batches=GRAD_ACCUM_EVERY,
        accelerator="gpu",
        devices=gpu_ids if gpu_ids else "auto",
        strategy=strategy,  # Use the strategy variable we set above (DeepSpeed or DDP)
        gradient_clip_val=0.5,
        callbacks=[checkpoint_callback, backup_checkpoint_callback, progress_bar, metrics_logger],
        val_check_interval=VALIDATE_EVERY,
        logger=csv_logger,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        limit_val_batches=4,
        max_epochs=max_epochs,  # Set max_epochs to ensure we can track progress
        check_val_every_n_epoch=None,  # Still validate based on steps, not epochs
        precision=precision,  # Use BF16 if requested
    )

    print(f"Starting training with {torch.cuda.device_count()} GPUs")
    print(f"Config: bs={BATCH_SIZE}, grad_accum={GRAD_ACCUM_EVERY}, lr={LEARNING_RATE}, seq_len={SEQ_LEN}")
    print(f"Will run for {NUM_BATCHES:,} steps per GPU ({total_requested_steps:,} total steps across {world_size} GPUs)")
    
    # Start training
    print("Starting training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    # Helper function to print detailed model information
    def print_model_details(model):
        """Print detailed information about model parameters"""
        total_params = sum(p.numel() for p in model.parameters())
        expected_params = calculate_model_size()
        
        print(f"\nDetailed Model Information:")
        print(f"Expected parameters: {expected_params:,}")
        print(f"Actual parameters: {total_params:,}")
        print(f"Difference: {total_params - expected_params:,}")
        
        # Group parameters by layer type
        param_groups = {}
        for name, param in model.named_parameters():
            # Extract the layer type (embedding, RNN, FF, etc.)
            if "token_emb" in name:
                group = "embedding"
            elif "mingru" in name or "to_hidden" in name or "to_hidden_and_gate" in name:
                group = "rnn"
            elif "ff" in name:
                group = "feedforward"
            elif "norm" in name:
                group = "normalization"
            elif "to_logits" in name:
                group = "output"
            else:
                group = "other"
                
            if group not in param_groups:
                param_groups[group] = 0
            param_groups[group] += param.numel()
        
        # Print parameter counts by group
        for group, count in param_groups.items():
            print(f"- {group}: {count:,} parameters ({count/total_params*100:.1f}%)")
    
    # Print final stats and save final model
    if trainer.is_global_zero:
        print("\nTraining completed.")
        print_model_details(model)
        print(f"Total steps: {trainer.global_step}")
        print(f"Total tokens: {model.global_tokens.item()}")
        elapsed = time.time() - model.start_time
        tokens_per_sec = model.global_tokens.item() / elapsed if elapsed > 0 else 0
        print(f"Average tokens/sec: {tokens_per_sec:.2f}")
        
        # Save final model explicitly
        final_model_path = os.path.join(checkpoint_dir, f"minlm-final-step-{trainer.global_step}.pt")
        torch.save({
            'step': trainer.global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizers[0].state_dict() if trainer.optimizers else None,
            'global_tokens': model.global_tokens.item(),
            'training_time': elapsed
        }, final_model_path)
        print(f"Final model saved to: {final_model_path}")
        
        # Save a helper script for loading the model
        load_script_path = os.path.join(checkpoint_dir, "load_model.py")
        with open(load_script_path, "w") as f:
            f.write('''
import torch
import json
import os
from minGRU_pytorch.minLM import minLM

def load_model(checkpoint_path, config_path=None):
    """
    Load a trained minLM model from checkpoint
    
    Args:
        checkpoint_path: Path to the model checkpoint
        config_path: Path to the model config file (optional)
    
    Returns:
        Loaded model
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load config if provided, otherwise use defaults
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Default config
        config = {
            "num_tokens": 256,
            "dim": 512,
            "depth": 6,
            "ff_mult": 4,
            "expansion": 1.5,
            "conv_kernel_size": 3,
            "use_lstm": False
        }
    
    # Create model with the same configuration
    model = minLM(
        num_tokens=config["num_tokens"],
        dim=config["dim"],
        depth=config["depth"],
        ff_mult=config["ff_mult"],
        expansion=config.get("expansion", 1.5),
        conv_kernel_size=config.get("conv_kernel_size", 3),
        use_lstm=config.get("use_lstm", False)
    )
    
    # Load model weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model

if __name__ == "__main__":
    # Example usage
    model = load_model(
        checkpoint_path="minlm-final-step-100000.pt",
        config_path="model_config.json"
    )
    print("Model loaded successfully!")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
''')
        print(f"Helper script for loading the model saved to: {load_script_path}")

if __name__ == "__main__":
    main()
