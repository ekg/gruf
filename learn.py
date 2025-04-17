import os
os.environ["NCCL_P2P_DISABLE"] = "1"

import gzip
import random
import numpy as np
import math
import time
import argparse
import re
import torch
import mmap
import datetime
import json
import shutil
from schedulefree import AdamWScheduleFree  # Import the Schedule-Free optimizer

# Make matplotlib optional
try:
    from matplotlib import pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plotting features will be disabled.")
from torch.utils.data import DataLoader, Dataset
import deepspeed
from tqdm import tqdm

# Set higher precision for float32 matrix multiplication
torch.set_float32_matmul_precision('high')

# Import the minLM model
from minGRU_pytorch.minLM import minLM

# Import configuration from config.py
from config import MODEL_CONFIG, TRAINING_CONFIG, calculate_model_size, get_parameter_count_str

# Set environment variables
os.environ["NCCL_DEBUG"] = "INFO"

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

# Text generation functions
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
    # Ensure prompt is a Long tensor
    if prompt.dtype != torch.long:
        prompt = prompt.long()
    prompt_seq_len, out = prompt.shape[-1], prompt.clone()
    sample_num_times = max(0, seq_len - prompt_seq_len)

    prev_hiddens = None

    for _ in range(sample_num_times):
        logits, next_prev_hiddens = net(out, return_prev_hiddens = True, prev_hiddens = prev_hiddens)
        logits = logits[:, -1]

        if hasattr(net, 'can_cache') and net.can_cache:
            prev_hiddens = next_prev_hiddens

        logits = top_k(logits, thres = filter_thres)
        sample = gumbel_sample(logits, temperature = temperature, dim = -1)
        
        # Ensure sample is Long type before concatenation
        sample = sample.long()
        out = torch.cat((out, sample), dim = -1)

    return out[..., prompt_seq_len:]

# Dataset classes
class MemoryMappedTextDataset(Dataset):
    def __init__(self, filepath, seq_len, offset=0, length=None, seed=42):
        super().__init__()
        self.filepath = filepath
        self.seq_len = seq_len
        self.offset = offset
        self.seed = seed
        
        # Get file size once
        self.file_size = os.path.getsize(filepath)
        
        # Set the effective length for this dataset
        if length is None:
            self.effective_length = self.file_size - offset
        else:
            self.effective_length = min(length, self.file_size - offset)
        
        # Calculate valid end position for random sampling
        # -seq_len-1 ensures we can always get seq_len+1 bytes
        self.valid_end = max(0, self.effective_length - seq_len - 1)
        
        # Open file and create memory map - will be initialized on first access
        self.file = None
        self.mm = None
        
        # Define length as number of possible starting positions
        self.samples_per_epoch = max(1, self.valid_end // seq_len)
    
    def _ensure_open(self):
        """Ensure the memory map is open, opening it if necessary"""
        if self.mm is None:
            self.file = open(self.filepath, 'r+b')
            self.mm = mmap.mmap(self.file.fileno(), 0)
    
    def __len__(self):
        return self.samples_per_epoch
    
    def __getitem__(self, index):
        self._ensure_open()
        
        # TESTING MODE: Always return the exact same sequence regardless of index
        # Use a fixed position based only on the seed (not the index)
        fixed_pos = self.offset
        
        # Directly read bytes from memory map without creating intermediate arrays
        self.mm.seek(fixed_pos)
        data = self.mm.read(self.seq_len + 1)  # +1 for the target
        
        # Convert to a writable buffer first, then to tensor
        writable_data = bytearray(data)
        tensor = torch.frombuffer(writable_data, dtype=torch.uint8).long()
        
        # Handle edge case if we didn't get enough data
        if tensor.size(0) < self.seq_len + 1:
            # Pad with zeros if needed
            padding = torch.zeros(self.seq_len + 1 - tensor.size(0), dtype=torch.long)
            tensor = torch.cat([tensor, padding])
        
        return tensor
    
    def __del__(self):
        # Clean up resources
        if hasattr(self, 'mm') and self.mm is not None:
            self.mm.close()
        if hasattr(self, 'file') and self.file is not None:
            self.file.close()

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len, seed=42):
        super().__init__()
        self.data = data
        self.seq_len = seq_len
        self.seed = seed
        # Define dataset length such that one epoch covers the full data
        # Each sample is seq_len tokens, so we need data_size/seq_len samples to cover all
        self.samples_per_epoch = max(1, self.data.size(0) // self.seq_len)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
        # TESTING MODE: Always return the exact same sequence regardless of index
        
        # Check if the dataset size is sufficient for the requested sequence length
        if self.data.size(0) <= self.seq_len:
            # Dataset is too small, return the full dataset padded if needed
            full_seq = self.data.clone().long()
            # If we need padding, add zeros at the end
            if full_seq.size(0) < self.seq_len + 1:
                padding = torch.zeros(self.seq_len + 1 - full_seq.size(0), dtype=torch.long)
                full_seq = torch.cat([full_seq, padding])
            return full_seq[:self.seq_len + 1]
        
        # Always use position 0 for absolute consistency
        fixed_start = 0
        
        # Always ensure we return a Long tensor
        full_seq = self.data[fixed_start : fixed_start + self.seq_len + 1].long()
        return full_seq  # DeepSpeed will handle device placement

# Trainer class
class MinLMTrainer:
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
        dropout=0.0,
        checkpoint_dir=None,
        world_size=1,
        global_rank=0,
        silent_mode=True,
        debug_gradients=False,
        checkpoint_every=100,
        permanent_save_interval=5000,
        args=None
    ):
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
        
        # Compile the model for better performance (if not disabled)
        if not (args is not None and getattr(args, 'no_compile', False)):
            self.model = torch.compile(self.model)
            if not silent_mode and global_rank == 0:
                print("Model compiled with torch.compile()")
        else:
            if not silent_mode and global_rank == 0:
                print("Model compilation disabled via --no_compile")
        # For tracking tokens per second
        self.total_tokens_processed = 0
        self.start_time = None
        self.global_tokens = torch.tensor(0, dtype=torch.long)
        
        # Gradient accumulation is handled by DeepSpeed
        self.grad_accum_steps = GRAD_ACCUM_EVERY
        
        # Directory for checkpoints
        self.checkpoint_dir = checkpoint_dir
        
        # Distributed training info
        self.world_size = world_size
        self.global_rank = global_rank
        
        # Store settings
        self.silent_mode = silent_mode
        self.debug_gradients = debug_gradients
        self.checkpoint_every = checkpoint_every
        self.permanent_save_interval = permanent_save_interval
        
        # Initialize metric tracking
        self.train_loss = 0.0
        self.val_loss = 0.0
        self.val_bpb = 0.0
        self.global_step = 0
        
        # Checkpoint tracking
        self.best_val_loss = float('inf')
        self.best_val_bpb = float('inf')
        self.best_checkpoints = []  # List of (path, val_loss) tuples to track top_k checkpoints
        self.recent_checkpoints = []  # List of recent checkpoint paths
        self.save_top_k = 3  # Number of best checkpoints to keep

    def init_deepspeed(self, train_dataloader, args):
        """Initialize DeepSpeed engine"""
        # Store the greedylr_debug flag for later use
        self.greedylr_debug = args.greedylr_debug if hasattr(args, 'greedylr_debug') else False
        # Create DeepSpeed config
        if args.deepspeed_config and os.path.exists(args.deepspeed_config):
            # Load config from file if provided
            with open(args.deepspeed_config, 'r') as f:
                ds_config = json.load(f)
            if self.global_rank == 0:
                print(f"Using DeepSpeed config from: {args.deepspeed_config}")
        else:
            # Create config from arguments
            ds_config = self.create_deepspeed_config(
                args.zero_stage, 
                args.precision, 
                args.offload_optimizer,
                args.offload_parameters,
                self.learning_rate,
                args.gradient_clip,
                args.tensor_parallel_size,
                MODEL_CONFIG["depth"],
                args
            )
            
        # Create custom optimizer if using Schedule-Free
        if args.schedulefree:
            # Create Schedule-Free optimizer (no warmup needed)
            if self.global_rank == 0 and not self.silent_mode:
                print(f"Initializing Schedule-Free optimizer with lr={self.learning_rate}, beta={args.sf_beta}, weight_decay={args.sf_weight_decay}")
                
            optimizer = AdamWScheduleFree(
                self.model.parameters(),
                lr=self.learning_rate,
                betas=(args.sf_beta, 0.999),
                weight_decay=args.sf_weight_decay
            )
            # Keep reference to the Schedule-Free optimizer for train/eval mode
            self.sf_optimizer = optimizer
            # Save the ScheduleFree optimizer for mode switching
            self.using_schedulefree = True
            
            # Verify optimizer's learning rate
            actual_lr = optimizer.param_groups[0]['lr']
            if self.global_rank == 0 and not self.silent_mode:
                print(f"Using Schedule-Free optimizer with beta={args.sf_beta}, weight_decay={args.sf_weight_decay}")
                print(f"Schedule-Free initial LR: {actual_lr} (configured: {self.learning_rate})")
                    
            # Need to flag that we're using Schedule-Free to avoid scheduler conflicts
            self.using_schedulefree = True
        else:
            # Regular optimizer will be created by DeepSpeed
            optimizer = None
            self.sf_optimizer = None
            self.using_schedulefree = False
        
        # CRITICAL FIX FOR BF16 + ZERO
        if args.precision == "bf16":
            # Ensure bf16 section exists and has proper settings
            ds_config.setdefault("bf16", {})
            ds_config["bf16"]["enabled"] = True
            ds_config["bf16"]["accumulate_grads_in_fp32"] = True
            
            # Critical setting from Domino code
            setattr(self.model, 'accumulate_allreduce_grads_in_fp32', True)
            
            # Force all parameters to have this attribute
            for param in self.model.parameters():
                setattr(param, 'accumulate_grads_in_fp32', True)
            
            # Configure ZeRO-1 specific options
            if args.zero_stage == 1:
                ds_config.setdefault("zero_optimization", {})
                ds_config["zero_optimization"]["reduce_bucket_size"] = 2e8
                ds_config["zero_optimization"]["allgather_bucket_size"] = 2e8
            
            # Disable stage3_gather_16bit_weights_on_model_save for bf16
            if "zero_optimization" in ds_config and "stage" in ds_config["zero_optimization"] and ds_config["zero_optimization"]["stage"] == 3:
                ds_config["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = False
            
            # Set gradient clipping to default 0.5 (if not already overridden)
            # Or disable it when explicitly set to 0
            if args.gradient_clip is None:
                ds_config["gradient_clipping"] = 0.5
            elif args.gradient_clip == 0:
                ds_config["gradient_clipping"] = 0  # Explicitly disable gradient clipping
            else:
                ds_config["gradient_clipping"] = args.gradient_clip
                
            if self.global_rank == 0:
                print("Configured for BF16 training with FP32 gradient accumulation")
                print(f"accumulate_allreduce_grads_in_fp32: {hasattr(self.model, 'accumulate_allreduce_grads_in_fp32')}")
        
        # Log the precision config
        if self.global_rank == 0 and not self.silent_mode:
            print(f"\nUsing precision: {args.precision}")
            if args.precision == "fp32":
                print("Explicitly configured for FP32 training")
                if "fp32" in ds_config:
                    print(f"FP32 config: {ds_config['fp32']}")
    
        # Initialize DeepSpeed engine
        if args.schedulefree:
            # For Schedule-Free, explicitly disable scheduler in config
            if "scheduler" in ds_config:
                del ds_config["scheduler"]
                if self.global_rank == 0 and not self.silent_mode:
                    print("Removed scheduler from DeepSpeed config for Schedule-Free compatibility")
                    
            model_engine, optimizer, _, _ = deepspeed.initialize(
                model=self.model,
                optimizer=optimizer,  # Pass the Schedule-Free optimizer
                config=ds_config
            )
        else:
            model_engine, optimizer, _, _ = deepspeed.initialize(
                model=self.model,
                model_parameters=self.model.parameters(),
                config=ds_config
            )
    
        self.model = model_engine
        self.optimizer = optimizer
    
        # Completely disable DeepSpeed's scheduler
        if hasattr(self.model, 'lr_scheduler'):
            self.model.lr_scheduler = None
            if self.global_rank == 0 and not self.silent_mode:
                print("Completely disabled DeepSpeed's native scheduler")
    
        # We no longer need to monkey patch since we now use Schedule-Free directly
        
        # No custom LR scheduler needed with Schedule-Free
        self.lr_scheduler = None
        if self.using_schedulefree and self.global_rank == 0 and not self.silent_mode:
            print("Schedule-Free optimizer is active - using built-in adaptive behavior instead of LR scheduler")
        
        # Print device info if main process
        if self.global_rank == 0:
            param_count = sum(1 for p in self.model.parameters())
            requires_grad = sum(1 for p in self.model.parameters() if p.requires_grad)
            print(f"DeepSpeed initialized with {param_count} parameters, {requires_grad} require grad")
            
            # Debug: print info about model's dtype
            for name, param in list(self.model.named_parameters())[:3]:  # Just the first few
                print(f"Parameter {name} dtype: {param.dtype}")
            
            # Debug BF16 settings
            if args.precision == "bf16":
                has_fp32_accum = hasattr(self.model, 'accumulate_allreduce_grads_in_fp32') and self.model.accumulate_allreduce_grads_in_fp32
                print(f"FP32 gradient accumulation enabled: {has_fp32_accum}")
    
    def _create_simple_scheduler_config(self, learning_rate, warmup_steps=0):
        """Create a simple scheduler configuration"""
        # Configure differently based on whether we're using Schedule-Free
        if hasattr(self, 'using_schedulefree') and self.using_schedulefree:
            # For Schedule-Free, return None as we don't want a scheduler
            return None
        else:
            # For standard optimizers, use a simple warmup scheduler
            return {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": learning_rate,
                    "warmup_num_steps": max(1, warmup_steps)
                }
            }
            
    def create_deepspeed_config(self, zero_stage, precision, offload_optimizer, offload_parameters, learning_rate, gradient_clip=None, tensor_parallel_size=1, depth=6, args=None):
        """Create DeepSpeed configuration"""
        config = {
            # Correctly set train_batch_size as the product of all components
            "train_batch_size": BATCH_SIZE * self.world_size * self.grad_accum_steps,
            "train_micro_batch_size_per_gpu": BATCH_SIZE,
            "gradient_accumulation_steps": self.grad_accum_steps,
            "steps_per_print": 500,  # Reduce logging frequency significantly
        
            # Add gradient clipping with default value 0.5 for all precision types
            # Disable gradient clipping when explicitly set to 0
            "gradient_clipping": 0.5 if gradient_clip is None else (gradient_clip if gradient_clip > 0 else 0),
        
            "optimizer": {
                "type": "Adam",  # Changed from AdamW to standard Adam to match Lightning's default
                "params": {
                    "lr": learning_rate,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8
                }
            },
            # Only add scheduler if not using Schedule-Free
            **({"scheduler": self._create_simple_scheduler_config(
                learning_rate,
                args.warmup_steps if args.warmup_steps else 0
            )} if not (hasattr(self, 'using_schedulefree') and self.using_schedulefree) else {}),
            "zero_optimization": {
                "stage": zero_stage,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,  # Increased for better bf16 performance
                "allgather_bucket_size": 5e8,  # Increased for better bf16 performance
                "round_robin_gradients": True
            },
            "zero_allow_untested_optimizer": True,
            "wall_clock_breakdown": False
        }
        
        # Configure precision settings properly
        if precision == "bf16":
            config["bf16"] = {
                "enabled": True,
                # Critical: accumulate gradients in fp32 for bf16
                "accumulate_grads_in_fp32": True
            }
            
            # For ZeRO-1, you need smaller buckets to avoid OOM
            if zero_stage == 1:
                config["zero_optimization"]["reduce_bucket_size"] = 2e8
                config["zero_optimization"]["allgather_bucket_size"] = 2e8
            
            # Important: Disable stage3_gather_16bit_weights_on_model_save for bf16
            if zero_stage == 3:
                config["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = False
            
            # Set gradient clipping to default 0.5 (if not already overridden)
            # Or disable it when explicitly set to 0
            if gradient_clip is None:
                config["gradient_clipping"] = 0.5
            elif gradient_clip == 0:
                config["gradient_clipping"] = 0  # Explicitly disable gradient clipping
            # Remove fp16 section to avoid confusion
            if "fp16" in config:
                config.pop("fp16")
            # Disable torch_autocast to use DeepSpeed's native bf16
            config["torch_autocast"] = {
                "enabled": False
            }
        elif precision == "fp16":
            config["fp16"] = {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            }
            # Remove bf16 section to avoid confusion
            if "bf16" in config:
                config.pop("bf16")
        else:
            # fp32 mode - remove both mixed precision sections
            if "fp16" in config:
                config.pop("fp16")
            if "bf16" in config:
                config.pop("bf16")
            if "torch_autocast" in config:
                config.pop("torch_autocast")
            
            # Explicitly configure for fp32
            config["fp32"] = {
                "enabled": True
            }
        
            # Ensure we're not using tensor parallelism configs that might default to fp16
            if "tensor_parallel" in config:
                config["tensor_parallel"]["tp_dtype"] = "fp32"
        
        # Add CPU offloading if requested (for ZeRO-2 and ZeRO-3)
        if zero_stage >= 2 and offload_optimizer:
            config["zero_optimization"]["offload_optimizer"] = {
                "device": "cpu",
                "pin_memory": True,
                "fast_init": True
            }
            
        # Parameter offloading only works with ZeRO-3
        if zero_stage == 3 and offload_parameters:
            config["zero_optimization"]["offload_param"] = {
                "device": "cpu",
                "pin_memory": True
            }
                
        # Special handling for ZeRO stages to handle model saving
        if zero_stage == 3:
            # Add stage3_gather_16bit_weights_on_model_save to save in fp16
            config["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = True
        elif zero_stage > 0:
            # For ZeRO-1 and ZeRO-2, also enable weight gathering on save
            config["zero_optimization"]["gather_16bit_weights_on_model_save"] = True
        
        # Configure tensor parallelism if requested
        if tensor_parallel_size > 1:
            config["tensor_parallel"] = {
                "autotp_size": tensor_parallel_size
            }
            
        # Add activation checkpointing
        config["activation_checkpointing"] = {
            "partition_activations": True,
            "cpu_checkpointing": True,
            "contiguous_memory_optimization": True,
            "number_checkpoints": min(depth, 8)
        }
            
        return config
        
    def train_step(self, batch):
        """Execute a single training step"""
        # Initialize start_time on first training step
        if self.start_time is None:
            self.start_time = time.time()
            if self.global_rank == 0:
                print(f"Starting tokens/s timing at step {self.global_step}")
                    
                # Debug: Print optimizer learning rate at start
                if hasattr(self, 'sf_optimizer') and self.sf_optimizer is not None:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"Schedule-Free initial training LR: {current_lr}")
                
        # Note: We no longer set sf_optimizer.train() here
        # ScheduleFree optimizer should be in train mode for the entire training loop
        
        # Ensure batch is a Long tensor before forward pass
        if batch.dtype != torch.long:
            batch = batch.long()
            
        # Ensure batch is on the right device
        if batch.device != self.model.device:
            batch = batch.to(self.model.device)
            
        # Forward pass - DeepSpeed handles loss scaling and backward
        loss = self.model(batch, return_loss=True)
        
        # Update step - DeepSpeed handles gradient accumulation internally
        self.model.backward(loss)
        
        # Debug gradient info periodically (only when enabled and on main process)
        if self.debug_gradients and self.global_rank == 0 and self.global_step % 50 == 0:
            # Get gradient norm of a parameter to check training health
            total_norm = 0.0
            for name, param in list(self.model.named_parameters())[:10]:
                if param.grad is not None:
                    param_norm = torch.norm(param.grad.detach()).item()
                    total_norm += param_norm ** 2
                    print(f"Step {self.global_step}, Parameter {name}, Grad norm: {param_norm:.6f}, Data type: {param.dtype}")
                else:
                    print(f"Step {self.global_step}, Parameter {name} has None gradient")
            
            total_norm = total_norm ** 0.5
            print(f"Step {self.global_step}, Total gradient norm: {total_norm:.6f}")
        
        # Store current learning rates before DeepSpeed's step()
        current_lrs = None
        if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
            current_lrs = [group['lr'] for group in self.optimizer.param_groups]
        
        self.model.step()
        
        # Ensure our learning rates weren't changed by DeepSpeed's step
        if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None and current_lrs is not None:
            for i, lr in enumerate(current_lrs):
                if self.optimizer.param_groups[i]['lr'] != lr:
                    # If DeepSpeed changed our LR, restore it
                    self.optimizer.param_groups[i]['lr'] = lr
                    if self.global_rank == 0 and not self.silent_mode:
                        print(f"Restored LR from {self.optimizer.param_groups[i]['lr']} to {lr}")
        
        # Track loss
        loss_val = loss.detach().float().item()
        self.train_loss = loss_val
        
        # With Schedule-Free, we don't need manual LR updates - it handles adaptivity internally
        
        # Schedule-Free optimizer handles learning rate internally
        
        # Update tokens processed count
        tokens_in_batch = batch.numel()
        self.total_tokens_processed += tokens_in_batch
        
        # Track tokens across all processes
        if self.world_size > 1:
            # Convert to tensor for all_reduce
            batch_tokens = torch.tensor(tokens_in_batch, device=batch.device)
            # Sum across all processes
            torch.distributed.all_reduce(batch_tokens, op=torch.distributed.ReduceOp.SUM)
            # Update global counter
            self.global_tokens += batch_tokens.item()
        else:
            self.global_tokens += tokens_in_batch
        
        # Track step
        self.global_step += 1
        
        # Log learning rate occasionally
        if self.global_rank == 0 and self.global_step % 100 == 0:
            current_lr = self.optimizer.param_groups[0]['lr']
            if not self.silent_mode:
                print(f"Step {self.global_step}: Current LR = {current_lr:.6f}")
        
        return loss_val
        
    def validation_step(self, batch):
        """Execute a single validation step"""
        with torch.no_grad():
            # Ensure batch is a Long tensor
            if batch.dtype != torch.long:
                batch = batch.long()
            # Move batch to device if needed
            if batch.device != self.model.device:
                batch = batch.to(self.model.device)
            loss = self.model(batch, return_loss=True)
            # Calculate bits per byte (bpb)
            bpb = loss / math.log(2)
            return {"val_loss": loss.item(), "bpb": bpb.item()}
    
    def validate(self, dataloader, max_batches=None):
        """Run validation on the entire validation dataset"""
        self.model.eval()
        # Note: We don't set sf_optimizer.eval() here anymore
        # The ScheduleFree mode is set by the caller
            
        val_losses = []
        val_bpbs = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if max_batches is not None and i >= max_batches:
                    break
                batch = batch.to(self.model.device)
                # Ensure batch is a Long tensor
                if batch.dtype != torch.long:
                    batch = batch.long()
                result = self.validation_step(batch)
                val_losses.append(result["val_loss"])
                val_bpbs.append(result["bpb"])
        
        # Calculate average
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_bpb = sum(val_bpbs) / len(val_bpbs)
        self.val_loss = avg_val_loss
        self.val_bpb = avg_bpb
        
        self.model.train()
        # Note: We don't set sf_optimizer.train() here anymore
        # The ScheduleFree mode is set by the caller
            
        return {"val_loss": avg_val_loss, "bpb": avg_bpb}
    
    def load_checkpoint(self, checkpoint_path):
        """Load model and optimizer state from checkpoint"""
        if self.global_rank == 0 and not self.silent_mode:
            print(f"Loading checkpoint from {checkpoint_path}")
        
        # Check if this is a directory (DeepSpeed checkpoint) or file (PyTorch checkpoint)
        if os.path.isdir(checkpoint_path):
            # Save reference to Schedule-Free optimizer before loading
            sf_optimizer = None
            if hasattr(self, 'sf_optimizer') and self.sf_optimizer is not None:
                sf_optimizer = self.sf_optimizer
                if self.global_rank == 0 and not self.silent_mode:
                    print("Preserving Schedule-Free optimizer instance for state restoration")
            
            # This is a DeepSpeed checkpoint directory
            # DeepSpeed handles loading both model and optimizer state
            success = self.model.load_checkpoint(checkpoint_path)
            if not success:
                raise ValueError(f"Failed to load DeepSpeed checkpoint from {checkpoint_path}")
            
            # For Schedule-Free: Load optimizer state from auxiliary file if it exists
            if sf_optimizer is not None:
                sf_state_path = os.path.join(checkpoint_path, "sf_optimizer_state.pt")
                if os.path.exists(sf_state_path):
                    if self.global_rank == 0 and not self.silent_mode:
                        print(f"Loading Schedule-Free optimizer state from {sf_state_path}")
                    sf_state = torch.load(sf_state_path, map_location='cpu')
                    sf_optimizer.load_state_dict(sf_state)
                    # Restore reference to optimizer 
                    self.sf_optimizer = sf_optimizer
                    # Point the engine optimizer to the same object
                    self.optimizer = sf_optimizer
                else:
                    if self.global_rank == 0:
                        print("WARNING: No Schedule-Free optimizer state found, using reinitialized state")
                
            # Try to extract step from the checkpoint path or tag file
            tag_file = os.path.join(checkpoint_path, "latest_tag")
            if os.path.exists(tag_file):
                with open(tag_file, 'r') as f:
                    tag_content = f.read().strip()
                    # Extract step if format is like "global_step{N}"
                    if tag_content.startswith("global_step"):
                        try:
                            self.global_step = int(tag_content.split("global_step")[1])
                        except (IndexError, ValueError):
                            # If parsing fails, keep the current step
                            pass
        else:
            # This is a vanilla PyTorch checkpoint file
            # Load checkpoint on CPU first to avoid OOM issues
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Load model state (DeepSpeed requires unwrapping the state_dict)
            missing_keys, unexpected_keys = self.model.module.load_state_dict(
                checkpoint['model_state_dict'], strict=False
            )
            
            if missing_keys and self.global_rank == 0 and not self.silent_mode:
                print(f"Warning: Missing keys in checkpoint: {missing_keys}")
            if unexpected_keys and self.global_rank == 0 and not self.silent_mode:
                print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
            
            # Set training state variables
            self.global_step = checkpoint.get('step', 0)
            self.global_tokens = torch.tensor(checkpoint.get('global_tokens', 0), dtype=torch.long)
            self.val_loss = checkpoint.get('val_loss', float('inf'))
            self.val_bpb = checkpoint.get('val_bpb', float('inf'))
            self.best_val_loss = checkpoint.get('val_loss', float('inf'))
            self.best_val_bpb = checkpoint.get('val_bpb', float('inf'))
            
            # Load Schedule-Free optimizer state if present
            if 'sf_optimizer_state' in checkpoint and hasattr(self, 'sf_optimizer') and self.sf_optimizer is not None:
                try:
                    if self.global_rank == 0 and not self.silent_mode:
                        print("Restoring Schedule-Free optimizer state from checkpoint")
                    self.sf_optimizer.load_state_dict(checkpoint['sf_optimizer_state'])
                    # Also update DeepSpeed's reference to the optimizer
                    self.optimizer = self.sf_optimizer
                except Exception as e:
                    if self.global_rank == 0:
                        print(f"WARNING: Failed to load Schedule-Free optimizer state: {e}")
        
        # Reset token counting and timing for correct tokens/s calculation on resume
        self.start_time = time.time()
        self.total_tokens_processed = 0
        
        # Important: We keep self.global_tokens as loaded from checkpoint for total count,
        # but reset the timing so tokens/s calculation starts fresh
        
        # Ensure Schedule-Free optimizer is in train mode after loading
        if hasattr(self, 'sf_optimizer') and self.sf_optimizer is not None:
            self.sf_optimizer.train()
            if self.global_rank == 0 and not self.silent_mode:
                print("Schedule-Free optimizer set to train mode after checkpoint load")
                # Debug: print current state of optimizer
                if hasattr(self.sf_optimizer, 'training'):
                    print(f"Schedule-Free optimizer training mode: {self.sf_optimizer.training}")
                # Print current learning rate to verify state
                print(f"Schedule-Free optimizer learning rate: {self.optimizer.param_groups[0]['lr']}")
        
        if self.global_rank == 0 and not self.silent_mode:
            print(f"✅ Checkpoint loaded. Resuming from step {self.global_step}")
            print(f"Token counter reset for accurate tokens/s calculation")
        
        return self.global_step
    
    def save_checkpoint(self, additional_info=None, is_periodic=False, is_permanent=False):
        """Save a checkpoint of the model"""
        if not self.checkpoint_dir or self.global_rank != 0:
            return
            
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Create checkpoint info
        checkpoint = {
            'step': self.global_step,
            'model_state_dict': self.model.module.state_dict(),  # Access the model inside DeepSpeed
            'global_tokens': self.global_tokens.item(),
            'training_time': time.time() - self.start_time,
            'val_loss': self.val_loss,
            'val_bpb': self.val_bpb
        }
        
        # Save Schedule-Free optimizer state if present
        if hasattr(self, 'sf_optimizer') and self.sf_optimizer is not None:
            # Get state dict from the Schedule-Free optimizer
            sf_state = self.sf_optimizer.state_dict()
            # Save as part of main checkpoint
            checkpoint['sf_optimizer_state'] = sf_state
        
        if additional_info:
            checkpoint.update(additional_info)
        
        # Format checkpoint filename with validation metrics
        val_loss_str = f"{self.val_loss:.4f}" if self.val_loss is not None else "NA"
        bpb_str = f"{self.val_bpb:.4f}" if self.val_bpb is not None else "NA"
        
        # Check if this should be a milestone checkpoint (every 10k steps)
        is_milestone = self.global_step % 10000 == 0 and self.global_step > 0
        
        # Add appropriate prefix for different types of checkpoints
        if is_permanent:
            prefix = "permanent-"
        elif is_milestone:
            prefix = "milestone-"
        else:
            prefix = ""
        
        # Save checkpoint with informative name
        filename = f"{prefix}minlm-step-{self.global_step}-loss-{val_loss_str}-bpb-{bpb_str}.pt"
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        
        # Save latest checkpoint (for resuming)
        latest_path = os.path.join(self.checkpoint_dir, "latest.pt")
        torch.save(checkpoint, latest_path)
        
        # Initialize recent_checkpoints list if it doesn't exist
        if not hasattr(self, 'recent_checkpoints'):
            self.recent_checkpoints = []
        
        # Add to recent checkpoints list if this is a regular checkpoint
        if not is_permanent and not is_milestone:
            self.recent_checkpoints.append(checkpoint_path)
            # Keep only the 5 most recent regular checkpoints
            if len(self.recent_checkpoints) > 5:
                old_checkpoints = self.recent_checkpoints[:-5]  # Get checkpoints beyond the 5 most recent
                self.recent_checkpoints = self.recent_checkpoints[-5:]  # Keep only 5 most recent
                
                # Delete old regular checkpoints
                for path in old_checkpoints:
                    if (os.path.exists(path) and "best" not in path and "latest" not in path 
                            and "permanent-" not in os.path.basename(path)
                            and "milestone-" not in os.path.basename(path)):
                        try:
                            os.remove(path)
                            if not self.silent_mode:
                                print(f"Removed old checkpoint {os.path.basename(path)}")
                        except OSError as e:
                            if not self.silent_mode:
                                print(f"Error removing checkpoint: {e}")
        
        # For permanent/milestone checkpoints, save Schedule-Free state and return
        if is_permanent or is_milestone:
            # For DeepSpeed checkpoints, also save Schedule-Free optimizer state separately
            # This is the standard path used by DeepSpeed
            ds_checkpoint_dir = os.path.join(self.checkpoint_dir, f"global_step{self.global_step}")
            
            # If this directory exists, it means DeepSpeed successfully saved a checkpoint
            # We need to save our Schedule-Free state there as well
            if os.path.exists(ds_checkpoint_dir) and hasattr(self, 'sf_optimizer') and self.sf_optimizer is not None:
                sf_state_path = os.path.join(ds_checkpoint_dir, "sf_optimizer_state.pt")
                # Get state dict from the Schedule-Free optimizer
                sf_state = self.sf_optimizer.state_dict()
                # Save to file
                torch.save(sf_state, sf_state_path)
                if not self.silent_mode:
                    print(f"Saved Schedule-Free optimizer state to {sf_state_path}")
            
            if not self.silent_mode:
                checkpoint_type = "permanent" if is_permanent else "milestone"
                print(f"Saved {checkpoint_type} checkpoint at step {self.global_step}")
            return checkpoint_path
        
        # Track best checkpoints (top k)
        if not is_periodic and self.val_loss is not None:
            # Check if this is a best model
            is_best = False
            if self.val_loss < self.best_val_loss:
                self.best_val_loss = self.val_loss
                self.best_val_bpb = self.val_bpb
                is_best = True
                
                # Also save as best model
                best_path = os.path.join(self.checkpoint_dir, "best.pt")
                torch.save(checkpoint, best_path)
                
                if hasattr(self, 'silent_mode') and not self.silent_mode:
                    print(f"New best model saved with val_loss: {self.val_loss:.4f}, bpb: {self.val_bpb:.4f}")
            
            # Add to best_checkpoints list and sort
            self.best_checkpoints.append((checkpoint_path, self.val_loss))
            self.best_checkpoints.sort(key=lambda x: x[1])  # Sort by val_loss (lower is better)
            
            # Keep only top_k best checkpoints
            if len(self.best_checkpoints) > self.save_top_k:
                # Get paths of checkpoints to remove (everything after top_k)
                to_remove = self.best_checkpoints[self.save_top_k:]
                self.best_checkpoints = self.best_checkpoints[:self.save_top_k]
                
                # Delete the checkpoints that didn't make the cut
                for path, _ in to_remove:
                    # Don't delete permanent checkpoints, milestone checkpoints, or special ones
                    if (os.path.exists(path) and "best" not in path and "latest" not in path 
                            and "permanent-" not in os.path.basename(path)
                            and "milestone-" not in os.path.basename(path)):
                        try:
                            os.remove(path)
                            if not self.silent_mode:
                                print(f"Removed checkpoint {os.path.basename(path)} to keep top {self.save_top_k}")
                        except OSError as e:
                            if not self.silent_mode:
                                print(f"Error removing checkpoint: {e}")
        
        return checkpoint_path
    
    def train(self, train_dataloader, val_dataloader, num_batches, validate_every, generate_every, val_batches=4):
        """Main training loop"""
        # Always set start time at the beginning of training or resuming
        # This ensures we get accurate tokens/s measurements
        self.start_time = time.time()
        
        # Create metrics log file
        if self.global_rank == 0 and self.checkpoint_dir:
            metrics_log_path = os.path.join(self.checkpoint_dir, "training_metrics.tsv")
            metrics_exists = os.path.exists(metrics_log_path)
            
            # If resuming, append to existing file rather than overwriting
            mode = 'a' if metrics_exists else 'w'
            with open(metrics_log_path, mode) as f:
                # Only write header for new files
                if not metrics_exists:
                    header = [
                        "step", "time", "tokens_processed", 
                        "tokens_per_sec", "train_loss", "val_loss", "bpb",
                        "current_lr", "batch_size", "grad_accum"
                    ]
                    f.write('\t'.join(header) + '\n')
                
                # Add resume marker if appending
                if metrics_exists:
                    f.write(f"# Resume training at step {self.global_step}\n")
                
        # If using ScheduleFree, put optimizer in train mode at the beginning
        if hasattr(self, 'sf_optimizer') and self.sf_optimizer is not None:
            self.sf_optimizer.train()
        
        # Initial validation - run on all ranks for tensor parallelism
        if self.global_rank == 0:
            # Save permanent checkpoint at step 0
            self.save_checkpoint({"initial": True}, is_periodic=False, is_permanent=True)
            if not self.silent_mode:
                print(f"Saved permanent initial checkpoint at step {self.global_step}")
    
        # Run validation on ALL ranks to ensure tensor parallel consistency
        val_results = self.validate(val_dataloader, max_batches=val_batches)
    
        # Only log metrics on rank 0
        if self.global_rank == 0:
            self._log_metrics(True)
        
        # Training loop
        self.model.train()
        pbar = tqdm(total=num_batches, disable=self.global_rank != 0)
        step = 0
        
        while step < num_batches:
            # Reset dataloader if needed
            train_iter = iter(train_dataloader)
            
            for batch in train_iter:
                if step >= num_batches:
                    break
                
                # Move batch to device and ensure correct type
                batch = batch.long().to(self.model.device)
                
                # Training step
                loss = self.train_step(batch)
                
                # Update progress bar
                if self.global_rank == 0:
                    elapsed = time.time() - self.start_time
                    # Use tokens processed in this session for accurate tokens/s reporting
                    tokens_per_sec = self.total_tokens_processed / elapsed if elapsed > 0 else 0
                    val_info = f"Val: {self.val_loss:.4f} BPB: {self.val_bpb:.4f} | " if self.val_loss > 0 else ""
                    current_lr = self.optimizer.param_groups[0]['lr']
                        
                    # Add GreedyLR concise status if available
                    lr_info = f"LR: {current_lr:.6f}"
                    if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None and hasattr(self.lr_scheduler, 'status_symbol'):
                        if hasattr(self.lr_scheduler, 'in_warmup') and self.lr_scheduler.in_warmup:
                            warmup_info = f"Warmup {self.lr_scheduler.warmup_counter}/{self.lr_scheduler.warmup}"
                            lr_info = f"LR: {current_lr:.6f} ↗ {warmup_info}"
                        else:
                            lr_info = f"LR: {current_lr:.6f} {self.lr_scheduler.status_symbol} {self.lr_scheduler.status_info}"
                            
                    pbar.set_description(f"Loss: {loss:.4f} | {lr_info} | {val_info}{tokens_per_sec:.2f} tok/s")
                    pbar.update(1)
                
                # Log progress on every step
                if self.global_rank == 0:
                    self._log_metrics(False)
                    
                # Save periodic checkpoint based on configured interval
                if self.global_rank == 0 and step > 0 and step % self.checkpoint_every == 0:
                    self.save_checkpoint({"periodic": True}, is_periodic=True)
                    if not self.silent_mode:
                        print(f"Saved periodic checkpoint at step {step}")
                
                # Save permanent checkpoint every permanent_save_interval steps
                if self.global_rank == 0 and step > 0 and step % self.permanent_save_interval == 0:
                    self.save_checkpoint({"permanent": True}, is_periodic=False, is_permanent=True)
                    if not self.silent_mode:
                        print(f"Saved permanent checkpoint at step {step}")
                
                # Validate periodically
                if step > 0 and step % validate_every == 0:
                    # Set ScheduleFree optimizer to eval mode if using it (on all ranks)
                    if hasattr(self, 'sf_optimizer') and self.sf_optimizer is not None:
                        self.sf_optimizer.eval()
                    
                    # Run validation on ALL ranks to ensure tensor parallel consistency
                    val_results = self.validate(val_dataloader, max_batches=val_batches)
                    
                    # Only do checkpoint and logging on rank 0
                    if self.global_rank == 0:
                        # Save checkpoint with validation results
                        self.save_checkpoint()
                        
                        # Log metrics
                        self._log_metrics(True)
                    
                    # Put ScheduleFree optimizer back to train mode on ALL ranks
                    if hasattr(self, 'sf_optimizer') and self.sf_optimizer is not None:
                        self.sf_optimizer.train()
                
                # Generate samples periodically if not skipped
                if generate_every > 0 and step > 0 and step % generate_every == 0 and self.global_rank == 0:
                    # Temporarily enable more verbose logging during generation
                    if self.checkpoint_dir:
                        import logging
                        ds_logger = logging.getLogger('deepspeed')
                        prev_level = ds_logger.level
                        
                        # Set console handler to INFO temporarily
                        for handler in ds_logger.handlers:
                            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                                prev_handler_level = handler.level
                                handler.setLevel(logging.INFO)
                                break
                    
                    # Generate the sample
                    self._generate_sample()
                    
                    # Restore logging level
                    if self.checkpoint_dir:
                        ds_logger.setLevel(prev_level)
                        for handler in ds_logger.handlers:
                            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                                handler.setLevel(prev_handler_level)
                                break
                
                step += 1
        
        pbar.close()
        
        # Final validation and checkpoint
        if self.global_rank == 0:
            val_results = self.validate(val_dataloader)
            
            # Save final checkpoint
            final_path = self.save_checkpoint({
                'final': True
            })
            if not self.silent_mode:
                print(f"Training complete! Final checkpoint saved to: {final_path}")
            
            # Log final metrics
            self._log_metrics(True)
    
    def find_learning_rate(self, train_dataloader, min_lr=1e-8, max_lr=10, num_iter=100, beta=0.98, show_progress=True):
        """
        Run the learning rate finder to determine the optimal learning rate.
        
        Args:
            train_dataloader: DataLoader for training data
            min_lr: Starting learning rate
            max_lr: Maximum learning rate to try
            num_iter: Number of iterations to run
            beta: Smoothing factor for loss (0.0 to 1.0)
            show_progress: Whether to show progress bar
            
        Returns:
            log_lrs: List of log10 of learning rates
            losses: List of smoothed losses
            suggested_lr: Suggested learning rate
        """
        if self.global_rank == 0 and not self.silent_mode:
            print(f"Running LR finder from {min_lr} to {max_lr} over {num_iter} iterations")
        
        # Initialize loss tracking
        losses = []
        log_lrs = []
        best_loss = float('inf')
        avg_loss = 0.0
        
        # Calculate the multiplication factor for LR
        mult_factor = (max_lr / min_lr) ** (1 / num_iter)
        
        # Record the initial learning rate
        initial_lr = self.optimizer.param_groups[0]['lr']
        
        # Save initial model parameters and optimizer state
        # For DeepSpeed, we need to save checkpoint first
        temp_dir = os.path.join(self.checkpoint_dir, "lr_finder_temp") if self.checkpoint_dir else "lr_finder_temp"
        
        # Make sure all ranks are synchronized when creating directory
        if self.global_rank == 0:
            os.makedirs(temp_dir, exist_ok=True)
        
        # Make sure all processes see the directory
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        
        # Now all ranks can save their part of the checkpoint
        if hasattr(self.model, 'save_checkpoint'):
            self.model.save_checkpoint(temp_dir, tag="init")
            # Ensure all ranks have completed the save
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

        # Set initial learning rate
        lr = min_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        # Create a progress bar if needed
        pbar = None
        if self.global_rank == 0 and show_progress:
            from tqdm import tqdm
            pbar = tqdm(total=num_iter, desc="Finding optimal learning rate")
        
        # Get an iterator for the dataloader
        train_iter = iter(train_dataloader)
        
        # Set ScheduleFree optimizer to train mode if using it
        if hasattr(self, 'sf_optimizer') and self.sf_optimizer is not None:
            self.sf_optimizer.train()
            
        # Run iterations with increasing learning rate
        for i in range(num_iter):
            # Get the next batch (resetting iterator if needed)
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                batch = next(train_iter)
            
            # Ensure batch is Long type and on the correct device
            batch = batch.long().to(self.model.device)
            
            # Forward pass and compute loss
            self.model.train()
            loss = self.model(batch, return_loss=True)
            
            # Compute smoothed loss
            if torch.is_tensor(loss):
                current_loss = loss.item()
            else:
                current_loss = loss
                
            avg_loss = beta * avg_loss + (1 - beta) * current_loss
            smoothed_loss = avg_loss / (1 - beta ** (i + 1))
            
            # Record best loss
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss
            
            # Stop if loss is exploding
            if i > 0 and smoothed_loss > 4 * best_loss:
                if self.global_rank == 0 and not self.silent_mode:
                    print(f"\nLoss exploded at learning rate {lr:.8f}. Stopping.")
                break
            
            # Record values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            
            # Backward pass and optimizer step
            self.model.backward(loss)
            self.model.step()
            
            # Increase learning rate for next iteration
            lr *= mult_factor
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({"loss": f"{smoothed_loss:.4f}", "lr": f"{lr:.8f}"})
        
        if pbar is not None:
            pbar.close()
        
        # Analyze results to suggest a learning rate
        suggested_lr = self._analyze_lr_find_results(log_lrs, losses)
        
        # Make sure every rank resets to the initial state
        if hasattr(self.model, 'load_checkpoint'):
            try:
                # All ranks try to load the checkpoint
                self.model.load_checkpoint(temp_dir, tag="init")
            except Exception as e:
                # If there's an error, log it but don't terminate
                if self.global_rank == 0 and not self.silent_mode:
                    print(f"Warning: Error when restoring model state after LR finder: {e}")
                    print("Continuing with current model state...")
        
        # Restore the original learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = initial_lr
            
        # Make sure all ranks are done before cleanup
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        
        # Clean up temp directory
        if self.global_rank == 0 and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Could not remove temporary directory: {e}")
        
        return log_lrs, losses, suggested_lr

    def _analyze_lr_find_results(self, log_lrs, losses):
        """
        Analyze the learning rate finder results to suggest an optimal learning rate.
        
        Args:
            log_lrs: List of log10 of learning rates
            losses: List of smoothed losses
            
        Returns:
            suggested_lr: The suggested learning rate
        """
        if len(log_lrs) <= 1 or len(losses) <= 1:
            if self.global_rank == 0 and not self.silent_mode:
                print("Not enough data points to suggest a learning rate.")
            return self.learning_rate  # Return the default value
        
        # Convert to numpy for analysis
        log_lrs = np.array(log_lrs)
        losses = np.array(losses)
        
        # Compute gradient
        # We need at least 2 points to compute a gradient
        derivatives = (losses[1:] - losses[:-1]) / (log_lrs[1:] - log_lrs[:-1])
        
        # Find the point with steepest negative gradient (if it exists)
        try:
            # Get the index of minimum gradient (steepest descent)
            min_grad_idx = np.argmin(derivatives)
            
            # The corresponding learning rate is our suggested value
            # We convert back from log scale
            suggested_lr = 10 ** log_lrs[min_grad_idx]
            
            # For safety, we typically pick a value slightly lower than the minimum
            # A common practice is to divide by 10
            safe_lr = suggested_lr / 10
            
            if self.global_rank == 0 and not self.silent_mode:
                print(f"Minimum gradient at LR: {suggested_lr:.8f}")
                print(f"Suggested safe LR: {safe_lr:.8f}")
                
            return safe_lr
        except (ValueError, IndexError):
            # If analysis fails, return the starting learning rate
            if self.global_rank == 0 and not self.silent_mode:
                print("Could not determine optimal learning rate from curve. Using default.")
            return self.learning_rate
            
    # Method removed as we no longer need LR scheduler calculations
    
    def _log_metrics(self, is_validation=False):
        """Log metrics to TSV file"""
        if not self.checkpoint_dir or self.global_rank != 0:
            return
            
        metrics_log_path = os.path.join(self.checkpoint_dir, "training_metrics.tsv")
        
        # Use a more robust approach with atomic file operations to prevent race conditions
        try:
            # Create a unique temporary file
            temp_path = f"{metrics_log_path}.tmp.{random.randint(0, 1000000)}"
            
            # Prepare the line to write
            elapsed = time.time() - self.start_time
            
            # Use the tokens processed since current session start for tokens/s calculation
            # This gives accurate performance metrics rather than using accumulated historical count
            tokens_per_sec = self.total_tokens_processed / elapsed if elapsed > 0 else 0
            
            # But use global_tokens for total count (includes previous runs)
            global_tokens = self.global_tokens.item()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Prepare values
            values = [
                str(self.global_step),
                f"{elapsed:.2f}",
                str(global_tokens),
                f"{tokens_per_sec:.2f}",
                f"{self.train_loss:.6f}",
                str(self.val_loss if is_validation else "NA"),
                str(self.val_bpb if is_validation else "NA"),
                f"{current_lr:.8f}",  # Use actual current LR, not initial LR
                str(BATCH_SIZE),
                str(self.grad_accum_steps)
            ]
            
            line = '\t'.join(values) + '\n'
            
            # Write to temporary file first
            with open(temp_path, 'w') as f:
                f.write(line)
            
            # Then append to the main log file atomically
            with open(metrics_log_path, 'a') as main_log:
                with open(temp_path, 'r') as temp_log:
                    main_log.write(temp_log.read())
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        except Exception as e:
            print(f"Warning: Could not write to metrics log: {e}")
            # Add more detailed error reporting
            import traceback
            print(f"Exception details: {traceback.format_exc()}")
    
    def _generate_sample(self, prime_length=PRIME_LENGTH, gen_length=GENERATE_LENGTH):
        """Generate a text sample during training"""
        if not hasattr(self, 'val_dataset') or self.val_dataset is None:
            print("No validation dataset provided for generation")
            return
        
        # Handle both memory-mapped and in-memory datasets
        if isinstance(self.val_dataset, MemoryMappedTextDataset):
            # Get a batch from the memory-mapped dataset
            self.val_dataset._ensure_open()  # Make sure memory map is open
            
            # Generate a random position within the validation area
            start_pos = random.randint(0, self.val_dataset.valid_end) + self.val_dataset.offset
            
            # Read the prime data from the memory map
            self.val_dataset.mm.seek(start_pos)
            data = self.val_dataset.mm.read(prime_length)
            
            # Convert to a writable buffer first, then to tensor
            writable_data = bytearray(data)
            prime = torch.frombuffer(writable_data, dtype=torch.uint8).long().unsqueeze(0).to(self.model.device)
        else:
            # For in-memory dataset
            rand_start = torch.randint(0, len(self.val_dataset.data) - prime_length - 1, (1,))
            prime = self.val_dataset.data[rand_start:rand_start + prime_length].long().unsqueeze(0).to(self.model.device)
        
        # Generate text
        if not self.silent_mode:
            print("\nGenerating sample text...")
            print(f"Prime: {decode_tokens(prime[0])}")
        
        # Set to eval mode for generation
        self.model.eval()
        # Set ScheduleFree optimizer to eval mode if using it
        was_training = False
        if hasattr(self, 'sf_optimizer') and self.sf_optimizer is not None:
            was_training = True
            self.sf_optimizer.eval()
            
        with torch.no_grad():
            generated = base_decoding(
                self.model, 
                prime, 
                gen_length, 
                temperature=0.8, 
                filter_thres=0.9
            )
            
        # Restore previous modes
        self.model.train()
        # Return to train mode if we were training before
        if was_training and hasattr(self, 'sf_optimizer') and self.sf_optimizer is not None:
            self.sf_optimizer.train()
        
        if not self.silent_mode:
            print(f"Generated: {decode_tokens(generated[0])}")

# Helper functions for command line arguments
def parse_gpu_ids(gpu_spec):
    """Parse a GPU specification string into a list of GPU ids"""
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
    """Parse a string with optional k, m, g suffix into a number"""
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

def round_to_multiple(n, multiple=32):
    """Round a number to the nearest multiple of a given value"""
    return multiple * round(n / multiple)

def solve_for_dimension(target_params, depth, vocab_size=256, ff_mult=4, expansion=1.5):
    """Solve for the dimension that gives the target parameter count"""
    from math import sqrt
    
    factor = 4 * expansion + 2 * ff_mult
    
    # Quadratic equation: a*dim^2 + b*dim - target_params = 0
    a = depth * factor
    b = 2 * vocab_size
    c = -target_params
    
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        raise ValueError("No solution exists for the given target parameter count")
    
    dim = (-b + sqrt(discriminant)) / (2*a)
    return round_to_multiple(dim)

def solve_for_depth(target_params, dim, vocab_size=256, ff_mult=4, expansion=1.5):
    """Solve for the depth that gives the target parameter count"""
    embed_params = 2 * dim * vocab_size
    factor = 4 * expansion + 2 * ff_mult
    layer_params = dim * dim * factor
    
    depth = (target_params - embed_params) / layer_params
    return max(1, round(depth))

def main():
    # Make variables global
    global SEQ_LEN
    global BATCH_SIZE
    global GRAD_ACCUM_EVERY
    global LEARNING_RATE
    global NUM_BATCHES
    
    # Set TensorFloat32 precision mode at the beginning of training
    torch.set_float32_matmul_precision('high')
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a minLM model with DeepSpeed")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint file to resume training from")
    parser.add_argument("--force_lr_on_resume", action="store_true",
                        help="Force reset learning rate to command-line value when resuming")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to the training data file (e.g., 'data/enwik8.gz')")
    parser.add_argument("--gpus", type=str, default=None, 
                        help="Comma-separated list or range of GPU IDs to use (e.g., '0,1,2' or '0-2')")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information during training")
    parser.add_argument("--generate", action="store_true",
                        help="Enable periodic text generation during training")
    
    # Model architecture arguments
    parser.add_argument("--dim", type=str, default=None,
                        help="Model hidden dimension (default: 512). Can use k/m/g suffix.")
    parser.add_argument("--checkpoint_every", type=int, default=100,
                        help="Save checkpoint every N steps (default: 100)")
    parser.add_argument("--permanent_save_interval", type=int, default=5000,
                        help="Save permanent checkpoints every N steps (default: 5000)")
    parser.add_argument("--depth", type=int, default=None,
                        help="Number of model layers (default: 6).")
    parser.add_argument("--ff_mult", type=float, default=None,
                        help=f"Feedforward multiplier (default: {MODEL_CONFIG['ff_mult']})")
    parser.add_argument("--expansion", type=float, default=None,
                        help=f"Expansion factor for minGRU/minLSTM (default: {MODEL_CONFIG['expansion']})")
    parser.add_argument("--embedding_dim", type=int, default=None,
                        help="Embedding dimension (vocabulary size, default: 256)")
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
                        help=f"Sequence length for training (default: {TRAINING_CONFIG['seq_len']}). Can use k suffix.")
    parser.add_argument("--steps", type=str, default=None,
                        help=f"Total training steps (default: {TRAINING_CONFIG['num_batches']}). Can use k suffix.")
    parser.add_argument("--output", type=str, default=None,
                        help="Directory to save checkpoints (default: auto-generated name)")
    
    # Warmup parameter (only used for non-ScheduleFree optimizers)
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Number of warmup steps (default: 0, not used with ScheduleFree)")
    # Note: Removed complex scheduler parameters that aren't needed for Schedule-Free
    
    # Learning rate finder parameters
    parser.add_argument("--find_lr", action="store_true",
                        help="Run the learning rate finder before training")
    parser.add_argument("--min_find_lr", type=float, default=1e-8,
                        help="Minimum learning rate for LR finder (default: 1e-8)")
    parser.add_argument("--max_find_lr", type=float, default=0.01,
                        help="Maximum learning rate for LR finder (default: 0.01)")
    parser.add_argument("--num_lr_find_iter", type=int, default=100,
                        help="Number of iterations for LR finder (default: 100)")
                        
    # Schedule-Free optimizer parameters
    parser.add_argument("--schedulefree", action="store_true",
                        help="Use Schedule-Free optimizer instead of standard Adam")
    parser.add_argument("--sf_beta", type=float, default=0.9,
                        help="Schedule-Free momentum parameter (default: 0.9)")
    parser.add_argument("--sf_weight_decay", type=float, default=0.01,
                        help="Weight decay value for Schedule-Free (default: 0.01)")
    # Precision options
    precision_group = parser.add_mutually_exclusive_group()
    precision_group.add_argument("--bf16", dest="precision", action="store_const", const="bf16", default="bf16",
                        help="Use BF16 precision (default)")
    precision_group.add_argument("--fp16", dest="precision", action="store_const", const="fp16",
                        help="Use FP16 precision instead of BF16")
    precision_group.add_argument("--fp32", dest="precision", action="store_const", const="fp32",
                        help="Use FP32 precision (no mixed precision)")
                        
    # DeepSpeed arguments
    parser.add_argument("--deepspeed_config", type=str, default=None,
                        help="Path to DeepSpeed JSON config file (overrides other DeepSpeed args)")
    parser.add_argument("--zero_stage", type=int, default=2, choices=[0, 1, 2, 3],
                        help="ZeRO optimization stage (0-3, default: 2)")
    parser.add_argument("--offload_optimizer", action="store_true",
                        help="Offload optimizer states to CPU (reduces GPU memory)")
    parser.add_argument("--offload_parameters", action="store_true",
                        help="Offload parameters to CPU (for ZeRO-3)")
    parser.add_argument("--no_compile", action="store_true",
                        help="Disable torch.compile() model compilation")
    parser.add_argument("--gradient_clip", type=float, default=None,
                        help="Gradient clipping value (default: 0.5 for all precision types, 0 to disable)")
    parser.add_argument("--debug_gradients", action="store_true",
                        help="Print detailed gradient norms during training")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Tensor parallel size for model parallelism (default: 1)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (set by deepspeed launcher)")
    
    args = parser.parse_args()
    
    # Set up distributed training
    deepspeed.init_distributed()
    
    # Get distributed training info
    local_rank = args.local_rank if args.local_rank >= 0 else 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_rank = int(os.environ.get("RANK", 0)) if world_size > 1 else 0
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Print CUDA information
    if global_rank == 0:
        print(f"CUDA AVAILABLE: {torch.cuda.is_available()}")
        print(f"GPU COUNT: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        print(f"World size: {world_size}")
        print(f"Global rank: {global_rank}")
        print(f"Local rank: {local_rank}")
    
    # Helper function to detect if a file is gzipped
    def is_gzip_file(filepath):
        with open(filepath, 'rb') as test_f:
            return test_f.read(2) == b'\x1f\x8b'
    
    # Load and prepare data
    if global_rank == 0:
        print(f"Loading data from {args.data}...")
    
    # Initialize variables to track whether data is in memory or memory-mapped
    using_mmap = False
    
    if is_gzip_file(args.data):
        if global_rank == 0:
            print("Detected gzip format, loading into memory...")
        with gzip.open(args.data) as file:
            data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
            # Use a percentage-based split (90% train, 10% validation)
            split_point = int(0.9 * len(data))
            np_train, np_valid = np.split(data, [split_point])
            data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)
            
            if global_rank == 0:
                print(f"Data loaded - Train: {data_train.shape}, Val: {data_val.shape}")
    else:
        if global_rank == 0:
            print("Detected raw format, using true memory mapping...")
        
        # Get file size
        file_size = os.path.getsize(args.data)
        
        # Calculate split point for train/val (90/10 split)
        split_point = int(0.9 * file_size)
        
        if global_rank == 0:
            print(f"File size: {file_size} bytes")
            print(f"Train portion: 0-{split_point} ({split_point} bytes)")
            print(f"Validation portion: {split_point}-{file_size} ({file_size - split_point} bytes)")
            print(f"Data will be accessed via memory mapping")
        
        # Mark that we're using memory mapping
        using_mmap = True
    
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
    
    # Override config values with command line arguments
    SEQ_LEN = seq_len_value
    BATCH_SIZE = batch_size_value
    GRAD_ACCUM_EVERY = grad_accum_value
    LEARNING_RATE = learning_rate_value
    NUM_BATCHES = total_requested_steps
    
    # Get embedding dimension from command line or use default
    embedding_dim = args.embedding_dim if args.embedding_dim is not None else MODEL_CONFIG["num_tokens"]
    
    # Get ff_mult and expansion values from command line or use defaults
    ff_mult_value = args.ff_mult if args.ff_mult is not None else MODEL_CONFIG["ff_mult"]
    expansion_value = args.expansion if args.expansion is not None else MODEL_CONFIG["expansion"]
    
    # Update model config with these values
    MODEL_CONFIG["ff_mult"] = ff_mult_value
    MODEL_CONFIG["expansion"] = expansion_value
    
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
                embedding_dim, 
                MODEL_CONFIG["ff_mult"],  # Now configurable from CLI
                MODEL_CONFIG["expansion"]  # Now configurable from CLI
            )
            if global_rank == 0:
                print(f"Target params: {target_params/1e6:.1f}M, Dimension: {dim}, Calculated depth: {depth}")
        elif dim_value is None and depth_value is not None:
            # If depth is specified but not dimension, solve for dimension
            depth = depth_value
            dim = solve_for_dimension(
                target_params, 
                depth, 
                embedding_dim, 
                MODEL_CONFIG["ff_mult"],  # Now configurable from CLI
                MODEL_CONFIG["expansion"]  # Now configurable from CLI
            )
            if global_rank == 0:
                print(f"Target params: {target_params/1e6:.1f}M, Calculated dimension: {dim}, Depth: {depth}")
        else:
            # If neither or both are specified
            if dim_value is not None and depth_value is not None:
                dim = round_to_multiple(dim_value)
                depth = depth_value
                if global_rank == 0:
                    print(f"Warning: Both dimension and depth specified with target params. Ignoring target params.")
            else:
                # Scale both according to parameter count
                base_params = 15 * 1024 * 1024  # 15M params reference
                base_depth = 6  # Reference depth
                
                # Calculate balanced depth based on parameter count
                if target_params >= base_params:
                    scaling_factor = (target_params / base_params) ** (1/3)
                    depth = max(base_depth, round(base_depth * scaling_factor))
                else:
                    scaling_factor = (target_params / base_params) ** (1/4)
                    depth = max(2, round(base_depth * scaling_factor))
                
                # Solve for dimension with the calculated depth
                dim = solve_for_dimension(
                    target_params, 
                    depth, 
                    MODEL_CONFIG["num_tokens"], 
                    MODEL_CONFIG["ff_mult"],  # Now configurable from CLI
                    MODEL_CONFIG["expansion"]  # Now configurable from CLI
                )
                if global_rank == 0:
                    print(f"Target params: {target_params/1e6:.1f}M, Balanced scaling - Dimension: {dim}, Depth: {depth}")
    else:
        # No target params specified, use explicit values or defaults
        dim = round_to_multiple(dim_value) if dim_value is not None else MODEL_CONFIG["dim"]
        depth = depth_value if depth_value is not None else MODEL_CONFIG["depth"]
        
    # Update model config with the calculated values
    MODEL_CONFIG["dim"] = dim
    MODEL_CONFIG["depth"] = depth
    
    # Check if BF16 is supported when requested
    if args.precision == "bf16" and torch.cuda.is_available():
        if global_rank == 0:
            print("Checking BF16 support for optimal training...")
        
        # Check if BF16 is supported
        if not torch.cuda.is_bf16_supported():
            if global_rank == 0:
                print("WARNING: BF16 is not supported on this device! Falling back to FP32.")
            args.precision = "fp32"
        else:
            if global_rank == 0:
                print("BF16 support confirmed!")
    
    # Create datasets and dataloaders
    if global_rank == 0:
        print(f"Creating datasets with sequence length: {SEQ_LEN}...")
    
    # Create appropriate datasets based on file type
    if not using_mmap:
        # For gzipped/in-memory data, use TextSamplerDataset
        train_dataset = TextSamplerDataset(data_train, SEQ_LEN, seed=42)  # Fixed seed
        val_dataset = TextSamplerDataset(data_val, SEQ_LEN, seed=42)  # Fixed seed
    else:
        # For raw data, use memory-mapped dataset
        # Training dataset uses first 90% of file
        train_dataset = MemoryMappedTextDataset(
            filepath=args.data,
            seq_len=SEQ_LEN,
            offset=0,
            length=split_point,
            seed=42  # Use fixed seed for consistent sampling
        )
        
        # Validation dataset uses last 10% of file
        val_dataset = MemoryMappedTextDataset(
            filepath=args.data,
            seq_len=SEQ_LEN,
            offset=split_point,
            length=file_size - split_point,
            seed=42  # Use fixed seed for consistent sampling
        )
    
    # Calculate optimal workers
    num_workers = min(4, os.cpu_count() or 2)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=num_workers,
        shuffle=False  # Disable shuffle - randomization happens in dataset
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=0,  # Zero workers ensures deterministic behavior
        shuffle=False,
        drop_last=True  # Prevents partial batches
    )
    
    # Generate unique run name
    RUN_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory name
    if global_rank == 0:
        if args.output:
            checkpoint_dir = args.output
        else:
            # Calculate expected model size
            expected_params = calculate_model_size(MODEL_CONFIG)
            params_str = f"{expected_params/1000000:.1f}M" if expected_params >= 1000000 else f"{expected_params/1000:.1f}K"
            checkpoint_dir = f"gruf_{params_str}_{RUN_TIMESTAMP}"
        
        # Create the directory
        print(f"Creating checkpoint directory: {checkpoint_dir}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Set up logging to file for DeepSpeed
        import logging
        ds_logger = logging.getLogger('deepspeed')
        ds_logger.setLevel(logging.INFO)
        
        # Remove any existing handlers to avoid duplicate logs
        for handler in ds_logger.handlers[:]:
            ds_logger.removeHandler(handler)
            
        # Add file handler
        log_file = os.path.join(checkpoint_dir, 'deepspeed.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        ds_logger.addHandler(file_handler)
        
        # Reduce console output - only show warnings and errors for non-verbose mode
        console_level = logging.INFO if args.verbose else logging.WARNING
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        ds_logger.addHandler(console_handler)
        
        # Add info about tensor parallelism
        if args.tensor_parallel_size > 1:
            ds_logger.info(f"Tensor Parallel training enabled with size: {args.tensor_parallel_size}")
        
        print(f"DeepSpeed logs will be written to: {log_file}")
        
        # Save model configuration
        config = {
            **MODEL_CONFIG, 
            **{
                "learning_rate": LEARNING_RATE, 
                "seq_len": SEQ_LEN, 
                "batch_size": BATCH_SIZE,
                "num_batches": NUM_BATCHES,
                "total_steps": total_requested_steps,
                "precision": args.precision,
                "zero_stage": args.zero_stage,
                "offload_optimizer": args.offload_optimizer,
                "offload_parameters": args.offload_parameters,
                "gradient_clip": args.gradient_clip,
                "tensor_parallel_size": args.tensor_parallel_size,
                "schedulefree": args.schedulefree,
                "warmup_steps": args.warmup_steps
            }
        }
        
        with open(os.path.join(checkpoint_dir, "model_config.json"), "w") as f:
            json.dump(config, f, indent=2)
            
        # Create the metrics TSV file
        metrics_log_path = os.path.join(checkpoint_dir, "training_metrics.tsv")
        with open(metrics_log_path, 'w') as f:
            header = [
                "step", "time", "tokens_processed", 
                "tokens_per_sec", "train_loss", "val_loss", "bpb",
                "current_lr", "batch_size", "grad_accum"
            ]
            f.write('\t'.join(header) + '\n')
    else:
        checkpoint_dir = ""
        
    # Synchronize checkpoint directory across processes
    if world_size > 1:
        if global_rank == 0:
            checkpoint_dir_tensor = torch.tensor([ord(c) for c in checkpoint_dir], dtype=torch.long).cuda()
            # Pad to fixed length
            padded_dir = torch.zeros(256, dtype=torch.long).cuda()
            padded_dir[:len(checkpoint_dir_tensor)] = checkpoint_dir_tensor
        else:
            padded_dir = torch.zeros(256, dtype=torch.long).cuda()
            
        # Broadcast from rank 0 to all other ranks
        torch.distributed.broadcast(padded_dir, 0)
        
        # Convert back to string
        if global_rank != 0:
            nonzero_indices = padded_dir.nonzero().squeeze(-1)
            if len(nonzero_indices) > 0:
                str_len = nonzero_indices[-1].item() + 1
                checkpoint_dir = ''.join([chr(i) for i in padded_dir[:str_len].tolist()])
            else:
                checkpoint_dir = ""
    
    if global_rank == 0:
        print(f"Creating model with dimension={dim}, depth={depth}...")
    
    # Update model config with embedding dimension
    MODEL_CONFIG["num_tokens"] = embedding_dim
    
    # Initialize the model trainer
    trainer = MinLMTrainer(
        num_tokens=embedding_dim,
        dim=MODEL_CONFIG["dim"],
        depth=MODEL_CONFIG["depth"],
        ff_mult=MODEL_CONFIG["ff_mult"],
        expansion=MODEL_CONFIG["expansion"],
        conv_kernel_size=MODEL_CONFIG["conv_kernel_size"],
        learning_rate=LEARNING_RATE,
        use_lstm=MODEL_CONFIG["use_lstm"],
        enable_conv=MODEL_CONFIG["enable_conv"],
        dropout=MODEL_CONFIG["dropout"],
        checkpoint_dir=checkpoint_dir,
        world_size=world_size,
        global_rank=global_rank,
        silent_mode=not args.verbose,
        debug_gradients=args.debug_gradients,
        checkpoint_every=args.checkpoint_every,
        permanent_save_interval=args.permanent_save_interval,
        args=args
    )
    
    # Store val_dataset for text generation
    trainer.val_dataset = val_dataset
    
    # Initialize DeepSpeed
    trainer.init_deepspeed(train_loader, args)
    
    # Check for tensor parallel groups consistency if tensor parallelism is enabled
    if args.tensor_parallel_size > 1 and hasattr(deepspeed, 'utils') and hasattr(deepspeed.utils, 'get_tensor_model_parallel_group'):
        if global_rank == 0 and not trainer.silent_mode:
            print("Tensor parallelism enabled - ensuring data consistency across TP groups")
        
        # Get the tensor parallel group
        tp_group = deepspeed.utils.get_tensor_model_parallel_group()
        
        # In TP training, all ranks in the same TP group should see the same data
        # We achieve this by using the same random seed across the TP group
        if tp_group is not None:
            # Get local rank within tensor parallel group
            tp_local_rank = deepspeed.utils.get_tensor_model_parallel_rank()
            # Get global ranks in this tensor parallel group
            tp_world_size = deepspeed.utils.get_tensor_model_parallel_world_size()
            
            if global_rank == 0 and not trainer.silent_mode:
                print(f"TP group size: {tp_world_size}, DP groups: {world_size // tp_world_size}")
                print(f"Using fixed random seed 42 for datasets to ensure consistency within TP groups")
    
    # Print effective batch size
    if global_rank == 0 and not trainer.silent_mode:
        print(f"\n--- Training Configuration ---")
        print(f"Model: {MODEL_CONFIG['depth']} layers, {MODEL_CONFIG['dim']} dimensions")
        print(f"FF Multiplier: {MODEL_CONFIG['ff_mult']}, Expansion Factor: {MODEL_CONFIG['expansion']}")
        print(f"Parameters: {get_parameter_count_str(MODEL_CONFIG)}")
        print(f"Batch size per GPU: {BATCH_SIZE}")
        # With tensor parallelism, the effective data parallel size is reduced
        dp_size = world_size // args.tensor_parallel_size if args.tensor_parallel_size > 0 else world_size
        print(f"Tensor Parallel size: {args.tensor_parallel_size}")
        print(f"Data Parallel size: {dp_size}")
        print(f"Global batch size: {BATCH_SIZE * dp_size}")
        print(f"Gradient accumulation: {GRAD_ACCUM_EVERY}")
        print(f"Effective batch size: {BATCH_SIZE * dp_size * GRAD_ACCUM_EVERY}")
        print(f"Learning rate: {LEARNING_RATE}")
        
        if args.schedulefree:
            print(f"Optimizer: Schedule-Free (beta={args.sf_beta}, weight_decay={args.sf_weight_decay})")
            print(f"Note: Schedule-Free handles adaptation internally (no warmup needed)")
        else:
            print(f"Optimizer: Adam with simple warmup")
            if args.warmup_steps:
                print(f"Warmup steps: {args.warmup_steps}")
        
        # Calculate warmup percentage for display
        warmup_pct = (args.warmup_steps / NUM_BATCHES) * 100 if args.warmup_steps else 0
        
        print(f"Warmup steps: {args.warmup_steps} ({warmup_pct:.1f}% of training)")
        print(f"Sequence length: {SEQ_LEN}")
        print(f"Training steps: {NUM_BATCHES}")
        print(f"ZeRO Stage: {args.zero_stage}")
        print(f"Optimizer offload: {args.offload_optimizer}")
        print(f"Parameter offload: {args.offload_parameters}")
        print(f"Gradient clipping: {args.gradient_clip if args.gradient_clip is not None else '0.5'} (default for all precision types)")
        print(f"Precision: {args.precision.upper()}")
        print(f"Schedule-Free: {args.schedulefree}")
        print(f"Debug gradients: {args.debug_gradients}")
        print(f"Tensor Parallel Size: {args.tensor_parallel_size}")
        print(f"-----------------------------\n")
    
    # If LR finder is enabled, run it before training
    if args.find_lr:
        if global_rank == 0:
            print("\nRunning learning rate finder...")
        
        # Run the learning rate finder
        log_lrs, losses, suggested_lr = trainer.find_learning_rate(
            train_loader,
            min_lr=args.min_find_lr,
            max_lr=args.max_find_lr,
            num_iter=args.num_lr_find_iter,
            show_progress=(global_rank == 0)
        )
        
        # For distributed training, broadcast the suggested_lr from rank 0 to all ranks
        if world_size > 1:
            suggested_lr_tensor = torch.tensor([suggested_lr], dtype=torch.float32, device=f"cuda:{local_rank}")
            torch.distributed.broadcast(suggested_lr_tensor, 0)
            suggested_lr = suggested_lr_tensor.item()
        
        # Save the results to a file
        if global_rank == 0:
            # Save results as CSV
            results_path = os.path.join(checkpoint_dir, "lr_finder_results.csv")
            with open(results_path, 'w') as f:
                f.write("log_lr,loss\n")
                for lr, loss in zip(log_lrs, losses):
                    f.write(f"{lr},{loss}\n")
            
            print(f"\nLearning rate finder results saved to: {results_path}")
            print(f"Suggested learning rate: {suggested_lr:.8f}")
            
            # Create the plot if matplotlib is available
            if MATPLOTLIB_AVAILABLE:
                try:
                    plt.figure(figsize=(10, 6))
                    plt.plot(log_lrs, losses)
                    plt.xlabel("log10(Learning Rate)")
                    plt.ylabel("Loss")
                    plt.title("Learning Rate Finder Results")
                    plt.grid(True)
                    
                    # Mark the suggested learning rate
                    try:
                        # Find the index that's closest to the suggested lr * 10 (pre-division for safety)
                        suggested_log = math.log10(suggested_lr * 10)
                        closest_idx = min(range(len(log_lrs)), key=lambda i: abs(log_lrs[i] - suggested_log))
                        
                        # Mark both the min gradient point and the suggested (safer) learning rate
                        plt.axvline(x=log_lrs[closest_idx], color='r', linestyle='--', label='Min Gradient LR')
                        plt.axvline(x=math.log10(suggested_lr), color='g', linestyle='--', label='Suggested LR')
                        plt.legend()
                    except Exception as e:
                        print(f"Error marking suggested learning rate on plot: {e}")
                    
                    plot_path = os.path.join(checkpoint_dir, "lr_finder_plot.png")
                    plt.savefig(plot_path)
                    print(f"Plot saved to: {plot_path}")
                except Exception as e:
                    print(f"Error creating plot: {e}")
            else:
                print("Matplotlib not available - skipping plot generation")
                # Print tabular data as a simple visualization
                print("\nLearning Rate Finder Results (text format):")
                print("-" * 50)
                print("| {:^15} | {:^15} |".format("Log LR", "Loss"))
                print("-" * 50)
                for i in range(0, len(log_lrs), max(1, len(log_lrs)//10)):  # Print ~10 rows
                    print("| {:^15.4f} | {:^15.4f} |".format(log_lrs[i], losses[i]))
                print("-" * 50)
            
            # Update the learning rate based on finder results
            if suggested_lr > 0:
                print(f"Updating learning rate from {LEARNING_RATE} to {suggested_lr}")
                LEARNING_RATE = suggested_lr
                # Update the trainer's learning rate
                trainer.learning_rate = suggested_lr
                
                # Actually update the optimizer's learning rate directly
                for param_group in trainer.optimizer.param_groups:
                    param_group['lr'] = suggested_lr
                
                # Verify the updated LR
                print(f"Verified optimizer learning rate: {trainer.optimizer.param_groups[0]['lr']}")
                
                # Set default warmup steps if not provided
                if args.warmup_steps is None:
                    # Set a reasonable default: 5% of training
                    args.warmup_steps = int(NUM_BATCHES * 0.05)
                    
                    # Make sure it's not too high - cap at 10% of total steps
                    if args.warmup_steps > NUM_BATCHES * 0.1:
                        args.warmup_steps = int(NUM_BATCHES * 0.1)
                    
                    print(f"Using default warmup_steps: {args.warmup_steps} ({(args.warmup_steps/NUM_BATCHES)*100:.1f}% of training)")
                
                print(f"\nLearning rate configuration:")
                print(f"  Learning rate: {suggested_lr:.6f}")
                if not args.schedulefree:
                    print(f"  Warmup steps: {args.warmup_steps} ({(args.warmup_steps/NUM_BATCHES)*100:.1f}% of training)")
    
    # Resume from checkpoint if specified
    if args.resume:
        # Load checkpoint
        step_offset = trainer.load_checkpoint(args.resume)
        
        # Make sure we're setting the correct optimizer mode before continuing
        if args.schedulefree and hasattr(trainer, 'sf_optimizer') and trainer.sf_optimizer is not None:
            # Print state to help debug
            if global_rank == 0:
                print("Ensuring ScheduleFree optimizer is in correct mode after resume")
            # Explicitly ensure we're in train mode
            trainer.sf_optimizer.train()
            
        # Reset the optimizer's state if there are issues
        if global_rank == 0:
            print(f"Current learning rate after resume: {trainer.optimizer.param_groups[0]['lr']}")
            
        # For older checkpoints or if issues are detected, we can force reset the learning rate
        if args.force_lr_on_resume and args.learning_rate is not None:
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = args.learning_rate
            if global_rank == 0:
                print(f"Forcing learning rate to {args.learning_rate} after resume")
        
        # Adjust remaining steps to account for already completed training
        if args.steps is not None:
            NUM_BATCHES = total_requested_steps - step_offset
            if NUM_BATCHES <= 0:
                if global_rank == 0:
                    print(f"Warning: All {total_requested_steps} requested steps have already been completed.")
                    print("Continuing with additional 1000 training steps.")
                NUM_BATCHES = 1000
        else:
            # If steps not specified, just continue with default number of batches
            if global_rank == 0:
                print(f"Resuming training for {NUM_BATCHES} more steps")
    
    # Start training
    trainer.train(
        train_loader,
        val_loader,
        NUM_BATCHES,
        VALIDATE_EVERY,
        GENERATE_EVERY if args.generate else 0,
        val_batches=4
    )

if __name__ == "__main__":
    main()
