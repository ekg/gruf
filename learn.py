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

# Dataset class
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
        # Always ensure we return a Long tensor
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
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
        debug_gradients=False
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
        
        # Initialize metric tracking
        self.train_loss = 0.0
        self.val_loss = 0.0
        self.val_bpb = 0.0
        self.global_step = 0
        
        # Checkpoint tracking
        self.best_val_loss = float('inf')
        self.best_val_bpb = float('inf')
        self.best_checkpoints = []  # List of (path, val_loss) tuples to track top_k checkpoints
        self.save_top_k = 3  # Number of best checkpoints to keep

    def init_deepspeed(self, train_dataloader, args):
        """Initialize DeepSpeed engine"""
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
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            model_parameters=self.model.parameters(),
            config=ds_config
        )
        
        self.model = model_engine
        self.optimizer = optimizer
        
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
    
    def _create_scheduler_config(self, scheduler_type, max_lr, min_lr, warmup_steps, 
                                total_steps, decay_rate, cycle_first_step_size, 
                                decay_step_size, decay_lr_rate):
        """Create scheduler configuration based on the selected type"""
        # Set default values if None
        warmup_steps = warmup_steps or max(100, int(total_steps * 0.06))  # Default to 6% of total steps
        min_lr = min_lr if min_lr is not None else max_lr * 0.01  # Default to 1% of max_lr
        decay_rate = decay_rate if decay_rate is not None else min_lr / max_lr
        cycle_first_step_size = cycle_first_step_size or warmup_steps
        decay_step_size = decay_step_size or (total_steps - warmup_steps)
        decay_lr_rate = decay_lr_rate if decay_lr_rate is not None else 0.9
        
        if scheduler_type == "auto":
            # Choose best scheduler based on training length
            if total_steps < 1000:
                scheduler_type = "warmup"
            elif total_steps > 10000:
                scheduler_type = "cosine"
            else:
                scheduler_type = "warmup_decay"
            
            if self.global_rank == 0 and not self.silent_mode:
                print(f"Auto-selected scheduler type: {scheduler_type}")
        
        if scheduler_type == "warmup":
            return {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": min_lr,
                    "warmup_max_lr": max_lr,
                    "warmup_num_steps": warmup_steps
                }
            }
        elif scheduler_type == "warmup_decay":
            return {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": min_lr,
                    "warmup_max_lr": max_lr,
                    "warmup_num_steps": warmup_steps,
                    "total_num_steps": total_steps,
                    "decay_rate": decay_rate  # Final LR = max_lr * decay_rate
                }
            }
        elif scheduler_type == "one_cycle":
            return {
                "type": "OneCycle",
                "params": {
                    "cycle_min_lr": min_lr,
                    "cycle_max_lr": max_lr,
                    "cycle_first_step_size": cycle_first_step_size,
                    "decay_step_size": decay_step_size,
                    "decay_lr_rate": decay_lr_rate
                }
            }
        elif scheduler_type == "cosine":
            return {
                "type": "WarmupCosineLR",
                "params": {
                    "warmup_min_ratio": min_lr / max_lr,  # Ratio of min to max LR during warmup
                    "warmup_num_steps": warmup_steps,
                    "total_num_steps": total_steps,
                    "cos_min_ratio": min_lr / max_lr,  # Final min ratio for cosine schedule
                    "warmup_type": "linear"
                }
            }
        elif scheduler_type == "constant":
            return {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": max_lr,  # No actual warmup, just constant LR
                    "warmup_max_lr": max_lr,
                    "warmup_num_steps": 1
                }
            }
        else:
            # Default to WarmupLR if an invalid type is specified
            print(f"WARNING: Unknown scheduler type '{scheduler_type}', using default WarmupLR")
            return {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": min_lr,
                    "warmup_max_lr": max_lr,
                    "warmup_num_steps": warmup_steps
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
                    # Removed weight_decay to match standard Adam in Lightning
                }
            },
            "scheduler": self._create_scheduler_config(
                args.lr_scheduler,
                learning_rate,
                args.min_lr,
                args.warmup_steps,
                NUM_BATCHES,
                args.decay_rate,
                args.cycle_first_step_size,
                args.decay_step_size,
                args.decay_lr_rate
            ),
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
        
        self.model.step()
        
        # Track loss
        loss_val = loss.detach().float().item()
        self.train_loss = loss_val
        
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
        return {"val_loss": avg_val_loss, "bpb": avg_bpb}
    
    def load_checkpoint(self, checkpoint_path):
        """Load model and optimizer state from checkpoint"""
        if self.global_rank == 0 and not self.silent_mode:
            print(f"Loading checkpoint from {checkpoint_path}")
        
        # Check if this is a directory (DeepSpeed checkpoint) or file (PyTorch checkpoint)
        if os.path.isdir(checkpoint_path):
            # This is a DeepSpeed checkpoint directory
            # DeepSpeed handles loading both model and optimizer state
            success = self.model.load_checkpoint(checkpoint_path)
            if not success:
                raise ValueError(f"Failed to load DeepSpeed checkpoint from {checkpoint_path}")
                
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
            
            # Note: With DeepSpeed, we don't manually load optimizer state
            # DeepSpeed will reinitialize the optimizer with our parameters
        
        if self.global_rank == 0 and not self.silent_mode:
            print(f"✅ Checkpoint loaded. Resuming from step {self.global_step}")
        
        return self.global_step
    
    def save_checkpoint(self, additional_info=None, is_periodic=False):
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
        
        if additional_info:
            checkpoint.update(additional_info)
        
        # Format checkpoint filename with validation metrics
        val_loss_str = f"{self.val_loss:.4f}" if self.val_loss is not None else "NA"
        bpb_str = f"{self.val_bpb:.4f}" if self.val_bpb is not None else "NA"
        
        # Save checkpoint with informative name
        filename = f"minlm-step-{self.global_step}-loss-{val_loss_str}-bpb-{bpb_str}.pt"
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        
        # Save latest checkpoint (for resuming)
        latest_path = os.path.join(self.checkpoint_dir, "latest.pt")
        torch.save(checkpoint, latest_path)
        
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
                    if os.path.exists(path) and "best" not in path and "latest" not in path:
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
        # Set start time if it's not already set (could be set when resuming)
        if self.start_time is None:
            self.start_time = time.time()
        
        # Create metrics log file
        if self.global_rank == 0 and self.checkpoint_dir:
            metrics_log_path = os.path.join(self.checkpoint_dir, "training_metrics.tsv")
            with open(metrics_log_path, 'w') as f:
                header = [
                    "step", "time", "tokens_processed", 
                    "tokens_per_sec", "train_loss", "val_loss", "bpb",
                    "current_lr", "batch_size", "grad_accum"
                ]
                f.write('\t'.join(header) + '\n')
        
        # Initial validation
        if self.global_rank == 0:
            val_results = self.validate(val_dataloader, max_batches=val_batches)
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
                    tokens_per_sec = self.global_tokens.item() / elapsed if elapsed > 0 else 0
                    val_info = f"Val: {self.val_loss:.4f} BPB: {self.val_bpb:.4f} | " if self.val_loss > 0 else ""
                    pbar.set_description(f"Loss: {loss:.4f} | {val_info}{tokens_per_sec:.2f} tok/s")
                    pbar.update(1)
                
                # Log progress
                if self.global_rank == 0 and step % 10 == 0:
                    self._log_metrics(False)
                    
                # Save periodic checkpoint every 1000 steps for safety
                if self.global_rank == 0 and step > 0 and step % 1000 == 0:
                    self.save_checkpoint({"periodic": True}, is_periodic=True)
                    if not self.silent_mode:
                        print(f"Saved periodic checkpoint at step {step}")
                
                # Validate periodically
                if step > 0 and step % validate_every == 0:
                    if self.global_rank == 0:
                        val_results = self.validate(val_dataloader, max_batches=val_batches)
                        
                        # Save checkpoint with validation results
                        self.save_checkpoint()
                        
                        # Log metrics
                        self._log_metrics(True)
                
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
        
        # Save initial model parameters and optimizer state
        # For DeepSpeed, we need to save checkpoint first
        temp_dir = os.path.join(self.checkpoint_dir, "lr_finder_temp") if self.checkpoint_dir else "lr_finder_temp"
        if self.global_rank == 0:
            os.makedirs(temp_dir, exist_ok=True)
        
        # Save the initial model state
        if hasattr(self.model, 'save_checkpoint'):
            self.model.save_checkpoint(temp_dir, tag="init")
        
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
        
        # Restore the original model state
        if hasattr(self.model, 'load_checkpoint'):
            self.model.load_checkpoint(temp_dir, tag="init")
        
        # Clean up temp directory
        if self.global_rank == 0 and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
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
            
    def calculate_optimal_lr_schedule(self, total_steps, max_lr, min_lr=None, model_dim=None, model_depth=None):
        """
        Calculate optimal learning rate schedule parameters based on model size and training steps.
        
        Args:
            total_steps: Total number of training steps
            max_lr: Maximum learning rate (from LR finder)
            min_lr: Minimum final learning rate (optional)
            model_dim: Model dimension (optional)
            model_depth: Model depth (optional)
            
        Returns:
            dict: Dictionary containing scheduler configuration
        """
        # Use model size to determine sensible defaults if available
        model_params = 0
        if model_dim is not None and model_depth is not None:
            # Calculate an estimate of model params
            model_params = model_dim * model_dim * model_depth * 4
        
        # Calculate min_lr if not provided
        if min_lr is None:
            # Scale down based on model size - larger models need higher final LR
            if model_params > 100_000_000:  # >100M params
                min_lr = max_lr * 0.05
            elif model_params > 10_000_000:  # >10M params
                min_lr = max_lr * 0.03
            else:
                min_lr = max_lr * 0.01  # Small models can go to 1% of max
        
        # Calculate warmup steps based on training length and model size
        # Larger models and longer training runs benefit from longer warmup
        warmup_pct = 0.06  # Default 6% for warmup
        
        # Adjust based on model size
        if model_params > 100_000_000:  # >100M params
            warmup_pct = 0.10  # 10% for large models
        elif model_params > 10_000_000:  # >10M params
            warmup_pct = 0.08  # 8% for medium models
            
        # Calculate actual steps
        warmup_steps = max(100, int(total_steps * warmup_pct))
        
        # For small/short trainings, cap warmup at 20% of total training
        if warmup_steps > total_steps * 0.2:
            warmup_steps = int(total_steps * 0.2)
            
        # Determine best scheduler based on training length
        scheduler_type = "warmup_decay"  # Default
        
        # For very short training runs, simple warmup might be enough
        if total_steps < 1000:
            scheduler_type = "warmup"
        # For longer runs, cosine is often better
        elif total_steps > 10000:
            scheduler_type = "cosine"
        
        # For one_cycle params (if that scheduler is chosen)
        cycle_first_step_size = warmup_steps
        decay_step_size = total_steps - warmup_steps
        decay_lr_rate = 0.9
        
        return {
            "scheduler_type": scheduler_type,
            "warmup_steps": warmup_steps,
            "warmup_pct": warmup_pct,
            "min_lr": min_lr,
            "max_lr": max_lr,
            "cycle_first_step_size": cycle_first_step_size,
            "decay_step_size": decay_step_size,
            "decay_lr_rate": decay_lr_rate,
            "decay_rate": min_lr / max_lr  # For warmup_decay
        }
    
    def _log_metrics(self, is_validation=False):
        """Log metrics to TSV file"""
        if not self.checkpoint_dir or self.global_rank != 0:
            return
            
        metrics_log_path = os.path.join(self.checkpoint_dir, "training_metrics.tsv")
        
        try:
            with open(metrics_log_path, 'a') as f:
                elapsed = time.time() - self.start_time
                global_tokens = self.global_tokens.item()
                tokens_per_sec = global_tokens / elapsed if elapsed > 0 else 0
                
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
                
                f.write('\t'.join(values) + '\n')
        except Exception as e:
            print(f"Warning: Could not write to metrics log: {e}")
    
    def _generate_sample(self, prime_length=PRIME_LENGTH, gen_length=GENERATE_LENGTH):
        """Generate a text sample during training"""
        if not hasattr(self, 'val_dataset') or self.val_dataset is None:
            print("No validation dataset provided for generation")
            return
            
        # Get a random sample from validation data
        rand_start = torch.randint(0, len(self.val_dataset.data) - prime_length - 1, (1,))
        # Ensure prime is a Long tensor before passing to the model
        prime = self.val_dataset.data[rand_start:rand_start + prime_length].long().unsqueeze(0).to(self.model.device)
        
        # Generate text
        if not self.silent_mode:
            print("\nGenerating sample text...")
            print(f"Prime: {decode_tokens(prime[0])}")
        
        self.model.eval()
        with torch.no_grad():
            generated = base_decoding(
                self.model, 
                prime, 
                gen_length, 
                temperature=0.8, 
                filter_thres=0.9
            )
        self.model.train()
        
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
    parser.add_argument("--depth", type=int, default=None,
                        help="Number of model layers (default: 6).")
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
    
    # Learning rate scheduler parameters
    parser.add_argument("--lr_scheduler", type=str, default="auto",
                        choices=["auto", "warmup", "warmup_decay", "one_cycle", "cosine", "constant"],
                        help="Learning rate scheduler type (default: auto - chooses based on LR finder)")
    parser.add_argument("--warmup_steps", type=int, default=None,
                        help="Number of warmup steps (default: auto-calculated based on total steps)")
    parser.add_argument("--warmup_pct", type=float, default=None,
                        help="Percentage of training for warmup phase (default: auto-calculated)")
    parser.add_argument("--min_lr", type=float, default=None,
                        help="Minimum learning rate for schedulers (default: auto-calculated from LR finder)")
    parser.add_argument("--decay_rate", type=float, default=None,
                        help="Final LR = max_lr * decay_rate (for warmup_decay, default: auto-calculated)")
    parser.add_argument("--cycle_first_step_size", type=int, default=None,
                        help="First cycle step size for OneCycle scheduler (default: auto-calculated)")
    parser.add_argument("--decay_step_size", type=int, default=None,
                        help="Decay step size for OneCycle scheduler (default: auto-calculated)")
    parser.add_argument("--decay_lr_rate", type=float, default=None,
                        help="Decay rate per step for OneCycle scheduler (default: auto-calculated)")
    
    # Learning rate finder parameters
    parser.add_argument("--find_lr", action="store_true",
                        help="Run the learning rate finder before training")
    parser.add_argument("--min_find_lr", type=float, default=1e-8,
                        help="Minimum learning rate for LR finder (default: 1e-8)")
    parser.add_argument("--max_find_lr", type=float, default=1.0,
                        help="Maximum learning rate for LR finder (default: 1.0)")
    parser.add_argument("--num_lr_find_iter", type=int, default=100,
                        help="Number of iterations for LR finder (default: 100)")
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
    
    if is_gzip_file(args.data):
        if global_rank == 0:
            print("Detected gzip format, loading into memory...")
        with gzip.open(args.data) as file:
            data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
            np_train, np_valid = np.split(data, [int(90e6)])
            data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)
    else:
        if global_rank == 0:
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
    
    if global_rank == 0:
        print(f"Data loaded - Train: {data_train.shape}, Val: {data_val.shape}")
    
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
                MODEL_CONFIG["ff_mult"], 
                MODEL_CONFIG["expansion"]
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
                MODEL_CONFIG["ff_mult"], 
                MODEL_CONFIG["expansion"]
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
                    MODEL_CONFIG["ff_mult"], 
                    MODEL_CONFIG["expansion"]
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
    
    train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
    val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
    
    # Calculate optimal workers
    num_workers = min(4, os.cpu_count() or 2)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=num_workers,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=num_workers,
        shuffle=False
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
                "lr_scheduler": args.lr_scheduler,
                "warmup_steps": args.warmup_steps,
                "min_lr": args.min_lr,
                "decay_rate": args.decay_rate
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
        debug_gradients=args.debug_gradients
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
    
    # Print effective batch size
    if global_rank == 0 and not trainer.silent_mode:
        print(f"\n--- Training Configuration ---")
        print(f"Model: {MODEL_CONFIG['depth']} layers, {MODEL_CONFIG['dim']} dimensions")
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
        print(f"LR Scheduler: {args.lr_scheduler}")
        
        # Calculate warmup percentage for display
        warmup_pct = (args.warmup_steps / NUM_BATCHES) * 100 if args.warmup_steps else 0
        min_lr_pct = (args.min_lr / LEARNING_RATE) * 100 if args.min_lr is not None else 0
        
        print(f"Warmup steps: {args.warmup_steps} ({warmup_pct:.1f}% of training)")
        if args.lr_scheduler in ['warmup_decay', 'cosine', 'one_cycle']:
            print(f"Min LR: {args.min_lr if args.min_lr is not None else (LEARNING_RATE * 0.01):.6f} "
                  f"({min_lr_pct:.1f}% of max)")
        print(f"Sequence length: {SEQ_LEN}")
        print(f"Training steps: {NUM_BATCHES}")
        print(f"ZeRO Stage: {args.zero_stage}")
        print(f"Optimizer offload: {args.offload_optimizer}")
        print(f"Parameter offload: {args.offload_parameters}")
        print(f"Gradient clipping: {args.gradient_clip if args.gradient_clip is not None else '0.5'} (default for all precision types)")
        print(f"Precision: {args.precision.upper()}")
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
                
                # Calculate optimal learning rate schedule if using auto mode
                if args.lr_scheduler == "auto":
                    schedule_params = trainer.calculate_optimal_lr_schedule(
                        total_steps=NUM_BATCHES,
                        max_lr=suggested_lr,
                        min_lr=args.min_lr,
                        model_dim=MODEL_CONFIG["dim"],
                        model_depth=MODEL_CONFIG["depth"]
                    )
                    
                    # Use calculated parameters but allow command-line overrides
                    if args.warmup_steps is None:
                        args.warmup_steps = schedule_params["warmup_steps"]
                    if args.min_lr is None:
                        args.min_lr = schedule_params["min_lr"]
                    if args.decay_rate is None:
                        args.decay_rate = schedule_params["decay_rate"]
                    if args.cycle_first_step_size is None:
                        args.cycle_first_step_size = schedule_params["cycle_first_step_size"]
                    if args.decay_step_size is None:
                        args.decay_step_size = schedule_params["decay_step_size"]
                    if args.decay_lr_rate is None:
                        args.decay_lr_rate = schedule_params["decay_lr_rate"]
                    
                    # Set scheduler type if using auto
                    if args.lr_scheduler == "auto":
                        args.lr_scheduler = schedule_params["scheduler_type"]
                    
                    print(f"\nAutomatic LR schedule configuration:")
                    print(f"  Scheduler type: {args.lr_scheduler}")
                    print(f"  Warmup steps: {args.warmup_steps} ({(args.warmup_steps/NUM_BATCHES)*100:.1f}% of training)")
                    print(f"  Max LR: {suggested_lr:.6f}")
                    print(f"  Min LR: {args.min_lr:.6f} ({(args.min_lr/suggested_lr)*100:.1f}% of max)")
    
    # Resume from checkpoint if specified
    if args.resume:
        # Load checkpoint
        step_offset = trainer.load_checkpoint(args.resume)
        
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
