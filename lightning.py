import os
import gzip
import random
import numpy as np
import math
import time
import argparse
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

# Import the minLM model and configuration
from minGRU_pytorch.minLM import minLM
from config import MODEL_CONFIG, TRAINING_CONFIG, calculate_model_size, get_parameter_count_str

# Override default depth to 6 (already has dim=512 by default)
MODEL_CONFIG["depth"] = 6

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
                
        loss = self.model(batch, return_loss=True)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        
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
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return {"val_loss": loss}
    
    def configure_optimizers(self):
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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a minLM model with PyTorch Lightning")
    parser.add_argument("--gpus", type=str, default=None, 
                        help="Comma-separated list or range of GPU IDs to use (e.g., '0,1,2' or '0-2' or '0,2-4')")
    
    # Model architecture arguments
    parser.add_argument("--dim", type=int, default=None,
                        help="Model hidden dimension (default: 512, will be rounded to multiple of 32)")
    parser.add_argument("--depth", type=int, default=None,
                        help="Number of model layers (default: 6)")
    parser.add_argument("--params", type=float, default=None,
                        help="Target parameter count in millions (e.g., 15 for 15M params). Will adjust dim or depth.")
    
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
    
    # Configure model architecture based on command line arguments
    if args.params is not None:
        # Convert millions to actual count
        target_params = args.params * 1e6
        
        if args.dim is not None and args.depth is None:
            # If dimension is specified but not depth, solve for depth
            dim = round_to_multiple(args.dim)
            depth = solve_for_depth(
                target_params, 
                dim, 
                MODEL_CONFIG["num_tokens"], 
                MODEL_CONFIG["ff_mult"], 
                MODEL_CONFIG["expansion"]
            )
            print(f"Target params: {args.params}M, Dimension: {dim}, Calculated depth: {depth}")
        elif args.dim is None and args.depth is not None:
            # If depth is specified but not dimension, solve for dimension
            depth = args.depth
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
            if args.dim is not None and args.depth is not None:
                print(f"Warning: Both dimension and depth specified with target params. Ignoring target params.")
                dim = round_to_multiple(args.dim)
                depth = args.depth
            else:
                # Default: keep depth=6, solve for dimension
                depth = MODEL_CONFIG["depth"]
                dim = solve_for_dimension(
                    target_params, 
                    depth, 
                    MODEL_CONFIG["num_tokens"], 
                    MODEL_CONFIG["ff_mult"], 
                    MODEL_CONFIG["expansion"]
                )
                print(f"Target params: {args.params}M, Calculated dimension: {dim}, Depth: {depth}")
    else:
        # No target params specified, use explicit values or defaults
        dim = round_to_multiple(args.dim) if args.dim is not None else MODEL_CONFIG["dim"]
        depth = args.depth if args.depth is not None else MODEL_CONFIG["depth"]
        
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
    
    # Create checkpoint directory
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Set up callbacks
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
    
    # Save model configuration for easy reloading
    def save_model_config():
        import json
        # Combine model and training configs
        config = {**MODEL_CONFIG, **{"learning_rate": LEARNING_RATE, "seq_len": SEQ_LEN, "batch_size": BATCH_SIZE}}
        with open(os.path.join(checkpoint_dir, "model_config.json"), "w") as f:
            json.dump(config, f, indent=2)
    
    # Save the config file
    save_model_config()
    
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
    print(f"Steps per epoch per GPU: {steps_per_epoch_per_gpu}")
    print(f"Steps per epoch total: {steps_per_epoch_global}")
    print(f"Tokens per epoch per GPU: {tokens_per_epoch_per_gpu:,}")
    print(f"Tokens per epoch total: {tokens_per_epoch_total:,}")
    print(f"Training for {max_epochs} epochs to reach approximately {NUM_BATCHES} steps")
    print(f"-----------------------------\n")

    # Create trainer
    trainer = pl.Trainer(
        max_steps=NUM_BATCHES,  # Still use max_steps as a hard limit
        accumulate_grad_batches=GRAD_ACCUM_EVERY,
        accelerator="gpu",
        devices=gpu_ids if gpu_ids else "auto",
        strategy=ddp_strategy,
        gradient_clip_val=0.5,
        callbacks=[checkpoint_callback, backup_checkpoint_callback, text_gen_callback, progress_bar],
        val_check_interval=VALIDATE_EVERY,
        logger=csv_logger,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        limit_val_batches=4,
        max_epochs=max_epochs,  # Set max_epochs to ensure we can track progress
        check_val_every_n_epoch=None,  # Still validate based on steps, not epochs
    )

    print(f"Starting training with {torch.cuda.device_count()} GPUs")
    print(f"Config: bs={BATCH_SIZE}, grad_accum={GRAD_ACCUM_EVERY}, lr={LEARNING_RATE}, seq_len={SEQ_LEN}")
    print(f"Will run for {NUM_BATCHES} steps")
    
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
