import os
import argparse
import gzip
import random
import torch
import numpy as np
import json
import re
from torch import Tensor
import time
import mmap

# Import the minLM model and configuration
from minGRU_pytorch.minLM import minLM
from config import MODEL_CONFIG

# Token decoding functions
def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

# Sampling helpers
def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature=1., dim=-1, keepdim=True):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim, keepdim=keepdim)

def top_k(logits, thres=0.9):
    import math
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(-1, ind, val)
    return probs

def load_model(checkpoint_path, config_path=None, use_bf16=False, use_fp16=False, device=None):
    """
    Load a trained minLM model from checkpoint
    
    Args:
        checkpoint_path: Path to the model checkpoint
        config_path: Path to the model config file (optional)
        use_bf16: Whether to load model in BF16 precision (default: False)
        use_fp16: Whether to load model in FP16 precision (default: False)
        device: Device to load model on (default: auto-detect)
    
    Returns:
        Loaded model
    """
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)  # First load to CPU to avoid OOM
    
    # Load config if provided, otherwise use defaults from MODEL_CONFIG
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Use defaults from MODEL_CONFIG
        config = MODEL_CONFIG
    
    print(f"Creating model with dimension={config['dim']}, depth={config['depth']}...")
    
    # Create model with the correct configuration
    model = minLM(
        num_tokens=config["num_tokens"],
        dim=config["dim"],
        depth=config["depth"],
        ff_mult=config["ff_mult"],
        expansion=config.get("expansion", 1.5),
        conv_kernel_size=config.get("conv_kernel_size", 3),
        use_lstm=config.get("use_lstm", False),
        enable_conv=config.get("enable_conv", False),
        dropout=config.get("dropout", 0.0)
    )
    
    # Load model weights - handling different checkpoint formats
    if 'state_dict' in checkpoint:
        # Standard Lightning checkpoint format
        pl_state_dict = checkpoint['state_dict']
            
        # The model in LightningMinLM is stored under 'model.' prefix
        model_state_dict = {}
        for key, value in pl_state_dict.items():
            # Remove the 'model.' prefix from keys
            if key.startswith('model.'):
                model_state_dict[key[6:]] = value
            
        model.load_state_dict(model_state_dict)
    elif 'model_state_dict' in checkpoint:
        # Our custom Lightning checkpoint format
        # First check if this is a compiled model (has '_orig_mod.' prefix)
        if any(key.startswith('_orig_mod.') for key in checkpoint['model_state_dict']):
            print("Detected compiled model checkpoint (_orig_mod. prefix)")
            # Remove the '_orig_mod.' prefix from all keys
            fixed_state_dict = {}
            for key, value in checkpoint['model_state_dict'].items():
                if key.startswith('_orig_mod.'):
                    fixed_state_dict[key[10:]] = value  # Remove '_orig_mod.' prefix
                else:
                    fixed_state_dict[key] = value
            model.load_state_dict(fixed_state_dict)
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint and 'state_dict' in checkpoint['model']:
        # Another possible Lightning format
        model.load_state_dict(checkpoint['model']['state_dict'])
    else:
        # Raw state dict - try direct loading
        try:
            model.load_state_dict(checkpoint)
        except (RuntimeError, KeyError) as e:
            # Handle DeepSpeed formats by checking for different module prefixes
            print(f"Direct loading failed, trying to match keys: {str(e)}")
            if isinstance(checkpoint, dict):
                # Try to adapt keys if they don't match directly
                ds_state_dict = {}
                # Check for module prefixes used by DeepSpeed
                for key, value in checkpoint.items():
                    if key.startswith('module.model.'):
                        # DeepSpeed might add 'module.' prefix
                        ds_state_dict[key[13:]] = value  # Remove 'module.model.'
                    elif key.startswith('model.'):
                        ds_state_dict[key[6:]] = value  # Remove 'model.'
                    elif key.startswith('_orig_mod.'):
                        # Handle compiled model saved with torch.compile
                        ds_state_dict[key[10:]] = value  # Remove '_orig_mod.' prefix
                    else:
                        ds_state_dict[key] = value  # Keep as is
                    
                # Try loading with adapted keys
                model.load_state_dict(ds_state_dict)
                print("Successfully loaded model with adapted keys")
    
    # Set model to evaluation mode
    model = model.eval()
    
    # Apply precision conversion
    if device == 'cuda':
        if use_bf16 and torch.cuda.is_available():
            model = model.to(torch.bfloat16)
            print("Model converted to BF16 precision")
        elif use_fp16 and torch.cuda.is_available():
            model = model.to(torch.float16)
            print("Model converted to FP16 precision")
    
    # Move model to device
    model = model.to(device)
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters on {device}")
    
    return model

def chunked_generation(
    model,
    prompt: torch.Tensor,
    generation_length: int,
    chunk_length: int,
    temperature: float = 1.0,
    filter_thres: float = 0.9,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    callback=None
):
    """
    Generate text in chunks using RNN hidden states.
    
    Args:
        model: The minLM model
        prompt: Starting prompt tensor
        generation_length: Total length to generate
        chunk_length: Length to process in each forward pass
        temperature: Temperature for sampling (higher = more random)
        filter_thres: Threshold for top-k filtering (higher = more diverse)
        device: Device to run generation on
        callback: Optional callback function to report progress
    
    Returns:
        Generated tokens
    """
    # Move prompt to device
    prompt = prompt.to(device)
    
    # Start with the prompt
    out = prompt.clone()
    
    # Track number of tokens to generate
    remaining_tokens = generation_length
    
    # Start with no hidden state
    prev_hiddens = None
    
    # Process the prompt efficiently to get initial hidden state
    with torch.no_grad():
        # Process the entire prompt at once to get the initial hidden state
        _, prev_hiddens = model(out, return_prev_hiddens=True)
    
    # Timing variables
    start_time = time.time()
    tokens_generated = 0
    
    # Efficient memory management
    torch.cuda.empty_cache()
    
    # Generate tokens in chunks
    with torch.no_grad():
        while remaining_tokens > 0:
            # Generate tokens efficiently in batches
            batch_size = min(chunk_length, remaining_tokens)
            
            # Generate tokens one at a time, efficiently maintaining RNN state
            for _ in range(batch_size):
                # Get logits and updated hidden state
                logits, next_prev_hiddens = model(
                    out[:, -1:],  # Only need the last token with RNN state
                    return_prev_hiddens=True, 
                    prev_hiddens=prev_hiddens
                )
                
                # Get logits for the generated token
                logits = logits[:, -1]
                
                # Update hidden state for next iteration
                prev_hiddens = next_prev_hiddens
                
                # Apply top-k filtering and sample
                filtered_logits = top_k(logits, thres=filter_thres)
                sample = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
                
                # Ensure sample is Long before concatenation
                sample_long = sample.long()
                # Append the new token to the output
                out = torch.cat((out, sample_long), dim=-1)
                
                tokens_generated += 1
                
                # Update progress regularly
                if tokens_generated % 5 == 0 and callback:
                    progress = tokens_generated / generation_length
                    elapsed = time.time() - start_time
                    tokens_per_sec = tokens_generated / elapsed if elapsed > 0 else 0
                    callback(progress, tokens_per_sec)
            
            # Update remaining tokens
            remaining_tokens -= batch_size
            
            # Calculate and show progress for the whole chunk
            elapsed = time.time() - start_time
            tokens_per_sec = tokens_generated / elapsed if elapsed > 0 else 0
            
            # Call progress callback if provided
            if callback:
                progress = (generation_length - remaining_tokens) / generation_length
                callback(progress, tokens_per_sec)
            
            # Periodically clear unused memory
            if tokens_generated % 100 == 0:
                torch.cuda.empty_cache()
    
    # Return only the newly generated tokens (excluding the prompt)
    return out[..., prompt.shape[-1]:]

def load_primer_text(primer_file=None, primer_length=None, val_dataset=None):
    """
    Load primer text either from a file, or randomly from validation dataset
    """
    if primer_file:
        # Load from file
        with open(primer_file, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Convert to tensor of byte values
        tokens = [ord(c) for c in text]
        if primer_length:
            tokens = tokens[:primer_length]
        return torch.tensor(tokens, dtype=torch.long)[None, ...]  # Add batch dimension
    
    elif val_dataset:
        # Random sample from validation set
        inp = random.choice(val_dataset)
        if primer_length:
            inp = inp[:primer_length]
        # Ensure this is a Long tensor
        return inp.long()[None, ...]  # Add batch dimension and convert to long
    
    else:
        # Default to a simple prompt if no primer is provided
        text = "The "
        tokens = [ord(c) for c in text]
        return torch.tensor(tokens, dtype=torch.long)[None, ...]  # Add batch dimension

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

def main():
    parser = argparse.ArgumentParser(description="Generate text using a trained minLM model")
    
    # Model and data parameters
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--config_path", type=str, default=None, help="Path to model config (optional)")
    parser.add_argument("--device", type=str, default="auto", help="Device to run on: 'cpu', 'cuda', 'cuda:0', etc. (default: 'auto')")
    parser.add_argument("--use-f32", dest="use_bf16", action="store_false", default=True,
                        help="Use FP32 precision instead of BF16 (default: BF16)")
    parser.add_argument("--use-fp16", action="store_true", default=False,
                        help="Use FP16 precision instead of BF16/FP32 (default: False)")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling (default: 1.0)")
    parser.add_argument("--top_k", type=float, default=0.9, help="Threshold for top-k filtering (default: 0.9)")
    parser.add_argument("--chunk_length", type=str, default="64", help="Process sequence in chunks of this length (default: 64). Can use k/m/g suffix.")
    parser.add_argument("--generation_length", type=str, default="512", help="Total number of tokens to generate (default: 512). Can use k/m/g suffix.")
    
    # Input parameters
    parser.add_argument("--primer_file", type=str, default=None, help="File containing primer text (optional)")
    parser.add_argument("--primer_text", type=str, default=None, help="Direct text to use as primer")
    parser.add_argument("--primer_length", type=str, default="128", help="Length of primer sequence (default: 128). Can use k/m/g suffix.")
    parser.add_argument("--random_primer", action="store_true", help="Use a random primer from validation set")
    parser.add_argument("--data", type=str, default="./data/enwik8.gz", help="Path to data file for random primer (default: ./data/enwik8.gz)")
    parser.add_argument("--output_file", type=str, default=None, help="Output file to write generated text (optional)")
    parser.add_argument("--memory_efficient", action="store_true", help="Use memory-efficient generation (slower but uses less VRAM)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Load the model
    print(f"Loading model from {args.model}...")
    model = load_model(
        checkpoint_path=args.model, 
        config_path=args.config_path, 
        use_bf16=args.use_bf16,
        use_fp16=args.use_fp16,
        device=device
    )
    
    # Helper function to detect if a file is gzipped
    def is_gzip_file(filepath):
        with open(filepath, 'rb') as test_f:
            return test_f.read(2) == b'\x1f\x8b'
    
    # Prepare validation dataset if using random primer
    val_dataset = None
    if args.random_primer:
        data_path = args.data
        print(f"Loading validation dataset for random primer from {data_path}...")
        
        if is_gzip_file(data_path):
            print("Detected gzip format, loading into memory...")
            with gzip.open(data_path) as file:
                data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
                _, np_valid = np.split(data, [int(90e6)])
                data_val = torch.from_numpy(np_valid)
        else:
            print("Detected raw format, using memory mapping...")
            # Get file size
            file_size = os.path.getsize(data_path)
            # Map the file into memory
            with open(data_path, 'r+b') as f:
                mm = mmap.mmap(f.fileno(), 0)
                # Create a numpy array using the memory map
                data = np.frombuffer(mm, dtype=np.uint8, count=min(int(95e6), file_size))
                # Get validation data
                train_size = min(int(90e6), len(data))
                np_valid = data[train_size:min(int(95e6), len(data))]
                data_val = torch.from_numpy(np_valid)
        
        # Create a simple dataset just for primer selection
        from torch.utils.data import Dataset
        
        class TextSamplerDataset(Dataset):
            def __init__(self, data, seq_len):
                super().__init__()
                # Ensure data is a Long tensor
                self.data = data.long() if data.dtype != torch.long else data
                self.seq_len = seq_len
            
            def __len__(self):
                return self.data.size(0) - self.seq_len
                
            def __getitem__(self, index):
                return self.data[index:index + self.seq_len + 1].long()
        
        val_dataset = TextSamplerDataset(data_val, args.primer_length)
    
    # Parse numerical arguments with potential suffixes
    chunk_length = int(parse_size_with_suffix(args.chunk_length))
    generation_length = int(parse_size_with_suffix(args.generation_length))
    primer_length = int(parse_size_with_suffix(args.primer_length)) if args.primer_length else None

    # If primer_text is provided directly, use that
    if args.primer_text:
        tokens = [ord(c) for c in args.primer_text]
        if primer_length:
            tokens = tokens[:primer_length]
        prompt = torch.tensor(tokens, dtype=torch.long)[None, ...]  # Add batch dimension
    else:
        # Otherwise get primer text from file or validation dataset
        prompt = load_primer_text(args.primer_file, primer_length, val_dataset)
    
    # Ensure prompt is a Long tensor before sending to device
    prompt = prompt.long().to(device)
    
    # Display the primer text
    primer_text = decode_tokens(prompt[0])
    print(f"\nPrimer text ({len(primer_text)} chars):\n{primer_text}")
    
    # Progress callback
    def progress_callback(progress, tokens_per_sec):
        percent_done = progress * 100
        print(f"Progress: {percent_done:.1f}% | Speed: {tokens_per_sec:.2f} tokens/sec", end="\r")
    
    print(f"\nGenerating {generation_length} tokens in chunks of {chunk_length}...")
    print(f"Temperature: {args.temperature}, Top-k threshold: {args.top_k}")
    
    # Memory optimization
    if args.memory_efficient:
        print("Using memory-efficient generation mode")
        # Clear CUDA cache before generation
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    
    # Generate text
    start_time = time.time()
    with torch.no_grad():
        generated = chunked_generation(
            model,
            prompt,
            generation_length,
            chunk_length,
            args.temperature,
            args.top_k,
            device,
            progress_callback
        )
        
    # Clear cache after generation
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    total_tokens = generation_length
    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
    
    # Decode the generated text
    generated_text = decode_tokens(generated[0])
    
    # Display the generated text
    print(f"\n\nGeneration complete! ({tokens_per_sec:.2f} tokens/sec)")
    print(f"Generated text ({len(generated_text)} chars):")
    print(generated_text)
    
    # Save to file if requested
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(f"PRIMER:\n{primer_text}\n\nGENERATED:\n{generated_text}")
        print(f"Output saved to {args.output_file}")

if __name__ == "__main__":
    main()
