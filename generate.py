import os
import argparse
import gzip
import random
import torch
import numpy as np
import json
from torch import Tensor
import time

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
    checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config if provided, otherwise use defaults from MODEL_CONFIG
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Use defaults from MODEL_CONFIG
        config = MODEL_CONFIG
    
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
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint and 'state_dict' in checkpoint['model']:
        # Another possible Lightning format
        model.load_state_dict(checkpoint['model']['state_dict'])
    else:
        # Raw state dict
        model.load_state_dict(checkpoint)
        
    model = model.eval()
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
    Generate text in chunks, feeding forward hidden state between chunks.
    
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
    
    # Timing variables
    start_time = time.time()
    tokens_generated = 0
    
    # Generate tokens in chunks
    while remaining_tokens > 0:
        # Determine how many tokens to generate in this chunk
        current_chunk_size = min(chunk_length, remaining_tokens)
        
        # Generate one chunk
        for _ in range(current_chunk_size):
            # Get logits and new hidden state
            logits, next_prev_hiddens = model(
                out, 
                return_prev_hiddens=True, 
                prev_hiddens=prev_hiddens
            )
            
            # Get logits for the last token
            logits = logits[:, -1]
            
            # Update hidden state for next iteration if model supports caching
            if model.can_cache:
                prev_hiddens = next_prev_hiddens
            
            # Apply top-k filtering and sample
            filtered_logits = top_k(logits, thres=filter_thres)
            sample = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
            
            # Append the new token to the output
            out = torch.cat((out, sample), dim=-1)
            
            tokens_generated += 1
        
        # Update remaining tokens
        remaining_tokens -= current_chunk_size
        
        # Calculate and show progress
        elapsed = time.time() - start_time
        tokens_per_sec = tokens_generated / elapsed if elapsed > 0 else 0
        
        # Call progress callback if provided
        if callback:
            progress = (generation_length - remaining_tokens) / generation_length
            callback(progress, tokens_per_sec)
    
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
        return inp[None, ...]  # Add batch dimension
    
    else:
        # Default to a simple prompt if no primer is provided
        text = "The "
        tokens = [ord(c) for c in text]
        return torch.tensor(tokens, dtype=torch.long)[None, ...]  # Add batch dimension

def main():
    parser = argparse.ArgumentParser(description="Generate text using a trained minLM model")
    
    # Model and data parameters
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--config_path", type=str, default=None, help="Path to model config (optional)")
    parser.add_argument("--device", type=str, default="auto", help="Device to run on: 'cpu', 'cuda', 'cuda:0', etc. (default: 'auto')")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling (default: 1.0)")
    parser.add_argument("--top_k", type=float, default=0.9, help="Threshold for top-k filtering (default: 0.9)")
    parser.add_argument("--chunk_length", type=int, default=64, help="Process sequence in chunks of this length (default: 64)")
    parser.add_argument("--generation_length", type=int, default=512, help="Total number of tokens to generate (default: 512)")
    
    # Input parameters
    parser.add_argument("--primer_file", type=str, default=None, help="File containing primer text (optional)")
    parser.add_argument("--primer_text", type=str, default=None, help="Direct text to use as primer")
    parser.add_argument("--primer_length", type=int, default=128, help="Length of primer sequence (default: 128)")
    parser.add_argument("--random_primer", action="store_true", help="Use a random primer from validation set")
    parser.add_argument("--output_file", type=str, default=None, help="Output file to write generated text (optional)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Load the model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, args.config_path)
    model = model.to(device)
    model.eval()
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Prepare validation dataset if using random primer
    val_dataset = None
    if args.random_primer:
        print("Loading validation dataset for random primer...")
        with gzip.open("./data/enwik8.gz") as file:
            data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
            _, np_valid = np.split(data, [int(90e6)])
            data_val = torch.from_numpy(np_valid)
            
        # Create a simple dataset just for primer selection
        from torch.utils.data import Dataset
        
        class TextSamplerDataset(Dataset):
            def __init__(self, data, seq_len):
                super().__init__()
                self.data = data
                self.seq_len = seq_len
            
            def __len__(self):
                return self.data.size(0) - self.seq_len
                
            def __getitem__(self, index):
                return self.data[index:index + self.seq_len + 1].long()
        
        val_dataset = TextSamplerDataset(data_val, args.primer_length)
    
    # If primer_text is provided directly, use that
    if args.primer_text:
        tokens = [ord(c) for c in args.primer_text]
        if args.primer_length:
            tokens = tokens[:args.primer_length]
        prompt = torch.tensor(tokens, dtype=torch.long)[None, ...]  # Add batch dimension
    else:
        # Otherwise get primer text from file or validation dataset
        prompt = load_primer_text(args.primer_file, args.primer_length, val_dataset)
    
    prompt = prompt.to(device)
    
    # Display the primer text
    primer_text = decode_tokens(prompt[0])
    print(f"\nPrimer text ({len(primer_text)} chars):\n{primer_text}")
    
    # Progress callback
    def progress_callback(progress, tokens_per_sec):
        percent_done = progress * 100
        print(f"Progress: {percent_done:.1f}% | Speed: {tokens_per_sec:.2f} tokens/sec", end="\r")
    
    print(f"\nGenerating {args.generation_length} tokens in chunks of {args.chunk_length}...")
    print(f"Temperature: {args.temperature}, Top-k threshold: {args.top_k}")
    
    # Generate text
    start_time = time.time()
    with torch.no_grad():
        generated = chunked_generation(
            model,
            prompt,
            args.generation_length,
            args.chunk_length,
            args.temperature,
            args.top_k,
            device,
            progress_callback
        )
    
    total_time = time.time() - start_time
    total_tokens = args.generation_length
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
