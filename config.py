"""
Central configuration for minGRU/minLM model parameters
"""

# Training configuration
TRAINING_CONFIG = {
    "num_batches": int(1e5),
    "batch_size": 2,
    "grad_accum_every": 8,
    "learning_rate": 1e-3,
    "validate_every": 128,
    "prime_length": 128,
    "generate_every": 128,
    "generate_length": 512,
    "seq_len": 1024 * 8,
}

# Model architecture configuration
MODEL_CONFIG = {
    "num_tokens": 256,
    "dim": 512,          # Hidden dimension size
    "depth": 6,          # Number of layers
    "ff_mult": 4,        # Feedforward multiplier
    "expansion": 1.5,    # Expansion factor for minGRU/minLSTM
    "conv_kernel_size": 3,
    "use_lstm": False,   # Whether to use minLSTM instead of minGRU
    "enable_conv": False, # Whether to use causal convolution
    "dropout": 0.0,      # Dropout rate
}

def calculate_model_size(config=None):
    """
    Calculate the number of parameters in the model based on configuration
    """
    if config is None:
        config = MODEL_CONFIG
    
    dim = config["dim"]
    depth = config["depth"]
    ff_mult = config["ff_mult"]
    expansion = config["expansion"]
    num_tokens = config["num_tokens"]
    
    # ByteVectorEncoder parameters (replacing token embedding)
    # 8-dim to model dim projection
    byte_to_model_params = 8 * dim
    
    # Each layer has:
    # 1. minGRU/minLSTM parameters
    rnn_input_size = dim
    rnn_hidden_size = int(dim * expansion)
    
    if config["use_lstm"]:
        # minLSTM: input -> (hidden, f_gate, i_gate)
        rnn_params_per_layer = rnn_input_size * (rnn_hidden_size * 3)
    else:
        # minGRU: input -> (hidden, gate)
        rnn_params_per_layer = rnn_input_size * (rnn_hidden_size * 2)
    
    # Optional projection out in minGRU/minLSTM
    if expansion != 1.0:
        rnn_params_per_layer += rnn_hidden_size * dim
    
    # 2. RMSNorm parameters
    norm_params_per_layer = dim * 2  # Two norms per layer
    
    # 3. FeedForward parameters
    ff_inner_dim = dim * ff_mult
    ff_params_per_layer = (dim * ff_inner_dim) + (ff_inner_dim * dim)
    
    # 4. Optional conv parameters
    conv_params_per_layer = 0
    if config["enable_conv"]:
        kernel_size = config["conv_kernel_size"]
        # Depthwise conv: dim * kernel_size * 1 (groups=dim)
        # Pointwise conv: dim * dim (1x1 conv)
        conv_params_per_layer = (dim * kernel_size) + (dim * dim)
    
    # Total parameters per layer
    params_per_layer = rnn_params_per_layer + norm_params_per_layer + ff_params_per_layer + conv_params_per_layer
    
    # Final norm, bottleneck to 8-dim, and projection to logits
    final_params = dim + (dim * 8) + (8 * num_tokens)
    
    # Total parameters
    total_params = byte_to_model_params + (params_per_layer * depth) + final_params
    
    return total_params

def get_parameter_count_str(config=None):
    """
    Get a formatted string with model parameter count
    """
    count = calculate_model_size(config)
    
    # Format with commas for large numbers and convert to millions if large enough
    if count >= 1_000_000:
        formatted = f"{count/1_000_000:.2f}M ({count:,})"
    else:
        formatted = f"{count:,}"
        
    return formatted
