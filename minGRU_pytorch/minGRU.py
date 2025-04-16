# https://arxiv.org/abs/2410.01201v1

import torch
import torch.nn.functional as F
from torch.nn import Linear, Identity, Module

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# appendix B
# https://github.com/glassroom/heinsen_sequence

def heinsen_associative_scan_log(log_coeffs, log_values):
    """
    Optimized version of the Heinsen associative scan in log space.
    Supports both single and batched inputs.
    
    Args:
        log_coeffs: Logarithm of coefficients, shape (batch, seq_len, dim) or (seq_len, dim)
        log_values: Logarithm of values, shape (batch, seq_len, dim) or (seq_len, dim)
        
    Returns:
        Scanned values, shape matches input
    """
    # Ensure inputs have consistent dimensions
    if log_coeffs.ndim == 2 and log_values.ndim == 2:
        # Handle unbatched inputs
        a_star = log_coeffs.cumsum(dim=0)
        log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim=0)
        log_h = a_star + log_h0_plus_b_star
        return log_h.exp()
    else:
        # Standard batched version
        a_star = log_coeffs.cumsum(dim=1)
        log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim=1)
        log_h = a_star + log_h0_plus_b_star
        return log_h.exp()

# Batched version using vmap for better memory efficiency
def batched_heinsen_scan_log(log_coeffs, log_values):
    """
    Memory-efficient implementation using torch.vmap.
    Processes each batch item separately without creating large intermediate tensors.
    
    Args:
        log_coeffs: Logarithm of coefficients, shape (batch, seq_len, dim)
        log_values: Logarithm of values, shape (batch, seq_len, dim)
        
    Returns:
        Scanned values, shape (batch, seq_len, dim)
    """
    # Define single-sequence function for vmapping
    def scan_single_sequence(lc, lv):
        a_star = lc.cumsum(dim=0)
        log_h0_plus_b_star = (lv - a_star).logcumsumexp(dim=0)
        log_h = a_star + log_h0_plus_b_star
        return log_h.exp()
    
    # Apply vmap to process each batch element independently
    return torch.vmap(scan_single_sequence)(log_coeffs, log_values)

# appendix B.3

def g(x):
    return torch.where(x >= 0, x + 0.5, x.sigmoid())

def log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

# log-space version of minGRU - B.3.1
# they enforce the hidden states to be positive

class minGRU(Module):
    def __init__(self, dim, expansion_factor = 1., proj_out = None):
        super().__init__()

        dim_inner = int(dim * expansion_factor)
        proj_out = default(proj_out, expansion_factor != 1.)

        self.to_hidden_and_gate = Linear(dim, dim_inner * 2, bias = False)
        self.to_out = Linear(dim_inner, dim, bias = False) if proj_out else Identity()

    def forward(self, x, prev_hidden = None, return_next_prev_hidden = False):
        seq_len = x.shape[1]
        batch_size = x.shape[0]
        hidden, gate = self.to_hidden_and_gate(x).chunk(2, dim = -1)

        if seq_len == 1:
            # handle sequential

            hidden = g(hidden)
            gate = gate.sigmoid()
            out = torch.lerp(prev_hidden, hidden, gate) if exists(prev_hidden) else (hidden * gate)
        else:
            # parallel

            log_coeffs = -F.softplus(gate)

            log_z = -F.softplus(-gate)
            log_tilde_h = log_g(hidden)
            log_values = log_z + log_tilde_h

            if exists(prev_hidden):
                log_values = torch.cat((prev_hidden.log(), log_values), dim = 1)
                log_coeffs = F.pad(log_coeffs, (0, 0, 1, 0))

            # Use the memory-efficient batched version for large batches/sequences
            if batch_size > 1 and seq_len > 16:  # Threshold where memory savings become significant
                out = batched_heinsen_scan_log(log_coeffs, log_values)
            else:
                # Use regular version for small batches
                out = heinsen_associative_scan_log(log_coeffs, log_values)
            
            out = out[:, -seq_len:]

        next_prev_hidden = out[:, -1:]

        out = self.to_out(out)

        if not return_next_prev_hidden:
            return out

        return out, next_prev_hidden
