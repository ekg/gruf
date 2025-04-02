import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList, RMSNorm

from minGRU_pytorch.minGRU import minGRU
from minGRU_pytorch.minLSTM import minLSTM

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

class ByteVectorEncoder(nn.Module):
    """Encode bytes as 8-dimensional vectors instead of using an embedding table"""
    def __init__(self, dim=8):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        # Convert byte values (0-255) to binary vectors
        batch_size, seq_len = x.shape
        # Create 8-bit binary representation
        binary = torch.zeros(batch_size, seq_len, 8, device=x.device)
        for i in range(8):
            binary[:, :, i] = (x >> i) & 1
        
        return binary.float()

# classes

def FeedForward(dim, mult = 4):
    dim_inner = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, dim_inner),
        nn.GELU(),
        nn.Linear(dim_inner, dim)
    )

# conv

class CausalDepthWiseConv1d(Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size = kernel_size, groups = dim),
            nn.Conv1d(dim, dim, kernel_size = 1)
        )
    def forward(self, x):
        x = x.transpose(1, 2) # b n d -> b d n
        x = F.pad(x, (self.kernel_size - 1, 0), value = 0.)
        x = self.net(x)
        return x.transpose(1, 2) # b d n -> b n d

# main class

class minLM(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        ff_mult = 4,
        expansion = 1.5,
        conv_kernel_size = 3,
        use_lstm = False,
        enable_conv = False,
        dropout = 0.
    ):
        super().__init__()
        # Replace token embedding with ByteVectorEncoder
        self.byte_encoder = ByteVectorEncoder(dim=8)
        # Projection from 8-dim binary vector to model dimension
        self.byte_to_model = nn.Linear(8, dim, bias=False)

        self.layers = ModuleList([])

        min_rnn_klass = minGRU if not use_lstm else minLSTM

        for _ in range(depth):
            self.layers.append(ModuleList([
                CausalDepthWiseConv1d(dim, conv_kernel_size) if enable_conv else None,
                RMSNorm(dim),
                min_rnn_klass(dim, expansion_factor = expansion),
                RMSNorm(dim),
                FeedForward(dim, mult = ff_mult),
                nn.Dropout(dropout) if dropout > 0. else None
            ]))

        self.norm = RMSNorm(dim)
        # At the output, add bottleneck back to 8 dimensions before final projection
        self.to_byte_vec = nn.Linear(dim, 8, bias=False)
        self.to_logits = nn.Linear(8, num_tokens, bias=False)

        self.can_cache = not enable_conv

    def forward(
        self,
        x,
        return_loss = False,
        return_prev_hiddens = False,
        prev_hiddens = None
    ):

        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        # Replace embedding lookup with byte vector encoding
        byte_vecs = self.byte_encoder(x)
        x = self.byte_to_model(byte_vecs)

        # handle previous hiddens, for recurrent decoding

        if exists(prev_hiddens):
            x = x[:, -1:]

        next_prev_hiddens = []
        prev_hiddens = iter(default(prev_hiddens, []))

        for conv, norm, mingru, ff_norm, ff, dropout in self.layers:

            # conv

            if exists(conv):
                assert len(list(prev_hiddens)) == 0, 'caching not supported for conv version'
                x = conv(x) + x

            # min gru

            prev_hidden = next(prev_hiddens, None)

            min_gru_out, next_prev_hidden = mingru(
                norm(x),
                prev_hidden,
                return_next_prev_hidden = True
            )

            x = min_gru_out + x
            next_prev_hiddens.append(next_prev_hidden)

            # feedforward

            x = ff(ff_norm(x)) + x
            
            # dropout
            
            if exists(dropout):
                x = dropout(x)

        embed = self.norm(x)
        # Add the bottleneck to 8-dim before logits
        byte_embed = self.to_byte_vec(embed)
        logits = self.to_logits(byte_embed)

        if not return_loss:
            if not return_prev_hiddens:
                return logits

            return logits, next_prev_hiddens

        loss = F.cross_entropy(
            logits.transpose(1, 2),
            labels
        )

        return loss
