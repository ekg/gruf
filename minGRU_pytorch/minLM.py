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

class OneHotByteEncoder(nn.Module):
    """Encode bytes as 256-dimensional one-hot vectors instead of using an embedding table"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Convert byte values (0-255) to one-hot vectors
        one_hot = F.one_hot(x, num_classes=256)
        
        # Use the same dtype as the weight parameters in the model
        weight_dtype = self.to_model.weight.dtype if hasattr(self, 'to_model') else None
        if weight_dtype is not None:
            one_hot = one_hot.to(dtype=weight_dtype)
        else:
            one_hot = one_hot.float()
        
        return one_hot

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
        # Projection from 256-dim one-hot vector to model dimension
        self.to_model = nn.Linear(256, dim, bias=False)
        # Replace token embedding with OneHotByteEncoder
        self.one_hot_encoder = OneHotByteEncoder()
        # Give encoder a reference to linear layer so it can match dtype
        self.one_hot_encoder.to_model = self.to_model

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
        # Direct output projection to logits (no bottleneck needed)
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)

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

        # Replace embedding lookup with one-hot encoding
        one_hot = self.one_hot_encoder(x)
        # Ensure one_hot is the same dtype as the linear layer weights
        if one_hot.dtype != self.to_model.weight.dtype:
            one_hot = one_hot.to(dtype=self.to_model.weight.dtype)
        x = self.to_model(one_hot)

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
        # Direct projection to logits
        logits = self.to_logits(embed)

        if not return_loss:
            if not return_prev_hiddens:
                return logits

            return logits, next_prev_hiddens

        loss = F.cross_entropy(
            logits.transpose(1, 2),
            labels
        )

        return loss
