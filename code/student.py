"""Round-one model: a near-parameter-matched SwiGLU replacement for the FFN.

Set ``ffn='gelu'`` to construct the untouched classroom GPT. The default SwiGLU
keeps the remaining architecture and initialization policy unchanged. Its
hidden width is configurable through ``ffn_hidden``; 128-wide models default
to 344 hidden units, versus 512 in the original two-projection GELU FFN.
"""
from torch import nn
from torch.nn import functional as F

from model import Block, GPT


class SwiGLU(nn.Module):
    """Two input projections, a SiLU gate, and one output projection."""

    def __init__(self, width, hidden):
        super().__init__()
        self.gate = nn.Linear(width, hidden)
        self.value = nn.Linear(width, hidden)
        self.output = nn.Linear(hidden, width)

    def forward(self, x):
        return self.output(F.silu(self.gate(x)) * self.value(x))


class SwiGLUBlock(Block):
    def __init__(self, width, heads, hidden):
        # Construct the final modules directly: do not initialize and discard
        # a baseline MLP, which would also consume its initialization RNG draws.
        nn.Module.__init__(self)
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.mlp = SwiGLU(width, hidden)


class SwiGLUGPT(GPT):
    def __init__(self, config):
        # Reuse GPT's features, forward, log-probability interface, and weight
        # initialization policy without constructing its original blocks first.
        nn.Module.__init__(self)
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        self.token = nn.Embedding(config['vocab'], width)
        self.pos = nn.Embedding(self.context, width)
        self.blocks = nn.ModuleList([
            SwiGLUBlock(width, config['heads'], config['ffn_hidden'])
            for _ in range(config['depth'])
        ])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(self.initialize)
        self.head.weight = self.token.weight


def build_model(config):
    config = dict(config)
    config.setdefault('ffn', 'swiglu')
    width = config['width']
    if config['ffn'] == 'gelu':
        hidden = config.setdefault('ffn_hidden', 4 * width)
        if hidden != 4 * width:
            raise ValueError('The original GELU baseline requires ffn_hidden=4*width.')
        return GPT(config)
    if config['ffn'] != 'swiglu':
        raise ValueError("ffn must be 'gelu' or 'swiglu'.")
    # Round (8/3)*width up to a multiple of eight. The third projection makes
    # this approximately match the original FFN's 4*width intermediate layer.
    hidden = config.setdefault('ffn_hidden', 8 * ((width + 2) // 3))
    if not isinstance(hidden, int) or isinstance(hidden, bool) or hidden < 1:
        raise ValueError('ffn_hidden must be a positive integer.')
    return SwiGLUGPT(config)
