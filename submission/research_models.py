"""Round-two research model: rotary position encoding beside the frozen round-one ones.

``build_model`` delegates every round-one configuration to ``student.build_model``
unchanged, so the completed E0-E3 checkpoints still load. Setting ``pos='rope'``
swaps the learned absolute position embedding for rotary position encoding, and
``dropout=p`` adds residual dropout inside each block.
"""
import torch
from torch import nn
from torch.nn import functional as F

from student import SwiGLUBlock, SwiGLUGPT
from student import build_model as build_round_one


def rope_tables(length, head_dim, device, base=10000.):
    """Cosine/sine tables for the adjacent-pair rotation of RoFormer (Su et al., 2021)."""
    inverse = 1. / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    angles = torch.outer(torch.arange(length, device=device, dtype=torch.float32), inverse)
    # Each frequency serves the adjacent pair (2i, 2i+1), so repeat it across both.
    angles = torch.repeat_interleave(angles, 2, dim=-1)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotate the last dimension of ``x`` in adjacent pairs; the angle depends only on position."""
    pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
    return (pairs.flatten(-2) * cos + rotated * sin).type_as(x)


class RoPEBlock(SwiGLUBlock):
    """SwiGLU block whose attention rotates queries and keys; values are untouched.

    ``dropout`` is residual dropout, applied to each sublayer output just before it
    is added back to the stream. It is registered after the linear layers because
    ``nn.Dropout`` holds no parameters and consumes no initialization draws, so the
    initialization RNG stream is identical for every ``p`` and paired arms start
    from bit-identical weights.
    """

    def __init__(self, width, heads, hidden, dropout=0.):
        super().__init__(width, heads, hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        cos, sin = rope_tables(length, width // self.heads, x.device)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.drop(self.proj(attended.transpose(1, 2).reshape(batch, length, width)))
        return x + self.drop(self.mlp(self.norm2(x)))


class RoPEGPT(SwiGLUGPT):
    """The round-one SwiGLU model with rotary instead of learned absolute positions."""

    def __init__(self, config):
        # Construct the modules directly for the same reason as SwiGLUGPT: do not
        # initialize and discard a position embedding that this variant will not use.
        nn.Module.__init__(self)
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        dropout = config.get('dropout', 0.)
        if not 0. <= dropout < 1.:
            raise ValueError('dropout must lie in [0, 1).')
        # ``temperature`` is a plain float attribute, not a submodule, so it holds no
        # parameters and consumes no initialization draws: every checkpoint trained
        # without it loads here unchanged and scores bit-identically at T=1.
        temperature = config.get('temperature', 1.)
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) \
                or not temperature > 0:
            raise ValueError('temperature must be a positive number.')
        self.temperature = float(temperature)
        self.token = nn.Embedding(config['vocab'], width)
        self.blocks = nn.ModuleList([
            RoPEBlock(width, config['heads'], config['ffn_hidden'], dropout)
            for _ in range(config['depth'])
        ])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(self.initialize)
        self.head.weight = self.token.weight

    def features(self, ids):
        x = self.token(ids)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def predict_log_probs(self, ids):
        """Evaluation interface with optional temperature calibration.

        Dividing log probabilities by T and normalizing again is exactly
        softmax(logits / T): the log-normalizer subtracted before the division is
        added back by the second normalization. The transform still reads only
        ``ids`` and returns a normalized distribution over the vocabulary. At T=1
        it is skipped rather than applied, so no existing checkpoint changes score.
        """
        logp = super().predict_log_probs(ids)
        if self.temperature == 1.:
            return logp
        return F.log_softmax(logp / self.temperature, dim=-1)


def build_model(config):
    config = dict(config)
    position = config.get('pos', 'learned')
    if position == 'learned':
        return build_round_one(config)
    if position != 'rope':
        raise ValueError("pos must be 'learned' or 'rope'.")
    if config.get('ffn', 'swiglu') != 'swiglu':
        raise ValueError('The RoPE variant is defined on the SwiGLU block only.')
    if (config['width'] // config['heads']) % 2:
        raise ValueError('Rotary encoding requires an even head dimension.')
    return RoPEGPT(config)
