"""Independent MP1 study: rotary positions and parameter-matched SwiGLU.

References: Su et al., arXiv:2104.09864; Shazeer, arXiv:2002.05202.
No pretrained weights or attention-output gate are used.
"""
import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F
from model import GPT


def swiglu_width(width):
    return 8 * math.floor((8 * width / 3) / 8 + 0.5)


class SwiGLU(nn.Module):
    def __init__(self, width):
        super().__init__()
        hidden = swiglu_width(width)
        self.value = nn.Linear(width, hidden)
        self.gate = nn.Linear(width, hidden)
        self.down = nn.Linear(hidden, width)

    def forward(self, x):
        return self.down(self.value(x) * F.silu(self.gate(x)))


class StudyBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        width, self.heads = config['width'], config['heads']
        self.rotary = config.get('position_encoding', 'learned') == 'rope'
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.mlp = (SwiGLU(width) if config.get('ffn_type', 'gelu') == 'swiglu'
                    else nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                       nn.Linear(4 * width, width)))
        if self.rotary:
            head_dim = width // self.heads
            freq = 10000.0 ** (-torch.arange(0, head_dim, 2).float() / head_dim)
            angles = torch.arange(config['context']).float()[:, None] * freq[None, :]
            # Deterministic tables only: no input-dependent state survives a call.
            self.register_buffer('rope_cos', angles.cos(), persistent=False)
            self.register_buffer('rope_sin', angles.sin(), persistent=False)

    def rotate(self, x):
        length = x.shape[-2]
        cos, sin = self.rope_cos[:length], self.rope_sin[:length]
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(
            batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        if self.rotary:
            q, k = self.rotate(q), self.rotate(k)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        return x + self.mlp(self.norm2(x))


class StudyGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config, self.context = dict(config), config['context']
        width = config['width']
        self.token = nn.Embedding(config['vocab'], width)
        self.pos = (nn.Embedding(self.context, width)
                    if config.get('position_encoding', 'learned') == 'learned' else None)
        self.blocks = nn.ModuleList([StudyBlock(config) for _ in range(config['depth'])])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(GPT.initialize)
        self.head.weight = self.token.weight

    def forward(self, ids):
        if ids.ndim != 2 or not 1 <= ids.shape[1] <= self.context:
            raise ValueError('Expected [batch, time] with 1 <= time <= context.')
        x = self.token(ids)
        if self.pos is not None:
            x = x + self.pos(torch.arange(ids.shape[1], device=ids.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    def predict_log_probs(self, ids):
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    if config.get('position_encoding', 'learned') not in ('learned', 'rope'):
        raise ValueError('Unknown position_encoding.')
    if config.get('ffn_type', 'gelu') not in ('gelu', 'swiglu'):
        raise ValueError('Unknown ffn_type.')
    if config['width'] % config['heads']:
        raise ValueError('width must be divisible by heads.')
    if config.get('position_encoding') == 'rope' and (config['width'] // config['heads']) % 2:
        raise ValueError('RoPE requires an even head dimension.')
    if config.get('position_encoding', 'learned') == 'learned' and config.get('ffn_type', 'gelu') == 'gelu':
        return GPT(config)
    return StudyGPT(config)


def build_seeded_model(config, seed):
    """Initialize shared tensors from the *untrained* official baseline.

    Isolated RNG contexts leave the caller's training/data random streams alone.
    SwiGLU tensors have distinct names; no semantically different tensors match.
    """
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        reference = GPT(config)
        torch.random.default_generator.manual_seed(seed + 10000)
        model = build_model(config)
        ref = reference.state_dict()
        with torch.no_grad():
            for name, tensor in model.state_dict().items():
                if name in ref and tensor.shape == ref[name].shape:
                    tensor.copy_(ref[name])
                else:
                    # Name-keyed private streams also match SwiGLU between C/D;
                    # removing position embeddings cannot shift these draws.
                    private_seed = int.from_bytes(hashlib.sha256(
                        f'{seed + 10000}:{name}'.encode()).digest()[:8], 'little')
                    private_rng = torch.Generator().manual_seed(private_seed)
                    if name.endswith('.bias'):
                        tensor.zero_()
                    else:
                        tensor.normal_(std=.02, generator=private_rng)
        model.head.weight = model.token.weight
    return model
