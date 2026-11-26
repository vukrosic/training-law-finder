"""A small modern decoder-only transformer.

RMSNorm, rotary position embeddings, QK-norm, SwiGLU feed-forward, no biases,
untied input embedding and output head. Parameter names start with `emb` and
`head` for the embedding and output layer, so the optimizer can route them to
Adam and every other matrix to Muon.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32 * self.weight.float()).type_as(x)


def rope_tables(seq_len, head_dim, base=10000.0, device=None):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    f = torch.outer(t, inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, Dh)
    T = x.shape[-2]
    c, s = cos[:T].to(x.dtype), sin[:T].to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, d, n_head, qk_norm=True):
        super().__init__()
        assert d % n_head == 0
        self.h, self.dh = n_head, d // n_head
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.qk_norm = qk_norm
        if qk_norm:
            self.qn, self.kn = RMSNorm(self.dh), RMSNorm(self.dh)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if self.qk_norm:
            q, k = self.qn(q), self.kn(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, d, n_head, hidden, qk_norm=True):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn = Attention(d, n_head, qk_norm)
        self.ff = SwiGLU(d, hidden)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        return x + self.ff(self.n2(x))


def ff_hidden(d, mult=8 / 3, multiple=32):
    return int(multiple * math.ceil(d * mult / multiple))


class GPT(nn.Module):
    def __init__(self, vocab, d=128, n_layer=4, n_head=4, seq_len=256, ff_mult=8 / 3, qk_norm=True):
        super().__init__()
        self.seq_len = seq_len
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(
            Block(d, n_head, ff_hidden(d, ff_mult), qk_norm) for _ in range(n_layer))
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        cos, sin = rope_tables(seq_len, d // n_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def forward(self, idx):
        x = self.emb(idx)
        for b in self.blocks:
            x = b(x, self.cos, self.sin)
        return self.head(self.norm(x))

    def n_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.emb.weight.numel() + self.head.weight.numel()
        return n
