"""Speedrun baseline: train a small GPT on FineWeb-Edu until the FLOP budget is used, then report validation bpb.

    python3 train.py                  # from speedrun/tracks/lm-fineweb-1e14/, after python3 prepare_data.py
    python3 train.py --budget 1e13    # shorter run, not a record

Model: RMSNorm, RoPE, QK-norm, SwiGLU, no biases, untied head. Muon on block matrices, Adam on the rest.
"""
import argparse
import contextlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TRACK = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(TRACK, "eval.py")):  # a copy in records/<run>/ finds the track folder
    TRACK = os.path.dirname(TRACK)
sys.path[:0] = [TRACK, os.path.join(os.path.dirname(os.path.dirname(TRACK)), "common")]  # eval.py, common/flops.py
import eval as ev  # noqa: E402
import flops  # noqa: E402

BUDGET = 1e14  # the speedrun budget, in FLOPs (README.md)
VOCAB, BOS = 1024, 1
D_MODEL, N_LAYERS, N_HEADS, D_MLP = 192, 4, 3, 512
SEQ_LEN, BATCH = 256, 64
MICRO = 32  # sequences per forward pass; smaller uses less memory, same FLOPs and same math
LR_MUON, LR_EMBED, LR_HEAD, LR_SCALAR = 0.05, 0.6, 0.008, 0.04
COOLDOWN = 0.4  # learning rate falls linearly to 0 over the last 40% of steps


class Rotary(nn.Module):
    def __init__(self, dim, max_len=4096):
        super().__init__()
        f = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        a = torch.outer(torch.arange(max_len).float(), f)
        self.register_buffer("cos", a.cos(), persistent=False)
        self.register_buffer("sin", a.sin(), persistent=False)

    def forward(self, x):  # x: B, H, T, hd
        T = x.size(2)
        c, s = self.cos[:T].to(x.dtype), self.sin[:T].to(x.dtype)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        hd = D_MODEL // N_HEADS
        self.n1, self.n2 = nn.RMSNorm(D_MODEL), nn.RMSNorm(D_MODEL)
        self.q, self.k, self.v, self.o = (nn.Linear(D_MODEL, D_MODEL, bias=False) for _ in range(4))
        self.qn, self.kn = nn.RMSNorm(hd), nn.RMSNorm(hd)
        self.rope = Rotary(hd)
        self.w1, self.w3 = nn.Linear(D_MODEL, D_MLP, bias=False), nn.Linear(D_MODEL, D_MLP, bias=False)
        self.w2 = nn.Linear(D_MLP, D_MODEL, bias=False)
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.w2.weight)

    def forward(self, x):
        B, T, _ = x.shape
        h = self.n1(x)
        q, k, v = (f(h).view(B, T, N_HEADS, -1).transpose(1, 2) for f in (self.q, self.k, self.v))
        q, k = self.rope(self.qn(q)), self.rope(self.kn(k))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(y.transpose(1, 2).reshape(B, T, D_MODEL))
        h = self.n2(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.blocks = nn.ModuleList(Block() for _ in range(N_LAYERS))
        self.norm = nn.RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)
        nn.init.zeros_(self.head.weight)

    def forward(self, idx):
        x = self.embed(idx)
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))


def newton_schulz(G, steps=5):
    """Approximate the nearest orthogonal matrix to G (Muon, Keller Jordan 2024)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    tall = X.size(0) > X.size(1)
    if tall:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if tall else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum=0.95):
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                buf = self.state[p].setdefault("buf", torch.zeros_like(p))
                buf.lerp_(p.grad, 1 - g["momentum"])
                u = newton_schulz(p.grad.lerp(buf, g["momentum"]))
                p.add_(u, alpha=-g["lr"] * max(1, p.size(0) / p.size(1)) ** 0.5)


class Batches:
    """Reads the fixed training stream in order: batch i is the next BATCH*SEQ_LEN+1 tokens."""

    def __init__(self):
        d = ev.DATA
        self.shards = [np.memmap(os.path.join(d, f), dtype=np.uint16, mode="r")
                       for f in sorted(os.listdir(d)) if f.startswith("train_") and f.endswith(".bin")]
        self.shard, self.pos = 0, 0

    def next(self, device):
        need, parts = BATCH * SEQ_LEN + 1, []
        while need:
            s = self.shards[self.shard]
            take = min(need, len(s) - self.pos)
            parts.append(s[self.pos:self.pos + take])
            need -= take
            self.pos += take
            if self.pos == len(s):
                assert self.shard + 1 < len(self.shards), "training stream used up"
                self.shard, self.pos = self.shard + 1, 0
        self.pos -= 1  # the last target is the next batch's first input
        if self.pos < 0:
            self.shard, self.pos = self.shard - 1, len(self.shards[self.shard - 1]) - 1
        t = torch.from_numpy(np.concatenate(parts).astype(np.int64))
        x, y = t[:-1].view(BATCH, SEQ_LEN), t[1:].view(BATCH, SEQ_LEN)
        return x.to(device), y.to(device)


def build():
    model = GPT()
    blocks = [p for n, p in model.named_parameters() if p.ndim == 2 and n.startswith("blocks.")]
    scalars = [p for n, p in model.named_parameters() if p.ndim < 2]
    muon = Muon(blocks, lr=LR_MUON)
    adam = torch.optim.Adam([dict(params=[model.embed.weight], lr=LR_EMBED),
                             dict(params=[model.head.weight], lr=LR_HEAD),
                             dict(params=scalars, lr=LR_SCALAR)], betas=(0.8, 0.95), eps=1e-10)
    return model, [muon, adam]


def autocast(device, dtype):
    return contextlib.nullcontext() if dtype == torch.float32 else torch.autocast(device, dtype=dtype)


def train_step(model, opts, x, y, dtype):
    total = 0.0
    for xm, ym in zip(x.split(MICRO), y.split(MICRO)):
        with autocast(x.device.type, dtype):
            loss = F.cross_entropy(model(xm).float().flatten(0, 1), ym.flatten()) * len(xm) / len(x)
        loss.backward()
        total = total + loss.detach()
    for o in opts:
        o.step()
    model.zero_grad(set_to_none=True)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=BUDGET)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
    torch.manual_seed(0)
    model, opts = build()  # built on cpu so the starting weights are the same on every machine
    model.to(device)
    for o in opts:
        for g in o.param_groups:
            g["base_lr"] = g["lr"]
    n_params = sum(p.numel() for p in model.parameters())
    n_matmul = n_params - model.embed.weight.numel()
    print(f"device {device}, params {n_params:,} ({n_matmul:,} without the embedding), budget {args.budget:.3g} FLOPs")

    step_flops = flops.count_step(build, train_step, (BATCH, SEQ_LEN))
    n_steps = int(args.budget // step_flops)
    tokens = n_steps * BATCH * SEQ_LEN
    est, est_attn = flops.estimate_6nd(n_matmul, tokens, SEQ_LEN, D_MODEL, N_LAYERS)
    attn = flops.attention_flops(BATCH, SEQ_LEN, D_MODEL, N_LAYERS)
    print(f"FLOPs per step {step_flops:.4g} (FlopCounterMode, of which attention {attn:.4g}), "
          f"steps {n_steps}, tokens {tokens:,}")
    print(f"FLOPs: counted {n_steps * step_flops:.4g}, 6ND {est:.4g}, 6ND + attention {est_attn:.4g}")

    data = Batches()
    t0 = time.time()
    for i in range(n_steps):
        frac = i / n_steps
        scale = min(1.0, (1 - frac) / COOLDOWN)
        for o in opts:
            for g in o.param_groups:
                g["lr"] = g["base_lr"] * scale
        x, y = data.next(device)
        loss = train_step(model, opts, x, y, dtype)
        if i % 100 == 0 or i == n_steps - 1:
            print(f"step {i}/{n_steps} loss {loss.item():.4f} flops {(i + 1) * step_flops:.4g} "
                  f"time {time.time() - t0:.0f}s", flush=True)
    if device == "mps":
        torch.mps.synchronize()
    train_time = time.time() - t0

    model.eval()
    t1 = time.time()
    docs = ev.val_docs()
    val = np.fromfile(os.path.join(ev.DATA, "val.bin"), dtype=np.uint16)
    nats = ev.total_nats(model.float(), val, SEQ_LEN, device)  # float32: bf16 eval on mps gives NaN
    n_bytes = ev.val_bytes(docs)
    result = dict(val_bpb=round(ev.bpb(nats, n_bytes), 4), flops=float(f"{n_steps * step_flops:.6g}"),
                  budget=args.budget, steps=n_steps, tokens=tokens, params=n_params, device=device,
                  train_seconds=round(train_time, 1), eval_seconds=round(time.time() - t1, 1),
                  val_tokens=len(val) - 1, val_bytes=n_bytes)
    print("result " + json.dumps(result))


if __name__ == "__main__":
    main()
