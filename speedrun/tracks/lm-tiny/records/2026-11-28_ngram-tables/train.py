"""lm-tiny record ngram-tables (settings below are the defaults; each can be overridden by an env var of the same name).

Speedrun baseline: train a small GPT on FineWeb-Edu until the FLOP budget is used, then report validation bpb.

    python3 train.py                  # from speedrun/tracks/lm-tiny/, after python3 prepare_data.py
    python3 train.py --budget 1e12 --val-docs 100   # search run: small budget, first 100 validation documents
    python3 train.py --record         # record run: also counts every step on this device, saves the
                                      # 16 MB artifact (out/model.xz) and scores the model loaded from it

FLOPs: one step is counted on the meta device before training (common/flops.py, every op charged, cached).

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
E = lambda k, d: type(d)(os.environ.get(k, d))
D_MODEL, N_LAYERS, N_HEADS, D_MLP = E("D_MODEL", 128), E("N_LAYERS", 6), E("N_HEADS", 2), E("D_MLP", 352)
SEQ_LEN, BATCH = E("SEQ_LEN", 128), E("BATCH", 64)
MICRO = min(BATCH, 32)
LR_MUON, LR_EMBED, LR_HEAD, LR_SCALAR = E("LR_MUON", 0.05), E("LR_EMBED", 0.3), E("LR_HEAD", 0.004), E("LR_SCALAR", 0.04)
COOLDOWN = E("COOLDOWN", 0.6)
BIGRAM = E("BIGRAM", 32768)
TRIGRAM = E("TRIGRAM", 32768)  # rows of a hashed (two back, previous, token) embedding, 0 = off
ACT = E("ACT", "swiglu")  # swiglu or relu2 (relu squared, no gate: w3 unused)
SOFTCAP = E("SOFTCAP", 0.0)  # logits = cap * tanh(logits / cap), 0 = off
SAVE16 = E("SAVE16", 0)  # store the hashed tables in bf16 in the artifact
SAVE8 = E("SAVE8", 1)  # store the hashed tables as int8 with one float16 scale per row
MUON_MOM = E("MUON_MOM", 0.85)


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
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h) if ACT == "swiglu" else F.relu(self.w1(h)).square())


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        if BIGRAM:
            self.bigram = nn.Embedding(BIGRAM, D_MODEL)
            nn.init.zeros_(self.bigram.weight)
        if TRIGRAM:
            self.trigram = nn.Embedding(TRIGRAM, D_MODEL)
            nn.init.zeros_(self.trigram.weight)
        self.blocks = nn.ModuleList(Block() for _ in range(N_LAYERS))
        self.norm = nn.RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)
        nn.init.zeros_(self.head.weight)

    def forward(self, idx):
        x = self.embed(idx)
        if BIGRAM:
            x = x + self.bigram((F.pad(idx[:, :-1], (1, 0), value=BOS) * 1031 + idx) % BIGRAM)
        if TRIGRAM:
            p1, p2 = F.pad(idx[:, :-1], (1, 0), value=BOS), F.pad(idx[:, :-2], (2, 0), value=BOS)
            x = x + self.trigram(((p2 * 1031 + p1) * 1033 + idx) % TRIGRAM)
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.norm(x))
        return SOFTCAP * torch.tanh(logits / SOFTCAP) if SOFTCAP else logits


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
                if p.grad is None:
                    continue
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
    muon = Muon(blocks, lr=LR_MUON, momentum=MUON_MOM)
    adam = torch.optim.Adam([dict(params=[model.embed.weight] + ([model.bigram.weight] if BIGRAM else []) + ([model.trigram.weight] if TRIGRAM else []), lr=LR_EMBED),
                             dict(params=[model.head.weight], lr=LR_HEAD),
                             dict(params=scalars, lr=LR_SCALAR)], betas=(0.8, 0.95), eps=1e-10)
    return model, [muon, adam]


def pack(state):
    """Artifact weights: hashed tables in int8 (SAVE8) or bf16 (SAVE16), the rest float32."""
    out = {}
    for k, v in state.items():
        v = v.detach().cpu()
        if k.endswith("gram.weight") and SAVE8:
            s = (v.abs().amax(1, keepdim=True) / 127).clamp_min(1e-12)
            out[k + ".q"], out[k + ".s"] = (v / s).round().to(torch.int8), s.half()
        else:
            out[k] = v.to(torch.bfloat16) if k.endswith("gram.weight") and SAVE16 else v
    return out


def unpack(state):
    out = {k: v for k, v in state.items() if not k.endswith((".q", ".s"))}
    for k in state:
        if k.endswith(".q"):
            out[k[:-2]] = state[k].float() * state[k[:-2] + ".s"].float()
    return out


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
    ap.add_argument("--full-count", action="store_true",
                    help="also count every op of every step on this device (about 2%% slower); needed only for "
                         "entries with FFT, decompositions, sorts, scans or scatters")
    ap.add_argument("--record", action="store_true",
                    help="record run: --full-count, save the 16 MB artifact, score the model loaded from it")
    ap.add_argument("--val-docs", type=int, default=0,
                    help="score only the first N validation documents (fast, for search; not a record)")
    args = ap.parse_args()
    assert not (args.record and args.val_docs), "a record is scored on the whole validation set"
    full = args.full_count or args.record
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
    torch.manual_seed(E("SEED", 0))
    model, opts = build()  # built on cpu so the starting weights are the same on every machine
    model.to(device)
    for o in opts:
        for g in o.param_groups:
            g["base_lr"] = g["lr"]
    n_params = sum(p.numel() for p in model.parameters())
    n_matmul = n_params - model.embed.weight.numel()
    print(f"device {device}, params {n_params:,} ({n_matmul:,} without the embedding), budget {args.budget:.3g} FLOPs")

    # Fast count: one step on the meta device, every op charged (common/flops.py), cached per code and shape.
    t = time.time()
    step_flops, rep = flops.fast_count(build, train_step, (BATCH, SEQ_LEN), key=flops.code_key(sys.modules[__name__]),
                                       cache=os.path.join(ev.OUT, "flops_cache.json"))
    n_steps = int(args.budget // step_flops)
    tokens = n_steps * BATCH * SEQ_LEN
    est, est_attn = flops.estimate_6nd(n_matmul, tokens, SEQ_LEN, D_MODEL, N_LAYERS)
    print(f"FLOPs per step {step_flops:.4g} (fast count, {time.time() - t:.2f}s), steps {n_steps}, tokens {tokens:,}; "
          f"total {n_steps * step_flops:.4g}, 6ND {est:.4g}, 6ND + attention {est_attn:.4g}")
    if rep["nonstandard"] and not full:
        print(f"WARNING: non-standard ops {sorted(rep['nonstandard'])}: the fast count charges them by formula on "
              f"the meta device; run with --full-count to see your real budget")
    elif not rep["nonstandard"]:
        print("standard ops, fast count is exact")

    meter = flops.Meter() if full else None
    data = Batches()
    t0 = time.time()
    for i in range(n_steps):
        scale = min(1.0, (1 - i / n_steps) / COOLDOWN)
        for o in opts:
            for g in o.param_groups:
                g["lr"] = g["base_lr"] * scale
        x, y = data.next(device)
        with meter.step() if meter else contextlib.nullcontext():
            loss = train_step(model, opts, x, y, dtype)
        if i % 100 == 0 or i == n_steps - 1:
            print(f"step {i}/{n_steps} loss {loss.item():.4f} flops {(i + 1) * step_flops:.4g} "
                  f"time {time.time() - t0:.0f}s", flush=True)
    if device == "mps":
        torch.mps.synchronize()
    train_time = time.time() - t0
    used = n_steps * step_flops
    result = dict(flops=float(f"{used:.6g}"), budget=args.budget, counter=flops.COUNTER, steps=n_steps,
                  tokens=tokens, params=n_params, device=device, train_seconds=round(train_time, 1))
    if meter:
        diff = meter.total / used - 1
        print(f"full count {meter.total:.4g} ({diff:+.2%} vs fast) " + json.dumps(meter.report()))
        if abs(diff) > 0.01:  # README: past 1% the full count is the score
            used = meter.total
        result.update(flops=float(f"{used:.6g}"), full_count=float(f"{meter.total:.6g}"))
    assert used <= args.budget, f"{used:.4g} FLOPs, over the budget {args.budget:.3g}"

    if args.record:  # the scored model is the one loaded back from the 16 MB artifact (eval.py)
        path = os.path.join(ev.OUT, "model.xz")
        size = ev.save_artifact(path, pack(model.state_dict()),
                                [os.path.abspath(__file__)], os.path.join(ev.DATA, "sp1024.model"))
        print(f"artifact {os.path.relpath(path, TRACK)}: {size:,} bytes (limit {ev.SIZE_LIMIT:,})")
        assert size <= ev.SIZE_LIMIT, "artifact over the size limit"
        model = build()[0]
        model.load_state_dict(unpack(ev.load_artifact(path)["state"]))
        model.to(device)
        result["artifact_bytes"] = size
    model.float().eval()  # float32: bf16 eval on mps gives NaN
    t1 = time.time()
    docs = ev.val_docs()
    val = np.fromfile(os.path.join(ev.DATA, "val.bin"), dtype=np.uint16)
    if args.val_docs:  # the stream up to the BOS of document N+1: the same bpb rule on fewer documents
        docs = docs[:args.val_docs]
        bos = np.flatnonzero(val == BOS)
        val = val[:bos[args.val_docs]] if args.val_docs < len(bos) else val
    nats = ev.total_nats(model, val, SEQ_LEN, device)
    n_bytes = ev.val_bytes(docs)
    result = dict(val_bpb=round(ev.bpb(nats, n_bytes), 4), **result, eval_seconds=round(time.time() - t1, 1),
                  val_docs=len(docs), val_tokens=len(val) - 1, val_bytes=n_bytes, nonstandard_ops=sorted(rep["nonstandard"]))
    print("result " + json.dumps(result))


if __name__ == "__main__":
    main()
