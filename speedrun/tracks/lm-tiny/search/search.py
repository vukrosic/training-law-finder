"""Record search for lm-tiny: train.py with every record idea behind one flag.

    python3 search/search.py                          # search size: mini model, 6e11 FLOPs, first 100 val docs
    python3 search/search.py --ve 2 --lr-table 0.3    # value embeddings in the first 2 blocks
    python3 search/search.py --bigram-buckets 8192    # bigram hash table (refused if the full-size artifact > 15.5 MB)
    python3 search/search.py --batch half             # half or quarter batch; the FLOP budget sets the steps
    python3 search/search.py --model w128             # w128, w160 (narrower) or l3 (3 layers)
    python3 search/search.py --vocab 4096             # needs python3 search/make_vocab.py first
    python3 search/search.py --size full --budget 1e14 --record ...   # confirm a winner at the real size
    python3 search/run_queue.py                       # run search/queue.txt in order, results in out/search/

Same math as train.py except the Muon step orthogonalizes same-shape matrices together (fewer kernel launches;
the mini baseline gives 2.5939 bpb vs 2.5942 for train.py's loop). --size full with no idea flags is the baseline.
Learning-rate flags change only the learning rates, so the fast-count cache is reused across an lr grid.
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

BUDGET = 6e11  # search budget; --budget 1e14 --size full for a record
VOCAB, BOS = 1024, 1
D_MODEL, N_LAYERS, N_HEADS, D_MLP = 192, 4, 3, 512
SEQ_LEN, BATCH = 256, 64
MICRO = 32  # sequences per forward pass; smaller uses less memory, same math, about the same FLOPs
LR_MUON, LR_EMBED, LR_HEAD, LR_SCALAR = 0.05, 0.6, 0.008, 0.04
LR_TABLE = 0.6  # Adam lr for value embeddings and the bigram table
COOLDOWN = 0.4  # learning rate falls linearly to 0 over the last 40% of steps
VE_LAYERS = 0  # value embeddings: blocks 0..VE_LAYERS-1 add a learned per-token vector to v (modded-nanogpt)
BIGRAM = 0  # bigram hash table rows: hash(previous token, token) -> a vector added to the input embedding
BIGRAM_DIM = 0  # 0: the table has D_MODEL columns; else BIGRAM_DIM columns and a linear map up to D_MODEL
TOK_DIR = ""  # token files: data/ for sp1024, data/sp4096/ for --vocab 4096

SIZES = dict(full=dict(D_MODEL=192, N_HEADS=3, D_MLP=512, N_LAYERS=4, BATCH=64, MICRO=32),
             mini=dict(D_MODEL=64, N_HEADS=1, D_MLP=176, N_LAYERS=4, BATCH=8, MICRO=8))
ARTIFACT_MAX = 15_500_000  # size check: 4 bytes per weight (lzma barely shrinks fp32) + code and tokenizer
EXTRA_BYTES = 300_000  # code + tokenizer, compressed (sp4096.model is about 0.1 MB)


def shapes(size, model, batch):
    """Constants for a size (full or mini), a model preset and a batch preset."""
    s = dict(SIZES[size])
    if model in ("w128", "w160"):
        frac = int(model[1:]) / 192
        d = 2 * round(s["D_MODEL"] * frac / 2)
        s.update(D_MODEL=d, N_HEADS=max(1, round(d / 64)), D_MLP=8 * round(s["D_MLP"] * d / s["D_MODEL"] / 8))
    elif model == "l3":
        s["N_LAYERS"] = 3
    s["BATCH"] = s["BATCH"] // dict(full=1, half=2, quarter=4)[batch]
    s["MICRO"] = min(s["MICRO"], s["BATCH"])
    assert s["D_MODEL"] % s["N_HEADS"] == 0 and (s["D_MODEL"] // s["N_HEADS"]) % 2 == 0, s
    return s


def count_params(s, vocab, ve, buckets, bdim):
    """Weights of the model these constants build (checked against the real model in main)."""
    d, hd, layers = s["D_MODEL"], s["D_MODEL"] // s["N_HEADS"], s["N_LAYERS"]
    block = 4 * d * d + 3 * d * s["D_MLP"] + 2 * d + 2 * hd
    n = 2 * vocab * d + d + layers * block + min(ve, layers) * vocab * d
    if buckets:
        n += buckets * (bdim or d) + (bdim * d if bdim else 0)
    return n


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
    def __init__(self, ve=False):
        super().__init__()
        hd = D_MODEL // N_HEADS
        if ve:
            self.ve = nn.Embedding(VOCAB, D_MODEL)
            nn.init.zeros_(self.ve.weight)
        self.n1, self.n2 = nn.RMSNorm(D_MODEL), nn.RMSNorm(D_MODEL)
        self.q, self.k, self.v, self.o = (nn.Linear(D_MODEL, D_MODEL, bias=False) for _ in range(4))
        self.qn, self.kn = nn.RMSNorm(hd), nn.RMSNorm(hd)
        self.rope = Rotary(hd)
        self.w1, self.w3 = nn.Linear(D_MODEL, D_MLP, bias=False), nn.Linear(D_MODEL, D_MLP, bias=False)
        self.w2 = nn.Linear(D_MLP, D_MODEL, bias=False)
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.w2.weight)

    def forward(self, x, idx):
        B, T, _ = x.shape
        h = self.n1(x)
        vv = self.v(h) + self.ve(idx) if hasattr(self, "ve") else self.v(h)
        q, k, v = (t.view(B, T, N_HEADS, -1).transpose(1, 2) for t in (self.q(h), self.k(h), vv))
        q, k = self.rope(self.qn(q)), self.rope(self.kn(k))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(y.transpose(1, 2).reshape(B, T, D_MODEL))
        h = self.n2(x)
        return x + self.w2(F.silu(self.w1(h)) * self.w3(h))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.blocks = nn.ModuleList(Block(i < VE_LAYERS) for i in range(N_LAYERS))
        if BIGRAM:
            self.bigram = nn.Embedding(BIGRAM, BIGRAM_DIM or D_MODEL)
            nn.init.zeros_(self.bigram.weight)
            if BIGRAM_DIM:
                self.bigram_up = nn.Linear(BIGRAM_DIM, D_MODEL, bias=False)
        self.norm = nn.RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)
        nn.init.zeros_(self.head.weight)

    def forward(self, idx):
        x = self.embed(idx)
        if BIGRAM:
            prev = F.pad(idx[:, :-1], (1, 0), value=VOCAB)  # first position: "no previous token" id
            g = self.bigram((prev * 1031 + idx) % BIGRAM)
            x = x + (self.bigram_up(g) if BIGRAM_DIM else g)
        for b in self.blocks:
            x = b(x, idx)
        return self.head(self.norm(x))


def newton_schulz(G, steps=5):
    """Approximate the nearest orthogonal matrix to each matrix in G (..., m, n) (Muon, Keller Jordan 2024)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    tall = X.size(-2) > X.size(-1)
    if tall:
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    return X.mT if tall else X


class Muon(torch.optim.Optimizer):
    """Same update as train.py, with same-shape matrices orthogonalized together (fewer kernel launches)."""
    def __init__(self, params, lr, momentum=0.95):
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            groups = {}
            for p in g["params"]:
                groups.setdefault(tuple(p.shape), []).append(p)
            for shape, ps in groups.items():
                bufs = [self.state[p].setdefault("buf", torch.zeros_like(p)) for p in ps]
                grads = [p.grad for p in ps]
                torch._foreach_lerp_(bufs, grads, 1 - g["momentum"])
                u = newton_schulz(torch.stack(grads).lerp(torch.stack(bufs), g["momentum"]))
                torch._foreach_add_(ps, list(u.unbind(0)), alpha=-g["lr"] * max(1, shape[0] / shape[1]) ** 0.5)


class Batches:
    """Reads the fixed training stream in order: batch i is the next BATCH*SEQ_LEN+1 tokens."""

    def __init__(self):
        d = TOK_DIR
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
    tables = [p for n, p in model.named_parameters() if n.endswith("ve.weight") or n == "bigram.weight"]
    blocks = [p for n, p in model.named_parameters() if p.ndim == 2 and (n.startswith("blocks.") or n == "bigram_up.weight")
              and not n.endswith("ve.weight")]
    scalars = [p for n, p in model.named_parameters() if p.ndim < 2]
    muon = Muon(blocks, lr=LR_MUON)
    adam = torch.optim.Adam([dict(params=[model.embed.weight], lr=LR_EMBED),
                             dict(params=[model.head.weight], lr=LR_HEAD),
                             dict(params=scalars, lr=LR_SCALAR)] + ([dict(params=tables, lr=LR_TABLE)] if tables else []), betas=(0.8, 0.95), eps=1e-10)
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
    ap.add_argument("--size", choices=sorted(SIZES), default="mini", help="mini: search size; full: record size")
    ap.add_argument("--model", choices=["base", "w128", "w160", "l3"], default="base",
                    help="w128/w160: width 128/160 at full size (scaled the same way at mini); l3: 3 layers")
    ap.add_argument("--batch", choices=["full", "half", "quarter"], default="full")
    ap.add_argument("--ve", type=int, default=0, metavar="LAYERS", help="value embeddings in the first LAYERS blocks")
    ap.add_argument("--bigram-buckets", type=int, default=0, metavar="N", help="bigram hash table with N rows")
    ap.add_argument("--bigram-dim", type=int, default=0, help="bigram table width (0: model width, no projection)")
    ap.add_argument("--vocab", type=int, choices=[1024, 4096], default=1024)
    for name in ("muon", "embed", "head", "scalar", "table"):
        ap.add_argument(f"--lr-{name}", type=float, default=None)
    ap.add_argument("--cooldown", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default=None, help="append the result to out/search/results.jsonl under this tag")
    ap.add_argument("--full-count", action="store_true",
                    help="also count every op of every step on this device (about 2%% slower); needed only for "
                         "entries with FFT, decompositions, sorts, scans or scatters")
    ap.add_argument("--record", action="store_true",
                    help="record run: --full-count, save the 16 MB artifact, score the model loaded from it")
    ap.add_argument("--val-docs", type=int, default=100,
                    help="score only the first N validation documents (fast, for search; not a record)")
    args = ap.parse_args()
    if args.record:
        args.val_docs = 0  # a record is scored on the whole validation set
    g = globals()
    g.update(shapes(args.size, args.model, args.batch), VE_LAYERS=args.ve, BIGRAM=args.bigram_buckets,
             BIGRAM_DIM=args.bigram_dim, VOCAB=args.vocab,
             TOK_DIR=ev.DATA if args.vocab == 1024 else os.path.join(ev.DATA, f"sp{args.vocab}"))
    for name in ("muon", "embed", "head", "scalar", "table"):
        if getattr(args, f"lr_{name}") is not None:
            g[f"LR_{name.upper()}"] = getattr(args, f"lr_{name}")
    if args.cooldown is not None:
        g["COOLDOWN"] = args.cooldown
    assert os.path.exists(os.path.join(TOK_DIR, "val.bin")), f"no tokens in {TOK_DIR}: run search/make_vocab.py"
    # size check before any training: the same flags at full size must fit the 16 MB artifact
    n_full = count_params(shapes("full", args.model, args.batch), VOCAB, VE_LAYERS, BIGRAM, BIGRAM_DIM)
    art_est = 4 * n_full + EXTRA_BYTES
    print(f"full-size artifact estimate {art_est / 1e6:.2f} MB ({n_full:,} weights; limit {ARTIFACT_MAX / 1e6} MB)")
    if art_est > ARTIFACT_MAX:
        sys.exit(f"too big for the 16 MB artifact: {art_est / 1e6:.2f} MB; use fewer --bigram-buckets or a --bigram-dim")
    full = args.full_count or args.record
    device = args.device or "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
    torch.manual_seed(0)
    model, opts = build()  # built on cpu so the starting weights are the same on every machine
    model.to(device)
    for o in opts:
        for g in o.param_groups:
            g["base_lr"] = g["lr"]
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == count_params(globals(), VOCAB, VE_LAYERS, BIGRAM, BIGRAM_DIM), "count_params is out of date"
    n_matmul = n_params - sum(p.numel() for n, p in model.named_parameters()
                              if n.endswith("embed.weight") or n.endswith("ve.weight") or n == "bigram.weight")
    print(f"device {device}, params {n_params:,} ({n_matmul:,} without the embedding), budget {args.budget:.3g} FLOPs")

    # Fast count: one step on the meta device, every op charged (common/flops.py), cached per code and shape.
    t = time.time()
    step_flops, rep = flops.fast_count(build, train_step, (BATCH, SEQ_LEN), key=flops.code_key(sys.modules[__name__]),
                                       cache=os.path.join(ev.OUT, "search", "flops_cache.json"))
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
        path = os.path.join(ev.OUT, "search", f"{args.tag or 'model'}.xz")
        size = ev.save_artifact(path, {k: v.detach().cpu() for k, v in model.state_dict().items()},
                                [os.path.abspath(__file__)], os.path.join(TOK_DIR, f"sp{VOCAB}.model"))
        print(f"artifact {os.path.relpath(path, TRACK)}: {size:,} bytes (limit {ev.SIZE_LIMIT:,})")
        assert size <= ev.SIZE_LIMIT, "artifact over the size limit"
        model = build()[0]
        model.load_state_dict(ev.load_artifact(path)["state"])
        model.to(device)
        result["artifact_bytes"] = size
    model.float().eval()  # float32: bf16 eval on mps gives NaN
    t1 = time.time()
    docs = ev.val_docs()
    val = np.fromfile(os.path.join(TOK_DIR, "val.bin"), dtype=np.uint16)
    if args.val_docs:  # the stream up to the BOS of document N+1: the same bpb rule on fewer documents
        docs = docs[:args.val_docs]
        bos = np.flatnonzero(val == BOS)
        val = val[:bos[args.val_docs]] if args.val_docs < len(bos) else val
    nats = ev.total_nats(model, val, SEQ_LEN, device)
    n_bytes = ev.val_bytes(docs)
    result = dict(val_bpb=round(ev.bpb(nats, n_bytes), 4), **result, eval_seconds=round(time.time() - t1, 1),
                  val_docs=len(docs), val_tokens=len(val) - 1, val_bytes=n_bytes, nonstandard_ops=sorted(rep["nonstandard"]))
    result["artifact_estimate"] = art_est
    print("result " + json.dumps(result))
    if args.tag:
        flags = " ".join(a for a in sys.argv[1:] if a != "--tag" and a != args.tag)
        os.makedirs(os.path.join(ev.OUT, "search"), exist_ok=True)
        with open(os.path.join(ev.OUT, "search", "results.jsonl"), "a") as f:
            f.write(json.dumps(dict(tag=args.tag, flags=flags, **result)) + "\n")


if __name__ == "__main__":
    main()
