"""Run packs of K same-shape models in one batched pass; one JSON row per model.

    python3 -I blaw.py --jobs jobs/x.jsonl --out out/x.jsonl --device mps

Pack line: {"pack", "opt": muon|adam, "d", "L", "heads", "seq", "batch", "steps",
            optional "sched" (cos|const|wsd|lin), "warm", "evals", "b2", "ffm", "ns",
            "members": [{"tag", "lr", optional "U" (0 = 32M pool), "off", "wd", "seed", "emb_mult"}]}
Row = pack settings + member settings + params, tokens, epochs, val, train (on the member's own
unique chunk), dec (val loss per token-frequency decile, 0 = most frequent), curve, timing.
"""
import argparse, json, math, os, sys, time
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.3"); os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.2")
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bgpt import BGPT, Opt  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))  # repo root: the first parent folder that has testbed/
while not os.path.isdir(os.path.join(ROOT, "testbed")) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)
DATA = None
V = None
POOL = 32_000_000
EVAL_TOK = 32768


class Store:
    def __init__(self, dev):
        tr = np.memmap(os.path.join(DATA, "train.bin"), dtype=np.uint16, mode="r")
        va = np.memmap(os.path.join(DATA, "val.bin"), dtype=np.uint16, mode="r")
        self.dev = dev
        self.pool = torch.from_numpy(np.asarray(tr[:POOL]).astype(np.int32)).to(dev)
        self.val = torch.from_numpy(np.asarray(va).astype(np.int32)).to(dev)
        cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"unigram_v{V}.npy")
        if not os.path.exists(cache):
            np.save(cache, np.bincount(np.asarray(tr), minlength=V))
        cnt = np.load(cache)
        order = np.argsort(-cnt)
        rank = np.empty(V, np.int64)
        rank[order] = np.arange(V)
        cum = np.cumsum(cnt[order]) / cnt.sum()
        self.decile = torch.from_numpy(np.minimum((cum * 10).astype(np.int64), 9)[rank]).to(dev)

    def fixed(self, arr, seq, n, start, stop):
        idx = torch.linspace(start, stop - seq - 2, n, device=self.dev).long()
        return arr[idx[:, None] + torch.arange(seq + 1, device=self.dev)].long()


def lr_mult(s, steps, warm, sched):
    if s < warm:
        return (s + 1) / warm
    p = (s - warm) / max(1, steps - warm)
    return {"cos": lambda: 0.5 * (1 + math.cos(math.pi * p)), "const": lambda: 1.0,
            "wsd": lambda: 1.0 if p < 0.8 else (1 - p) / 0.2, "lin": lambda: 1 - p}[sched]()


def run_pack(P, S, dev):
    M = P["members"]
    K, seq, B, steps = len(M), P.get("seq", 64), P.get("batch", 32), P["steps"]
    d, L = P["d"], P["L"]
    heads = P.get("heads", max(1, d // 32))
    model = BGPT(K, V, d, L, heads, seq, P.get("ffm", 8 / 3), dev, [m.get("seed", i) for i, m in enumerate(M)])
    lr = torch.tensor([m["lr"] for m in M], device=dev, dtype=torch.float32)
    wd = torch.tensor([m.get("wd", 0.0) for m in M], device=dev, dtype=torch.float32)
    em = torch.tensor([m.get("emb_mult", 1.0) for m in M], device=dev, dtype=torch.float32)
    opt = Opt(model, P["opt"], lr, wd, P.get("b2", 0.95), ns_steps=P.get("ns", 5), emb_mult=em,
              mom=P.get("mom", 0.95), nest=P.get("nest", True), coef=P.get("coef", "ns"), mscale=P.get("mscale", "r02"), pexp=P.get("pexp"), pnorm=P.get("pnorm", "spec"),
              pend=P.get("pend"), total=P["steps"])
    U = torch.tensor([m.get("U", 0) or POOL for m in M], device=dev, dtype=torch.float32)
    off = torch.tensor([m.get("off", 0) for m in M], device=dev, dtype=torch.float32)
    assert all((m.get("off", 0) + (m.get("U", 0) or POOL)) <= POOL for m in M)
    ar = torch.arange(seq + 1, device=dev)
    warm = max(1, int(P.get("warm", 0.05) * steps))
    sched = P.get("sched", "cos")
    nw = EVAL_TOK // seq
    Wv = S.fixed(S.val, seq, nw, 0, len(S.val))
    ne = P.get("evals", 0)
    marks = set(np.unique(np.geomspace(max(1, steps // 64), steps, ne).astype(int)).tolist()) if ne else set()
    curves = [[] for _ in range(K)]
    t0 = time.time()
    for s in range(steps):
        st = (off[:, None] + torch.floor(torch.rand(K, B, device=dev) * (U[:, None] - seq - 1))).long()
        idx = S.pool[st[..., None] + ar].long()
        loss = model.loss(idx, chunk=max(256, 8192 // K))
        loss.sum().backward()
        opt.step(lr_mult(s, steps, warm, sched))
        if (s + 1) in marks and s + 1 < steps:
            cl = model.token_losses(Wv[None].expand(K, -1, -1)[:, : nw // 4]).mean(1).tolist()
            for k in range(K):
                curves[k].append([s + 1, round(cl[k], 4)])
    if dev.type == "mps":
        torch.mps.synchronize()
    ts = time.time() - t0
    tlv = model.token_losses(Wv[None].expand(K, -1, -1))            # [K, nw*seq]
    tgt = Wv[:, 1:].reshape(-1)
    dec = S.decile[tgt]
    dsum = torch.zeros(K, 10, device=dev).index_add_(1, dec, tlv)
    dcnt = torch.zeros(10, device=dev).index_add_(0, dec, torch.ones_like(tgt, dtype=torch.float32))
    decl = (dsum / dcnt).tolist()
    val = tlv.mean(1).tolist()
    Wt = torch.stack([S.fixed(S.pool, seq, nw // 2, m.get("off", 0), m.get("off", 0) + (m.get("U", 0) or POOL))
                      for m in M])
    tr = model.token_losses(Wt).mean(1).tolist()
    tokens = steps * B * seq
    base = {k: v for k, v in P.items() if k != "members"}
    rows = []
    for k, m in enumerate(M):
        r = dict(base)
        r.update(m)
        Uk = m.get("U", 0) or POOL
        ok = math.isfinite(val[k])
        if ne:
            curves[k].append([steps, round(val[k], 4) if ok else None])
        r.update(vocab=V, params=model.n_params(False), nonemb=model.n_params(True), K=K, tokens=tokens, U_eff=Uk,
                 epochs=round(tokens / Uk, 4), val=round(val[k], 5) if ok else None,
                 train=round(tr[k], 5) if math.isfinite(tr[k]) else None,
                 dec=[round(x, 4) for x in decl[k]] if ok else None, curve=curves[k] or None,
                 diverged=not ok, pack_s=round(ts, 2), device=str(dev))
        rows.append(r)
    return rows, ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--budget", type=float, default=0, help="seconds; stop before a pack that would overrun (0 = no limit)")
    a = ap.parse_args()
    t_start, last = time.time(), 0.0
    global DATA, V
    V = a.vocab
    DATA = os.path.join(ROOT, "data", f"fineweb_v{V}")
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    S = Store(dev)
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["pack"] for l in open(a.out) if l.strip()}
    for l in open(a.jobs):
        if not l.strip():
            continue
        P = json.loads(l)
        if P["pack"] in done:
            continue
        if a.budget and time.time() - t_start + 1.3 * last > a.budget:
            print("budget reached", flush=True)
            break
        torch.manual_seed(abs(hash(P["pack"])) % 2**31)
        rows, ts = run_pack(P, S, dev)
        last = ts
        if dev.type == "mps":
            torch.mps.empty_cache()
        with open(a.out, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        v = [r["val"] for r in rows]
        print(f'{P["pack"]:30s} K={len(rows):2d} N={rows[0]["nonemb"]:7d} tok={rows[0]["tokens"]:8d} '
              f'val min={min(x for x in v if x is not None) if any(x is not None for x in v) else float("nan"):.4f} '
              f'{ts:.1f}s' + (f' mem={torch.mps.current_allocated_memory()>>20}/{torch.mps.driver_allocated_memory()>>20}MB' if dev.type == 'mps' else ''), flush=True)


if __name__ == "__main__":
    main()
