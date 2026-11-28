"""Micro language-model runs on FineWeb-edu (vocab 4096) for law finding.

One process loads the data once and runs a list of jobs, appending one JSON row per job.
    python3 -I tl.py --jobs jobs/x.jsonl --out out/x.jsonl --device cpu --threads 4

Job keys: tag, opt (muon|adam), d, L, heads, seq, batch, steps, lr, optional:
  U       unique training tokens (first U tokens of train; 0 = all 73.6M)
  off     start offset of the unique chunk (default 0)
  sched   cos|const|wsd   warm (frac)   wd (Muon wd)   emb_lr  head_lr
  evals   number of log-spaced evaluation points for a loss curve (default 0)
  ffm     ff multiplier    b2   Adam beta2
Row: val loss, train-chunk loss, val loss per token-frequency decile, curve, timings.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "testbed"))
from model import GPT  # noqa: E402
from muon import build_optimizers  # noqa: E402

DATA = os.path.join(ROOT, "data", "fineweb_v4096")
VOCAB = 4096
EVAL_TOK = 32768


class Store:
    def __init__(self):
        self.train = np.memmap(os.path.join(DATA, "train.bin"), dtype=np.uint16, mode="r")
        self.val = np.memmap(os.path.join(DATA, "val.bin"), dtype=np.uint16, mode="r")
        cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unigram.npy")
        if os.path.exists(cache):
            cnt = np.load(cache)
        else:
            cnt = np.bincount(np.asarray(self.train), minlength=VOCAB)
            np.save(cache, cnt)
        self.cnt = cnt
        order = np.argsort(-cnt)            # decile 0 = most frequent tokens
        rank = np.empty(VOCAB, np.int64)
        rank[order] = np.arange(VOCAB)
        # deciles by token mass, so each decile holds ~10% of training tokens
        cum = np.cumsum(cnt[order]) / cnt.sum()
        dec_of_rank = np.minimum((cum * 10).astype(np.int64), 9)
        self.decile = torch.from_numpy(dec_of_rank[rank])
        self.pool = {}

    def pool_on(self, dev, n=32_000_000):
        if dev.type not in self.pool:
            self.pool[dev.type] = torch.from_numpy(np.asarray(self.train[:n]).astype(np.int64)).to(dev)
        return self.pool[dev.type]

    def windows(self, arr, seq, n, rng=None, start=0, stop=None):
        stop = len(arr) if stop is None else stop
        if rng is None:  # fixed, evenly spaced
            idx = np.linspace(start, stop - seq - 2, n).astype(np.int64)
        else:
            idx = rng.integers(start, stop - seq - 1, n)
        return np.stack([arr[i:i + seq + 1] for i in idx]).astype(np.int64)


def lr_mult(s, steps, warm, sched):
    if s < warm:
        return (s + 1) / warm
    p = (s - warm) / max(1, steps - warm)
    if sched == "cos":
        return 0.5 * (1 + math.cos(math.pi * p))
    if sched == "const":
        return 1.0
    if sched == "wsd":
        return 1.0 if p < 0.8 else (1 - p) / 0.2
    if sched == "lin":
        return 1 - p
    raise ValueError(sched)


@torch.no_grad()
def evaluate(model, W, dev, dec=None):
    model.eval()
    tot, n = 0.0, 0
    dsum = torch.zeros(10, dtype=torch.float64)
    dcnt = torch.zeros(10, dtype=torch.float64)
    for i in range(0, len(W), 64):
        b = W[i:i + 64].to(dev)
        lg = model(b[:, :-1]).float()
        ce = F.cross_entropy(lg.reshape(-1, VOCAB), b[:, 1:].reshape(-1), reduction="none").cpu()
        tot += ce.sum().item()
        n += ce.numel()
        if dec is not None:
            dd = dec[b[:, 1:].reshape(-1).cpu()]
            dsum.index_add_(0, dd, ce.double())
            dcnt.index_add_(0, dd, torch.ones_like(ce, dtype=torch.float64))
    model.train()
    out = tot / n
    if dec is None:
        return out, None
    return out, [round(x, 4) for x in (dsum / dcnt.clamp(min=1)).tolist()]


def run(job, S, dev):
    seq, B, steps = job.get("seq", 64), job.get("batch", 32), job["steps"]
    d, L = job["d"], job["L"]
    torch.manual_seed(job.get("seed", 0))
    rng = np.random.default_rng(job.get("seed", 0))
    pool = S.pool_on(dev)
    U = job.get("U", 0) or len(pool)
    off = job.get("off", 0)
    assert off + U <= len(pool)
    gen = torch.Generator(device=dev).manual_seed(job.get("seed", 0)) if dev.type == "cpu" else None
    ar = torch.arange(seq + 1, device=dev)
    model = GPT(VOCAB, d, L, job.get("heads", max(1, d // 32)), seq, job.get("ffm", 8 / 3)).to(dev)
    lr = job["lr"]
    if job["opt"] == "muon":
        opts = build_optimizers(model, lr, job.get("emb_lr"), job.get("head_lr"), job.get("wd", 0.0))
    else:
        opts = [torch.optim.Adam([dict(params=list(model.parameters()), lr=lr, base_lr=lr)],
                                 betas=(0.9, job.get("b2", 0.95)), eps=1e-10)]
    warm = max(1, int(job.get("warm", 0.05) * steps))
    sched = job.get("sched", "cos")
    Wv = torch.from_numpy(S.windows(S.val, seq, EVAL_TOK // seq))
    Wt = torch.from_numpy(S.windows(S.train, seq, EVAL_TOK // seq, start=off, stop=off + U))
    ne = job.get("evals", 0)
    marks = set(np.unique(np.geomspace(max(1, steps // 64), steps, ne).astype(int)).tolist()) if ne else set()
    curve = []
    t0 = time.time()
    diverged = False
    for s in range(steps):
        st = torch.randint(off, off + U - seq - 1, (B,), device=dev, generator=gen)
        b = pool[st[:, None] + ar]
        m = lr_mult(s, steps, warm, sched)
        for o in opts:
            for g in o.param_groups:
                g["lr"] = g["base_lr"] * m
        loss = F.cross_entropy(model(b[:, :-1]).float().reshape(-1, VOCAB), b[:, 1:].reshape(-1))
        loss.backward()
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
        if s % 100 == 0 and not math.isfinite(loss.item()):
            diverged = True
            break
        if (s + 1) in marks and s + 1 < steps:
            curve.append([s + 1, round(evaluate(model, Wv, dev)[0], 4)])
    ts = time.time() - t0
    vl, dec = evaluate(model, Wv, dev, S.decile)
    tl, _ = evaluate(model, Wt, dev)
    if ne:
        curve.append([steps, round(vl, 4)])
    row = dict(job)
    tokens = steps * B * seq
    row.update(params=model.n_params(False), nonemb=model.n_params(True), tokens=tokens,
               U_eff=U, epochs=round(tokens / U, 4), val=round(vl, 5), train=round(tl, 5),
               dec=dec, curve=curve or None, diverged=diverged or not math.isfinite(vl),
               train_s=round(ts, 2), total_s=round(time.time() - t0, 2), device=str(dev))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    S = Store()
    dev = torch.device(a.device)
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["tag"] for l in open(a.out) if l.strip()}
    for l in open(a.jobs):
        if not l.strip():
            continue
        j = json.loads(l)
        if j["tag"] in done:
            continue
        r = run(j, S, dev)
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(f'{r["tag"]:34s} N={r["nonemb"]:7d} tok={r["tokens"]:8d} ep={r["epochs"]:7.2f} '
              f'val={r["val"]:.4f} tr={r["train"]:.4f} {r["total_s"]:.1f}s', flush=True)


if __name__ == "__main__":
    main()
