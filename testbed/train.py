"""Train one small language model and print one JSON row of results.

    python3 testbed/train.py --d 128 --layers 4 --steps 500 --tag d128_s500 --out results.jsonl

Every knob is a flag, so a job is just a set of flags (see queue/run_jobs.py).
The row holds the config, parameter counts, tokens seen, final validation loss
and bits per byte, an optional loss curve, and optional planted-fact recall.

Learning-rate schedules (peak lr, warmup is a fraction of the steps):
    cos    warmup, then cosine from peak to min_lr_frac * peak
    const  warmup, then flat
    wsd    warmup, flat, then linear decay to min_lr_frac * peak over the last `decay` share
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT  # noqa: E402
from muon import build_optimizers  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default="", help="append the JSON row to this file")
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "fineweb_v8192"))
    # model
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ff_mult", type=float, default=8 / 3)
    ap.add_argument("--qk_norm", type=int, default=1)
    # training
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.01, help="peak lr (Muon and Adam)")
    ap.add_argument("--emb_lr", type=float, default=None)
    ap.add_argument("--head_lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=0.0, help="Muon decoupled weight decay")
    ap.add_argument("--sched", choices=["cos", "const", "wsd"], default="cos")
    ap.add_argument("--warmup", type=float, default=0.05, help="fraction of steps")
    ap.add_argument("--decay", type=float, default=0.2, help="wsd decay share")
    ap.add_argument("--min_lr_frac", type=float, default=0.0)
    # eval
    ap.add_argument("--eval_tokens", type=int, default=131072)
    ap.add_argument("--eval_every", type=int, default=0, help="0 = only at the end")
    # planted facts
    ap.add_argument("--facts", type=int, default=0)
    ap.add_argument("--facts_per_bin", type=int, default=64)
    ap.add_argument("--facts_kmax", type=int, default=16)
    ap.add_argument("--facts_start", type=float, default=0.0)
    ap.add_argument("--facts_end", type=float, default=1.0)
    ap.add_argument("--facts_dump", default="", help="write per-person recall rows here")
    # system
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--save", default="", help="save the final weights here")
    return ap.parse_args(argv)


def lr_mult(step, a):
    T = a.steps
    w = int(round(a.warmup * T))
    if step < w:
        return (step + 1) / w
    if a.sched == "const":
        return 1.0
    if a.sched == "cos":
        p = (step - w) / max(1, T - w)
        return a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p))
    d0 = T - int(round(a.decay * T))
    if step < d0:
        return 1.0
    return 1.0 - (1.0 - a.min_lr_frac) * (step - d0 + 1) / max(1, T - d0)


class Data:
    def __init__(self, path, seq, batch, seed):
        self.train = np.memmap(os.path.join(path, "train.bin"), dtype=np.uint16, mode="r")
        self.val = np.memmap(os.path.join(path, "val.bin"), dtype=np.uint16, mode="r")
        self.seq, self.batch = seq, batch
        self.rng = np.random.default_rng(seed)

    def batch_np(self):
        ix = self.rng.integers(0, len(self.train) - self.seq - 1, self.batch)
        return np.stack([self.train[i:i + self.seq + 1] for i in ix]).astype(np.int64)


@torch.no_grad()
def evaluate(model, val_windows, val_bytes, device):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(val_windows), 16):
        w = torch.from_numpy(val_windows[i:i + 16]).to(device)
        logits = model(w[:, :-1]).float()
        tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)), w[:, 1:].reshape(-1),
                               reduction="sum").item()
        n += w[:, 1:].numel()
    model.train()
    loss = tot / n
    return loss, loss / math.log(2) * n / val_bytes


def main(argv=None):
    a = get_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    device = a.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    t0 = time.time()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.data, "tokenizer.json"))
    vocab = tok.get_vocab_size()
    data = Data(a.data, a.seq, a.batch, a.seed)

    # fixed validation windows and their exact UTF-8 byte count (for bits per byte)
    nwin = max(1, a.eval_tokens // a.seq)
    val_windows = np.stack([data.val[i * a.seq:i * a.seq + a.seq + 1]
                            for i in range(nwin)]).astype(np.int64)
    val_bytes = sum(len(tok.decode(w[1:].tolist(), skip_special_tokens=False).encode("utf-8"))
                    for w in val_windows)

    world = None
    if a.facts:
        from facts import FactWorld
        world = FactWorld(tok, a.facts_per_bin, a.facts_kmax, a.facts_start, a.facts_end,
                          a.steps, seed=a.seed)

    model = GPT(vocab, a.d, a.layers, a.heads, a.seq, a.ff_mult, bool(a.qk_norm)).to(device)
    opts = build_optimizers(model, a.lr, a.emb_lr, a.head_lr, a.wd)
    curve, recent, dropped = [], [], 0
    for step in range(a.steps):
        buf = data.batch_np()
        if world is not None:
            bt = world.step_tokens(step)
            if bt:
                flat = buf.reshape(-1)
                if len(bt) > flat.size:
                    dropped += len(bt) - flat.size
                    bt = bt[:flat.size]
                flat[:len(bt)] = bt
        b = torch.from_numpy(buf).to(device)
        logits = model(b[:, :-1])
        loss = F.cross_entropy(logits.float().reshape(-1, vocab), b[:, 1:].reshape(-1))
        loss.backward()
        m = lr_mult(step, a)
        for o in opts:
            for g in o.param_groups:
                g["lr"] = g["base_lr"] * m
            o.step()
            o.zero_grad(set_to_none=True)
        recent.append(loss.item())
        recent = recent[-20:]
        if not math.isfinite(recent[-1]):
            break
        if a.eval_every and (step + 1) % a.eval_every == 0 and step + 1 < a.steps:
            vl, _ = evaluate(model, val_windows, val_bytes, device)
            pt = dict(step=step + 1, val_loss=round(vl, 4))
            if world is not None:
                pt["facts_acc"] = world.evaluate(model, device)[0]["acc"]
            curve.append(pt)

    vl, bpb = evaluate(model, val_windows, val_bytes, device)
    row = dict(tag=a.tag, config={k: v for k, v in vars(a).items()
                                  if k not in ("out", "tag", "save", "facts_dump", "device", "threads")},
               vocab=vocab, params=model.n_params(False), nonemb_params=model.n_params(True),
               tokens=a.steps * a.batch * a.seq, val_loss=round(vl, 5), val_bpb=round(bpb, 5),
               train_loss=round(float(np.mean(recent)), 5), diverged=not math.isfinite(vl))
    if curve:
        curve.append(dict(step=a.steps, val_loss=round(vl, 4)))
        row["curve"] = curve
    if world is not None:
        summary, rows = world.evaluate(model, device)
        row["facts"] = summary
        if dropped:
            row["facts"]["dropped_tokens"] = dropped
        if a.facts_dump:
            with open(a.facts_dump, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
    row["seconds"] = round(time.time() - t0, 1)
    row["finished"] = round(time.time(), 1)
    if a.save:
        torch.save(model.state_dict(), a.save)
    line = json.dumps(row)
    print(line)
    if a.out:
        with open(a.out, "a") as f:
            f.write(line + "\n")
    return row


if __name__ == "__main__":
    main()
