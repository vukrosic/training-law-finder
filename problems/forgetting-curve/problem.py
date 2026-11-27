"""Problem: how much of a set of facts does a language model still know at each step after it last saw them?

Data: data/llm.jsonl.gz, 1,283 real training runs (decoder-only, Muon, 1,024-token vocabulary, widths 24-2048,
1-12 layers, 512-16,384 steps, constant / cosine / WSD schedules, weight decay 0-1).
64 facts are shown only inside a window of training (A = 20%-45% of steps, B = 45%-70%); after that the model
trains on plain text and slowly forgets them. Each run is read at 8 points after the window
(1/16, 1/8, 1/4, 3/8, 1/2, 5/8, 3/4 and all of the way to the end), so each run gives 8 rows.
Target: kept_now, the share of the learned fact bits still there at that reading (mean of the 5 readings around it,
which removes a step-to-step wobble).
Grouped by width (widths with at least 5 runs), so a formula must predict model sizes it never saw.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from core.rows import problem_rows, vocab  # noqa: E402

GROUP_BY = "width"
TARGET = "kept_now"   # column target() reads; never shown to a formula
MIN_GROUP = 5
FLOOR = 0.0043   # pooled SD of kept_now over runs with identical settings: 384 (setting, reading) groups, 832 rows
FRACS = [1 / 16, 1 / 8, 1 / 4, 3 / 8, 1 / 2, 5 / 8, 3 / 4, 1.0]

# Used only by tools/build_problem_data.py to make data/llm.jsonl.gz from data/llm/runs.jsonl.gz.
SOURCE = {"llm": "runs.jsonl.gz"}
COLUMNS = ["tag", "machine", "width", "layers", "heads", "hidden", "batch", "seq", "steps", "sched", "peak", "mlr",
           "wd", "min_lr_frac", "warmup", "decay", "momentum", "window", "fwlo", "fwhi", "params", "nonemb_params",
           "tokens"]


def select(r):
    """Plain forgetting runs: protocol lf (no branch, no replay, no optimizer change), dense Muon, 64 facts,
    clipping 1.0, no bf16, const / cos / wsd schedules, with a kept curve."""
    return (r.get("protocol") == "lf" and r.get("method") == "dense" and r.get("opt") == "muon"
            and vocab(r) == 1024 and r.get("F") == 64 and r.get("clip") == 1.0 and not r.get("amp")
            and r.get("sched") in ("const", "cos", "wsd") and bool(r.get("kept")) and bool(r.get("lr")))


def project(r):
    """Keep the settings and 8 readings: steps since the window (rd_t), lr sum so far (rd_S), lr there (rd_lr)
    and the smoothed kept share (rd_kept). S_end = lr sum over all steps after the window."""
    out = {k: r.get(k) for k in COLUMNS}
    kept, lr = r["kept"], r["lr"]
    n = len(kept)
    S, s = [], 0.0
    for x in lr:
        s += x
        S.append(s)
    idx = [max(1, round(f * (n - 1))) for f in FRACS]
    out["n_after"] = n
    out["S_end"] = round(S[-1], 6)
    out["rd_t"] = idx
    out["rd_S"] = [round(S[i], 6) for i in idx]
    out["rd_lr"] = [round(lr[i], 8) for i in idx]
    out["rd_kept"] = [round(sum(kept[max(0, i - 2):i + 3]) / len(kept[max(0, i - 2):i + 3]), 5) for i in idx]
    return out


def rows(family=None):
    """One row per (run, reading)."""
    out = []
    for r in problem_rows("forgetting-curve", family):
        base = {k: v for k, v in r.items() if not k.startswith("rd_")}
        for j, f in enumerate(FRACS):
            d = dict(base)
            d.update(frac=f, t_now=r["rd_t"][j], S_now=r["rd_S"][j], lr_now=r["rd_lr"][j], kept_now=r["rd_kept"][j])
            out.append(d)
    return out


def target(r):
    return r["kept_now"]


def group(r):
    return r["width"]
