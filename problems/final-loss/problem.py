"""Problem: predict a language model's final validation loss from its training settings.

Data: data/llm.jsonl.gz, a frozen copy of every dense Muon run with a 1,024-token vocabulary
(diverged runs removed, widths 32-1024, 1-12 layers, 160-2048 steps, constant / cosine / WSD schedules,
many learning rates). tools/build_problem_data.py made it from the full run table with select() below.
Target: val_loss (nats per token on held-out FineWeb text) at the end of training.
Grouped by width, so a formula must predict model sizes it never saw.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from core.rows import problem_rows, vocab  # noqa: E402

GROUP_BY = "width"
FLOOR = 0.0001  # 115 duplicate tags (CPU, Kaggle, Mac, GPU) agree within 0.00006 RMS

# Used only by tools/build_problem_data.py to make data/llm.jsonl.gz from data/llm/runs.jsonl.gz.
SOURCE = {"llm": "runs.jsonl.gz"}
COLUMNS = ["tag", "machine", "device", "width", "layers", "heads", "hidden", "batch", "seq", "steps", "sched",
           "peak", "mlr", "wd", "min_lr_frac", "warmup", "decay", "momentum", "params", "nonemb_params",
           "tokens", "val_loss", "val_bpb"]


def select(r):
    return (r.get("val_loss") is not None and math.isfinite(r["val_loss"]) and r["val_loss"] < 8
            and r.get("method") == "dense" and r.get("opt") == "muon" and vocab(r) == 1024)


def rows(family=None):
    return problem_rows("final-loss", family)


def target(r):
    return r["val_loss"]


def group(r):
    return r["width"]
