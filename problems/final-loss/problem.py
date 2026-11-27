"""Problem: predict a language model's final validation loss from its training settings.

Data: data/llm.jsonl.gz, a frozen copy of the plain dense Muon runs with a 1,024-token vocabulary
(select() below says what "plain" means): 1,163 runs, widths 24-1024 (each with 5+ runs), 1-12 layers,
512-16,384 steps, constant / cosine / WSD schedules, many learning rates.
tools/build_problem_data.py made it from the full run table with select() below.
Target: val_loss (nats per token on held-out FineWeb text) at the end of training.
Grouped by width (only widths with at least 5 runs), so a formula must predict model sizes it never saw.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from core.rows import problem_rows, vocab  # noqa: E402

GROUP_BY = "width"
TARGET = "val_loss"   # column target() reads; never shown to a formula
MIN_GROUP = 5  # widths with fewer runs are left out when the data is built: one run cannot weigh as much as 10,000
FLOOR = 0.0019  # spread of rows with the same visible settings: pooled SD over 36 such settings (80 rows)

# Used only by tools/build_problem_data.py to make data/llm.jsonl.gz from data/llm/runs.jsonl.gz.
SOURCE = {"llm": "runs.jsonl.gz"}
COLUMNS = ["tag", "machine", "device", "width", "layers", "heads", "hidden", "batch", "seq", "steps", "sched",
           "peak", "mlr", "wd", "min_lr_frac", "warmup", "decay", "momentum", "params", "nonemb_params",
           "tokens", "val_loss", "val_bpb"]


def select(r):
    """Plain runs only, so two rows with the same visible settings are the same training run.

    protocol "lf" = no branch from a checkpoint, no replayed facts, no optimizer change after the
    fact window. Window A only (the window position changes the text). Clipping at 1.0, no bf16.
    Rewarm schedules (crw, cosrw) are left out: their shape depends on columns a formula cannot see.
    """
    return (r.get("val_loss") is not None and math.isfinite(r["val_loss"]) and r["val_loss"] < 8
            and r.get("method") == "dense" and r.get("opt") == "muon" and vocab(r) == 1024
            and r.get("protocol") == "lf" and r.get("window") == "A" and r.get("clip") == 1.0
            and not r.get("amp") and r.get("sched") in ("const", "cos", "wsd"))


def rows(family=None):
    return problem_rows("final-loss", family)


def target(r):
    return r["val_loss"]


def group(r):
    return r["width"]
