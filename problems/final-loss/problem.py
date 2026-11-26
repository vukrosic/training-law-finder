"""Problem: predict a language model's final validation loss from its training settings.

Data: every dense Muon run with a 1,024-token vocabulary in data/llm/runs.jsonl.gz (diverged runs removed,
widths 32-1024, 1-12 layers, 160-2048 steps, constant / cosine / WSD schedules, many learning rates).
Target: val_loss (nats per token on held-out FineWeb text) at the end of training.
Grouped by width, so a formula must predict model sizes it never saw.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from core.rows import load, vocab  # noqa: E402

GROUP_BY = "width"
FLOOR = 0.0001  # 115 duplicate tags (CPU, Kaggle, Mac, GPU) agree within 0.00006 RMS


def rows():
    return load(where=lambda r: r.get("val_loss") is not None and math.isfinite(r["val_loss"]) and r["val_loss"] < 8 and r.get("method") == "dense"
                and r.get("opt") == "muon" and vocab(r) == 1024)


def target(r):
    return r["val_loss"]


def group(r):
    return r["width"]
