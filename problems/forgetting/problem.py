"""Problem: how much of a set of facts does a language model still know at the end of training?

Data: data.jsonl.gz, 375 real training runs (decoder-only, Muon, widths 64-1024, 3-12 layers).
Facts are shown only inside a window of training ("A" = 20%-45% of steps, "B" = 45%-70%); after that the model
keeps training on plain text and slowly forgets them. Target: `kept` at the last eval, the share
of the fact knowledge (in bits) still there at the end of training.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from core.rows import read_jsonl  # noqa: E402

GROUP_BY = "width"
FLOOR = 0.0126   # RMSE between two measurements of the same setting; below this is noise


def rows():
    return read_jsonl(os.path.join(HERE, "data.jsonl.gz"))


def target(r):
    return r["kept"][-1]


def group(r):
    return r["width"]
