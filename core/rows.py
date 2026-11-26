"""Load the shared run table.

data/llm/runs.jsonl.gz holds every training run we have (23,874 rows), one JSON object per line.
data/llm/sim.jsonl.gz holds 4,725 runs of the fast simulator. Columns: data/llm/SCHEMA.md.

    from core.rows import load
    rows = load(opt="muon", method="dense")            # keep rows whose fields equal these values
    rows = load(where=lambda r: r["val_loss"] is not None)
"""
import gzip
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "data", "llm", "runs.jsonl.gz")
SIM = os.path.join(ROOT, "data", "llm", "sim.jsonl.gz")


def read_jsonl(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def vocab(r):
    return (r.get("cfg_all") or {}).get("vocab")


def load(path=RUNS, where=None, **eq):
    rows = read_jsonl(path)
    if eq:
        rows = [r for r in rows if all(r.get(k) == v for k, v in eq.items())]
    if where:
        rows = [r for r in rows if where(r)]
    return rows
