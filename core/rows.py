"""Load run data.

Two kinds of data:

1. Problem data, small and frozen, kept in git: problems/<problem>/data/<model_family>.jsonl.gz
   (for example problems/forgetting/data/llm.jsonl.gz). Scores are computed on these files only,
   so a score never changes when new runs arrive.

       from core.rows import problem_rows
       rows = problem_rows("forgetting")                 # every model family, combined
       rows = problem_rows("forgetting", family="llm")   # one model family

   Each row gets a "model_family" field (the file name), so a formula can treat families apart.

2. The full run tables, large, kept on Hugging Face (vukrosic/training-laws-research) and
   downloaded into data/<model_family>/ the first time you ask for them:

       from core.rows import load
       rows = load()                                     # data/llm/runs.jsonl.gz, about 24,000 runs
       rows = load(opt="muon", method="dense")           # keep rows whose fields equal these values
       rows = load(where=lambda r: r["val_loss"] is not None)

   Columns: data/llm/SCHEMA.md and core/columns.yaml.
"""
import glob
import gzip
import json
import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF = "https://huggingface.co/datasets/vukrosic/training-laws-research/resolve/main/"
RUNS = os.path.join(ROOT, "data", "llm", "runs.jsonl.gz")
SIM = os.path.join(ROOT, "data", "llm", "sim.jsonl.gz")
OTHER = os.path.join(ROOT, "data", "llm", "other_searches.jsonl.gz")


def read_jsonl(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def fetch(path):
    """Download a full run table from Hugging Face if it is not on disk yet."""
    if not os.path.exists(path):
        name = os.path.basename(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"downloading {name} from Hugging Face (once) ...")
        urllib.request.urlretrieve(HF + name, path + ".part")
        os.replace(path + ".part", path)
    return path


def vocab(r):
    return (r.get("cfg_all") or {}).get("vocab")


def load(path=RUNS, where=None, **eq):
    rows = read_jsonl(fetch(path))
    if eq:
        rows = [r for r in rows if all(r.get(k) == v for k, v in eq.items())]
    if where:
        rows = [r for r in rows if where(r)]
    return rows


def families(problem):
    return sorted(os.path.basename(f)[:-len(".jsonl.gz")]
                  for f in glob.glob(os.path.join(ROOT, "problems", problem, "data", "*.jsonl.gz")))


def problem_rows(problem, family=None):
    have = families(problem)
    if family is not None and family not in have:
        raise SystemExit(f"problem {problem} has no data for model family '{family}'; it has: {', '.join(have)}")
    rows = []
    for fam in ([family] if family else have):
        for r in read_jsonl(os.path.join(ROOT, "problems", problem, "data", fam + ".jsonl.gz")):
            r["model_family"] = fam
            rows.append(r)
    return rows
