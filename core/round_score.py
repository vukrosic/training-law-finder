"""Score a prediction round: python3 core/round_score.py rounds/<NNN>-<problem>

Reads predictions.jsonl ({"tag", "settings", "pred": {name: value}}) and results.jsonl ({"tag", "actual"})
and prints RMSE and mean bias (predicted - actual) per formula over the runs that have results.
"""
import json
import math
import os
import sys


def read(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main(d):
    preds = {r["tag"]: r["pred"] for r in read(os.path.join(d, "predictions.jsonl"))}
    actual = {r["tag"]: r["actual"] for r in read(os.path.join(d, "results.jsonl"))}
    tags = [t for t in preds if t in actual]
    print(f"{len(tags)} of {len(preds)} predicted runs have results")
    names = sorted({n for t in tags for n in preds[t]})
    for n in names:
        e = [preds[t][n] - actual[t] for t in tags if n in preds[t]]
        print(f"{n:24s} RMSE {math.sqrt(sum(x * x for x in e) / len(e)):.4f}  bias {sum(e) / len(e):+.4f}  n={len(e)}")


if __name__ == "__main__":
    main(sys.argv[1])
