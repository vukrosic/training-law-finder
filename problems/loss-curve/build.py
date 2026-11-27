"""Build data/llm.jsonl.gz from testbed/train.py result lines that carry an "lc" entry (--curve_points 32).

    python3 problems/loss-curve/build.py results_a.jsonl [results_b.jsonl ...]

Keeps runs with 32 finite readings, a final val_loss under 8, no divergence, and the same lr for the Muon
and Adam parts. Drops widths with fewer than MIN_GROUP runs. Prints runs per width, the floor (if two runs
share their settings) and the next commands. Remove data/synthetic.jsonl.gz and empty scores.jsonl before
scoring on real runs: rows() combines every data file.
"""
import gzip
import json
import math
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import problem  # noqa: E402


def ff_hidden(d, mult, multiple=32):
    return int(multiple * math.ceil(d * mult / multiple))


def convert(r):
    c, lc = r["config"], r["lc"]
    T = c["steps"]
    return {"tag": r["tag"], "machine": "gpu" if r.get("device") == "cuda" else "mac", "device": r.get("device"),
            "width": c["d"], "layers": c["layers"], "heads": c["heads"], "hidden": ff_hidden(c["d"], c["ff_mult"]),
            "batch": c["batch"], "seq": c["seq"], "steps": T, "sched": c["sched"], "peak": c["lr"], "mlr": c["lr"],
            "wd": c["wd"], "min_lr_frac": c["min_lr_frac"], "warmup": int(round(c["warmup"] * T)),
            "decay": c["decay"], "params": r["params"], "nonemb_params": r["nonemb_params"], "tokens": r["tokens"],
            "lc_step": lc["step"], "lc_lr": lc["lr_mult"], "lc_loss": lc["val_loss"], "lc_band": lc["band_loss"],
            "lc_band_share": lc["band_share"]}


def keep(r):
    lc, c = r.get("lc"), r.get("config", {})
    return (lc is not None and len(lc["val_loss"]) == problem.POINTS
            and all(math.isfinite(x) for x in lc["val_loss"]) and not r.get("diverged")
            and r.get("val_loss") is not None and r["val_loss"] < 8
            and c.get("emb_lr") is None and c.get("head_lr") is None)


def main(paths):
    runs = {}
    for p in paths:
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                if keep(r):
                    runs[r["tag"]] = convert(r)   # a tag seen twice keeps its last line
    n = Counter(r["width"] for r in runs.values())
    out = [r for r in runs.values() if n[r["width"]] >= problem.MIN_GROUP]
    dst = os.path.join(HERE, "data", "llm.jsonl.gz")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with gzip.open(dst, "wt") as f:
        for r in sorted(out, key=lambda r: r["tag"]):
            f.write(json.dumps(r) + "\n")
    print(f"{len(out)} runs ({len(runs) - len(out)} dropped by MIN_GROUP) -> {os.path.relpath(dst)}")
    print("runs per width: " + "  ".join(f"{w}:{n[w]}" for w in sorted(n)))
    print("floor: " + str(problem.floor(problem.rows("llm"))))
    print("next: rm data/synthetic.jsonl.gz, empty scores.jsonl, set FLOOR, score every formula with --record,"
          " then python3 core/leaderboard.py loss-curve")


if __name__ == "__main__":
    main(sys.argv[1:])
