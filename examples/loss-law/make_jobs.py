"""Random draws of model width and training steps for the loss-law example.

    python3 examples/loss-law/make_jobs.py fit     > examples/loss-law/jobs_fit.jsonl
    python3 examples/loss-law/make_jobs.py predict > examples/loss-law/jobs_predict.jsonl

`fit` draws 20 runs at widths 48-128 and 100-800 steps for fitting.
`predict` draws 6 new runs, half of them at width 160 (wider than any fit run)
and up to 1000 steps, to test pre-registered predictions outside the fit range.
Random draws instead of a crossed grid: every run adds a new (width, steps) pair.
"""
import json
import math
import random
import sys

HEADS = {48: 2, 64: 2, 96: 3, 128: 4, 160: 5}


def job(rng, widths, lo, hi, prefix):
    d = rng.choice(widths)
    steps = int(round(math.exp(rng.uniform(math.log(lo), math.log(hi))) / 25) * 25)
    return dict(tag=f"{prefix}_d{d}_s{steps}", prio=1 if prefix == "fit" else 2,
                args=dict(d=d, heads=HEADS[d], steps=steps))


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "fit"
    rng = random.Random({"fit": 1, "predict": 2}[which])
    jobs, seen = [], set()
    n, widths, lo, hi = (20, [48, 64, 96, 128], 100, 800) if which == "fit" else (6, [96, 128, 160, 160], 300, 1000)
    while len(jobs) < n:
        j = job(rng, widths, lo, hi, which)
        if j["tag"] not in seen:
            seen.add(j["tag"])
            jobs.append(j)
    for j in jobs:
        print(json.dumps(j))


if __name__ == "__main__":
    main()
