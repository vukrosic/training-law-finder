"""Write data/synthetic.jsonl.gz: a small MADE-UP fixture so the scorer and tools/check.py run before real runs exist.

    python3 problems/loss-curve/make_fixture.py

Settings: the grid lines of prereg/loss-curve-grid.jsonl at widths 32, 64, 128, 256, 512 with at most 2,048
steps and 3 layers (blocks A, B, D, E): 90 runs. Curves come from an invented law, not from training:

    L(t) = 1.6 + 40 N^-0.35 + 0.6 (c_w * peak * area(t) + 0.05)^-0.4 + 8 * peak * lr_mult(t) + noise
    c_w = (width / 64)^0.3, N = non-embedding parameters, noise = normal with SD 0.003

Band losses are the curve plus fixed offsets. None of these numbers say anything about real training;
they only check that the code runs. Delete this file's output when data/llm.jsonl.gz exists.
"""
import gzip
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import problem  # noqa: E402

WIDTHS = {32, 64, 128, 256, 512}


def main():
    rng = np.random.default_rng(0)
    out = []
    for line in open(os.path.join(ROOT, "prereg", "loss-curve-grid.jsonl")):
        g = json.loads(line)
        d = g["draw"]
        if d["width"] not in WIDTHS or d["steps"] > 2048 or d["layers"] != 3 or g["block"] not in "ABDE":
            continue
        T, w = d["steps"], d["width"]
        mult = np.array([problem.lr_mult(s, T, 20, d["sched"], d["decay"], d["min_lr_frac"]) for s in range(T)])
        area = np.cumsum(mult)
        steps = sorted({max(1, int(round(i * T / problem.POINTS))) for i in range(1, problem.POINTS + 1)})
        cw = (w / 64) ** 0.3
        loss, lrs, bands = [], [], []
        for st in steps:
            L = (1.6 + 40 * d["nonemb_params"] ** -0.35 + 0.6 * (cw * d["peak"] * area[st - 1] + 0.05) ** -0.4
                 + 8 * d["peak"] * mult[st - 1] + rng.normal(0, 0.003))
            loss.append(round(float(L), 5))
            lrs.append(round(float(mult[st - 1]), 6))
            bands.append([round(float(L + o), 5) for o in (-1.5, -0.3, 0.4, 1.2)])
        out.append({"tag": "synthetic_" + g["tag"], "machine": "mac", "device": "cpu", "width": w,
                    "layers": d["layers"], "heads": d["heads"], "hidden": 2 * w, "batch": 64, "seq": 64, "steps": T,
                    "sched": d["sched"], "peak": d["peak"], "mlr": d["peak"], "wd": 0.0,
                    "min_lr_frac": d["min_lr_frac"], "warmup": 20, "decay": d["decay"], "params": d["params"],
                    "nonemb_params": d["nonemb_params"], "tokens": d["tokens"], "lc_step": steps, "lc_lr": lrs,
                    "lc_loss": loss, "lc_band": bands, "lc_band_share": [0.45, 0.35, 0.15, 0.05]})
    dst = os.path.join(HERE, "data", "synthetic.jsonl.gz")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with gzip.GzipFile(dst, "wb", mtime=0) as f:
        f.write("".join(json.dumps(r) + "\n" for r in out).encode())
    print(f"{len(out)} made-up runs -> {os.path.relpath(dst, ROOT)}")


if __name__ == "__main__":
    main()
