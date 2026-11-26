"""hA: end+z2 plus a capped batch term and a width hinge for window A (frozen 2026-11-26, internal fit on 900 rows).

    kept = 1 / (1 + exp(q * (z + ln S)))

Same clock S and features as end_z2.py, plus two terms:
  lB = ln(min(batch, 256)/64)          larger batches forget less, up to a cap of 256
  hA = max(0, ln(width/256)) * [window A]   wide models with facts early in training (window A) keep less than the width terms alone say
q = exp(lnq0). 12 constants.

Internal leave-one-width-out (900 rows, 125 at width >= 384): held-out RMSE 0.0495 on wide rows, 0.0396 on narrow rows.
The batch constant (-0.3592 in hA, -0.2894 in a+hA) comes from the internal table, where batch runs 16-1024; the
repo table has batch 64 only, so fit() keeps it fixed and refits the rest.
"""
import math

import numpy as np
from scipy.optimize import least_squares

CONSTANTS = 12
FROZEN = [0.6529, -0.6265, 0.9592, 0.4384, -1.1127, -0.3611, -0.1219, 0.1857, 0.1713, -0.3592, 0.15, 0.2710]  # theta (11), lnq0


def x(r):
    lp, ld = math.log(r["peak"] / 8e-3), math.log(r["width"] / 256)
    return np.array([1.0, lp, r["wd"], r["min_lr_frac"], math.log(r["steps"] / 1024), ld,
                     math.log(r["layers"] / 8), lp * ld, ld * ld, math.log(min(r["batch"], 256) / 64),
                     max(0.0, ld) * (r["window"] == "A")])


def kept(p, r):
    z = x(r) @ np.asarray(p[:11])
    return 1.0 / (1.0 + math.exp(min(50.0, math.exp(p[11]) * (z + math.log(sum(r["lr"]))))))


FREE = [i for i in range(len(FROZEN)) if i != 9]   # the batch constant (index 9) stays at its internal value: every repo row has batch 64


def fit(rows, y):
    def full(f):
        p = list(FROZEN)
        for i, v in zip(FREE, f):
            p[i] = v
        return p
    return full(least_squares(lambda f: [kept(full(f), r) - t for r, t in zip(rows, y)], [FROZEN[i] for i in FREE]).x)


def predict(params, row):
    return kept(params, row)
