"""a+hA: hA.py with a steepness that changes with width, q = q0 * (width/256)^c (frozen 2026-11-27 as a second candidate).

    kept = 1 / (1 + exp(q * (z + ln S))),   ln q = lnq0 + c * ln(width/256)

z is the same as in hA.py. 13 constants. Internal leave-one-width-out (900 rows): held-out RMSE 0.0482 on wide rows,
0.0387 on narrow rows, but worse than hA on held-out width 2048 (0.0349 vs 0.0271, 4 rows).
The batch constant (-0.3592 in hA, -0.2894 in a+hA) comes from the internal table, where batch runs 16-1024; the
repo table has batch 64 only, so fit() keeps it fixed and refits the rest.
"""
import math

import numpy as np
from scipy.optimize import least_squares

CONSTANTS = 13
FROZEN = [0.6237, -0.6575, 1.0393, 0.2822, -1.1073, -0.4232, -0.1474, 0.1523, 0.2021, -0.2894, 0.1095, 0.3132, 0.1041]  # theta (11), lnq0, c


def x(r):
    lp, ld = math.log(r["peak"] / 8e-3), math.log(r["width"] / 256)
    return np.array([1.0, lp, r["wd"], r["min_lr_frac"], math.log(r["steps"] / 1024), ld,
                     math.log(r["layers"] / 8), lp * ld, ld * ld, math.log(min(r["batch"], 256) / 64),
                     max(0.0, ld) * (r["window"] == "A")])


def kept(p, r):
    z = x(r) @ np.asarray(p[:11]); q = math.exp(p[11] + p[12] * math.log(r["width"] / 256))
    return 1.0 / (1.0 + math.exp(min(50.0, q * (z + math.log(sum(r["lr"]))))))


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
