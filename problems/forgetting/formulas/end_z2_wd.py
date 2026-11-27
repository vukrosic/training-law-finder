"""end+z2 times the published weight decay bound (2026-11-27).

    kept = B / (1 + exp(q * (z + ln S)))

B = product, over the steps after the facts, of (1 - learning rate * weight decay). This is the
cumulative weight decay of Bordt et al. (ICML 2025, arXiv:2410.03249), the formula in cum_wd.py.
It has no fitted constant, and B = 1 when weight decay is 0.
S, z and q are those of end_z2.py. 10 constants, the same count as end_z2.

Why: weight decay shrinks every weight by a known share, so the share B is an upper limit on what
is kept. end_z2 handles weight decay only through one term in z. With B in front, a prediction can
never be above the published limit, and the weight decay term in z falls from 0.85 to 0.41.

Held-out error in points, end_z2 -> this formula, one group of each setting held out in turn:
width 4.41 -> 4.11, layers 4.31 -> 4.17, peak 4.15 -> 3.69, schedule 2.76 -> 2.81,
steps 5.15 -> 5.70, window 8.46 -> 8.98.
Weight decay values the fit never saw: 0.1: 2.2 -> 2.2, 0.3: 6.1 -> 4.4, 1.0: 6.3 -> 3.9.
"""
import math

import numpy as np
from scipy.optimize import least_squares

CONSTANTS = 10
START = [0.6090888304082575, -0.5929143343302276, 0.8482437722622462, 0.5477102470699494,
         -1.1231337626121998, -0.3217546827755414, -0.12444372936842882, 0.17762544026882962,
         0.23118595535279032, 0.24836649590480928]   # the end_z2 constants: theta (9), lnq0


def x(r):
    lp, ld = math.log(r["peak"] / 8e-3), math.log(r["width"] / 256)
    return np.array([1.0, lp, r["wd"], r["min_lr_frac"], math.log(r["steps"] / 1024), ld,
                     math.log(r["layers"] / 8), lp * ld, ld * ld])


def kept(p, r):
    lr = np.asarray(r["lr"])
    z = x(r) @ np.asarray(p[:9])
    bound = float(np.prod(1.0 - lr * r["wd"]))
    return bound / (1.0 + math.exp(min(50.0, math.exp(p[9]) * (z + math.log(lr.sum())))))


def fit(rows, y):
    res = least_squares(lambda p: [kept(p, r) - t for r, t in zip(rows, y)], START)
    return res.x


def predict(params, row):
    return kept(params, row)
