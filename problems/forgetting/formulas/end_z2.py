"""end+z2: the best forgetting formula so far (frozen 2026-11-26).

    kept = 1 / (1 + exp(q * (z + ln S)))

S = sum of the learning rate over every step after the facts stop (the forgetting clock).
z = theta . x, the "onset": how much learning rate it takes before facts start to go.
x = [1, ln(peak/8e-3), wd, min_lr_frac, ln(steps/1024), ln(width/256), ln(layers/8),
     ln(peak/8e-3)*ln(width/256), ln(width/256)^2]
q = exp(lnq0), the steepness of the drop. 10 constants.

Fit on all 375 rows, in-sample RMSE 0.0375. Known weakness: it overpredicts width 1024+ with
3 layers by +0.02 to +0.05 (three runs predicted before they ran).
"""
import math

import numpy as np
from scipy.optimize import least_squares

CONSTANTS = 10
FROZEN = [0.6090888304082575, -0.5929143343302276, 0.8482437722622462, 0.5477102470699494,
          -1.1231337626121998, -0.3217546827755414, -0.12444372936842882, 0.17762544026882962,
          0.23118595535279032, 0.24836649590480928]   # theta (9), lnq0


def x(r):
    lp, ld = math.log(r["peak"] / 8e-3), math.log(r["width"] / 256)
    return np.array([1.0, lp, r["wd"], r["min_lr_frac"], math.log(r["steps"] / 1024), ld,
                     math.log(r["layers"] / 8), lp * ld, ld * ld])


def kept(p, r):
    z = x(r) @ np.asarray(p[:9])
    return 1.0 / (1.0 + math.exp(min(50.0, math.exp(p[9]) * (z + math.log(sum(r["lr"]))))))


def fit(rows, y):
    res = least_squares(lambda p: [kept(p, r) - t for r, t in zip(rows, y)], FROZEN)
    return res.x


def predict(params, row):
    return kept(params, row)
