"""end+z3: end_z2 with two more terms in the onset z (2026-11-27).

    kept = 1 / (1 + exp(q * (z + ln S)))

S = sum of the learning rate over every step after the facts stop (the forgetting clock).
z = theta . x, with the nine terms of end_z2 and two new ones:
    sqrt(wd) * ln(peak/8e-3)   weight decay costs more at a high peak learning rate
    ln(share of S that falls in the last quarter of the steps after the facts)
                               learning rate spent late forgets less than the same amount spent early
q = exp(lnq0). 12 constants.

Held-out error in points, end_z2 -> end_z3, one group of each setting held out in turn:
width 4.41 -> 3.66, layers 4.31 -> 3.54, steps 5.15 -> 4.99, peak 4.15 -> 3.24,
schedule 5.92 -> 6.06, weight decay 13.91 -> 11.32, window 8.47 -> 8.90.
Score 1 falls by 0.0075 for two constants, which is under 0.005 per constant.
"""
import math

import numpy as np
from scipy.optimize import least_squares

CONSTANTS = 12
START = [0.6090888304082575, -0.5929143343302276, 0.8482437722622462, 0.5477102470699494,
         -1.1231337626121998, -0.3217546827755414, -0.12444372936842882, 0.17762544026882962,
         0.23118595535279032, 0.0, 0.0, 0.24836649590480928]   # end_z2 constants, new terms at 0


def x(r):
    lp, ld = math.log(r["peak"] / 8e-3), math.log(r["width"] / 256)
    lr = r["lr"]
    late = sum(lr[len(lr) - (len(lr) + 3) // 4:]) / sum(lr)
    return np.array([1.0, lp, r["wd"], r["min_lr_frac"], math.log(r["steps"] / 1024), ld,
                     math.log(r["layers"] / 8), lp * ld, ld * ld,
                     math.sqrt(r["wd"]) * lp, math.log(max(late, 1e-4))])


def kept(p, r):
    z = x(r) @ np.asarray(p[:11])
    return 1.0 / (1.0 + math.exp(min(50.0, math.exp(p[11]) * (z + math.log(sum(r["lr"]))))))


def fit(rows, y):
    X = np.array([x(r) for r in rows]); c = np.array([math.log(sum(r["lr"])) for r in rows]); y = np.asarray(y)
    f = lambda p: 1.0 / (1.0 + np.exp(np.minimum(50.0, np.exp(p[11]) * (X @ p[:11] + c)))) - y
    return least_squares(f, START).x


def predict(params, row):
    return kept(params, row)
