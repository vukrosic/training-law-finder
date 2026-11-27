"""clock_z plus two settings the end-of-training data could not see: the window position and the batch.

    kept = 1 / (1 + exp(q * (z + ln S)))

S = sum of the learning rate from the end of the fact window to this reading (S_now).
z = theta . x with
x = [1, ln(peak/8e-3), wd, min_lr_frac, ln(steps/1024), ln(width/256), ln(layers/8),
     ln(peak/8e-3)*ln(width/256), ln(width/256)^2, fwhi, ln(batch/64)]
q = exp(lnq0). 12 constants, fit from zero.
"""
import math

import numpy as np
from scipy.optimize import least_squares

CONSTANTS = 12


def x(r):
    lp, ld = math.log(r["peak"] / 8e-3), math.log(r["width"] / 256)
    return np.array([1.0, lp, r["wd"], r["min_lr_frac"], math.log(r["steps"] / 1024), ld,
                     math.log(r["layers"] / 8), lp * ld, ld * ld, r["fwhi"], math.log(r["batch"] / 64)])


def kept(p, r):
    z = x(r) @ np.asarray(p[:11])
    return 1.0 / (1.0 + math.exp(min(50.0, math.exp(p[11]) * (z + math.log(r["S_now"])))))


def fit(rows, y):
    return least_squares(lambda p: [kept(p, r) - t for r, t in zip(rows, y)], np.zeros(12)).x


def predict(params, row):
    return kept(params, row)
