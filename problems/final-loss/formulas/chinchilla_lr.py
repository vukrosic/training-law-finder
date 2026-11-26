"""Chinchilla plus a learning-rate bowl and a schedule term.

L = E + A / N^a + B / D^b + c * (ln peak - ln p*)^2 + m * min_lr_frac
The bowl says loss rises when the peak learning rate is far from a best value p* (in log space).
min_lr_frac = 1 is a constant schedule, 0 decays to zero; decay usually lowers final loss.
8 constants.
"""
import math

import numpy as np
from scipy.optimize import least_squares

CONSTANTS = 8


def cols(rows):
    return np.array([[math.log(r["nonemb_params"]), math.log(r["tokens"]), math.log(r["peak"]),
                      r["min_lr_frac"]] for r in rows])


def loss(p, X):
    E, a0, a, b0, b, c, lp, m = p
    return E + np.exp(a0 - a * X[:, 0]) + np.exp(b0 - b * X[:, 1]) + c * (X[:, 2] - lp) ** 2 + m * X[:, 3]


def fit(rows, y):
    X, y = cols(rows), np.array(y)
    return least_squares(lambda p: loss(p, X) - y, [2.0, 3.0, 0.3, 3.0, 0.3, 0.05, math.log(0.01), 0.05],
                         bounds=([0, -20, 0, -20, 0, 0, -12, -2], [10, 40, 3, 40, 3, 5, 2, 2])).x


def predict(params, row):
    return float(loss(params, cols([row]))[0])
