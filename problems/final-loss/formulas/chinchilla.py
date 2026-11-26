"""Chinchilla form (Hoffmann et al. 2022): L = E + A / N^a + B / D^b.

N = non-embedding parameters, D = training tokens. 5 constants. Ignores learning rate and schedule,
which vary a lot in this table, so it sets the bar that a law with optimizer terms has to beat.
"""
import math

import numpy as np
from scipy.optimize import least_squares

CONSTANTS = 5


def loss(p, lnN, lnD):
    E, a0, a, b0, b = p
    return E + np.exp(a0 - a * lnN) + np.exp(b0 - b * lnD)


def cols(rows):
    return (np.array([math.log(r["nonemb_params"]) for r in rows]), np.array([math.log(r["tokens"]) for r in rows]))


def fit(rows, y):
    lnN, lnD = cols(rows)
    y = np.array(y)
    return least_squares(lambda p: loss(p, lnN, lnD) - y, [2.0, 3.0, 0.3, 3.0, 0.3],
                         bounds=([0, -20, 0, -20, 0], [10, 40, 3, 40, 3])).x


def predict(params, row):
    lnN, lnD = cols([row])
    return float(loss(params, lnN, lnD)[0])
