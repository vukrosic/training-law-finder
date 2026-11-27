"""Chinchilla plus a learning-rate bowl whose center and steepness move with model size.

L = E + A / N^a + B / D^b + c * (N / N0)^g * (ln peak - ln p* - k * ln(width / e^5))^2 + m * min_lr_frac

The best peak learning rate falls as the model gets wider (k < 0, about width^-0.67), and the bowl gets
steeper as the model grows (g > 0): big models pay more for a wrong learning rate. Both match what
Wortsman et al. 2023 report for small-scale proxies, and muP-style lr-versus-width laws.
N = non-embedding parameters, D = training tokens, N0 = e^13 (a scale constant, not fitted).
10 constants.
"""
import math

import numpy as np
from scipy.optimize import least_squares

CONSTANTS = 10


def cols(rows):
    return np.array([[math.log(r["nonemb_params"]), math.log(r["tokens"]), math.log(r["peak"]),
                      r["min_lr_frac"], math.log(r["width"])] for r in rows])


def loss(p, X):
    E, a0, a, b0, b, c, lp, m, k, g = p
    bowl = c * np.exp(g * (X[:, 0] - 13)) * (X[:, 2] - lp - k * (X[:, 4] - 5)) ** 2
    return E + np.exp(a0 - a * X[:, 0]) + np.exp(b0 - b * X[:, 1]) + bowl + m * X[:, 3]


def fit(rows, y):
    X, y = cols(rows), np.array(y)
    return least_squares(lambda p: loss(p, X) - y, [2.0, 3.0, 0.3, 3.0, 0.3, 0.05, math.log(0.01), 0.05, 0.0, 0.0],
                         bounds=([0, -20, 0, -20, 0, 0, -12, -2, -3, -1], [10, 40, 3, 40, 3, 5, 2, 2, 3, 1])).x


def predict(params, row):
    return float(loss(params, cols([row]))[0])
