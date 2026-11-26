"""The simplest forgetting curve: kept = 1 / (1 + (r * S)^q).

S = sum of the learning rate after the facts stop. Two constants (r, q), no model settings.
This is the bar every richer formula has to beat.
"""
import math

from scipy.optimize import least_squares

CONSTANTS = 2


def kept(p, r):
    return 1.0 / (1.0 + math.exp(min(50.0, math.exp(p[1]) * (p[0] + math.log(sum(r["lr"]))))))


def fit(rows, y):
    return least_squares(lambda p: [kept(p, r) - t for r, t in zip(rows, y)], [0.0, 0.0]).x


def predict(params, row):
    return kept(params, row)
