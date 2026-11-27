"""The learning-rate clock: kept = 1 / (1 + exp(q * (c + ln S))).

S = sum of the learning rate from the end of the fact window to this reading (S_now). Two constants
(c, ln q), no model settings. Same curve as problems/forgetting/formulas/lr_clock.py, read at every point
of the curve instead of only the end.
"""
import math

from scipy.optimize import least_squares

CONSTANTS = 2


def kept(p, r):
    return 1.0 / (1.0 + math.exp(min(50.0, math.exp(p[1]) * (p[0] + math.log(r["S_now"])))))


def fit(rows, y):
    return least_squares(lambda p: [kept(p, r) - t for r, t in zip(rows, y)], [0.0, 0.0]).x


def predict(params, row):
    return kept(params, row)
