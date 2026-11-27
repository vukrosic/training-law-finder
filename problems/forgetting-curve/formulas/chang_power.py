"""Published baseline: forgetting as a power law in steps since the facts were last seen
(Chang et al. 2024, "How Do Large Language Models Acquire Factual Knowledge During Pretraining?", arXiv:2406.11813).

    kept = (1 + t / tau) ^ (-alpha)

t = steps since the fact window ended (t_now). Two constants (ln tau, alpha), no model settings and no
learning rate: the paper measures retention against steps of training, so this is its curve.
"""
import math

from scipy.optimize import least_squares

CONSTANTS = 2


def kept(p, r):
    return (1.0 + r["t_now"] / math.exp(p[0])) ** (-p[1])


def fit(rows, y):
    return least_squares(lambda p: [kept(p, r) - t for r, t in zip(rows, y)], [5.0, 0.5],
                         bounds=([-5, 0], [15, 10])).x


def predict(params, row):
    return kept(params, row)
