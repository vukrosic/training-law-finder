"""Published: scaling law with learning-rate annealing (Tissue et al. 2024, arXiv:2408.11029), size form.

    L(s) = L0 + B * (N / e^13)^-beta + A * S1(s)^-alpha - C * (N / e^13)^gamma * S2(s)

S1(s) = sum of the learning rate over the steps taken ("forward area").
S2(s) = sum over i <= s of m_i, with m_i = lambda * m_(i-1) + (lr_(i-1) - lr_i) ("annealing area").
lambda = 0.999, the paper's fixed value (not fitted). As in the paper, warmup is not modeled: warmup steps
count at the peak learning rate. N = non-embedding parameters. The N terms are the paper's extension to
model size (B, beta, gamma), with the base law's L0, A, alpha, C: 7 constants.
Fitted by least squares on the loss (the paper uses a Huber loss on log loss).
The learning rate per step comes from the schedule settings with the testbed's lr_mult() rules.
"""
import functools
import math

import numpy as np
from scipy.optimize import least_squares

CONSTANTS = 7
LAMBDA = 0.999


def lr_mult(step, steps, warmup, sched, decay, min_lr_frac):
    if step < warmup:
        return (step + 1) / warmup
    if sched == "const":
        return 1.0
    if sched == "cos":
        p = (step - warmup) / max(1, steps - warmup)
        return min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p))
    d0 = steps - int(round(decay * steps))
    if step < d0:
        return 1.0
    return 1.0 - (1.0 - min_lr_frac) * (step - d0 + 1) / max(1, steps - d0)


@functools.lru_cache(maxsize=None)
def areas(sched, steps, warmup, decay, min_lr_frac):
    """Cumulative S1 and S2 per steps taken, in units of the peak (index s-1 = after s steps)."""
    lr = np.array([1.0 if s < warmup else lr_mult(s, steps, warmup, sched, decay, min_lr_frac) for s in range(steps)])
    drop = np.concatenate([[0.0], lr[:-1] - lr[1:]])
    m, s2 = 0.0, np.empty(steps)
    for i in range(steps):
        m = LAMBDA * m + drop[i]
        s2[i] = m
    return np.cumsum(lr), np.cumsum(s2)


def cols(rows):
    out = []
    for r in rows:
        s1, s2 = areas(r["sched"], r["steps"], r["warmup"], r["decay"], r["min_lr_frac"])
        i = r["cv_step"] - 1
        out.append([math.log(r["nonemb_params"]) - 13, r["peak"] * s1[i], r["peak"] * s2[i]])
    return np.array(out)


def loss(p, X):
    L0, B, beta, A, alpha, C, gamma = p
    lnN, S1, S2 = X.T
    return L0 + B * np.exp(-beta * lnN) + A * S1 ** -alpha - C * np.exp(gamma * lnN) * S2


P0 = [2.0, 1.0, 0.3, 1.0, 0.5, 1.0, 0.0]
LO = [0, 0, 0, 0, 0, 0, -3]
HI = [10, 50, 3, 50, 3, 1e3, 3]


def fit(rows, y):
    X, y = cols(rows), np.array(y)
    return least_squares(lambda p: loss(p, X) - y, P0, bounds=(LO, HI)).x


def predict(params, row):
    return float(loss(params, cols([row]))[0])
