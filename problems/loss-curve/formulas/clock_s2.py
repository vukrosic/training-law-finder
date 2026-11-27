"""Candidate: one lr clock plus Tissue et al.'s annealing term (arXiv:2408.11029), with warmup modeled.

    L(s) = L0 + B * (N / e^13)^-beta + A * tau(s)^-alpha - C * (N / e^13)^gamma * S2(s)
    tau(s) = sum over the steps taken of lr_i^p,   lr_i = peak * lr_mult(i)
    S2(s)  = sum over i <= s of m_i,   m_i = lambda * m_(i-1) + max(0, lr_(i-1) - lr_i)

Same clock as clock_lr; the second term is tissue_anneal's annealing area instead of the current lr. Two
changes from tissue_anneal: the clock is tau (lr^p) instead of the lr area, and warmup steps count at their
real lr. Warmup raises the lr, so max(0, .) keeps it out of S2.
lambda is fitted from LAMBDA_GRID (tissue_anneal fixes 0.999, set for runs of thousands of steps; these runs
have 128-1,024 steps). N = non-embedding parameters.
Constants: L0, B, beta, A, alpha, C, gamma, p, lambda = 9. p and lambda by grid, the rest by least squares.
"""
import functools
import math

import numpy as np
from scipy.optimize import least_squares

CONSTANTS = 9
P_GRID = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
LAMBDA_GRID = (0.9, 0.99, 0.999)


def lr_mult(step, steps, warmup, sched, decay, min_lr_frac):
    """testbed/train.py lr_mult with warmup given in steps. step is 0-based."""
    if step < warmup:
        return (step + 1) / warmup
    if sched == "const":
        return 1.0
    if sched == "cos":
        q = (step - warmup) / max(1, steps - warmup)
        return min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + math.cos(math.pi * q))
    d0 = steps - int(round(decay * steps))
    if step < d0:
        return 1.0
    return 1.0 - (1.0 - min_lr_frac) * (step - d0 + 1) / max(1, steps - d0)


@functools.lru_cache(maxsize=None)
def mults(sched, steps, warmup, decay, min_lr_frac):
    return np.array([lr_mult(s, steps, warmup, sched, decay, min_lr_frac) for s in range(steps)])


@functools.lru_cache(maxsize=None)
def clock(sched, steps, warmup, decay, min_lr_frac, peak, p):
    """tau after each number of steps taken (index s-1 = after s steps)."""
    return np.cumsum((peak * mults(sched, steps, warmup, decay, min_lr_frac)) ** p)


@functools.lru_cache(maxsize=None)
def anneal(sched, steps, warmup, decay, min_lr_frac, peak, lam):
    """S2 after each number of steps taken, in lr units."""
    lr = peak * mults(sched, steps, warmup, decay, min_lr_frac)
    drop = np.maximum(0.0, np.concatenate([[0.0], lr[:-1] - lr[1:]]))
    m, s2 = 0.0, np.empty(steps)
    for i in range(steps):
        m = lam * m + drop[i]
        s2[i] = m
    return np.cumsum(s2)


def cols(rows, p, lam):
    out = []
    for r in rows:
        key = (r["sched"], r["steps"], r["warmup"], r["decay"], r["min_lr_frac"], r["peak"])
        i = r["cv_step"] - 1
        out.append([math.log(r["nonemb_params"]) - 13, math.log(clock(*key, p)[i]), anneal(*key, lam)[i]])
    return np.array(out)


def loss(q, X):
    L0, B, beta, A, alpha, C, gamma = q
    lnN, ln_tau, S2 = X.T
    return L0 + B * np.exp(-beta * lnN) + A * np.exp(-alpha * ln_tau) - C * np.exp(gamma * lnN) * S2


Q0 = [3.0, 0.5, 0.3, 1.0, 0.3, 1.0, 0.0]
LO = [0, 0, 0, 0, 0, 0, -3]
HI = [10, 50, 3, 50, 3, 1e4, 3]


def fit(rows, y):
    y = np.array(y)
    best = None
    for p in P_GRID:
        for lam in LAMBDA_GRID:
            X = cols(rows, p, lam)
            r = least_squares(lambda q: loss(q, X) - y, Q0, bounds=(LO, HI))
            if best is None or r.cost < best[0]:
                best = (r.cost, p, lam, r.x)
    return best[1:]


def predict(params, row):
    p, lam, q = params
    return float(loss(q, cols([row], p, lam))[0])
