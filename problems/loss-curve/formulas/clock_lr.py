"""Candidate (prereg/loss-curve.md, the next candidate after the kill line): one lr clock plus a current-lr term.

    L(s) = L0 + B * (N / e^13)^-beta + A * tau(s)^-alpha + C * (N / e^13)^gamma * lr(s)
    tau(s) = sum over the steps taken of lr_i^p,   lr_i = peak * lr_mult(i)

tau is the single clock of the kill line (one p for every run). The kill line found that at the same tau a
run whose lr is falling sits 0.07-0.09 nats lower. The last term says why: the loss carries an extra amount
proportional to the lr in use now, which goes away as the lr decays. Its size may change with model size (C,
gamma). Warmup is modeled (lr ramps from peak / warmup to peak), unlike tissue_anneal.
N = non-embedding parameters. Constants: L0, B, beta, A, alpha, C, gamma, p = 8.
p is fitted by trying each value in P_GRID; the other 7 by least squares on the loss.
"""
import functools
import math

import numpy as np
from scipy.optimize import least_squares

CONSTANTS = 8
P_GRID = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


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


def cols(rows, p):
    out = []
    for r in rows:
        key = (r["sched"], r["steps"], r["warmup"], r["decay"], r["min_lr_frac"])
        i = r["cv_step"] - 1
        tau = clock(*key, r["peak"], p)[i]
        lr = r["peak"] * mults(*key)[i]
        out.append([math.log(r["nonemb_params"]) - 13, math.log(tau), lr])
    return np.array(out)


def loss(q, X):
    L0, B, beta, A, alpha, C, gamma = q
    lnN, ln_tau, lr = X.T
    return L0 + B * np.exp(-beta * lnN) + A * np.exp(-alpha * ln_tau) + C * np.exp(gamma * lnN) * lr


Q0 = [3.0, 0.5, 0.3, 1.0, 0.3, 10.0, 0.0]
LO = [0, 0, 0, 0, 0, 0, -3]
HI = [10, 50, 3, 50, 3, 1e4, 3]


def fit(rows, y):
    y = np.array(y)
    best = None
    for p in P_GRID:
        X = cols(rows, p)
        r = least_squares(lambda q: loss(q, X) - y, Q0, bounds=(LO, HI))
        if best is None or r.cost < best[0]:
            best = (r.cost, p, r.x)
    return best[1], best[2]


def predict(params, row):
    p, q = params
    return float(loss(q, cols([row], p))[0])
