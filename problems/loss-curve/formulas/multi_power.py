"""Published: multi-power law (Luo et al. 2025, arXiv:2503.12811, ICLR 2025), plus a model-size offset.

    L(t) = L0 + Bn * (N / e^13)^-bn + A * (S1(t) + S_W)^-alpha - LD(t)
    LD(t) = B * sum over k <= t of (lr_(k-1) - lr_k) * G(lr_k^-gamma * S_k(t)),   G(x) = 1 - (C x + 1)^-beta
    S1(t) = sum of the learning rate over the steps after warmup, S_W = peak * warmup / 2,
    S_k(t) = sum of the learning rate from step k to t; steps k count after warmup, lr_0 = peak.

The published law has 7 constants (L0, A, alpha, B, C, beta, gamma) and no model-size input. The score holds
out whole widths, so this entry adds the smallest size term, an offset Bn * N^-bn (N = non-embedding
parameters): 9 constants. That term is ours, not the paper's.
For speed the sum in LD(t) is taken over at most 128 equal blocks of steps: each block adds its summed lr
drop at the lr and S_k(t) of its middle step. The learning rate per step comes from the schedule settings
with the testbed's lr_mult() rules.
"""
import functools
import math

import numpy as np
from scipy.optimize import least_squares

CONSTANTS = 9
BINS = 128


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
def schedule(sched, steps, warmup, decay, min_lr_frac):
    """lr multipliers of the steps after warmup."""
    return np.array([lr_mult(s, steps, warmup, sched, decay, min_lr_frac) for s in range(warmup, steps)])


@functools.lru_cache(maxsize=None)
def reading(sched, steps, warmup, decay, min_lr_frac, step):
    """(S1, drops[BINS], lr[BINS], S_k[BINS]) in units of the peak, after `step` steps taken."""
    lr = schedule(sched, steps, warmup, decay, min_lr_frac)
    t = max(0, step - warmup)
    d, e, sk = np.zeros(BINS), np.ones(BINS), np.zeros(BINS)
    if t == 0:
        return 0.0, d, e, sk
    eta = lr[:t]
    drop = np.concatenate([[1.0], eta[:-1]]) - eta
    tail = np.cumsum(eta[::-1])[::-1]          # S_k(t) for k = 1..t
    edges = np.linspace(0, t, min(BINS, t) + 1).astype(int)
    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1]
        if hi > lo:
            mid = (lo + hi - 1) // 2
            d[b], e[b], sk[b] = drop[lo:hi].sum(), eta[mid], tail[mid]
    return float(eta.sum()), d, e, sk


def cols(rows):
    lnN, S, D, E, K = [], [], [], [], []
    for r in rows:
        s1, d, e, sk = reading(r["sched"], r["steps"], r["warmup"], r["decay"], r["min_lr_frac"], r["cv_step"])
        p = r["peak"]
        lnN.append(math.log(r["nonemb_params"]) - 13)
        S.append(p * (s1 + 0.5 * min(r["warmup"], r["cv_step"])))
        D.append(p * d)
        E.append(np.maximum(p * e, 1e-12))
        K.append(p * sk)
    return np.array(lnN), np.array(S), np.array(D), np.array(E), np.array(K)


def loss(p, X):
    L0, Bn, bn, A, alpha, B, C, beta, gamma = p
    lnN, S, D, E, K = X
    G = 1 - (C * E ** -gamma * K + 1) ** -beta
    return L0 + Bn * np.exp(-bn * lnN) + A * S ** -alpha - B * (D * G).sum(1)


P0 = [2.0, 1.0, 0.3, 1.0, 0.5, 10.0, 1.0, 0.5, 0.5]
LO = [0, 0, 0, 0, 0, 0, 0, 0, 0]
HI = [10, 50, 3, 50, 3, 1e4, 1e4, 5, 3]


def fit(rows, y):
    X, y = cols(rows), np.array(y)
    return least_squares(lambda p: loss(p, X) - y, P0, bounds=(LO, HI)).x


def predict(params, row):
    return float(loss(params, cols([row]))[0])
