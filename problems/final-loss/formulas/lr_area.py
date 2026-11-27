"""Chinchilla where the learning rate enters through the schedule's mean learning rate.

    lam = peak * mean_lr_share            (mean_lr_share: average of lr(step) / peak over the run)
    L = E + A / N^(a + a_D * (ln D - 14)) + B / (D * (lam / e^-5)^h)^b
          + c * (N / e^13)^g * (ln lam - ln p* - k * (ln width - 5))^2 + m * min_lr_frac

Three ideas:
1. The bowl is in the mean learning rate, not the peak: a cosine run (mean share about 0.55) wants a
   higher peak than a WSD run (about 0.9), which wants a higher peak than a constant run. Fitting a free
   exponent on mean_lr_share gives 1.02, so the share enters with power 1 and costs no constant.
2. The data term counts tokens times a power of the learning rate (Tissue et al. 2024 use the lr area
   as the clock): runs at a higher mean learning rate move further per token.
3. The model-size exponent depends on tokens (a_D): in these short runs a bigger model helps less when
   it sees few tokens, as in the Kaplan et al. 2020 joint law.
The bowl center falls with width (k) and the bowl gets steeper with size (g), as in width_bowl.py.
N = non-embedding parameters, D = training tokens. 12 constants.

mean_lr_share is computed from the schedule settings with the trainer's own lr_at() rules
(families/llm/testbed-facts-v1/env/train.py). cosrw is treated as cos and crw as const, because their
restart settings (rwp, rww) are not in the data.
"""
import functools
import math

import numpy as np
from scipy.optimize import least_squares

CONSTANTS = 12


@functools.lru_cache(maxsize=None)
def mean_lr_share(sched, steps, warmup, decay, min_lr_frac):
    s = np.arange(steps)
    lr = np.ones(steps)
    if sched == "wsd":
        d0 = steps - int(round(decay * steps))
        m = s >= d0
        lr[m] = 1 - (1 - min_lr_frac) * (s[m] - d0 + 1) / (steps - d0)
    elif sched in ("cos", "cosrw"):
        p = (s - warmup) / max(1, steps - warmup)
        lr = min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + np.cos(np.pi * p))
    w = s < warmup
    lr[w] = (s[w] + 1) / warmup
    return float(lr.mean())


def cols(rows):
    return np.array([[math.log(r["nonemb_params"]), math.log(r["tokens"]),
                      math.log(r["peak"] * mean_lr_share(r["sched"], r["steps"], r["warmup"], r["decay"], r["min_lr_frac"])),
                      r["min_lr_frac"], math.log(r["width"])] for r in rows])


def loss(p, X):
    E, a0, a, b0, b, c, lp, m, k, g, h, aD = p
    lnN, lnD, lnlam, mlf, lnw = X.T
    data = np.exp(b0 - b * (lnD + h * (lnlam + 5)))
    size = np.exp(a0 - (a + aD * (lnD - 14)) * lnN)
    bowl = c * np.exp(g * (lnN - 13)) * (lnlam - lp - k * (lnw - 5)) ** 2
    return E + size + data + bowl + m * mlf


P0 = [2.0, 3.0, 0.3, 3.0, 0.3, 0.05, math.log(0.01), 0.05, 0.0, 0.0, 0.0, 0.0]
LO = [0, -20, 0, -20, 0, 0, -12, -2, -3, -1, -3, -1]
HI = [10, 40, 3, 40, 3, 5, 2, 2, 3, 1, 3, 1]


def fit(rows, y):
    X, y = cols(rows), np.array(y)
    return least_squares(lambda p: loss(p, X) - y, P0, bounds=(LO, HI)).x


def predict(params, row):
    return float(loss(params, cols([row]))[0])
