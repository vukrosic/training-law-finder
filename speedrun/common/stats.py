"""Uncertainty from a track's fixed evaluation set, and the rule for a new record.

The score of a track is a mean over fixed evaluation items (rollouts, games, sequences). The uncertainty comes only
from which items are in that set, never from repeated training runs:
- se_of_mean: resample the items with replacement 1,000 times (numpy generator seeded with 0) and take the
  standard deviation of the resampled means.
- decimals: the place of the first significant digit of that standard error (0.04 -> 2, 0.4 -> 1, 4 -> 0).
  A track stores the decimals of its baseline in track.json, so all its records use the same precision.
- record_test: a new record must beat the current record by more than 2 standard errors of the paired
  difference: per item, new minus current, on the same items, bootstrapped the same way.
"""
import math

import numpy as np

N_BOOT, SEED = 1000, 0


def se_of_mean(values):
    v = np.asarray(values, dtype=np.float64)
    idx = np.random.default_rng(SEED).integers(0, len(v), (N_BOOT, len(v)))
    return float(v[idx].mean(1).std(ddof=1))


def decimals(se):
    """Place of the first significant digit of se: 0.04 -> 2, 0.4 -> 1, 4 -> 0."""
    return max(0, -math.floor(math.log10(se))) if se > 0 else 0


def fmt(mean, se, d):
    return f"{mean:.{d}f} ± {se:.{d}f}"


def record_test(new, current, lower_is_better):
    """Per-item values of a new and the current record on the same items, in the same order.
    Returns (gain, needs, is_record): gain = mean improvement, needs = 2 bootstrap standard errors of it."""
    new, current = np.asarray(new, dtype=np.float64), np.asarray(current, dtype=np.float64)
    d = current - new if lower_is_better else new - current
    gain, needs = float(d.mean()), 2 * se_of_mean(d)
    return gain, needs, gain > needs
