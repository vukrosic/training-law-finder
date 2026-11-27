"""Published baseline: the cumulative weight decay of Bordt et al. (ICML 2025, arXiv:2410.03249).

    kept = product, over the steps after the facts, of (1 - learning rate * weight decay)

This is the share of an update made at the end of the fact window that is still in the weights at the
end of training (their equation 4). It has no fitted constant. The paper uses it as a bound, not as a
prediction: measured forgetting is faster. On this table the bound holds in all 110 runs with weight
decay (kept is below it every time). It says nothing is forgotten in the 265 runs without weight decay,
which keep 18% on average, so as a prediction it scores far worse than the constant in mean.py.
"""
import numpy as np

CONSTANTS = 0


def fit(rows, y):
    return []


def predict(params, row):
    return float(np.prod(1.0 - np.asarray(row["lr"]) * row["wd"]))
