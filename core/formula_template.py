"""Copy this file to start a new formula, then score it:

    cp core/formula_template.py problems/forgetting/formulas/my_idea.py
    python3 core/score.py forgetting problems/forgetting/formulas/my_idea.py

Change fit() and predict(). Keep CONSTANTS equal to the number of values fit() learns.
predict() may read any column of the row that is known before training ends
(settings, not measurements of the run you are predicting); problems/<name>/README.md lists them.
"""
import numpy as np
from scipy.optimize import least_squares

CONSTANTS = 2


def features(row):
    return np.array([1.0, np.log(row["width"])])


def fit(rows, y):
    X = np.array([features(r) for r in rows])
    y = np.array(y)
    res = least_squares(lambda th: X @ th - y, np.zeros(X.shape[1]))
    return res.x


def predict(params, row):
    return float(features(row) @ params)
