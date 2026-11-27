"""Predict the training-set mean for every run. One constant."""
CONSTANTS = 1


def fit(rows, y):
    return sum(y) / len(y)


def predict(params, row):
    return params
