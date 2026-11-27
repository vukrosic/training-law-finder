"""Predict the training-set mean for every run. One constant. Any formula must beat this."""
CONSTANTS = 1


def fit(rows, y):
    return sum(y) / len(y)


def predict(params, row):
    return params
