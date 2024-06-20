import numpy as np
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    zero_one_loss,
)

__all__ = [
    "instance_informedness_np",
    "a_mean",
    "f1_measure",
    "precision",
    "recall",
    "subset_01_loss",
    "hamming",
]


def instance_informedness_np(y_true, y_pred, **kwargs):
    tp = (y_pred & y_true).sum(axis=1)
    tn = (~y_pred & ~y_true).sum(axis=1)
    cp = y_true.sum(axis=1)
    cn = (~y_true).sum(axis=1)
    return np.nanmean(tp / cp + tn / cn - 1)


def a_mean(y_true, y_pred, **kwargs):
    return (instance_informedness_np(y_true=y_true, y_pred=y_pred) + 1.0) / 2.0


def f1_measure(y_true, y_pred, **kwargs):
    return f1_score(y_true, y_pred, average="samples")


def precision(y_true, y_pred, **kwargs):
    return precision_score(y_true, y_pred, average="samples")


def recall(y_true, y_pred, **kwargs):
    return recall_score(y_true, y_pred, average="samples")


def subset_01_loss(y_true, y_pred, **kwargs):
    return zero_one_loss(y_true, y_pred)


def hamming(y_true, y_pred, **kwargs):
    return hamming_loss(y_true, y_pred)
