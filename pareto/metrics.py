import torch

__all__ = ["instance_informedness"]


def nanmean(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


def instance_informedness(y_pred, y_true):
    tp = (y_pred & y_true).sum(axis=1)
    tn = (~y_pred & ~y_true).sum(axis=1)
    cp = y_true.sum(axis=1)
    cn = (~y_true).sum(axis=1)
    return nanmean(tp / cp + tn / cn - 1)
