import matplotlib.pyplot as plt
import numpy as np


__all__ = [
    'plot_predictiveness_curve',
]


def _normalize(arr):
    return (arr-arr.min()) / (arr.max()-arr.min())


def plot_predictiveness_curve(risks, normalize=False):
    """
    Plot predictiveness curve.

    Parameters
    ----------
    risks : array_like, shape = [n_samples]
        Risks or probabilities for something happens

    normalize : boolean, default False
        If the risk data is not normalized to the 0-1 range, normalize it.
    """
    if normalize:
        risks = _normalize(risks)
