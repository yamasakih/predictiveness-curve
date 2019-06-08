import matplotlib.pyplot as plt
import numpy as np


__all__ = [
    'plot_predictiveness_curve',
]


def _normalize(arr):
    return (arr-arr.min()) / (arr.max()-arr.min())


def plot_predictiveness_curve(risks, labels, points=100, figsize=(5, 10),
                              normalize=False, **kwargs):
    """
    Plot predictiveness curve.

    Parameters
    ----------
    risks : array_like, shape = [n_samples]
        Risks or probabilities for something happens

    labels : array_like, shape = [n_samples]
        Labels of 0 or 1 for sample data. 0 means negative and 1 means
        positive.

    normalize : boolean, default False
        If the risk data is not normalized to the 0-1 range, normalize it.

    points : int, default 100.
        Determine the fineness of the plotted points. The larger the number,
        the finer the detail.

    figsize : tuple, default (5, 10).
        Width, height in inches. If not provided, defaults to = (5, 10).
    """
    risks = np.array(risks)
    labels = np.array(labels)
    points = np.linspace(0, 1, points)

    if normalize:
        risks = _normalize(risks)

    labels = labels[np.argsort(risks)]
    risks = np.sort(risks)
    num_positive = labels.sum()
    risk_percentiles = []
    true_positive_fractions = []
    for p in points:
        risk_percentiles.append((risks <= p).sum() / len(risks))
        true_positive_fractions.append(labels[risks >= p].sum() / num_positive)

    fontsize=14
    plt.figure(figsize=figsize)
    plt.subplot(2, 1, 1)
    plt.plot(risk_percentiles, points)
    plt.ylabel('Risk', fontsize=fontsize)
    plt.subplot(2, 1, 2)
    plt.plot(risk_percentiles, true_positive_fractions)
    plt.xlabel('Risk percentiles', fontsize=fontsize)
    plt.ylabel('TPR', fontsize=fontsize)
