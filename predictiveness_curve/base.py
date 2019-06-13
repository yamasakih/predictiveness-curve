import matplotlib.pyplot as plt
import numpy as np


__all__ = [
    'plot_predictiveness_curve',
]


def _normalize(arr):
    return (arr-arr.min()) / (arr.max()-arr.min())


def plot_predictiveness_curve(risks, labels, classes=[0, 1], normalize=False,
    points=100, figsize=(4.5, 10), fontsize=14, **kwargs):
    """
    Plot predictiveness curve.

    Parameters
    ----------
    risks : array_like, shape = [n_samples]
        Risks or probabilities for something happens

    labels : array_like, shape = [n_samples]
        Labels for sample data. The argument classes can set negative and
        postive values respectively. In default, 0 means negative and 1 means
        positive.

    classes : array_like, default [0, 1]
        Represents the names of the negative class and the positive class.
        Give in the order of [negative, positive]. In default, 0 means negative
        and 1 means positive.

    normalize : boolean, default False
        If the risk data is not normalized to the 0-1 range, normalize it.

    points : int, default 100.
        Determine the fineness of the plotted points. The larger the number,
        the finer the detail.

    figsize : tuple, default (4.5, 10).
        Width, height in inches. If not provided, defaults to = (4.5, 10).

    fontsize : int, default 14.
        Font size for labels in plots.

    **kwargs : matplotlib.pyplot.Line2D properties, optional
        This function internally calls matplotlib.pyplot.plot. The argument
        kwargs is passed to this function.
        See https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot
        for details.
    """
    risks = np.array(risks)
    labels = np.array(labels)
    points = np.linspace(0, 1, points)

    if not np.all(np.unique(labels)==np.unique(classes)):
        raise ValueError('The values of labels and classes do not match.')

    default_classes = [0, 1]
    if not np.array_equal(classes, default_classes):
        labels = (labels == classes[1]).astype('int16')

    if normalize:
        risks = _normalize(risks)

    labels = labels[np.argsort(risks)]
    risks = np.sort(risks)
    num_positive = labels.sum()

    calculate_risk_percentiles = np.frompyfunc(
        lambda p: np.count_nonzero(risks<=p)/len(risks), 1, 1)
    calculate_true_positive_fractions = np.frompyfunc(
        lambda p: np.count_nonzero(labels[risks>=p])/num_positive, 1, 1)

    risk_percentiles = calculate_risk_percentiles(points)
    true_positive_fractions = calculate_true_positive_fractions(points)

    margin = 0.03
    lim = (0 - margin, 1 + margin)
    plt.figure(figsize=figsize)

    plt.subplot(2, 1, 1)
    plt.plot(np.append(0, risk_percentiles),
             np.append(0, points), **kwargs)
    plt.ylabel('Risk', fontsize=fontsize)
    plt.xlim(lim)
    plt.ylim(lim)
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(np.append(0, risk_percentiles),
             np.append(1, true_positive_fractions), **kwargs)
    plt.xlabel('Risk percentiles', fontsize=fontsize)
    plt.ylabel('TPR', fontsize=fontsize)
    plt.xlim(lim)
    plt.ylim(lim)
    plt.grid(True)
