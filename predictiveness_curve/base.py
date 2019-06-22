import matplotlib.pyplot as plt
import numpy as np


__all__ = [
    'calculate_enrichment_factor',
    'convert_label_to_zero_or_one',
    'plot_predictiveness_curve',
]


def _normalize(arr):
    return (arr-arr.min()) / (arr.max()-arr.min())


def _set_axes(ax, lim, fontsize):
    ax.set_xlim(left=lim[0], right=lim[1])
    ax.set_ylim(bottom=lim[0], top=lim[1])
    ax.grid(True)
    axis = ax.xaxis
    axis.label.set_fontsize(fontsize)
    axis = ax.yaxis
    axis.label.set_fontsize(fontsize)


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

    Returns
    -------
    figure : matplotlib.figure.Figure
        A figure instance is returned. You can also assign this figure instant 
        attribute and customize it yourself.
    """
    risks = np.array(risks)
    labels = np.array(labels)
    points = np.linspace(0, 1, points)

    if not np.all(np.unique(labels)==np.unique(classes)):
        raise ValueError('The values of labels and classes do not match')

    default_classes = [0, 1]
    if not np.array_equal(classes, default_classes):
        labels = convert_label_to_zero_or_one(labels, classes)

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
    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(2, 1, 1)
    _set_axes(ax, lim, fontsize)
    ax.plot(np.append(0, risk_percentiles), np.append(0, points), **kwargs)
    ax.yaxis.set_label_text('Risk percentiles')

    ax = fig.add_subplot(2, 1, 2)
    _set_axes(ax, lim, fontsize)
    ax.plot(np.append(0, risk_percentiles),
            np.append(1, true_positive_fractions), **kwargs)
    ax.xaxis.set_label_text('Risk percentiles')
    ax.yaxis.set_label_text('TPR')
    return fig


def calculate_enrichment_factor(scores, labels, classes=[0, 1], threshold=0.01):
    """
    Calculate enrichment factor.

    Parameters
    ----------
    scores : array_like, shape = [n_samples]
        Scores, risks or probabilities for something happens

    labels : array_like, shape = [n_samples]
        Labels for sample data. The argument classes can set negative and
        postive values respectively. In default, 0 means negative and 1 means
        positive.

    classes : array_like, default [0, 1]
        Represents the names of the negative class and the positive class.
        Give in the order of [negative, positive]. In default, 0 means negative
        and 1 means positive.

    threshold : int, float, array_like of int or float, default is 0.01
        If the value of threshold is 1 or more, it means percent, and if it is
        less than 1, it simply assumes that it represents a ratio. In addition,
        it returns one value for int or float, and broadcast for array_like.

    Returns
    -------
    enrichment factors : float or ndarray
        Return enrichment factors. If threshold is int or float, return one
        value. If threshold is array_like, return ndarray. 
    """
    def f(threshold):
        n = int(np.floor(scores.size * threshold))
        return (np.count_nonzero(labels[-n:]) / n) / positive_ratio

    scores = np.array(scores)
    labels = np.array(labels)
    threshold = np.array(threshold)

    if np.any(threshold <= 0) | np.any(threshold > 100):
        raise ValueError('Invalid value for threshold. Threshold should be '
                         'either positive and smaller a int or ints than 100 '
                         'or a float in the (0, 1) range')
    elif threshold.dtype.kind == 'f' and np.any(threshold > 1):
        raise ValueError('Invalid value for threshold. Threshold should be '
                         'either positive and a float or floats in the (0, 1) '
                         'range')
    elif threshold.dtype.kind == 'i':
        threshold = threshold.astype('float32') / 10

    if not np.all(np.unique(labels)==np.unique(classes)):
        raise ValueError('The values of labels and classes do not match')

    default_classes = [0, 1]
    if not np.array_equal(classes, default_classes):
        labels = convert_label_to_zero_or_one(labels, classes)

    labels = labels[np.argsort(scores)]
    scores = np.sort(scores)
    positive_ratio = np.count_nonzero(labels) / scores.size

    _calculate_enrichment_factor = np.frompyfunc(f, 1, 1)
    return _calculate_enrichment_factor(threshold)


def convert_label_to_zero_or_one(labels, classes):
    """
    Convert label data of specified class into 0 or 1 data.

    Parameters
    ----------
    labels : array_like, shape = [n_samples]
        Labels for sample data.

    classes : array_like
        Represents the names of the negative class and the positive class.
        Give in the order of [negative, positive].

    Returns
    -------
    converted label : ndarray, shape = [n_samples]
        Return ndarray which converted labels to 0 and 1.
    """
    return (labels == classes[1]).astype('int16') 
