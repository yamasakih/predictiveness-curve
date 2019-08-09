from typing import Sequence
import warnings

import numpy as np

from bokeh.plotting import figure, show
from bokeh.layouts import column

__all__ = [
    "calculate_enrichment_factor",
    "convert_label_to_zero_or_one",
    "plot_predictiveness_curve",
]


def _normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())


def _set_axes(ax, lim, fontsize: int):
    ax.set_xlim(left=lim[0], right=lim[1])
    ax.grid(True)
    ax.xaxis.label.set_fontsize(fontsize)
    ax.yaxis.label.set_fontsize(fontsize)


def plot_predictiveness_curve(
        risks,
        labels,
        classes: Sequence = [0, 1],
        normalize: bool = False,
        points: int = 100,
        figsize: Sequence = (400, 340),
        fontsize: int = 12,
        kind: str = "TPR",
        xlabel: str = None,
        top_ylabel: str = None,
        bottom_ylabel: str = None,
        **kwargs,
):
    """
    Plot predictiveness curve. Predictiveness curve is a method to display two
    graphs simultaneously. In both figures, the x-axis is risk percentile, the
    y-axis of one figure is the value of risk, and the y-axis of the other
    figure is true positive fractions. See Am. J. Epidemiol. 2008; 167:362–368
    for details.

    The plot of EF at the threshold value where the product with the sample
    data is less than 1 are not displayed.

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

    points : int, default 100
        Determine the fineness of the plotted points. The larger the number,
        the finer the detail.

    figsize : tuple, default (400, 340)
        Width, height in inches. If not provided, defaults to = (4.5, 10).

    fontsize : int, default 12
        Font size for labels in plots.

    kind : str, default TPR
        * TPR : plot risk percentile vs TPR at bottom.
        * EF  : plot risk percentile vs EF at bottom. The risk percentile of
          the upper plot is also in descending order.

    xlabel : str, default Risk percentiles
        Set the label for the x-axis.

    top_ylabel : str, default Risk
        Set the label for the y-axis in the top plot.

    bottom_ylabel : str, default value of kind.
        Set the label for the y-axis in the bottom plot.

    **kwargs : bokeh.plotting.figure.Figure.line properties, optional
        The argument kwargs is passed to this function.
        See
        https://bokeh.pydata.org/en/latest/docs/reference/plotting.html#bokeh.plotting.figure.Figure.line
        for details.

    Returns
    -------
    figure : matplotlib.figure.Figure
        A figure instance is returned. You can also assign this figure instant
        attribute and customize it yourself.
    """
    risks = np.array(risks)
    labels = np.array(labels)
    thresholds = np.linspace(0, 1, points + 1)[1:]
    points = np.linspace(0, 1, points + 1)

    if not np.all(np.unique(labels) == np.unique(classes)):
        raise ValueError("The values of labels and classes do not match")

    default_classes = [0, 1]  # Sequence
    if not np.array_equal(classes, default_classes):
        labels = convert_label_to_zero_or_one(labels, classes)

    if normalize:
        risks = _normalize(risks)

    if xlabel is None:
        xlabel = "Risk percentiles"
    if top_ylabel is None:
        top_ylabel = "Risk"
    if bottom_ylabel is None:
        bottom_ylabel = kind

    labels = labels[np.argsort(risks)]
    risks = np.sort(risks)
    num_positive: int = labels.sum()

    if kind.upper() == "TPR":

        def f(point):
            count: int = np.count_nonzero(risks <= point)
            return count / len(risks) if count > 0 else 0

        calculate_risk_percentiles = np.frompyfunc(f, 1, 1)
        risk_percentiles = calculate_risk_percentiles(points)
        risk_percentiles = np.append(0, risk_percentiles)
        points = np.append(0, points)

    elif kind.upper() == "EF":

        def f(point):
            count: int = np.count_nonzero(risks >= point)
            return count / len(risks) if count > 0 else 0

        labels = labels[::-1]
        risks = risks[::-1]
        calculate_risk_percentiles = np.frompyfunc(f, 1, 1)
        risk_percentiles = calculate_risk_percentiles(points)
        risk_percentiles = np.append(risk_percentiles, 0)
        points = np.append(points, 1)

    else:
        raise ValueError(f"kind must be either TPR or EF, not {kind}")

    margin: float = 0.03
    lim: Sequence = (0 - margin, 1 + margin)

    fig_top = figure(
        plot_width=figsize[0],
        plot_height=figsize[1],
        x_range=(lim[0], lim[1]),
        y_range=(lim[0], lim[1]),
        y_axis_label=top_ylabel,
    )
    fig_top.yaxis.axis_label_text_font_size = f"{fontsize}pt"
    fig_top.line(risk_percentiles, points, **kwargs)

    fig_bottom = figure(
        plot_width=figsize[0],
        plot_height=figsize[1],
        x_range=fig_top.x_range,
        x_axis_label=xlabel,
        y_axis_label=bottom_ylabel,
    )
    fig_bottom.xaxis.axis_label_text_font_size = f"{fontsize}pt"
    fig_bottom.yaxis.axis_label_text_font_size = f"{fontsize}pt"

    if kind.upper() == "TPR":
        calculate_true_positive_fractions = np.frompyfunc(
            lambda p: np.count_nonzero(labels[risks >= p]) / num_positive, 1,
            1)
        true_positive_fractions = calculate_true_positive_fractions(points)

        fig_bottom.y_range = fig_top.y_range
        fig_bottom.line(risk_percentiles, true_positive_fractions, **kwargs)

    elif kind.upper() == "EF":
        n = np.floor(risks.shape[0] * thresholds).astype("int32")
        if np.any(n == 0):
            warning_message = (
                "The plot of EF at the threshold value where the product with "
                "the sample data is less than 1 is not displayed.")
            warnings.warn(warning_message)
            thresholds = thresholds[n != 0]
        enrichment_factors = calculate_enrichment_factor(risks,
                                                         labels,
                                                         threshold=thresholds)

        fig_bottom.line(thresholds, enrichment_factors, **kwargs)

    return show(column(fig_top, fig_bottom))


def calculate_enrichment_factor(scores, labels, classes=[0, 1],
                                threshold=0.01):
    """
    Calculate enrichment factor. Returns one as the value of enrichment factor
    when the product of sample data and threshold is less than one.

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
        if n == 0:
            return np.nan
        return (np.count_nonzero(labels[-n:]) / n) / positive_ratio

    scores = np.array(scores)
    labels = np.array(labels)
    threshold = np.array(threshold)

    if np.any(threshold <= 0) | np.any(threshold > 100):
        raise ValueError("Invalid value for threshold. Threshold should be "
                         "either positive and smaller a int or ints than 100 "
                         "or a float in the (0, 1) range")
    elif threshold.dtype.kind == "f" and (np.any(threshold <= 0)
                                          or np.any(threshold > 1)):
        raise ValueError("Invalid value for threshold. Threshold should be "
                         "either positive and a float or floats in the (0, 1) "
                         "range")
    elif threshold.dtype.kind == "i":
        threshold = threshold.astype("float32") / 100

    if not np.all(np.unique(labels) == np.unique(classes)):
        raise ValueError("The values of labels and classes do not match")

    default_classes: Sequence = [0, 1]
    if not np.array_equal(classes, default_classes):
        labels = convert_label_to_zero_or_one(labels, classes)

    labels = labels[np.argsort(scores)]
    scores = np.sort(scores)
    positive_ratio = np.count_nonzero(labels) / scores.size

    _calculate_enrichment_factor = np.frompyfunc(f, 1, 1)
    enrichment_factors = _calculate_enrichment_factor(threshold)
    if isinstance(enrichment_factors, float):
        return_float: bool = True
        enrichment_factors = np.array([enrichment_factors], dtype="float32")
    else:
        return_float: bool = False
        enrichment_factors = enrichment_factors.astype("float32")
    if np.any(np.isnan(enrichment_factors)):
        warning_message = (
            "Returns one as the value of enrichment factor because "
            "the product of sample data and threshold is less than one")
        warnings.warn(warning_message)
        enrichment_factors[np.isnan(enrichment_factors)] = 1.0
    if return_float:
        enrichment_factors = enrichment_factors[0]
    return enrichment_factors


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
    if not np.all(np.unique(labels) == np.unique(classes)):
        raise ValueError("The values of labels and classes do not match")
    labels = np.asarray(labels)
    return (labels == classes[1]).astype("int16")
