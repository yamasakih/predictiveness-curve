import matplotlib.pyplot as plt
import numpy as np
import pytest

from predictiveness_curve import plot_predictiveness_curve
from predictiveness_curve.base import _normalize
from predictiveness_curve.base import _set_axes


@pytest.mark.parametrize('arr, expect', [
    (np.array([0, 0, 1]), np.array([0, 0, 1])),
    (np.array([0, 1, 1, 1, 1]), np.array([0, 1, 1, 1, 1])),
    (np.array([2, 3, 3, 3, 3]), np.array([0, 1, 1, 1, 1])),
    (np.array([-2, -1, -2, -2, -1]), np.array([0, 1, 0, 0, 1])),
    (np.array([0, 1, 2]), np.array([0, 0.5, 1])),
    (np.array([0, 1, 2, 3, 4]), np.array([0, 0.25, 0.5, 0.75, 1])),
    (np.array([-2, -1, 0, 1, 2]), np.array([0, 0.25, 0.5, 0.75, 1])),
    (np.array([-2, -1, 0, 1, 8]), np.array([0, 0.1, 0, 0.3, 1])),
])
def test_normalize(arr, expect):
    arr = np.array([0, 0, 1])
    actual = _normalize(arr)
    expect = np.array([0, 0, 1])
    np.testing.assert_array_equal(actual, expect)


@pytest.mark.parametrize('lim, fontsize', [
    ((0, 100), 20),
    ((0, 1), 15),
    ([0, 10], 6),
    (np.array([10, 20]), 8),
])
def test_set_axes(lim, fontsize):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    _set_axes(ax, lim, fontsize)

    actual = ax.get_xlim()
    expect = lim
    assert np.array_equal(actual, expect)

    actual = ax.get_ylim()
    expect = (0.0, 1.0)
    assert np.array_equal(actual, expect)

    actual = ax.xaxis.label.get_fontsize()
    expect = fontsize
    assert actual == expect

    actual = ax.yaxis.label.get_fontsize()
    expect = fontsize
    assert actual == expect


def test_predictiveness_curve():
    scores = np.array([10, 9, 5, 1, 8, 7, 6, 4, 3, 2])
    labels = np.zeros((10, ), dtype='int32')
    labels[:4] = 1

    fig = plot_predictiveness_curve(scores, labels, kind='TPR')
    axes = fig.get_axes()

    actual = len(axes)
    expect = 2
    assert actual == expect

    ax = axes[0]
    actual = ax.get_xlim()
    expect = (-0.03, 1.03)
    assert np.array_equal(actual, expect)

    actual = ax.get_ylim()
    expect = (-0.03, 1.03)
    assert np.array_equal(actual, expect)

    actual = ax.xaxis.label.get_fontsize()
    expect = 14.0
    assert actual == expect

    actual = ax.yaxis.label.get_fontsize()
    expect = 14.0
    assert actual == expect

    actual = ax.xaxis.get_label_text()
    expect = ''
    assert actual == expect

    actual = ax.yaxis.get_label_text()
    expect = 'Risk'
    assert actual == expect

    ax = axes[1]
    actual = ax.get_xlim()
    expect = (-0.03, 1.03)
    assert np.array_equal(actual, expect)

    actual = ax.get_ylim()
    expect = (-0.03, 1.03)
    assert np.array_equal(actual, expect)

    actual = ax.xaxis.label.get_fontsize()
    expect = 14.0
    assert actual == expect

    actual = ax.yaxis.label.get_fontsize()
    expect = 14.0
    assert actual == expect

    actual = ax.xaxis.get_label_text()
    expect = 'Risk percentiles'
    assert actual == expect

    actual = ax.yaxis.get_label_text()
    expect = 'TPR'
    assert actual == expect
    """
    fig = plot_predictiveness_curve(scores, labels, kind='EF')
    axes = fig.get_axes()

    actual = len(axes)
    expect = 2
    assert actual == expect

    ax = axes[0]
    actual = ax.get_xlim()
    expect = (-0.03, 1.03)
    assert np.array_equal(actual, expect)

    actual = ax.get_ylim()
    expect = (-0.03, 1.03)
    assert np.array_equal(actual, expect)

    actual = ax.xaxis.label.get_fontsize()
    expect = 14.0
    assert actual == expect

    actual = ax.yaxis.label.get_fontsize()
    expect = 14.0
    assert actual == expect

    actual = ax.xaxis.get_label_text()
    expect = ''
    assert actual == expect

    actual = ax.yaxis.get_label_text()
    expect = 'Risk'
    assert actual == expect

    ax = axes[1]
    actual = ax.get_xlim()
    expect = (-0.03, 1.03)
    assert np.array_equal(actual, expect)

    actual = ax.get_ylim()
    expect = (-0.03, 1.03)
    assert np.array_equal(actual, expect)

    actual = ax.xaxis.label.get_fontsize()
    expect = 14.0
    assert actual == expect

    actual = ax.yaxis.label.get_fontsize()
    expect = 14.0
    assert actual == expect

    actual = ax.xaxis.get_label_text()
    expect = 'Risk percentiles'
    assert actual == expect

    actual = ax.yaxis.get_label_text()
    expect = 'TPR'
    assert actual == expect
    """


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
