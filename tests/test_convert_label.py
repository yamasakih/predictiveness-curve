import numpy as np
import pandas as pd
import pytest

from predictiveness_curve import convert_label_to_zero_or_one


@pytest.mark.parametrize('labels', [[1, 0, 0], (1, 0, 0),
                                    np.array([1, 0, 0]),
                                    pd.Series([1, 0, 0])])
def test_zero_and_one_label(labels):
    classes = [0, 1]
    expect = np.array([1, 0, 0])
    actual = convert_label_to_zero_or_one(labels, classes)
    np.testing.assert_array_equal(actual, expect)


@pytest.mark.parametrize('labels', [['a', 'b', 'b'], ('a', 'b', 'b'),
                                    np.array(['a', 'b', 'b']),
                                    pd.Series(['a', 'b', 'b'])])
def test_a_anb_b_labels(labels):
    classes = ['a', 'b']
    actual = convert_label_to_zero_or_one(labels, classes)
    expect = np.array([0, 1, 1])
    np.testing.assert_array_equal(actual, expect)


@pytest.mark.parametrize('labels', [['a', 'b', 'b'], ('a', 'b', 'b'),
                                    np.array(['a', 'b', 'b']),
                                    pd.Series(['a', 'b', 'b'])])
def test_b_anb_a_labels(labels):
    classes = ['b', 'a']
    actual = convert_label_to_zero_or_one(labels, classes)
    expect = np.array([1, 0, 0])
    np.testing.assert_array_equal(actual, expect)


@pytest.mark.parametrize(
    'classes', [[3, 5],
                (3, 5), np.array([3, 5]),
                pd.Series([3, 5])])
def test_classes_type(classes):
    labels = np.array([5, 5, 3, 3, 5])
    actual = convert_label_to_zero_or_one(labels, classes)
    expect = np.array([1, 1, 0, 0, 1])
    np.testing.assert_array_equal(actual, expect)
