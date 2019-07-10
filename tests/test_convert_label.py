import numpy as np

from predictiveness_curve import convert_label_to_zero_or_one


def test_zero_and_one_label():
    # labels is numpy.ndarray
    labels = np.array([1, 0, 0])
    classes = [0, 1]
    actual = convert_label_to_zero_or_one(labels, classes)
    expect = np.array([1, 0, 0])
    np.testing.assert_array_equal(actual, expect)

    actual = convert_label_to_zero_or_one(labels, classes)
    expect = np.array([1, 0, 0])
    np.testing.assert_array_equal(actual, expect)

    # labels is list
    labels = [1, 0, 0]
    classes = [0, 1]
    actual = convert_label_to_zero_or_one(labels, classes)
    expect = np.array([1, 0, 0])
    np.testing.assert_array_equal(actual, expect)

    actual = convert_label_to_zero_or_one(labels, classes)
    expect = np.array([1, 0, 0])
    np.testing.assert_array_equal(actual, expect)
