import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from predictiveness_curve import calculate_enrichment_factor


def test_calculate_enrichiment_factor():
    labels = np.zeros((10000, ), dtype='int32')
    labels[:200] = 1
    scores = np.zeros((10000, ), dtype='float32')
    scores[:30] = 100
    scores[-470:] = 100
    actual = calculate_enrichment_factor(scores, labels, threshold=0.05)
    expect = 3.0
    assert actual == expect

    labels = np.zeros((10000, ), dtype='int32')
    labels[:200] = 1
    scores = np.zeros((10000, ), dtype='float32')
    scores[:30] = 100
    scores[-470:] = 1000
    actual = calculate_enrichment_factor(scores, labels, threshold=0.05)
    expect = 3.0
    assert actual == expect

    labels = np.zeros((10, ), dtype='int32')
    labels[:4] = 1
    scores = np.array([10, 9, 5, 1, 8, 7, 6, 4, 3, 2])
    actual = calculate_enrichment_factor(scores, labels, threshold=0.2)
    expect = 2.5
    assert actual == expect

    labels = np.zeros((10, ), dtype='int32')
    labels[:4] = 1
    scores = np.array([9, 8, 5, 4, 10, 7, 6, 3, 2, 1])
    actual = calculate_enrichment_factor(scores, labels, threshold=0.2)
    expect = 1.25
    assert actual == expect


@pytest.mark.parametrize('scores, labels', [
    ([10, 9, 5, 1, 8, 7, 6, 4, 3, 2], [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
    ((10, 9, 5, 1, 8, 7, 6, 4, 3, 2), (1, 1, 1, 1, 0, 0, 0, 0, 0, 0)),
    (pd.Series([10, 9, 5, 1, 8, 7, 6, 4, 3, 2
                ]), pd.Series([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])),
    (np.array([10, 9, 5, 1, 8, 7, 6, 4, 3, 2
               ]), np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])),
    (np.array([10, 9, 5, 1, 8, 7, 6, 4, 3, 2
               ]), pd.Series([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])),
    ([10, 9, 5, 1, 8, 7, 6, 4, 3, 2],
        np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])),
])
def test_input_type(scores, labels):
    actual = calculate_enrichment_factor(scores, labels, threshold=0.2)
    expect = 2.5
    assert actual == expect


def test_calculate_enrichiment_factor_for_probabilities_():
    X, y = load_breast_cancer(return_X_y=True)
    training_X, test_X, training_y, test_y = train_test_split(X,
                                                              y,
                                                              test_size=0.5,
                                                              random_state=42)
    clsf = RandomForestClassifier(n_estimators=100, random_state=42)
    clsf.fit(training_X, training_y)
    probabilities = clsf.predict_proba(test_X)[:, 1]
    actual = calculate_enrichment_factor(probabilities, test_y)
    expect = 1.5240
    assert pytest.approx(actual, 0.0001) == expect

    actual = calculate_enrichment_factor(probabilities,
                                         test_y,
                                         threshold=[0.01, 0.05, 0.6, 0.8, 1])
    expect = np.array([1.5240, 1.5240, 1.5151, 1.25, 1.0])
    np.testing.assert_almost_equal(actual, expect, decimal=4)


@pytest.mark.parametrize('threshold', [
    (150),
    (120),
    (101),
    (0),
    (-10),
    (-50),
    ([150, 120]),
    ([150, 80, 60, 20]),
    ([120, 50, 0, -100, 250]),
])
def test_int_threshold_error(threshold):
    scores = np.array([0, 0, 0, 0, 0])
    labels = np.array([0, 0, 1, 1, 1])
    classes = np.array([0, 1])
    with pytest.raises(ValueError) as exc_info:
        calculate_enrichment_factor(scores, labels, classes,
                                    threshold=threshold)
        expect = ('Invalid value for threshold. Threshold should be '
                  'either positive and smaller a int or ints than 100 '
                  'or a float in the (0, 1) range')
        assert exc_info.values.args[0] == expect


@pytest.mark.parametrize('threshold', [
    (150.0),
    (1.5),
    (1.01),
    (-0.01),
    (-0.5),
    (-10.0),
    ([80, 0.5]),
    ([0.0, 0.1, 0.2, 0.3, 0.4])
])
def test_float_threshold_error(threshold):
    scores = np.array([0, 0, 0, 0, 0])
    labels = np.array([0, 0, 1, 1, 1])
    classes = np.array([0, 1])
    with pytest.raises(ValueError) as exc_info:
        calculate_enrichment_factor(scores, labels, classes,
                                    threshold=threshold)
        expect = ('Invalid value for threshold. Threshold should be '
                  'either positive and smaller a int or ints than 100 '
                  'or a float in the (0, 1) range')
        assert exc_info.values.args[0] == expect


def test_int_threshold():
    scores = np.array([10, 9, 5, 1, 8, 7, 6, 4, 3, 2])
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    classes = np.array([0, 1])
    threshold = 20
    actual = calculate_enrichment_factor(scores, labels, classes,
                                         threshold=threshold)
    expect = 2.5
    assert actual == expect

    threshold = [20, 50]
    actual = calculate_enrichment_factor(scores, labels, classes,
                                         threshold=threshold)
    expect = np.array([2.5, 1.0])
    np.testing.assert_almost_equal(actual, expect, decimal=1)


def test_labels_and_classes_unmatch():
    scores = np.array([0, 0, 0, 0, 0])
    labels = np.array([0, 1, 1, 1, 2])
    classes = np.array([0, 1])
    with pytest.raises(ValueError,
                       match='The values of labels and classes do not match'):
        calculate_enrichment_factor(scores, labels, classes)


def test_warning():
    scores = np.array([10, 9, 5, 1, 8, 7, 6, 4, 3, 2])
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])

    with pytest.warns(UserWarning) as exc_info:
        actual = calculate_enrichment_factor(scores, labels, threshold=0.01)
        expect = 1
        assert actual == expect

        actual = exc_info[0].message.args[0]
        expect = ('Returns one as the value of enrichment factor because the '
                  'product of sample data and threshold is less than one')
        assert actual == expect


def test_converting_label():
    scores = np.array([10, 9, 5, 1, 8, 7, 6, 4, 3, 2])
    labels = np.array(['c', 'c', 'c', 'c', 'a', 'a', 'a', 'a', 'a', 'a'])
    classes = ['a', 'c']
    threshold = 20
    actual = calculate_enrichment_factor(scores, labels, classes,
                                         threshold=threshold)
    expect = 2.5
    assert actual == expect

    threshold = [20, 50]
    actual = calculate_enrichment_factor(scores, labels, classes,
                                         threshold=threshold)
    expect = np.array([2.5, 1.0])
    np.testing.assert_almost_equal(actual, expect, decimal=1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
