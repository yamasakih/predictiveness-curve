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
