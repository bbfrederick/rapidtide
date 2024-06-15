#!/usr/bin/env python

import numpy as np
from sklearn.linear_model import LinearRegression

from rapidtide.fit import mlregress_alt


def test_mlregress_alt():
    # Test with a simple dataset
    X = np.array([[1, 2], [3, 4]])
    y = np.array([5, 6])
    intercept = True
    expected_coeffs = np.array([[0, 1], [2, 3]])
    expected_intercept = 4
    expected_R = 0.707

    coeffs, R = mlregress_alt(X, y, intercept)

    print(coeffs, expected_coeffs)
    assert np.allclose(coeffs, expected_coeffs)
    assert np.isclose(R, expected_R)

def test_mlregress_alt_intercept():
    # Test with a simple dataset and no intercept
    X = np.array([[1, 2], [3, 4]])
    y = np.array([5, 6])
    intercept = False
    expected_coeffs = np.array([[1, 2], [3, 4]])
    expected_intercept = None
    expected_R = 0.707

    coeffs, R = mlregress_alt(X, y, intercept)

    assert np.allclose(coeffs, expected_coeffs)
    assert np.isclose(R, expected_R)

def test_mlregress_alt_no_intercept():
    # Test with a simple dataset and no intercept
    X = np.array([[1, 2], [3, 4]])
    y = np.array([5, 6])
    intercept = None
    expected_coeffs = np.array([[1, 2], [3, 4]])
    expected_intercept = None
    expected_R = 0.707

    coeffs, R = mlregress_alt(X, y, intercept)

    assert np.allclose(coeffs, expected_coeffs)
    assert np.isclose(R, expected_R)

test_mlregress_alt()
test_mlregress_alt_intercept()
test_mlregress_alt_nointercept()

