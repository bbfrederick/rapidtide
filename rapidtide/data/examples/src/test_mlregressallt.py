#!/usr/bin/env python

import numpy as np
from sklearn.linear_model import LinearRegression

from rapidtide.fit import mlregress, olsregress


def test_olsregress():
    # Test with a simple dataset
    X = np.array([[1, 2], [3, 4]])
    y = np.array([5, 6])
    intercept = True
    expected_coeffs = np.array([[0, 1], [2, 3]])
    expected_intercept = 4
    expected_R = 0.707

    mlcoeffs, mlR = mlregress(X, y, intercept=intercept)
    print(f"{mlcoeffs=}, {mlR=}")

    olscoeffs, olsR = olsregress(X, y, intercept=intercept)
    print(f"{olscoeffs=}, {olsR=}")

    print(mlcoeffs, expected_coeffs)
    print(olscoeffs, expected_coeffs)
    assert np.allclose(mlcoeffs, expected_coeffs)
    assert np.isclose(mlR, expected_R)
    assert np.allclose(olscoeffs, expected_coeffs)
    assert np.isclose(olsR, expected_R)

def test_olsregress_intercept():
    # Test with a simple dataset and no intercept
    X = np.array([[1, 2], [3, 4]])
    y = np.array([5, 6])
    intercept = False
    expected_coeffs = np.array([[1, 2], [3, 4]])
    expected_intercept = None
    expected_R = 0.707

    coeffs, R = mlregress(X, y, intercept)

    assert np.allclose(coeffs, expected_coeffs)
    assert np.isclose(R, expected_R)

def test_olsregress_nointercept():
    # Test with a simple dataset and no intercept
    X = np.array([[1, 2], [3, 4]])
    y = np.array([5, 6])
    intercept = False
    expected_coeffs = np.array([[1, 2], [3, 4]])
    expected_intercept = False
    expected_R = 0.707

    mlcoeffs, mlR = mlregress(X, y, intercept=intercept)
    print(f"{mlcoeffs=}, {mlR=}")

    olscoeffs, olsR = olsregress(X, y, intercept=intercept)
    print(f"{olscoeffs=}, {olsR=}")

    print(mlcoeffs, expected_coeffs)
    print(olscoeffs, expected_coeffs)
    assert np.allclose(mlcoeffs, expected_coeffs)
    assert np.isclose(mlR, expected_R)
    assert np.allclose(olscoeffs, expected_coeffs)
    assert np.isclose(olsR, expected_R)

test_olsregress_nointercept()
test_olsregress_intercept()
test_olsregress()

