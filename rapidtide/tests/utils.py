"""
Utility functions for testing rapidtide.
"""

import os

import numpy as np


def get_test_data_path():
    """
    Returns the path to test datasets, terminated with separator. Test-related
    data are kept in tests folder in "data".
    Based on function by Yaroslav Halchenko used in Neurosynth Python package.
    """
    thisdir, thisfile = os.path.split(os.path.join(os.path.realpath(__file__)))
    return os.path.join(thisdir, 'data') + os.path.sep


def mse(vec1, vec2):
    """
    Compute mean-squared error.
    """
    return np.mean(np.square(vec2 - vec1))
