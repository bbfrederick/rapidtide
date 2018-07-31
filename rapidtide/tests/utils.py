"""
Utility functions for testing rapidtide.
"""

from os.path import abspath, dirname, join, sep

import numpy as np


def get_test_data_path():
    """
    Returns the path to test datasets, terminated with separator. Test-related
    data are kept in tests folder in "data".
    Based on function by Yaroslav Halchenko used in Neurosynth Python package.
    """
    return abspath(join(dirname(__file__), 'data') + sep)


def mse(vec1, vec2):
    """
    Compute mean-squared error.
    """
    return np.mean(np.square(vec2 - vec1))
