#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
#   Copyright 2016-2021 Blaise Frederick
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#
"""
Utility functions for testing rapidtide.
"""

import os

import numpy as np


def get_rapidtide_root():
    """
    Returns the path to the base rapidtide directory, terminated with separator.
    Based on function by Yaroslav Halchenko used in Neurosynth Python package.
    """
    thisdir, thisfile = os.path.split(os.path.join(os.path.realpath(__file__)))
    return os.path.join(thisdir, "..") + os.path.sep


def get_scripts_path():
    """
    Returns the path to test datasets, terminated with separator. Test-related
    data are kept in tests folder in "testdata".
    Based on function by Yaroslav Halchenko used in Neurosynth Python package.
    """
    return os.path.realpath(os.path.join(get_rapidtide_root(), "scripts")) + os.path.sep


def get_test_data_path():
    """
    Returns the path to test datasets, terminated with separator. Test-related
    data are kept in tests folder in "testdata".
    Based on function by Yaroslav Halchenko used in Neurosynth Python package.
    """
    return os.path.realpath(os.path.join(get_rapidtide_root(), "tests", "testdata")) + os.path.sep


def get_test_target_path():
    """
    Returns the path to test comparison data, terminated with separator. Test-related
    data are kept in tests folder in "testtargets".
    Based on function by Yaroslav Halchenko used in Neurosynth Python package.
    """
    return (
        os.path.realpath(os.path.join(get_rapidtide_root(), "tests", "testtargets")) + os.path.sep
    )


def get_test_temp_path():
    """
    Returns the path to test temporary directory, terminated with separator.
    Based on function by Yaroslav Halchenko used in Neurosynth Python package.
    """
    return os.path.realpath(os.path.join(get_rapidtide_root(), "tests", "tmp")) + os.path.sep


def get_examples_path():
    """
    Returns the path to examples src directory, where larger test files live, terminated with separator. Test-related
    data are kept in tests folder in "data".
    Based on function by Yaroslav Halchenko used in Neurosynth Python package.
    """
    return (
        os.path.realpath(os.path.join(get_rapidtide_root(), "data", "examples", "src"))
        + os.path.sep
    )


def create_dir(thedir, debug=False):
    # create a directory if it doesn't exist
    try:
        os.makedirs(thedir)
        if debug:
            print(thedir, "created")
    except OSError:
        if debug:
            print(thedir, "exists")
        else:
            pass


def mse(ndarr1, ndarr2):
    """
    Compute mean-squared error.
    """
    return np.mean(np.square(ndarr2 - ndarr1))
