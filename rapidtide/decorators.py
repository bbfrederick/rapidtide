#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2016-2025 Blaise Frederick
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
from typing import Callable

# ----------------------------------------- Conditional imports ---------------------------------------
donotbeaggressive = True

try:
    from numba import jit
except ImportError:
    donotusenumba = True
else:
    donotusenumba = False


def getdecoratorvars():
    return donotusenumba, donotbeaggressive


def conditionaljit() -> Callable:
    """
    Wrap functions in jit if numba is enabled.

    This function creates a decorator that conditionally applies Numba's jit
    decorator to functions. If the `donotusenumba` flag is True, the original
    function is returned unchanged. Otherwise, the function is compiled with
    `jit(nopython=True)` for optimal performance.

    Returns
    -------
    Callable
        A decorator function that can be applied to other functions.

    Notes
    -----
    This decorator provides a convenient way to conditionally enable Numba
    compilation based on a global flag. It's useful for debugging and
    development where you want to disable JIT compilation temporarily.

    Examples
    --------
    >>> @conditionaljit()
    ... def my_function(x):
    ...     return x * 2
    ...
    >>> result = my_function(5)
    >>> print(result)
    10
    """

    def resdec(f):
        if donotusenumba:
            return f
        return jit(f, nopython=True)

    return resdec


def conditionaljit2() -> Callable:
    """Return a decorator that conditionally applies numba JIT compilation (conservative mode).

    Returns
    -------
    decorator
        A decorator that applies numba JIT compilation with nopython=True if numba
        is enabled and aggressive optimization is allowed, otherwise returns the function unchanged.
        This is more conservative than conditionaljit() as it also checks the donotbeaggressive flag.
    """

    def resdec(f):
        if donotusenumba or donotbeaggressive:
            return f
        return jit(f, nopython=True)

    return resdec
