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
import warnings

import numpy as np
from scipy import fftpack

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        import pyfftw
    except ImportError:
        pyfftwpresent = False
    else:
        pyfftwpresent = True

if pyfftwpresent:
    fftpack = pyfftw.interfaces.scipy_fftpack
    pyfftw.interfaces.cache.enable()


def primefacs(thelen: int) -> list:
    """
    Compute the prime factorization of a given integer.

    Parameters
    ----------
    thelen : int
        The positive integer to factorize. Must be greater than 0.

    Returns
    -------
    list
        A list of prime factors of `thelen`, sorted in ascending order.
        Each factor appears as many times as its multiplicity in the
        prime factorization.

    Notes
    -----
    This function implements trial division algorithm to find prime factors.
    The algorithm starts with the smallest prime (2) and continues with
    increasing integers until the square root of the remaining number.
    The final remaining number (if greater than 1) is also a prime factor.

    Examples
    --------
    >>> primefacs(12)
    [2, 2, 3]

    >>> primefacs(17)
    [17]

    >>> primefacs(100)
    [2, 2, 5, 5]
    """
    i = 2
    factors = []
    while i * i <= thelen:
        if thelen % i:
            i += 1
        else:
            factors.append(i)
            thelen //= i
    factors.append(thelen)
    return factors


lencache: dict = {}


def optfftlen(
    thelen: np.uint64, padlen: np.uint64 = 0, _depth: int = 0, debug: bool = False
) -> np.uint64:
    """
    Calculate optimal FFT length for given input length.

    This function currently returns the input length as-is, but is designed
    to be extended for optimal FFT length calculation based on hardware
    constraints or performance considerations.

    Parameters
    ----------
    thelen : int
        The input length for which to calculate optimal FFT length.
        Must be a positive integer.
    padlen : optional, int
        Number of points the data is symmetrically padded with.  Ensure that
        the optimal length accounts for symmetric padding.  Default is 0.
    _depth: optional, int
         How deep we are in the recursion
    debug :optional, bool
        Print out detailed information about how the length is calculated.  Default is False.

    Returns
    -------
    int
        The optimal FFT length. If not using pyfftw, this
        simply returns the input `thelen` value.

    Notes
    -----
    In a more complete implementation, this function would calculate
    the optimal FFT length by finding the smallest number >= thelen
    that has only small prime factors (2, 3, 5, 7) for optimal
    performance on most FFT implementations.

    Examples
    --------
    >>> optfftlen(1024)
    1024
    >>> optfftlen(1000)
    1000
    """
    cachekey = f"{thelen}_{padlen}"
    if debug:
        print(
            f"entering optfftlen with {thelen=}, {padlen=}, {_depth=}, {cachekey=} totallen={thelen + 2 * padlen}"
        )
    if pyfftwpresent:
        try:
            thelen = lencache[cachekey]
            if debug:
                print(f"cache hit for {cachekey} ({thelen})")
        except KeyError:
            if padlen == 0:
                thelen = pyfftw.interfaces.scipy_fft.next_fast_len(thelen)
            else:
                startlen = thelen
                optpadded = pyfftw.interfaces.scipy_fft.next_fast_len(thelen + 2 * padlen)
                if debug:
                    print(f"{optpadded=}")
                if (optpadded - thelen) % 2 == 0:
                    thelen = optpadded
                else:
                    # we get here if the optimal value - the initial value is not divisible by 2
                    # so we need to start one greater than that and and go to the next highest value.
                    if _depth < 500:
                        newpadlen = int((optpadded + 1 - startlen) // 2)
                        thelen = optfftlen(
                            thelen, padlen=newpadlen, _depth=(_depth + 1)
                        )
                    else:
                        thelen = initval

    lencache[cachekey] = thelen
    if debug:
        print(f"optfftlen returning {thelen=}")
    return thelen


def showfftcache() -> None:
    print("FFT length cache entries:")
    for key, value in enumerate(lencache):
        print(f"\t{key}: {value}, {lencache[value]}, {primefacs(lencache[value])}")
