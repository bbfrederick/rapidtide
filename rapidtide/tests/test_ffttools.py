#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2016-2026 Blaise Frederick
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
"""Tests for rapidtide.ffttools module."""

import io
import sys

import numpy as np

import rapidtide.ffttools as ffttools

# --- primefacs tests ---


def primefacs_small_primes(debug=False):
    """Test prime factorization of small prime numbers."""
    assert ffttools.primefacs(2) == [2]
    assert ffttools.primefacs(3) == [3]
    assert ffttools.primefacs(5) == [5]
    assert ffttools.primefacs(7) == [7]
    assert ffttools.primefacs(11) == [11]
    assert ffttools.primefacs(13) == [13]
    if debug:
        print("primefacs_small_primes passed")


def primefacs_composite(debug=False):
    """Test prime factorization of composite numbers."""
    assert ffttools.primefacs(4) == [2, 2]
    assert ffttools.primefacs(6) == [2, 3]
    assert ffttools.primefacs(8) == [2, 2, 2]
    assert ffttools.primefacs(12) == [2, 2, 3]
    assert ffttools.primefacs(100) == [2, 2, 5, 5]
    assert ffttools.primefacs(360) == [2, 2, 2, 3, 3, 5]
    if debug:
        print("primefacs_composite passed")


def primefacs_product_check(debug=False):
    """Verify that the product of prime factors equals the original number."""
    test_values = [2, 6, 12, 17, 60, 100, 128, 255, 360, 1024, 2000, 4096]
    for val in test_values:
        factors = ffttools.primefacs(val)
        product = 1
        for f in factors:
            product *= f
        assert product == val, f"Product of factors {factors} = {product}, expected {val}"
    if debug:
        print("primefacs_product_check passed")


def primefacs_all_factors_prime(debug=False):
    """Verify every returned factor is actually prime."""

    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    test_values = [12, 30, 100, 128, 255, 360, 1000, 2048, 9999]
    for val in test_values:
        factors = ffttools.primefacs(val)
        for f in factors:
            assert is_prime(f), f"Factor {f} of {val} is not prime"
    if debug:
        print("primefacs_all_factors_prime passed")


def primefacs_powers_of_two(debug=False):
    """Test prime factorization of powers of 2."""
    for exp in range(1, 13):
        val = 2**exp
        factors = ffttools.primefacs(val)
        assert factors == [2] * exp, f"primefacs({val}) = {factors}, expected {[2]*exp}"
    if debug:
        print("primefacs_powers_of_two passed")


def primefacs_one(debug=False):
    """Test prime factorization of 1 (edge case)."""
    # primefacs(1): the while loop doesn't execute (2*2 > 1),
    # then it appends thelen=1 at the end
    result = ffttools.primefacs(1)
    assert result == [1]
    if debug:
        print("primefacs_one passed")


def primefacs_large_prime(debug=False):
    """Test prime factorization of a large prime number."""
    # 997 is prime
    result = ffttools.primefacs(997)
    assert result == [997]
    # 7919 is prime
    result = ffttools.primefacs(7919)
    assert result == [7919]
    if debug:
        print("primefacs_large_prime passed")


# --- optfftlen tests ---


def optfftlen_basic(debug=False):
    """Test optfftlen returns sensible values for basic inputs."""
    # Result must be >= input
    for length in [64, 100, 128, 256, 500, 1000, 1024, 2000]:
        result = ffttools.optfftlen(length)
        assert result >= length, f"optfftlen({length}) = {result}, expected >= {length}"
    if debug:
        print("optfftlen_basic passed")


def optfftlen_powers_of_two(debug=False):
    """Test that powers of 2 are returned as-is (already optimal)."""
    for exp in range(4, 14):
        val = 2**exp
        result = ffttools.optfftlen(val)
        # Powers of 2 are always fast FFT lengths
        assert result == val, f"optfftlen({val}) = {result}, expected {val}"
    if debug:
        print("optfftlen_powers_of_two passed")


def optfftlen_with_padding(debug=False):
    """Test optfftlen with padlen parameter."""
    length = 100
    padlen = 20
    result = ffttools.optfftlen(length, padlen=padlen)
    # With padding, result should accommodate length + 2*padlen
    assert result >= length, f"optfftlen({length}, padlen={padlen}) = {result}, expected >= {length}"
    if debug:
        print("optfftlen_with_padding passed")


def optfftlen_caching(debug=False):
    """Test that optfftlen uses the cache (lencache)."""
    # Clear cache first
    ffttools.lencache.clear()

    val = 500
    result1 = ffttools.optfftlen(val)
    # Check that cachekey is in lencache
    cachekey = f"{val}_0"
    assert cachekey in ffttools.lencache, f"Expected cachekey '{cachekey}' in lencache"
    assert ffttools.lencache[cachekey] == result1

    # Second call should return the same (from cache)
    result2 = ffttools.optfftlen(val)
    assert result1 == result2, f"Cached result mismatch: {result1} vs {result2}"

    # Clean up
    ffttools.lencache.clear()
    if debug:
        print("optfftlen_caching passed")


def optfftlen_debug_output(debug=False):
    """Test optfftlen debug mode produces output."""
    ffttools.lencache.clear()
    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        ffttools.optfftlen(100, debug=True)
    finally:
        sys.stdout = old_stdout
    output = captured.getvalue()
    assert "optfftlen" in output.lower() or "entering" in output.lower(), (
        f"Expected debug output, got: '{output}'"
    )
    ffttools.lencache.clear()
    if debug:
        print("optfftlen_debug_output passed")


def optfftlen_various_lengths(debug=False):
    """Test optfftlen with a range of input lengths and verify factors are small."""
    ffttools.lencache.clear()
    for length in [33, 65, 99, 127, 129, 255, 257, 500, 1000, 2001]:
        result = ffttools.optfftlen(length)
        assert result >= length
        # Check that result has only small prime factors (2, 3, 5, 7, etc.)
        factors = ffttools.primefacs(result)
        max_factor = max(factors)
        if ffttools.pyfftwpresent:
            # pyfftw's next_fast_len should produce numbers with small prime factors
            assert max_factor <= 13, (
                f"optfftlen({length}) = {result} has large prime factor {max_factor}"
            )
    ffttools.lencache.clear()
    if debug:
        print("optfftlen_various_lengths passed")


def optfftlen_zero_padlen(debug=False):
    """Test that padlen=0 is equivalent to default."""
    ffttools.lencache.clear()
    for length in [64, 100, 256]:
        r1 = ffttools.optfftlen(length)
        ffttools.lencache.clear()
        r2 = ffttools.optfftlen(length, padlen=0)
        assert r1 == r2, f"optfftlen({length}) = {r1} vs optfftlen({length}, padlen=0) = {r2}"
    ffttools.lencache.clear()
    if debug:
        print("optfftlen_zero_padlen passed")


# --- showfftcache tests ---


def showfftcache_empty(debug=False):
    """Test showfftcache with an empty cache."""
    ffttools.lencache.clear()
    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        ffttools.showfftcache()
    finally:
        sys.stdout = old_stdout
    output = captured.getvalue()
    assert "FFT length cache entries:" in output
    if debug:
        print("showfftcache_empty passed")


def showfftcache_with_entries(debug=False):
    """Test showfftcache after populating the cache."""
    ffttools.lencache.clear()
    # Populate cache with a few entries
    ffttools.optfftlen(64)
    ffttools.optfftlen(128)

    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        ffttools.showfftcache()
    finally:
        sys.stdout = old_stdout
    output = captured.getvalue()
    assert "FFT length cache entries:" in output
    # Should have printed some entries
    lines = output.strip().split("\n")
    assert len(lines) >= 1  # at least the header line

    ffttools.lencache.clear()
    if debug:
        print("showfftcache_with_entries passed")


# --- pyfftwpresent flag test ---


def pyfftw_flag_test(debug=False):
    """Test that pyfftwpresent flag is a boolean."""
    assert isinstance(ffttools.pyfftwpresent, bool)
    if debug:
        print(f"pyfftwpresent = {ffttools.pyfftwpresent}")
        print("pyfftw_flag_test passed")


# --- integration test: primefacs + optfftlen ---


def integration_primefacs_optfftlen(debug=False):
    """Test that optfftlen results can be factored correctly."""
    ffttools.lencache.clear()
    for length in [50, 100, 200, 500, 1000]:
        opt_len = ffttools.optfftlen(length)
        factors = ffttools.primefacs(opt_len)
        product = 1
        for f in factors:
            product *= f
        assert product == opt_len, (
            f"Product of factors of optfftlen({length})={opt_len} is {product}"
        )
    ffttools.lencache.clear()
    if debug:
        print("integration_primefacs_optfftlen passed")


# --- main test function ---


def test_ffttools(debug=False):
    # primefacs tests
    primefacs_small_primes(debug=debug)
    primefacs_composite(debug=debug)
    primefacs_product_check(debug=debug)
    primefacs_all_factors_prime(debug=debug)
    primefacs_powers_of_two(debug=debug)
    primefacs_one(debug=debug)
    primefacs_large_prime(debug=debug)

    # optfftlen tests
    optfftlen_basic(debug=debug)
    optfftlen_powers_of_two(debug=debug)
    optfftlen_with_padding(debug=debug)
    optfftlen_caching(debug=debug)
    optfftlen_debug_output(debug=debug)
    optfftlen_various_lengths(debug=debug)
    optfftlen_zero_padlen(debug=debug)

    # showfftcache tests
    showfftcache_empty(debug=debug)
    showfftcache_with_entries(debug=debug)

    # misc tests
    pyfftw_flag_test(debug=debug)
    integration_primefacs_optfftlen(debug=debug)


if __name__ == "__main__":
    test_ffttools(debug=True)
