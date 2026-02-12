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
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rapidtide.refineregressor import (
    _packvoxeldata,
    _procOneVoxelTimeShift,
    _unpackvoxeldata,
    alignvoxels,
    dorefine,
    findecho,
    makerefinemask,
    prenorm,
)

# ==================== Helpers ====================


def _make_broadband_signal(n, seed=42):
    """Create a broadband LFO signal (sum of sinusoids) for realistic testing."""
    rng = np.random.RandomState(seed)
    t = np.arange(n, dtype=np.float64)
    signal = np.zeros(n)
    for _ in range(20):
        freq = rng.uniform(0.01, 0.15)
        phase = rng.uniform(0, 2 * np.pi)
        signal += np.sin(2 * np.pi * freq * t + phase)
    return signal


# ==================== _packvoxeldata ====================


def test_packvoxeldata_basic(debug=False):
    features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    labels = np.array([10.0, 20.0])
    extra1 = "param1"
    extra2 = "param2"

    result = _packvoxeldata(0, (features, labels, extra1, extra2))
    np.testing.assert_array_equal(result[0], [1.0, 2.0, 3.0])
    assert result[1] == 10.0
    assert result[2] == "param1"
    assert result[3] == "param2"

    if debug:
        print(f"packvoxeldata result: {result}")


def test_packvoxeldata_second_voxel(debug=False):
    features = np.array([[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]])
    labels = np.array([10.0, 20.0, 30.0])

    result = _packvoxeldata(2, (features, labels, 0.5, 2.0))
    np.testing.assert_array_equal(result[0], [7.0, 8.0])
    assert result[1] == 30.0

    if debug:
        print("test_packvoxeldata_second_voxel passed")


# ==================== _unpackvoxeldata ====================


def test_unpackvoxeldata_basic(debug=False):
    nvoxels = 5
    ntimepoints = 10
    voxelproducts = [np.zeros((nvoxels, ntimepoints)) for _ in range(4)]

    data1 = np.arange(ntimepoints, dtype=np.float64)
    data2 = np.ones(ntimepoints) * 2.0
    data3 = np.ones(ntimepoints) * 3.0
    data4 = np.ones(ntimepoints) * 4.0

    retvals = (2, data1, data2, data3, data4)
    _unpackvoxeldata(retvals, voxelproducts)

    np.testing.assert_array_equal(voxelproducts[0][2, :], data1)
    np.testing.assert_array_equal(voxelproducts[1][2, :], data2)
    np.testing.assert_array_equal(voxelproducts[2][2, :], data3)
    np.testing.assert_array_equal(voxelproducts[3][2, :], data4)
    # Other rows should remain zero
    np.testing.assert_array_equal(voxelproducts[0][0, :], np.zeros(ntimepoints))

    if debug:
        print("test_unpackvoxeldata_basic passed")


def test_unpackvoxeldata_multiple_voxels(debug=False):
    nvoxels = 3
    ntimepoints = 5
    voxelproducts = [np.zeros((nvoxels, ntimepoints)) for _ in range(4)]

    for vox in range(nvoxels):
        retvals = (vox, np.ones(ntimepoints) * (vox + 1), np.ones(ntimepoints) * (vox + 10),
                   np.ones(ntimepoints) * (vox + 100), np.ones(ntimepoints) * (vox + 1000))
        _unpackvoxeldata(retvals, voxelproducts)

    for vox in range(nvoxels):
        assert voxelproducts[0][vox, 0] == vox + 1
        assert voxelproducts[1][vox, 0] == vox + 10
        assert voxelproducts[2][vox, 0] == vox + 100
        assert voxelproducts[3][vox, 0] == vox + 1000

    if debug:
        print("test_unpackvoxeldata_multiple_voxels passed")


# ==================== _procOneVoxelTimeShift ====================


def test_procOneVoxelTimeShift_zero_shift(debug=False):
    """With zero lag time and zero offset, output should match detrended input."""
    n = 100
    fmritc = np.sin(np.linspace(0, 4 * np.pi, n))
    lagtime = 0.0
    padtrs = 10
    fmritr = 2.0

    voxelargs = (fmritc, lagtime, padtrs, fmritr)
    vox, shiftedtc, wts, paddedshiftedtc, paddedwts = _procOneVoxelTimeShift(
        0, voxelargs, detrendorder=0, offsettime=0.0
    )

    assert vox == 0
    assert len(shiftedtc) == n
    assert len(paddedshiftedtc) == n + 2 * padtrs
    # With zero shift, shifted tc should be very close to input
    np.testing.assert_allclose(shiftedtc, fmritc, atol=1e-10)

    if debug:
        print(f"Zero shift max diff: {np.max(np.abs(shiftedtc - fmritc))}")


def test_procOneVoxelTimeShift_nonzero_shift(debug=False):
    """With nonzero lag, the output should be shifted."""
    n = 200
    fmritc = _make_broadband_signal(n)
    lagtime = 2.0  # seconds
    padtrs = 20
    fmritr = 1.0

    voxelargs = (fmritc, lagtime, padtrs, fmritr)
    vox, shiftedtc, wts, paddedshiftedtc, paddedwts = _procOneVoxelTimeShift(
        5, voxelargs, detrendorder=0
    )

    assert vox == 5
    assert len(shiftedtc) == n
    # The shifted TC should differ from the original
    assert not np.allclose(shiftedtc, fmritc, atol=0.01)

    if debug:
        print(f"Nonzero shift correlation: {np.corrcoef(shiftedtc, fmritc)[0, 1]:.4f}")


def test_procOneVoxelTimeShift_detrending(debug=False):
    """With detrendorder > 0, linear trends should be removed."""
    n = 100
    # Create a signal with a strong linear trend
    fmritc = np.linspace(0, 10, n) + np.sin(np.linspace(0, 4 * np.pi, n))
    lagtime = 0.0
    padtrs = 10
    fmritr = 2.0

    voxelargs = (fmritc, lagtime, padtrs, fmritr)
    _, shiftedtc, _, _, _ = _procOneVoxelTimeShift(
        0, voxelargs, detrendorder=1, offsettime=0.0
    )

    # After detrending, mean should be near zero
    assert abs(np.mean(shiftedtc)) < 0.5

    if debug:
        print(f"After detrend: mean={np.mean(shiftedtc):.6f}")


def test_procOneVoxelTimeShift_with_offset(debug=False):
    """Offset time should shift the result differently than without."""
    n = 200
    fmritc = _make_broadband_signal(n)
    lagtime = 1.0
    padtrs = 20
    fmritr = 1.0

    voxelargs = (fmritc, lagtime, padtrs, fmritr)
    _, shifted_no_offset, _, _, _ = _procOneVoxelTimeShift(
        0, voxelargs, detrendorder=0, offsettime=0.0
    )
    _, shifted_with_offset, _, _, _ = _procOneVoxelTimeShift(
        0, voxelargs, detrendorder=0, offsettime=0.5
    )

    # Different offset times should produce different results
    assert not np.allclose(shifted_no_offset, shifted_with_offset, atol=0.001)

    if debug:
        print("test_procOneVoxelTimeShift_with_offset passed")


# ==================== findecho ====================


def test_findecho(debug=False):
    """Test AR parameter estimation via Levinson-Durbin."""
    rng = np.random.RandomState(42)
    nvoxels = 5
    ntimepoints = 200
    nlags = 3

    # Create AR(1) processes for predictable results
    shiftedtcs = np.zeros((nvoxels, ntimepoints))
    for v in range(nvoxels):
        shiftedtcs[v, 0] = rng.randn()
        for t in range(1, ntimepoints):
            shiftedtcs[v, t] = 0.5 * shiftedtcs[v, t - 1] + rng.randn()

    sigmav = np.zeros(nvoxels)
    arcoefs = np.zeros((nvoxels, nlags))
    # levinson_durbin returns pacf with nlags+1 elements, sigma with nlags+1,
    # and phi as (nlags+1, nlags+1) 2D array
    pacf = np.zeros((nvoxels, nlags + 1))
    sigma = np.zeros((nvoxels, nlags + 1))
    phi = np.zeros((nvoxels, nlags + 1, nlags + 1))

    findecho(nlags, shiftedtcs, sigmav, arcoefs, pacf, sigma, phi)

    # sigmav should be positive (variance estimate)
    assert np.all(sigmav > 0)
    # AR coefficients should be populated
    assert np.any(arcoefs != 0)
    # First AR coefficient should be close to 0.5 for AR(1) with coeff 0.5
    for v in range(nvoxels):
        assert abs(arcoefs[v, 0] - 0.5) < 0.3  # rough check

    if debug:
        print(f"sigmav: {sigmav}")
        print(f"arcoefs[:,0]: {arcoefs[:, 0]}")


# ==================== makerefinemask ====================


def test_makerefinemask_basic(debug=False):
    """Test basic mask creation with default parameters."""
    shape = (5, 5, 5)
    lagstrengths = np.ones(shape) * 0.5
    lagtimes = np.ones(shape) * 2.0
    lagsigma = np.ones(shape) * 1.0
    lagmask = np.ones(shape)

    volumetotal, maskarray, locfails, ampfails, lagfails, sigfails, numinmask = makerefinemask(
        lagstrengths, lagtimes, lagsigma, lagmask,
        ampthresh=0.3, lagminthresh=0.5, lagmaxthresh=5.0, sigmathresh=100,
    )

    assert volumetotal > 0
    assert maskarray is not None
    assert maskarray.shape == shape
    assert numinmask == np.prod(shape)

    if debug:
        print(f"volumetotal={volumetotal}, numinmask={numinmask}")


def test_makerefinemask_ampthresh_filters(debug=False):
    """Voxels below ampthresh should be excluded."""
    shape = (5, 5, 5)
    lagstrengths = np.ones(shape) * 0.2  # below threshold
    lagtimes = np.ones(shape) * 2.0
    lagsigma = np.ones(shape) * 1.0
    lagmask = np.ones(shape)

    volumetotal, maskarray, _, ampfails, _, _, _ = makerefinemask(
        lagstrengths, lagtimes, lagsigma, lagmask,
        ampthresh=0.5,
        lagminthresh=0.5, lagmaxthresh=5.0,
    )

    # All voxels below ampthresh, so mask should be empty
    assert volumetotal == 0
    assert maskarray is None

    if debug:
        print(f"ampfails={ampfails}")


def test_makerefinemask_lag_filters(debug=False):
    """Voxels with lag outside range should be excluded."""
    shape = (5, 5, 5)
    lagstrengths = np.ones(shape) * 0.8
    lagtimes = np.ones(shape) * 10.0  # outside lagmaxthresh=5
    lagsigma = np.ones(shape) * 1.0
    lagmask = np.ones(shape)

    volumetotal, maskarray, _, _, lagfails, _, _ = makerefinemask(
        lagstrengths, lagtimes, lagsigma, lagmask,
        ampthresh=0.3, lagminthresh=0.5, lagmaxthresh=5.0,
    )

    assert volumetotal == 0
    assert maskarray is None

    if debug:
        print(f"lagfails={lagfails}")


def test_makerefinemask_sigma_filters(debug=False):
    """Voxels with sigma above sigmathresh should be excluded."""
    shape = (5, 5, 5)
    lagstrengths = np.ones(shape) * 0.8
    lagtimes = np.ones(shape) * 2.0
    lagsigma = np.ones(shape) * 200.0  # above sigmathresh=100
    lagmask = np.ones(shape)

    volumetotal, maskarray, _, _, _, sigfails, _ = makerefinemask(
        lagstrengths, lagtimes, lagsigma, lagmask,
        ampthresh=0.3, lagminthresh=0.5, lagmaxthresh=5.0, sigmathresh=100,
    )

    assert volumetotal == 0
    assert maskarray is None

    if debug:
        print(f"sigfails={sigfails}")


def test_makerefinemask_lagmaskside_upper(debug=False):
    """Test upper lag mask side."""
    shape = (5, 5, 5)
    lagstrengths = np.ones(shape) * 0.8
    lagtimes = np.ones(shape) * 2.0
    lagsigma = np.ones(shape) * 1.0
    lagmask = np.ones(shape)

    volumetotal, maskarray, _, _, _, _, _ = makerefinemask(
        lagstrengths, lagtimes, lagsigma, lagmask,
        ampthresh=0.3, lagmaskside="upper", lagminthresh=0.5, lagmaxthresh=5.0,
    )

    assert volumetotal > 0
    assert maskarray is not None

    if debug:
        print(f"upper side volumetotal={volumetotal}")


def test_makerefinemask_lagmaskside_lower(debug=False):
    """Test lower lag mask side with negative lags."""
    shape = (5, 5, 5)
    lagstrengths = np.ones(shape) * 0.8
    lagtimes = np.ones(shape) * -2.0  # negative lags
    lagsigma = np.ones(shape) * 1.0
    lagmask = np.ones(shape)

    volumetotal, maskarray, _, _, _, _, _ = makerefinemask(
        lagstrengths, lagtimes, lagsigma, lagmask,
        ampthresh=0.3, lagmaskside="lower", lagminthresh=0.5, lagmaxthresh=5.0,
    )

    assert volumetotal > 0

    if debug:
        print(f"lower side volumetotal={volumetotal}")


def test_makerefinemask_bipolar(debug=False):
    """Test bipolar mode uses absolute correlation values."""
    shape = (5, 5, 5)
    lagstrengths = np.ones(shape) * -0.8  # negative correlations
    lagtimes = np.ones(shape) * 2.0
    lagsigma = np.ones(shape) * 1.0
    lagmask = np.ones(shape)

    # Without bipolar, negative strengths fail ampthresh
    vol_nobipolar, mask_nobipolar, _, _, _, _, _ = makerefinemask(
        lagstrengths, lagtimes, lagsigma, lagmask,
        ampthresh=0.3, lagminthresh=0.5, lagmaxthresh=5.0, bipolar=False,
    )

    # With bipolar, absolute values are used
    vol_bipolar, mask_bipolar, _, _, _, _, _ = makerefinemask(
        lagstrengths, lagtimes, lagsigma, lagmask,
        ampthresh=0.3, lagminthresh=0.5, lagmaxthresh=5.0, bipolar=True,
    )

    assert vol_nobipolar == 0
    assert vol_bipolar > 0

    if debug:
        print(f"nobipolar={vol_nobipolar}, bipolar={vol_bipolar}")


def test_makerefinemask_includemask(debug=False):
    """Test that includemask restricts which voxels are considered."""
    shape = (5, 5, 5)
    lagstrengths = np.ones(shape) * 0.8
    lagtimes = np.ones(shape) * 2.0
    lagsigma = np.ones(shape) * 1.0
    lagmask = np.ones(shape)

    # Only include a small subregion
    includemask = np.zeros(shape)
    includemask[2, 2, 2] = 1

    volumetotal, maskarray, _, _, _, _, _ = makerefinemask(
        lagstrengths, lagtimes, lagsigma, lagmask,
        ampthresh=0.3, lagminthresh=0.5, lagmaxthresh=5.0,
        includemask=includemask,
    )

    assert volumetotal == 1

    if debug:
        print(f"includemask volumetotal={volumetotal}")


def test_makerefinemask_excludemask(debug=False):
    """Test that excludemask removes specified voxels."""
    shape = (5, 5, 5)
    lagstrengths = np.ones(shape) * 0.8
    lagtimes = np.ones(shape) * 2.0
    lagsigma = np.ones(shape) * 1.0
    lagmask = np.ones(shape)

    # Exclude everything except one voxel
    excludemask = np.ones(shape)
    excludemask[2, 2, 2] = 0

    volumetotal, maskarray, _, _, _, _, _ = makerefinemask(
        lagstrengths, lagtimes, lagsigma, lagmask,
        ampthresh=0.3, lagminthresh=0.5, lagmaxthresh=5.0,
        excludemask=excludemask,
    )

    assert volumetotal == 1

    if debug:
        print(f"excludemask volumetotal={volumetotal}")


def test_makerefinemask_fixdelay(debug=False):
    """When fixdelay=True, lag thresholds should not be applied."""
    shape = (5, 5, 5)
    lagstrengths = np.ones(shape) * 0.8
    lagtimes = np.ones(shape) * 100.0  # way outside normal range
    lagsigma = np.ones(shape) * 1.0
    lagmask = np.ones(shape)

    # Without fixdelay, these lags would be filtered out
    vol_nofixdelay, _, _, _, _, _, _ = makerefinemask(
        lagstrengths, lagtimes, lagsigma, lagmask,
        ampthresh=0.3, lagminthresh=0.5, lagmaxthresh=5.0, fixdelay=False,
    )

    # With fixdelay, lag thresholds are ignored
    vol_fixdelay, mask_fixdelay, _, _, _, _, _ = makerefinemask(
        lagstrengths, lagtimes, lagsigma, lagmask,
        ampthresh=0.3, lagminthresh=0.5, lagmaxthresh=5.0, fixdelay=True,
    )

    assert vol_nofixdelay == 0
    assert vol_fixdelay > 0

    if debug:
        print(f"nofixdelay={vol_nofixdelay}, fixdelay={vol_fixdelay}")


def test_makerefinemask_negative_ampthresh(debug=False):
    """Negative ampthresh should be treated as a percentile."""
    shape = (5, 5, 5)
    rng = np.random.RandomState(42)
    lagstrengths = rng.uniform(0.1, 1.0, shape)
    lagtimes = np.ones(shape) * 2.0
    lagsigma = np.ones(shape) * 1.0
    lagmask = np.ones(shape)

    volumetotal, maskarray, _, _, _, _, _ = makerefinemask(
        lagstrengths, lagtimes, lagsigma, lagmask,
        ampthresh=-0.5,  # 50th percentile
        lagminthresh=0.5, lagmaxthresh=5.0,
    )

    # Should keep roughly half the voxels (those above median)
    assert 0 < volumetotal < np.prod(shape)

    if debug:
        print(f"negative ampthresh volumetotal={volumetotal}")


def test_makerefinemask_cleanrefined(debug=False):
    """When cleanrefined=True, shiftmask should be locationmask instead of refinemask."""
    shape = (5, 5, 5)
    lagstrengths = np.ones(shape) * 0.8
    lagtimes = np.ones(shape) * 2.0
    lagsigma = np.ones(shape) * 1.0
    lagmask = np.ones(shape)
    # Make a few voxels fail amp threshold
    lagstrengths[0, 0, 0] = 0.1

    vol_clean, mask_clean, _, _, _, _, _ = makerefinemask(
        lagstrengths, lagtimes, lagsigma, lagmask,
        ampthresh=0.3, lagminthresh=0.5, lagmaxthresh=5.0,
        cleanrefined=True,
    )

    vol_noclean, mask_noclean, _, _, _, _, _ = makerefinemask(
        lagstrengths, lagtimes, lagsigma, lagmask,
        ampthresh=0.3, lagminthresh=0.5, lagmaxthresh=5.0,
        cleanrefined=False,
    )

    # cleanrefined uses locationmask (more voxels) vs refinemask
    assert vol_clean >= vol_noclean

    if debug:
        print(f"clean={vol_clean}, noclean={vol_noclean}")


# ==================== prenorm ====================


def test_prenorm_mean(debug=False):
    """Test mean normalization."""
    rng = np.random.RandomState(42)
    nvoxels = 10
    ntimepoints = 50
    shiftedtcs = rng.randn(nvoxels, ntimepoints) + 5.0  # positive mean
    refinemask = np.ones(nvoxels)
    lagtimes = np.ones(nvoxels) * 2.0
    lagmaxthresh = 5.0
    lagstrengths = np.ones(nvoxels) * 0.8
    R2vals = np.ones(nvoxels) * 0.6

    original = shiftedtcs.copy()
    prenorm(shiftedtcs, refinemask, lagtimes, lagmaxthresh, lagstrengths, R2vals,
            "mean", "R")

    # After mean normalization and R weighting, values should change
    assert not np.allclose(shiftedtcs, original)

    if debug:
        print("test_prenorm_mean passed")


def test_prenorm_var(debug=False):
    """Test variance normalization."""
    rng = np.random.RandomState(42)
    nvoxels = 10
    ntimepoints = 50
    shiftedtcs = rng.randn(nvoxels, ntimepoints) * 3.0
    refinemask = np.ones(nvoxels)
    lagtimes = np.ones(nvoxels) * 2.0
    lagmaxthresh = 5.0
    lagstrengths = np.ones(nvoxels) * 0.8
    R2vals = np.ones(nvoxels) * 0.6

    prenorm(shiftedtcs, refinemask, lagtimes, lagmaxthresh, lagstrengths, R2vals,
            "var", "R")

    assert np.all(np.isfinite(shiftedtcs))

    if debug:
        print("test_prenorm_var passed")


def test_prenorm_std(debug=False):
    """Test std normalization."""
    rng = np.random.RandomState(42)
    nvoxels = 10
    ntimepoints = 50
    shiftedtcs = rng.randn(nvoxels, ntimepoints) * 3.0
    refinemask = np.ones(nvoxels)
    lagtimes = np.ones(nvoxels) * 2.0
    lagmaxthresh = 5.0
    lagstrengths = np.ones(nvoxels) * 0.8
    R2vals = np.ones(nvoxels) * 0.6

    prenorm(shiftedtcs, refinemask, lagtimes, lagmaxthresh, lagstrengths, R2vals,
            "std", "R")

    assert np.all(np.isfinite(shiftedtcs))

    if debug:
        print("test_prenorm_std passed")


def test_prenorm_invlag(debug=False):
    """Test inverse lag normalization."""
    rng = np.random.RandomState(42)
    nvoxels = 10
    ntimepoints = 50
    shiftedtcs = rng.randn(nvoxels, ntimepoints)
    refinemask = np.ones(nvoxels)
    lagtimes = np.linspace(0.5, 4.0, nvoxels)
    lagmaxthresh = 5.0
    lagstrengths = np.ones(nvoxels) * 0.8
    R2vals = np.ones(nvoxels) * 0.6

    prenorm(shiftedtcs, refinemask, lagtimes, lagmaxthresh, lagstrengths, R2vals,
            "invlag", "R")

    assert np.all(np.isfinite(shiftedtcs))

    if debug:
        print("test_prenorm_invlag passed")


def test_prenorm_default_norm(debug=False):
    """Test with unknown normalization type (should use unit divisor)."""
    rng = np.random.RandomState(42)
    nvoxels = 10
    ntimepoints = 50
    shiftedtcs = rng.randn(nvoxels, ntimepoints)
    original = shiftedtcs.copy()
    refinemask = np.ones(nvoxels)
    lagtimes = np.ones(nvoxels) * 2.0
    lagmaxthresh = 5.0
    lagstrengths = np.ones(nvoxels) * 0.8
    R2vals = np.ones(nvoxels) * 0.6

    prenorm(shiftedtcs, refinemask, lagtimes, lagmaxthresh, lagstrengths, R2vals,
            "unknown_method", "R")

    # With unit normalization and R weighting (all 0.8), result = original * 0.8
    np.testing.assert_allclose(shiftedtcs, original * 0.8, atol=1e-10)

    if debug:
        print("test_prenorm_default_norm passed")


def test_prenorm_R2_weighting(debug=False):
    """Test R2 weighting mode."""
    rng = np.random.RandomState(42)
    nvoxels = 10
    ntimepoints = 50
    shiftedtcs = rng.randn(nvoxels, ntimepoints)
    original = shiftedtcs.copy()
    refinemask = np.ones(nvoxels)
    lagtimes = np.ones(nvoxels) * 2.0
    lagmaxthresh = 5.0
    lagstrengths = np.ones(nvoxels) * 0.8
    R2vals = np.ones(nvoxels) * 0.5

    prenorm(shiftedtcs, refinemask, lagtimes, lagmaxthresh, lagstrengths, R2vals,
            "unknown_method", "R2")

    # With unit normalization and R2 weighting (all 0.5)
    np.testing.assert_allclose(shiftedtcs, original * 0.5, atol=1e-10)

    if debug:
        print("test_prenorm_R2_weighting passed")


def test_prenorm_default_weighting(debug=False):
    """Test default (sign-based) weighting."""
    rng = np.random.RandomState(42)
    nvoxels = 10
    ntimepoints = 50
    shiftedtcs = rng.randn(nvoxels, ntimepoints)
    original = shiftedtcs.copy()
    refinemask = np.ones(nvoxels)
    lagtimes = np.ones(nvoxels) * 2.0
    lagmaxthresh = 5.0
    lagstrengths = np.ones(nvoxels) * 0.8  # all positive
    R2vals = np.ones(nvoxels) * 0.5

    prenorm(shiftedtcs, refinemask, lagtimes, lagmaxthresh, lagstrengths, R2vals,
            "unknown_method", "other")

    # Default weighting: sign of lagstrengths * refinemask = 1.0 * 1.0 = 1.0
    np.testing.assert_allclose(shiftedtcs, original * 1.0, atol=1e-10)

    if debug:
        print("test_prenorm_default_weighting passed")


def test_prenorm_masked_voxels(debug=False):
    """Voxels outside refinemask should be zeroed."""
    rng = np.random.RandomState(42)
    nvoxels = 10
    ntimepoints = 50
    shiftedtcs = rng.randn(nvoxels, ntimepoints) + 5.0
    refinemask = np.zeros(nvoxels)
    refinemask[:5] = 1  # only first 5 active
    lagtimes = np.ones(nvoxels) * 2.0
    lagmaxthresh = 5.0
    lagstrengths = np.ones(nvoxels) * 0.8
    R2vals = np.ones(nvoxels) * 0.6

    prenorm(shiftedtcs, refinemask, lagtimes, lagmaxthresh, lagstrengths, R2vals,
            "mean", "R")

    # Masked-out voxels should be zeroed
    for v in range(5, 10):
        assert np.all(shiftedtcs[v, :] == 0.0)

    if debug:
        print("test_prenorm_masked_voxels passed")


# ==================== dorefine ====================


def test_dorefine_unweighted_average(debug=False):
    """Test unweighted average refinement."""
    rng = np.random.RandomState(42)
    nvoxels = 20
    ntimepoints = 100

    signal = _make_broadband_signal(ntimepoints, seed=0)
    shiftedtcs = np.tile(signal, (nvoxels, 1)) + 0.1 * rng.randn(nvoxels, ntimepoints)
    refinemask = np.ones(nvoxels)
    weights = np.ones((nvoxels, ntimepoints))
    lagstrengths = np.ones(nvoxels) * 0.8
    lagtimes = np.ones(nvoxels) * 2.0

    # Create a mock prefilter
    mock_prefilter = MagicMock()
    mock_prefilter.apply = MagicMock(side_effect=lambda freq, data: data)

    with tempfile.TemporaryDirectory() as tmpdir:
        outputname = os.path.join(tmpdir, "test_refine")
        volumetotal, outputdata = dorefine(
            shiftedtcs, refinemask, weights, mock_prefilter,
            fmritr=2.0, passnum=1, lagstrengths=lagstrengths,
            lagtimes=lagtimes, refinetype="unweighted_average",
            fmrifreq=0.5, outputname=outputname,
        )

    assert volumetotal == nvoxels
    assert len(outputdata) == ntimepoints
    # Output should be close to the common signal
    corr = np.corrcoef(outputdata, signal)[0, 1]
    assert corr > 0.9

    if debug:
        print(f"unweighted average corr with original: {corr:.4f}")


def test_dorefine_weighted_average(debug=False):
    """Test weighted average refinement."""
    rng = np.random.RandomState(42)
    nvoxels = 20
    ntimepoints = 100

    signal = _make_broadband_signal(ntimepoints, seed=0)
    shiftedtcs = np.tile(signal, (nvoxels, 1)) + 0.1 * rng.randn(nvoxels, ntimepoints)
    refinemask = np.ones(nvoxels)
    weights = np.ones((nvoxels, ntimepoints)) * 0.5
    lagstrengths = np.ones(nvoxels) * 0.8
    lagtimes = np.ones(nvoxels) * 2.0

    mock_prefilter = MagicMock()
    mock_prefilter.apply = MagicMock(side_effect=lambda freq, data: data)

    with tempfile.TemporaryDirectory() as tmpdir:
        outputname = os.path.join(tmpdir, "test_refine")
        volumetotal, outputdata = dorefine(
            shiftedtcs, refinemask, weights, mock_prefilter,
            fmritr=2.0, passnum=1, lagstrengths=lagstrengths,
            lagtimes=lagtimes, refinetype="weighted_average",
            fmrifreq=0.5, outputname=outputname,
        )

    assert volumetotal == nvoxels
    assert len(outputdata) == ntimepoints
    assert np.all(np.isfinite(outputdata))

    if debug:
        print("test_dorefine_weighted_average passed")


def test_dorefine_pca(debug=False):
    """Test PCA refinement."""
    rng = np.random.RandomState(42)
    nvoxels = 30
    ntimepoints = 100

    signal = _make_broadband_signal(ntimepoints, seed=0)
    shiftedtcs = np.tile(signal, (nvoxels, 1)) + 0.5 * rng.randn(nvoxels, ntimepoints)
    refinemask = np.ones(nvoxels)
    weights = np.ones((nvoxels, ntimepoints))
    lagstrengths = np.ones(nvoxels) * 0.8
    lagtimes = np.ones(nvoxels) * 2.0

    mock_prefilter = MagicMock()
    mock_prefilter.apply = MagicMock(side_effect=lambda freq, data: data)

    with tempfile.TemporaryDirectory() as tmpdir:
        outputname = os.path.join(tmpdir, "test_refine")
        volumetotal, outputdata = dorefine(
            shiftedtcs, refinemask, weights, mock_prefilter,
            fmritr=2.0, passnum=1, lagstrengths=lagstrengths,
            lagtimes=lagtimes, refinetype="pca",
            fmrifreq=0.5, outputname=outputname,
            pcacomponents=0.8,
        )

    assert volumetotal == nvoxels
    assert len(outputdata) == ntimepoints

    if debug:
        corr = np.corrcoef(outputdata, signal)[0, 1]
        print(f"PCA refinement corr: {corr:.4f}")


def test_dorefine_ica(debug=False):
    """Test ICA refinement."""
    rng = np.random.RandomState(42)
    nvoxels = 30
    ntimepoints = 100

    signal = _make_broadband_signal(ntimepoints, seed=0)
    shiftedtcs = np.tile(signal, (nvoxels, 1)) + 0.5 * rng.randn(nvoxels, ntimepoints)
    refinemask = np.ones(nvoxels)
    weights = np.ones((nvoxels, ntimepoints))
    lagstrengths = np.ones(nvoxels) * 0.8
    lagtimes = np.ones(nvoxels) * 2.0

    mock_prefilter = MagicMock()
    mock_prefilter.apply = MagicMock(side_effect=lambda freq, data: data)

    with tempfile.TemporaryDirectory() as tmpdir:
        outputname = os.path.join(tmpdir, "test_refine")
        volumetotal, outputdata = dorefine(
            shiftedtcs, refinemask, weights, mock_prefilter,
            fmritr=2.0, passnum=1, lagstrengths=lagstrengths,
            lagtimes=lagtimes, refinetype="ica",
            fmrifreq=0.5, outputname=outputname,
        )

    assert volumetotal == nvoxels
    assert len(outputdata) == ntimepoints

    if debug:
        print("test_dorefine_ica passed")


def test_dorefine_partial_mask(debug=False):
    """Test that refinemask properly selects subset of voxels."""
    rng = np.random.RandomState(42)
    nvoxels = 20
    ntimepoints = 100

    signal = _make_broadband_signal(ntimepoints, seed=0)
    shiftedtcs = np.tile(signal, (nvoxels, 1)) + 0.1 * rng.randn(nvoxels, ntimepoints)
    refinemask = np.zeros(nvoxels)
    refinemask[:10] = 1  # only use first 10 voxels
    weights = np.ones((nvoxels, ntimepoints))
    lagstrengths = np.ones(nvoxels) * 0.8
    lagtimes = np.ones(nvoxels) * 2.0

    mock_prefilter = MagicMock()
    mock_prefilter.apply = MagicMock(side_effect=lambda freq, data: data)

    with tempfile.TemporaryDirectory() as tmpdir:
        outputname = os.path.join(tmpdir, "test_refine")
        volumetotal, outputdata = dorefine(
            shiftedtcs, refinemask, weights, mock_prefilter,
            fmritr=2.0, passnum=1, lagstrengths=lagstrengths,
            lagtimes=lagtimes, refinetype="unweighted_average",
            fmrifreq=0.5, outputname=outputname,
        )

    assert volumetotal == 10

    if debug:
        print(f"partial mask volumetotal={volumetotal}")


def test_dorefine_bipolar(debug=False):
    """Test bipolar mode flips negative lag strengths."""
    rng = np.random.RandomState(42)
    nvoxels = 20
    ntimepoints = 100

    signal = _make_broadband_signal(ntimepoints, seed=0)
    shiftedtcs = np.tile(signal, (nvoxels, 1)) + 0.05 * rng.randn(nvoxels, ntimepoints)
    # Invert the timecourses of voxels that will have negative lagstrengths,
    # simulating anti-correlated voxels
    shiftedtcs[10:, :] *= -1.0
    refinemask = np.ones(nvoxels)
    weights = np.ones((nvoxels, ntimepoints))
    lagstrengths = np.ones(nvoxels) * 0.8
    lagstrengths[10:] = -0.8  # half have negative correlation
    lagtimes = np.ones(nvoxels) * 2.0

    mock_prefilter = MagicMock()
    mock_prefilter.apply = MagicMock(side_effect=lambda freq, data: data)

    with tempfile.TemporaryDirectory() as tmpdir:
        outputname = os.path.join(tmpdir, "test_refine")
        volumetotal, outputdata = dorefine(
            shiftedtcs, refinemask, weights, mock_prefilter,
            fmritr=2.0, passnum=1, lagstrengths=lagstrengths,
            lagtimes=lagtimes, refinetype="unweighted_average",
            fmrifreq=0.5, outputname=outputname,
            bipolar=True,
        )

    assert volumetotal == nvoxels
    # With bipolar, the inverted voxels get flipped back, so average should match signal
    corr = np.corrcoef(outputdata, signal)[0, 1]
    assert corr > 0.9

    if debug:
        print(f"bipolar corr: {corr:.4f}")


def test_dorefine_cleanrefined(debug=False):
    """Test cleanrefined mode removes discard data via regression."""
    rng = np.random.RandomState(42)
    nvoxels = 20
    ntimepoints = 100

    signal = _make_broadband_signal(ntimepoints, seed=0)
    shiftedtcs = np.tile(signal, (nvoxels, 1)) + 0.1 * rng.randn(nvoxels, ntimepoints)
    refinemask = np.zeros(nvoxels)
    refinemask[:15] = 1  # 15 in mask, 5 discarded
    weights = np.ones((nvoxels, ntimepoints))
    lagstrengths = np.ones(nvoxels) * 0.8
    lagtimes = np.ones(nvoxels) * 2.0

    mock_prefilter = MagicMock()
    mock_prefilter.apply = MagicMock(side_effect=lambda freq, data: data)

    with tempfile.TemporaryDirectory() as tmpdir:
        outputname = os.path.join(tmpdir, "test_refine")
        volumetotal, outputdata = dorefine(
            shiftedtcs, refinemask, weights, mock_prefilter,
            fmritr=2.0, passnum=1, lagstrengths=lagstrengths,
            lagtimes=lagtimes, refinetype="unweighted_average",
            fmrifreq=0.5, outputname=outputname,
            cleanrefined=True,
        )

    assert volumetotal == 15
    assert len(outputdata) == ntimepoints
    assert np.all(np.isfinite(outputdata))

    if debug:
        print("test_dorefine_cleanrefined passed")


# ==================== alignvoxels ====================


def test_alignvoxels_single_proc(debug=False):
    """Test alignvoxels with single processor."""
    nvoxels = 10
    ntimepoints = 50
    padtrs = 10
    fmritr = 2.0

    rng = np.random.RandomState(42)
    fmridata = rng.randn(nvoxels, ntimepoints)
    lagtimes = np.ones(nvoxels) * 1.0
    lagmask = np.ones(nvoxels)

    shiftedtcs = np.zeros((nvoxels, ntimepoints))
    weights = np.zeros((nvoxels, ntimepoints))
    paddedshiftedtcs = np.zeros((nvoxels, ntimepoints + 2 * padtrs))
    paddedweights = np.zeros((nvoxels, ntimepoints + 2 * padtrs))

    volumetotal = alignvoxels(
        fmridata, fmritr, shiftedtcs, weights, paddedshiftedtcs, paddedweights,
        lagtimes, lagmask,
        detrendorder=1, offsettime=0.0,
        nprocs=1, showprogressbar=False, padtrs=padtrs,
    )

    assert volumetotal == nvoxels
    # Shifted TCs should be populated for masked voxels
    assert np.any(shiftedtcs != 0)
    assert np.any(paddedshiftedtcs != 0)

    if debug:
        print(f"alignvoxels volumetotal={volumetotal}")


def test_alignvoxels_partial_mask(debug=False):
    """Test that only masked voxels are processed."""
    nvoxels = 10
    ntimepoints = 50
    padtrs = 10
    fmritr = 2.0

    rng = np.random.RandomState(42)
    fmridata = rng.randn(nvoxels, ntimepoints)
    lagtimes = np.ones(nvoxels) * 1.0
    lagmask = np.zeros(nvoxels)
    lagmask[:5] = 1  # only first 5 voxels

    shiftedtcs = np.zeros((nvoxels, ntimepoints))
    weights = np.zeros((nvoxels, ntimepoints))
    paddedshiftedtcs = np.zeros((nvoxels, ntimepoints + 2 * padtrs))
    paddedweights = np.zeros((nvoxels, ntimepoints + 2 * padtrs))

    volumetotal = alignvoxels(
        fmridata, fmritr, shiftedtcs, weights, paddedshiftedtcs, paddedweights,
        lagtimes, lagmask,
        detrendorder=1, offsettime=0.0,
        nprocs=1, showprogressbar=False, padtrs=padtrs,
    )

    assert volumetotal == 5
    # Unmasked voxels should remain zero
    for v in range(5, 10):
        assert np.all(shiftedtcs[v, :] == 0)

    if debug:
        print(f"partial mask volumetotal={volumetotal}")


# ==================== Main test entry point ====================


def test_refineregressor(debug=False):
    test_packvoxeldata_basic(debug=debug)
    test_packvoxeldata_second_voxel(debug=debug)
    test_unpackvoxeldata_basic(debug=debug)
    test_unpackvoxeldata_multiple_voxels(debug=debug)
    test_procOneVoxelTimeShift_zero_shift(debug=debug)
    test_procOneVoxelTimeShift_nonzero_shift(debug=debug)
    test_procOneVoxelTimeShift_detrending(debug=debug)
    test_procOneVoxelTimeShift_with_offset(debug=debug)
    test_findecho(debug=debug)
    test_makerefinemask_basic(debug=debug)
    test_makerefinemask_ampthresh_filters(debug=debug)
    test_makerefinemask_lag_filters(debug=debug)
    test_makerefinemask_sigma_filters(debug=debug)
    test_makerefinemask_lagmaskside_upper(debug=debug)
    test_makerefinemask_lagmaskside_lower(debug=debug)
    test_makerefinemask_bipolar(debug=debug)
    test_makerefinemask_includemask(debug=debug)
    test_makerefinemask_excludemask(debug=debug)
    test_makerefinemask_fixdelay(debug=debug)
    test_makerefinemask_negative_ampthresh(debug=debug)
    test_makerefinemask_cleanrefined(debug=debug)
    test_prenorm_mean(debug=debug)
    test_prenorm_var(debug=debug)
    test_prenorm_std(debug=debug)
    test_prenorm_invlag(debug=debug)
    test_prenorm_default_norm(debug=debug)
    test_prenorm_R2_weighting(debug=debug)
    test_prenorm_default_weighting(debug=debug)
    test_prenorm_masked_voxels(debug=debug)
    test_dorefine_unweighted_average(debug=debug)
    test_dorefine_weighted_average(debug=debug)
    test_dorefine_pca(debug=debug)
    test_dorefine_ica(debug=debug)
    test_dorefine_partial_mask(debug=debug)
    test_dorefine_bipolar(debug=debug)
    test_dorefine_cleanrefined(debug=debug)
    test_alignvoxels_single_proc(debug=debug)
    test_alignvoxels_partial_mask(debug=debug)


if __name__ == "__main__":
    test_refineregressor(debug=True)
