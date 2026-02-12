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
import numpy as np
import pytest

from rapidtide.wiener import _procOneVoxelWiener, wienerpass

# ==================== _procOneVoxelWiener ====================


def test_procOneVoxelWiener_basic(debug=False):
    """Test basic Wiener processing with a known linear relationship."""
    rng = np.random.RandomState(42)
    n = 100
    lagtc = rng.randn(n)
    # inittc = 2 * lagtc + 3 + small noise
    inittc = 2.0 * lagtc + 3.0 + 0.01 * rng.randn(n)

    result = _procOneVoxelWiener(0, lagtc, inittc)

    vox, intercept, sqrt_R2, R2, fitcoff, ratio, datatoremove, residual = result

    assert vox == 0
    # fitcoff should be close to 2.0
    assert abs(fitcoff - 2.0) < 0.1
    # R2 should be close to 1.0
    assert R2 > 0.99
    # sqrt_R2 should be sqrt of R2
    assert abs(sqrt_R2 - np.sqrt(R2)) < 1e-10
    # datatoremove should be fitcoff * lagtc
    np.testing.assert_allclose(datatoremove, fitcoff * lagtc, atol=1e-10)
    # residual should be inittc - datatoremove
    np.testing.assert_allclose(residual, inittc - datatoremove, atol=1e-10)

    if debug:
        print(f"fitcoff={fitcoff:.4f}, R2={R2:.6f}, intercept={intercept:.4f}")


def test_procOneVoxelWiener_voxel_index(debug=False):
    """Test that voxel index is passed through correctly."""
    lagtc = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    inittc = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

    result = _procOneVoxelWiener(42, lagtc, inittc)
    assert result[0] == 42

    if debug:
        print("test_procOneVoxelWiener_voxel_index passed")


def test_procOneVoxelWiener_float32(debug=False):
    """Test that output arrays respect rt_floattype."""
    lagtc = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    inittc = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

    result = _procOneVoxelWiener(0, lagtc, inittc, rt_floattype=np.float32)
    datatoremove = result[6]
    residual = result[7]
    assert datatoremove.dtype == np.float32
    assert residual.dtype == np.float32

    if debug:
        print("test_procOneVoxelWiener_float32 passed")


def test_procOneVoxelWiener_float64(debug=False):
    """Test float64 output type."""
    lagtc = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    inittc = np.array([3.0, 6.0, 9.0, 12.0, 15.0])

    result = _procOneVoxelWiener(0, lagtc, inittc, rt_floattype=np.float64)
    datatoremove = result[6]
    residual = result[7]
    assert datatoremove.dtype == np.float64
    assert residual.dtype == np.float64

    if debug:
        print("test_procOneVoxelWiener_float64 passed")


def test_procOneVoxelWiener_uncorrelated(debug=False):
    """Test with uncorrelated signals — R2 should be low."""
    rng = np.random.RandomState(42)
    n = 200
    lagtc = rng.randn(n)
    inittc = rng.randn(n)

    result = _procOneVoxelWiener(0, lagtc, inittc)
    R2 = result[3]
    # R2 should be near zero for uncorrelated signals
    assert R2 < 0.1

    if debug:
        print(f"Uncorrelated R2: {R2:.6f}")


def test_procOneVoxelWiener_perfect_fit(debug=False):
    """Test with perfectly correlated signals (no noise)."""
    lagtc = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    inittc = 3.0 * lagtc + 1.0

    result = _procOneVoxelWiener(0, lagtc, inittc)
    fitcoff = result[4]
    R2 = result[3]
    intercept = result[1]

    assert abs(fitcoff - 3.0) < 0.01
    assert R2 > 0.999
    assert abs(intercept - 1.0) < 0.01

    if debug:
        print(f"Perfect fit: fitcoff={fitcoff:.4f}, R2={R2:.6f}, intercept={intercept:.4f}")


def test_procOneVoxelWiener_ratio(debug=False):
    """Test that ratio = slope / intercept."""
    lagtc = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    inittc = 2.0 * lagtc + 4.0

    result = _procOneVoxelWiener(0, lagtc, inittc)
    intercept = result[1]
    fitcoff = result[4]
    ratio = result[5]

    expected_ratio = fitcoff / intercept
    assert abs(ratio - expected_ratio) < 1e-10

    if debug:
        print(f"Ratio: {ratio:.4f}, expected: {expected_ratio:.4f}")


def test_procOneVoxelWiener_residual_orthogonal(debug=False):
    """Test that the residual is approximately orthogonal to the lagtc for well-conditioned data."""
    rng = np.random.RandomState(42)
    n = 500
    lagtc = rng.randn(n)
    inittc = 2.0 * lagtc + 3.0 + 0.5 * rng.randn(n)

    result = _procOneVoxelWiener(0, lagtc, inittc)
    residual = result[7]

    # Correlation between residual and lagtc should be near zero
    corr = np.corrcoef(residual, lagtc)[0, 1]
    assert abs(corr) < 0.1

    if debug:
        print(f"Residual-lagtc correlation: {corr:.6f}")


# ==================== wienerpass ====================

def test_wienerpass_singleproc(debug=False):
    """Test wienerpass with single processor (exercises the else branch)."""
    rng = np.random.RandomState(42)
    nvoxels = 5
    ntimepoints = 50
    fmri_data = rng.randn(nvoxels, ntimepoints) + 10.0  # mean > threshval
    lagtc_data = rng.randn(nvoxels, ntimepoints)

    # Set up the global arrays that wienerpass writes into
    import rapidtide.wiener as wiener_mod

    wiener_mod.meanvalue = np.zeros(nvoxels)
    wiener_mod.rvalue = np.zeros(nvoxels)
    wiener_mod.r2value = np.zeros(nvoxels)
    wiener_mod.fitcoff = np.zeros(nvoxels)
    wiener_mod.fitNorm = np.zeros(nvoxels)
    wiener_mod.datatoremove = np.zeros((nvoxels, ntimepoints))
    wiener_mod.filtereddata = np.zeros((nvoxels, ntimepoints))

    optiondict = {
        "nprocs": 1,
        "showprogressbar": False,
        "mp_chunksize": 10,
    }

    volumetotal = wienerpass(
        numspatiallocs=nvoxels,
        fmri_data=fmri_data,
        threshval=0.0,
        lagtc=lagtc_data,
        optiondict=optiondict,
        wienerdeconv=np.array([1.0]),
        wpeak=np.array([0.5]),
        resampref_y=np.array([1.0]),
    )

    assert volumetotal == nvoxels
    # Check that output arrays were populated for above-threshold voxels
    assert np.any(wiener_mod.fitcoff != 0)
    assert np.any(wiener_mod.rvalue != 0)

    if debug:
        print(f"volumetotal={volumetotal}")
        print(f"fitcoff={wiener_mod.fitcoff}")
        print(f"rvalue={wiener_mod.rvalue}")


def test_wienerpass_multiproc(debug=False):
    """Test wienerpass with multiple processors (exercises the else branch)."""
    rng = np.random.RandomState(42)
    nvoxels = 5
    ntimepoints = 50
    fmri_data = rng.randn(nvoxels, ntimepoints) + 10.0  # mean > threshval
    lagtc_data = rng.randn(nvoxels, ntimepoints)

    # Set up the global arrays that wienerpass writes into
    import rapidtide.wiener as wiener_mod

    wiener_mod.meanvalue = np.zeros(nvoxels)
    wiener_mod.rvalue = np.zeros(nvoxels)
    wiener_mod.r2value = np.zeros(nvoxels)
    wiener_mod.fitcoff = np.zeros(nvoxels)
    wiener_mod.fitNorm = np.zeros(nvoxels)
    wiener_mod.datatoremove = np.zeros((nvoxels, ntimepoints))
    wiener_mod.filtereddata = np.zeros((nvoxels, ntimepoints))

    optiondict = {
        "nprocs": 4,
        "showprogressbar": False,
        "mp_chunksize": 10,
    }

    volumetotal = wienerpass(
        numspatiallocs=nvoxels,
        fmri_data=fmri_data,
        threshval=0.0,
        lagtc=lagtc_data,
        optiondict=optiondict,
        wienerdeconv=np.array([1.0]),
        wpeak=np.array([0.5]),
        resampref_y=np.array([1.0]),
    )

    assert volumetotal == nvoxels
    # Check that output arrays were populated for above-threshold voxels
    assert np.any(wiener_mod.fitcoff != 0)
    assert np.any(wiener_mod.rvalue != 0)

    if debug:
        print(f"volumetotal={volumetotal}")
        print(f"fitcoff={wiener_mod.fitcoff}")
        print(f"rvalue={wiener_mod.rvalue}")


def test_wienerpass_threshold_mask(debug=False):
    """Test that voxels below threshval are skipped."""
    rng = np.random.RandomState(42)
    nvoxels = 6
    ntimepoints = 50
    fmri_data = rng.randn(nvoxels, ntimepoints)
    # Make first 3 voxels high mean, last 3 low mean
    fmri_data[:3, :] += 20.0
    fmri_data[3:, :] -= 20.0
    lagtc_data = rng.randn(nvoxels, ntimepoints)

    import rapidtide.wiener as wiener_mod

    wiener_mod.meanvalue = np.zeros(nvoxels)
    wiener_mod.rvalue = np.zeros(nvoxels)
    wiener_mod.r2value = np.zeros(nvoxels)
    wiener_mod.fitcoff = np.zeros(nvoxels)
    wiener_mod.fitNorm = np.zeros(nvoxels)
    wiener_mod.datatoremove = np.zeros((nvoxels, ntimepoints))
    wiener_mod.filtereddata = np.zeros((nvoxels, ntimepoints))

    optiondict = {
        "nprocs": 1,
        "showprogressbar": False,
        "mp_chunksize": 10,
    }

    volumetotal = wienerpass(
        numspatiallocs=nvoxels,
        fmri_data=fmri_data,
        threshval=5.0,
        lagtc=lagtc_data,
        optiondict=optiondict,
        wienerdeconv=np.array([1.0]),
        wpeak=np.array([0.5]),
        resampref_y=np.array([1.0]),
    )

    assert volumetotal == nvoxels
    # Below-threshold voxels should remain at zero
    for v in range(3, 6):
        assert wiener_mod.fitcoff[v] == 0.0
        assert wiener_mod.rvalue[v] == 0.0
    # Above-threshold voxels should have been processed
    for v in range(3):
        assert wiener_mod.rvalue[v] != 0.0

    if debug:
        print(f"fitcoff: {wiener_mod.fitcoff}")


def test_wienerpass_consistency_with_procOneVoxelWiener(debug=False):
    """Verify wienerpass single-proc results match direct _procOneVoxelWiener calls."""
    rng = np.random.RandomState(42)
    nvoxels = 3
    ntimepoints = 60
    fmri_data = rng.randn(nvoxels, ntimepoints) + 10.0
    lagtc_data = rng.randn(nvoxels, ntimepoints)

    import rapidtide.wiener as wiener_mod

    wiener_mod.meanvalue = np.zeros(nvoxels)
    wiener_mod.rvalue = np.zeros(nvoxels)
    wiener_mod.r2value = np.zeros(nvoxels)
    wiener_mod.fitcoff = np.zeros(nvoxels)
    wiener_mod.fitNorm = np.zeros(nvoxels)
    wiener_mod.datatoremove = np.zeros((nvoxels, ntimepoints))
    wiener_mod.filtereddata = np.zeros((nvoxels, ntimepoints))

    optiondict = {
        "nprocs": 1,
        "showprogressbar": False,
        "mp_chunksize": 10,
    }

    wienerpass(
        numspatiallocs=nvoxels,
        fmri_data=fmri_data,
        threshval=0.0,
        lagtc=lagtc_data,
        optiondict=optiondict,
        wienerdeconv=np.array([1.0]),
        wpeak=np.array([0.5]),
        resampref_y=np.array([1.0]),
    )

    # Compare with direct calls
    for v in range(nvoxels):
        direct = _procOneVoxelWiener(v, lagtc_data[v, :], fmri_data[v, :].copy())
        assert abs(wiener_mod.fitcoff[v] - direct[4]) < 1e-10
        assert abs(wiener_mod.r2value[v] - direct[3]) < 1e-10
        np.testing.assert_allclose(wiener_mod.datatoremove[v, :], direct[6], atol=1e-10)
        np.testing.assert_allclose(wiener_mod.filtereddata[v, :], direct[7], atol=1e-10)

    if debug:
        print("wienerpass results match direct _procOneVoxelWiener calls")


# ==================== Main test entry point ====================


def test_wiener(debug=False):
    test_procOneVoxelWiener_basic(debug=debug)
    test_procOneVoxelWiener_voxel_index(debug=debug)
    test_procOneVoxelWiener_float32(debug=debug)
    test_procOneVoxelWiener_float64(debug=debug)
    test_procOneVoxelWiener_uncorrelated(debug=debug)
    test_procOneVoxelWiener_perfect_fit(debug=debug)
    test_procOneVoxelWiener_ratio(debug=debug)
    test_procOneVoxelWiener_residual_orthogonal(debug=debug)
    test_wienerpass_singleproc(debug=debug)
    test_wienerpass_multiproc(debug=debug)
    test_wienerpass_threshold_mask(debug=debug)
    test_wienerpass_consistency_with_procOneVoxelWiener(debug=debug)


if __name__ == "__main__":
    test_wiener(debug=True)
