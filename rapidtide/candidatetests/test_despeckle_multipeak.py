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

import numpy as np

import rapidtide.fit as tide_fit
from rapidtide.notreadyforprimetime.fitSimFuncMap import _build_peakdict_for_candidates
from rapidtide.simfuncfit import _find_and_try_peaks


def _make_multi_peak_corr(corrtimeaxis, peak_lags, peak_amplitudes, sigma=0.5):
    """Create a synthetic correlation function with multiple Gaussian peaks."""
    corr = np.zeros_like(corrtimeaxis)
    for lag, amp in zip(peak_lags, peak_amplitudes):
        corr += amp * np.exp(-0.5 * ((corrtimeaxis - lag) / sigma) ** 2)
    return corr


def test_build_peakdict_for_candidates(debug=False):
    """Test that _build_peakdict_for_candidates finds peaks in flagged voxels."""
    corrtimeaxis = np.linspace(-10, 10, 201)
    # Voxel 0: two peaks at lag=2 and lag=-3
    corr0 = _make_multi_peak_corr(corrtimeaxis, [2.0, -3.0], [0.8, 0.5])
    # Voxel 1: single peak at lag=0
    corr1 = _make_multi_peak_corr(corrtimeaxis, [0.0], [0.9])
    # Voxel 2: three peaks
    corr2 = _make_multi_peak_corr(corrtimeaxis, [1.0, -2.0, 5.0], [0.7, 0.6, 0.4])
    corrout = np.vstack([corr0, corr1, corr2])

    # Only voxels 0 and 2 are candidates
    candidate_mask = np.array([True, False, True], dtype=bool)

    peakdict = _build_peakdict_for_candidates(
        candidate_mask, corrout, corrtimeaxis, bipolar=False
    )

    # Should have entries for voxels 0 and 2, not 1
    assert "0" in peakdict
    assert "1" not in peakdict
    assert "2" in peakdict

    # Voxel 0 should have 2 peaks
    assert len(peakdict["0"]) == 2
    # Voxel 2 should have 3 peaks
    assert len(peakdict["2"]) == 3

    # Check peak format: [lag, strength, abs(strength)]
    for entry in peakdict["0"]:
        assert len(entry) == 3
        assert entry[2] == abs(entry[1])

    if debug:
        print(f"Voxel 0 peaks: {peakdict['0']}")
        print(f"Voxel 2 peaks: {peakdict['2']}")


def test_build_peakdict_empty_candidates(debug=False):
    """Test that _build_peakdict_for_candidates returns empty dict when no candidates."""
    corrtimeaxis = np.linspace(-10, 10, 201)
    corr0 = _make_multi_peak_corr(corrtimeaxis, [0.0], [0.5])
    corrout = corr0.reshape(1, -1)
    candidate_mask = np.array([False], dtype=bool)

    peakdict = _build_peakdict_for_candidates(
        candidate_mask, corrout, corrtimeaxis, bipolar=False
    )
    assert len(peakdict) == 0


def test_find_and_try_peaks_selects_nearest_to_target(debug=False):
    """Test that _find_and_try_peaks selects the peak nearest to target_lag."""
    from unittest.mock import MagicMock

    corrtimeaxis = np.linspace(-10, 10, 201)
    # Two peaks: one at lag=2.0 (strong), one at lag=-3.0 (weak)
    corr = _make_multi_peak_corr(corrtimeaxis, [2.0, -3.0], [0.8, 0.5])

    # Create a mock fitter
    fitter = MagicMock()
    fitter.corrtimeaxis = corrtimeaxis
    fitter.bipolar = False

    # When fit is called, return success for the peak nearest to target
    def mock_fit(corrfunc):
        # Return (maxindex, maxlag, maxval, maxsigma, maskval, failreason, peakstart, peakend)
        # Get the current guess from setguess calls
        guess = fitter.setguess.call_args[1].get("maxguess", 0.0)
        if abs(guess - 2.0) < 1.0:
            return (100, 2.0, 0.8, 0.5, 1, np.uint32(0), 95, 105)
        elif abs(guess - (-3.0)) < 1.0:
            return (50, -3.0, 0.5, 0.5, 1, np.uint32(0), 45, 55)
        else:
            return (0, 0.0, 0.0, 0.0, 0, np.uint32(0x40), 0, 0)

    fitter.fit = mock_fit

    # Target lag is 2.5 — should select the peak at 2.0 (nearest)
    result = _find_and_try_peaks(
        corr, fitter, target_lag=2.5, despeckle_thresh=5.0
    )
    maxlag = result[1]
    failreason = result[5]
    assert failreason == 0
    assert abs(maxlag - 2.0) < 0.1

    if debug:
        print(f"Selected lag: {maxlag}, failreason: {failreason}")


def test_find_and_try_peaks_falls_back_to_second_peak(debug=False):
    """Test that when the nearest peak fails, the next nearest is tried."""
    from unittest.mock import MagicMock

    corrtimeaxis = np.linspace(-10, 10, 201)
    # Two peaks: one at lag=2.0, one at lag=-1.0
    corr = _make_multi_peak_corr(corrtimeaxis, [2.0, -1.0], [0.8, 0.6])

    fitter = MagicMock()
    fitter.corrtimeaxis = corrtimeaxis
    fitter.bipolar = False

    call_count = [0]

    def mock_fit(corrfunc):
        call_count[0] += 1
        guess = fitter.setguess.call_args[1].get("maxguess", 0.0)
        # First peak (nearest to target=1.5, which is lag=2.0) fails
        if abs(guess - 2.0) < 1.0:
            return (100, 2.0, 0.8, 0.5, 0, np.uint32(0x0100), 95, 105)  # FML_FITAMPLOW
        # Second peak (lag=-1.0) succeeds
        elif abs(guess - (-1.0)) < 1.0:
            return (70, -1.0, 0.6, 0.5, 1, np.uint32(0), 65, 75)
        else:
            return (0, 0.0, 0.0, 0.0, 0, np.uint32(0x40), 0, 0)

    fitter.fit = mock_fit

    result = _find_and_try_peaks(
        corr, fitter, target_lag=1.5, despeckle_thresh=5.0
    )
    maxlag = result[1]
    failreason = result[5]
    assert failreason == 0
    assert abs(maxlag - (-1.0)) < 0.1
    assert call_count[0] >= 2  # Should have tried at least 2 peaks

    if debug:
        print(f"Selected lag: {maxlag}, calls: {call_count[0]}")


def test_find_and_try_peaks_accepts_width_only_nearmiss(debug=False):
    """Test that width-only failures are accepted as near-misses."""
    from unittest.mock import MagicMock

    corrtimeaxis = np.linspace(-10, 10, 201)
    corr = _make_multi_peak_corr(corrtimeaxis, [3.0], [0.7])

    fitter = MagicMock()
    fitter.corrtimeaxis = corrtimeaxis
    fitter.bipolar = False

    def mock_fit(corrfunc):
        # Only width flag set (FML_FITWIDTHHIGH = 0x0800)
        return (120, 3.0, 0.7, 2.0, 1, np.uint32(0x0800), 115, 125)

    fitter.fit = mock_fit

    result = _find_and_try_peaks(
        corr, fitter, target_lag=3.0, despeckle_thresh=5.0
    )
    maxlag = result[1]
    failreason = result[5]
    # Should accept width-only failure as near-miss
    assert abs(maxlag - 3.0) < 0.1
    assert failreason == 0x0800

    if debug:
        print(f"Selected lag: {maxlag}, failreason: {hex(failreason)}")


def test_find_and_try_peaks_no_peaks_falls_back(debug=False):
    """Test fallback when no peaks are found (flat correlation)."""
    from unittest.mock import MagicMock

    corrtimeaxis = np.linspace(-10, 10, 201)
    # Flat correlation — no peaks
    corr = np.ones_like(corrtimeaxis) * 0.1

    fitter = MagicMock()
    fitter.corrtimeaxis = corrtimeaxis
    fitter.bipolar = False

    def mock_fit(corrfunc):
        return (100, 0.0, 0.1, 1.0, 1, np.uint32(0), 95, 105)

    fitter.fit = mock_fit

    result = _find_and_try_peaks(
        corr, fitter, target_lag=0.0, despeckle_thresh=5.0
    )
    # Should still return a result (fell back to single guess at target)
    assert result is not None
    assert result[5] == 0  # failreason

    if debug:
        print(f"Fallback result: lag={result[1]}, failreason={result[5]}")


def test_icm_with_local_peakdict(debug=False):
    """Test that ICM works with locally-built peakdict (non-hybrid mode)."""
    from rapidtide.notreadyforprimetime.fitSimFuncMap import (
        _optimize_despeckle_labels_icm,
    )

    corrtimeaxis = np.linspace(-10, 10, 201)
    # Voxel 1 has wrong peak; neighbors are at lag=0
    corr_wrong = _make_multi_peak_corr(corrtimeaxis, [0.0, 7.0], [0.6, 0.8])
    corrout = np.zeros((3, len(corrtimeaxis)))
    corrout[0] = _make_multi_peak_corr(corrtimeaxis, [0.0], [0.9])
    corrout[1] = corr_wrong
    corrout[2] = _make_multi_peak_corr(corrtimeaxis, [0.0], [0.9])

    # Build local peakdict for the candidate voxel
    candidate_mask_valid = np.array([False, True, False], dtype=bool)
    local_peakdict = _build_peakdict_for_candidates(
        candidate_mask_valid, corrout, corrtimeaxis, bipolar=False
    )

    assert "1" in local_peakdict
    assert len(local_peakdict["1"]) >= 2  # Should find both peaks

    # Run ICM: voxel 1 (lag=7) should be corrected toward 0 (neighbor consensus)
    lagmap_flat = np.array([0.0, 7.0, 0.0], dtype=float)
    candidate_mask_flat = np.array([False, True, False], dtype=bool)
    validmask_flat = np.array([True, True, True], dtype=bool)
    validvoxels = np.array([0, 1, 2], dtype=int)

    optimized, info = _optimize_despeckle_labels_icm(
        lagmap_flat,
        candidate_mask_flat,
        validmask_flat,
        validvoxels,
        local_peakdict,
        nativespaceshape=(3,),
        max_candidates=3,
        max_iters=4,
        data_weight=0.5,
        smooth_weight=1.0,
    )

    # Voxel 1 should have moved from 7.0 toward 0.0
    assert abs(optimized[1]) < abs(lagmap_flat[1])

    if debug:
        print(f"Original lag[1]={lagmap_flat[1]}, optimized={optimized[1]}")
        print(f"ICM info: {info}")


def test_despeckle_multipeak(debug=False):
    """Run all sub-tests."""
    test_build_peakdict_for_candidates(debug=debug)
    test_build_peakdict_empty_candidates(debug=debug)
    test_find_and_try_peaks_selects_nearest_to_target(debug=debug)
    test_find_and_try_peaks_falls_back_to_second_peak(debug=debug)
    test_find_and_try_peaks_accepts_width_only_nearmiss(debug=debug)
    test_find_and_try_peaks_no_peaks_falls_back(debug=debug)
    test_icm_with_local_peakdict(debug=debug)


if __name__ == "__main__":
    test_despeckle_multipeak(debug=True)
