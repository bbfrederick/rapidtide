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
"""
Tests for the resample module - covers all resampling and time shifting functions.
"""

import os
import tempfile
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import rapidtide.io as tide_io
import rapidtide.resample as tide_resample
from rapidtide.tests.utils import create_dir, get_test_temp_path

# ============================================================================
# Helper functions for creating test data
# ============================================================================


def create_sine_wave(length=100, freq=1.0, samplerate=10.0, phase=0.0):
    """Create a simple sine wave for testing."""
    t = np.arange(length) / samplerate
    return t, np.sin(2 * np.pi * freq * t + phase)


def create_test_timecourse_file(filepath, length=100, samplerate=10.0):
    """Create a test BIDS TSV timecourse file."""
    t, data = create_sine_wave(length=length, samplerate=samplerate)
    tide_io.writebidstsv(
        filepath,
        data,
        samplerate,
        starttime=0.0,
        columns=["signal"],
    )
    return t, data


# ============================================================================
# Tests for congrid function
# ============================================================================


class TestCongrid:
    """Tests for the congrid function."""

    def test_congrid_basic_kaiser(self):
        """Test basic congrid operation with kaiser kernel."""
        xaxis = np.linspace(0, 10, 100)
        loc = 5.5
        val = 1.0
        width = 2.0

        vals, weights, indices = tide_resample.congrid(xaxis, loc, val, width, kernel="kaiser")

        assert vals is not None
        assert weights is not None
        assert indices is not None
        assert len(vals) == len(weights)
        assert len(vals) == len(indices)
        # Values should be proportional to weights
        assert np.allclose(vals, val * weights)

    def test_congrid_basic_gauss(self):
        """Test basic congrid operation with gaussian kernel."""
        xaxis = np.linspace(0, 10, 100)
        loc = 5.5
        val = 2.0
        width = 2.5

        vals, weights, indices = tide_resample.congrid(xaxis, loc, val, width, kernel="gauss")

        assert vals is not None
        assert weights is not None
        assert indices is not None
        assert np.allclose(vals, val * weights)

    def test_congrid_old_kernel(self):
        """Test congrid with old kernel."""
        xaxis = np.linspace(0, 10, 100)
        loc = 5.0
        val = 1.0
        width = 2.0

        vals, weights, indices = tide_resample.congrid(xaxis, loc, val, width, kernel="old")

        assert vals is not None
        assert weights is not None
        assert indices is not None

    def test_congrid_different_widths(self):
        """Test congrid with different valid width values."""
        xaxis = np.linspace(0, 10, 100)
        loc = 5.0
        val = 1.0

        # Valid widths are half-integral values between 1.5 and 5.0
        valid_widths = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        for width in valid_widths:
            vals, weights, indices = tide_resample.congrid(xaxis, loc, val, width, kernel="kaiser")
            assert vals is not None, f"Failed for width={width}"

    def test_congrid_cyclic(self):
        """Test congrid with cyclic boundary conditions."""
        xaxis = np.linspace(0, 10, 100)
        # Location near the end
        loc = 9.9
        val = 1.0
        width = 2.0

        vals, weights, indices = tide_resample.congrid(
            xaxis, loc, val, width, kernel="kaiser", cyclic=True
        )

        assert vals is not None
        # With cyclic=True, indices should wrap around
        assert len(indices) > 0

    def test_congrid_non_cyclic(self):
        """Test congrid with non-cyclic boundary conditions."""
        xaxis = np.linspace(0, 10, 100)
        loc = 5.0
        val = 1.0
        width = 2.0

        vals, weights, indices = tide_resample.congrid(
            xaxis, loc, val, width, kernel="kaiser", cyclic=False
        )

        assert vals is not None

    def test_congrid_cache_behavior(self):
        """Test congrid caching behavior."""
        xaxis = np.linspace(0, 10, 100)
        loc = 5.25
        val = 1.0
        width = 2.0

        # Call twice with same offset - should use cache
        vals1, weights1, indices1 = tide_resample.congrid(
            xaxis, loc, val, width, kernel="kaiser", cache=True
        )
        vals2, weights2, indices2 = tide_resample.congrid(
            xaxis, loc, val, width, kernel="kaiser", cache=True
        )

        assert np.allclose(vals1, vals2)
        assert np.allclose(weights1, weights2)


# ============================================================================
# Tests for FastResampler class
# ============================================================================


class TestFastResampler:
    """Tests for the FastResampler class."""

    def test_init_basic(self):
        """Test basic FastResampler initialization."""
        timeaxis = np.linspace(0, 10, 100)
        timecourse = np.sin(2 * np.pi * 0.5 * timeaxis)

        resampler = tide_resample.FastResampler(
            timeaxis, timecourse, padtime=5.0, upsampleratio=10
        )

        assert resampler.timeaxis is not None
        assert resampler.timecourse is not None
        assert resampler.hires_x is not None
        assert resampler.hires_y is not None
        assert len(resampler.hires_x) > len(timeaxis)

    def test_init_univariate_method(self):
        """Test FastResampler with univariate method."""
        timeaxis = np.linspace(0, 10, 100)
        timecourse = np.sin(2 * np.pi * 0.5 * timeaxis)

        resampler = tide_resample.FastResampler(
            timeaxis, timecourse, padtime=5.0, upsampleratio=10, method="univariate"
        )

        assert resampler.method == "univariate"
        assert resampler.hires_y is not None

    def test_init_poly_method(self):
        """Test FastResampler with poly method."""
        timeaxis = np.linspace(0, 10, 101)
        timecourse = np.sin(2 * np.pi * 0.5 * timeaxis)

        resampler = tide_resample.FastResampler(
            timeaxis, timecourse, padtime=1.0, upsampleratio=2, method="poly"
        )

        assert resampler.method == "poly"
        assert resampler.hires_y is not None

    def test_init_fourier_method(self):
        """Test FastResampler with fourier method."""
        timeaxis = np.linspace(0, 10, 101)
        timecourse = np.sin(2 * np.pi * 0.5 * timeaxis)

        resampler = tide_resample.FastResampler(
            timeaxis, timecourse, padtime=1.0, upsampleratio=2, method="fourier"
        )

        assert resampler.method == "fourier"
        assert resampler.hires_y is not None

    def test_getdata(self):
        """Test the getdata method."""
        timeaxis = np.linspace(0, 10, 100)
        timecourse = np.sin(2 * np.pi * 0.5 * timeaxis)

        resampler = tide_resample.FastResampler(
            timeaxis, timecourse, padtime=5.0, upsampleratio=10
        )

        ta, tc, hx, hy, inv_step = resampler.getdata()

        assert np.allclose(ta, timeaxis)
        assert np.allclose(tc, timecourse)
        assert len(hx) == len(hy)
        assert inv_step == pytest.approx(1.0 / resampler.initstep)

    def test_info(self, capsys):
        """Test the info method."""
        timeaxis = np.linspace(0, 10, 100)
        timecourse = np.sin(2 * np.pi * 0.5 * timeaxis)

        resampler = tide_resample.FastResampler(
            timeaxis, timecourse, padtime=5.0, upsampleratio=10
        )

        resampler.info()
        captured = capsys.readouterr()

        assert "upsampleratio" in captured.out
        assert "padtime" in captured.out
        assert "initstep" in captured.out
        assert "method" in captured.out

    def test_info_with_prefix(self, capsys):
        """Test the info method with prefix."""
        timeaxis = np.linspace(0, 10, 100)
        timecourse = np.sin(2 * np.pi * 0.5 * timeaxis)

        resampler = tide_resample.FastResampler(
            timeaxis, timecourse, padtime=5.0, upsampleratio=10
        )

        resampler.info(prefix="  ")
        captured = capsys.readouterr()

        # Check that prefix is applied
        assert "  self.upsampleratio" in captured.out

    def test_save(self, tmp_path):
        """Test the save method."""
        timeaxis = np.linspace(0, 10, 100)
        timecourse = np.sin(2 * np.pi * 0.5 * timeaxis)

        resampler = tide_resample.FastResampler(
            timeaxis, timecourse, padtime=5.0, upsampleratio=10
        )

        outputpath = str(tmp_path / "test_resampler_output")
        resampler.save(outputpath)

        # Check that files were created
        assert os.path.exists(outputpath + ".tsv.gz") or os.path.exists(outputpath + ".tsv")

    def test_yfromx(self):
        """Test the yfromx method."""
        timeaxis = np.linspace(0, 10, 100)
        timecourse = np.sin(2 * np.pi * 0.5 * timeaxis)

        resampler = tide_resample.FastResampler(
            timeaxis, timecourse, padtime=5.0, upsampleratio=100
        )

        # Request values at new time points within the valid range
        new_timeaxis = np.linspace(1, 9, 50)
        result = resampler.yfromx(new_timeaxis)

        assert len(result) == len(new_timeaxis)
        # The result should approximately match the original sine wave
        expected = np.sin(2 * np.pi * 0.5 * new_timeaxis)
        assert np.allclose(result, expected, atol=0.1)

    def test_yfromx_preserves_signal(self):
        """Test that yfromx approximately preserves the original signal."""
        timeaxis = np.linspace(0, 10, 100)
        timecourse = np.sin(2 * np.pi * 0.5 * timeaxis)

        resampler = tide_resample.FastResampler(
            timeaxis, timecourse, padtime=5.0, upsampleratio=100
        )

        # Request values at original time points
        result = resampler.yfromx(timeaxis)

        # Should closely match original
        assert np.allclose(result, timecourse, atol=0.1)


# ============================================================================
# Tests for FastResamplerFromFile function
# ============================================================================


class TestFastResamplerFromFile:
    """Tests for the FastResamplerFromFile function."""

    def test_from_file_basic(self, tmp_path):
        """Test creating FastResampler from a BIDS TSV file."""
        filepath = str(tmp_path / "test_timecourse")
        t, data = create_test_timecourse_file(filepath, length=100, samplerate=10.0)

        # The file will have .tsv.gz extension added
        actual_file = filepath + ".tsv.gz"
        if not os.path.exists(actual_file):
            actual_file = filepath + ".tsv"

        resampler = tide_resample.FastResamplerFromFile(actual_file)

        assert resampler is not None
        assert resampler.hires_y is not None

    def test_from_file_with_kwargs(self, tmp_path):
        """Test FastResamplerFromFile with additional kwargs."""
        filepath = str(tmp_path / "test_timecourse2")
        t, data = create_test_timecourse_file(filepath, length=100, samplerate=10.0)

        actual_file = filepath + ".tsv.gz"
        if not os.path.exists(actual_file):
            actual_file = filepath + ".tsv"

        resampler = tide_resample.FastResamplerFromFile(
            actual_file, padtime=10.0, upsampleratio=50
        )

        assert resampler.padtime == 10.0
        assert resampler.upsampleratio == 50


# ============================================================================
# Tests for doresample function
# ============================================================================


class TestDoresample:
    """Tests for the doresample function."""

    def test_doresample_cubic(self):
        """Test doresample with cubic interpolation."""
        orig_x = np.linspace(0, 10, 100)
        orig_y = np.sin(2 * np.pi * 0.5 * orig_x)
        new_x = np.linspace(0, 10, 200)

        result = tide_resample.doresample(orig_x, orig_y, new_x, method="cubic")

        assert result is not None
        assert len(result) == len(new_x)

    def test_doresample_quadratic(self):
        """Test doresample with quadratic interpolation."""
        orig_x = np.linspace(0, 10, 100)
        orig_y = np.sin(2 * np.pi * 0.5 * orig_x)
        new_x = np.linspace(0, 10, 200)

        result = tide_resample.doresample(orig_x, orig_y, new_x, method="quadratic")

        assert result is not None
        assert len(result) == len(new_x)

    def test_doresample_univariate(self):
        """Test doresample with univariate spline interpolation."""
        orig_x = np.linspace(0, 10, 100)
        orig_y = np.sin(2 * np.pi * 0.5 * orig_x)
        new_x = np.linspace(0, 10, 200)

        result = tide_resample.doresample(orig_x, orig_y, new_x, method="univariate")

        assert result is not None
        assert len(result) == len(new_x)

    def test_doresample_upsample(self):
        """Test upsampling (more output points than input)."""
        orig_x = np.linspace(0, 10, 50)
        orig_y = np.sin(2 * np.pi * 0.5 * orig_x)
        new_x = np.linspace(0, 10, 200)

        result = tide_resample.doresample(orig_x, orig_y, new_x, method="cubic")

        assert len(result) == 200

    def test_doresample_downsample(self):
        """Test downsampling (fewer output points than input)."""
        orig_x = np.linspace(0, 10, 200)
        orig_y = np.sin(2 * np.pi * 0.5 * orig_x)
        new_x = np.linspace(0, 10, 50)

        result = tide_resample.doresample(orig_x, orig_y, new_x, method="cubic")

        assert len(result) == 50

    def test_doresample_with_padding(self):
        """Test doresample with padding."""
        orig_x = np.linspace(0, 10, 100)
        orig_y = np.sin(2 * np.pi * 0.5 * orig_x)
        new_x = np.linspace(0, 10, 200)

        result = tide_resample.doresample(
            orig_x, orig_y, new_x, method="cubic", padlen=10, padtype="reflect"
        )

        assert result is not None
        assert len(result) == len(new_x)

    def test_doresample_preserves_signal(self):
        """Test that resampling approximately preserves signal characteristics."""
        orig_x = np.linspace(0, 10, 100)
        orig_y = np.sin(2 * np.pi * 0.5 * orig_x)
        new_x = np.linspace(0, 10, 200)

        result = tide_resample.doresample(orig_x, orig_y, new_x, method="univariate")

        # The resampled signal should match expected sine wave
        expected = np.sin(2 * np.pi * 0.5 * new_x)
        assert np.allclose(result, expected, atol=0.1)

    def test_doresample_with_antialias(self):
        """Test doresample with antialiasing filter."""
        # Use longer data to satisfy padding requirements for antialiasing
        orig_x = np.linspace(0, 100, 2000)
        orig_y = np.sin(2 * np.pi * 0.05 * orig_x)
        new_x = np.linspace(0, 100, 500)

        result = tide_resample.doresample(orig_x, orig_y, new_x, method="cubic", antialias=True)

        assert result is not None
        assert len(result) == len(new_x)


# ============================================================================
# Tests for arbresample function
# ============================================================================


class TestArbresample:
    """Tests for the arbresample function."""

    def test_arbresample_upsample(self):
        """Test upsampling with arbresample."""
        # Use longer data to satisfy padding requirements
        inputdata = np.sin(np.linspace(0, 40 * np.pi, 2000))
        init_freq = 10.0
        final_freq = 20.0

        result = tide_resample.arbresample(inputdata, init_freq, final_freq)

        assert result is not None
        # Output should have approximately twice as many points
        assert len(result) == pytest.approx(len(inputdata) * 2, rel=0.1)

    def test_arbresample_downsample(self):
        """Test downsampling with arbresample."""
        # Use longer data to satisfy padding requirements
        inputdata = np.sin(np.linspace(0, 40 * np.pi, 2000))
        init_freq = 20.0
        final_freq = 10.0

        result = tide_resample.arbresample(inputdata, init_freq, final_freq)

        assert result is not None
        # Output should have approximately half as many points
        assert len(result) == pytest.approx(len(inputdata) / 2, rel=0.1)

    def test_arbresample_same_freq(self):
        """Test arbresample when initial and final frequencies are the same."""
        inputdata = np.sin(np.linspace(0, 4 * np.pi, 100))
        init_freq = 10.0
        final_freq = 10.0

        result = tide_resample.arbresample(inputdata, init_freq, final_freq, decimate=True)

        assert result is not None
        assert len(result) == len(inputdata)

    def test_arbresample_with_decimate(self):
        """Test arbresample with decimate option."""
        inputdata = np.sin(np.linspace(0, 4 * np.pi, 100))
        init_freq = 20.0
        final_freq = 10.0

        result = tide_resample.arbresample(inputdata, init_freq, final_freq, decimate=True)

        assert result is not None
        assert len(result) < len(inputdata)

    def test_arbresample_with_antialias(self):
        """Test arbresample with antialiasing."""
        inputdata = np.sin(np.linspace(0, 4 * np.pi, 100))
        init_freq = 20.0
        final_freq = 10.0

        result = tide_resample.arbresample(
            inputdata, init_freq, final_freq, decimate=True, antialias=True
        )

        assert result is not None

    def test_arbresample_without_antialias(self):
        """Test arbresample without antialiasing."""
        inputdata = np.sin(np.linspace(0, 4 * np.pi, 100))
        init_freq = 20.0
        final_freq = 10.0

        result = tide_resample.arbresample(
            inputdata, init_freq, final_freq, decimate=True, antialias=False
        )

        assert result is not None

    def test_arbresample_two_step(self):
        """Test arbresample using two-step resampling (decimate=False)."""
        # Use longer data to satisfy padding requirements
        inputdata = np.sin(np.linspace(0, 40 * np.pi, 2000))
        init_freq = 10.0
        final_freq = 15.0

        result = tide_resample.arbresample(inputdata, init_freq, final_freq, decimate=False)

        assert result is not None


# ============================================================================
# Tests for upsample function
# ============================================================================


class TestUpsample:
    """Tests for the upsample function."""

    def test_upsample_basic(self):
        """Test basic upsampling."""
        # Use longer data to satisfy filtering requirements
        inputdata = np.sin(np.linspace(0, 40 * np.pi, 2000))
        Fs_init = 10.0
        Fs_higher = 20.0

        result = tide_resample.upsample(inputdata, Fs_init, Fs_higher)

        assert result is not None
        # Output should have approximately twice as many points
        assert len(result) >= len(inputdata)

    def test_upsample_integer_factor(self):
        """Test upsampling with integer factor option."""
        # Use longer data to satisfy filtering requirements
        inputdata = np.sin(np.linspace(0, 40 * np.pi, 2000))
        Fs_init = 10.0
        Fs_higher = 20.0

        result = tide_resample.upsample(inputdata, Fs_init, Fs_higher, intfac=True)

        assert result is not None
        # With intfac=True, should have exactly 2x points
        assert len(result) == 2 * len(inputdata)

    def test_upsample_with_filter(self):
        """Test upsampling with filtering."""
        # Use longer data to satisfy filtering requirements
        inputdata = np.sin(np.linspace(0, 40 * np.pi, 2000))
        Fs_init = 10.0
        Fs_higher = 40.0

        result = tide_resample.upsample(inputdata, Fs_init, Fs_higher, dofilt=True)

        assert result is not None

    def test_upsample_without_filter(self):
        """Test upsampling without filtering."""
        inputdata = np.sin(np.linspace(0, 4 * np.pi, 100))
        Fs_init = 10.0
        Fs_higher = 40.0

        result = tide_resample.upsample(inputdata, Fs_init, Fs_higher, dofilt=False)

        assert result is not None

    def test_upsample_preserves_signal(self):
        """Test that upsampling preserves signal characteristics."""
        # Create a low-frequency sine wave - use longer data
        orig_x = np.linspace(0, 100, 2000)
        inputdata = np.sin(2 * np.pi * 0.05 * orig_x)
        Fs_init = 20.0
        Fs_higher = 100.0

        result = tide_resample.upsample(inputdata, Fs_init, Fs_higher, dofilt=True)

        # The upsampled signal should still be approximately sinusoidal
        assert result is not None
        assert len(result) > len(inputdata)


# ============================================================================
# Tests for dotwostepresample function
# ============================================================================


class TestDotwostepresample:
    """Tests for the dotwostepresample function."""

    def test_dotwostepresample_basic(self):
        """Test basic two-step resampling."""
        # Use longer data to satisfy padding requirements
        orig_x = np.linspace(0, 100, 2000)
        orig_y = np.sin(2 * np.pi * 0.05 * orig_x)
        intermed_freq = 50.0
        final_freq = 20.0

        result = tide_resample.dotwostepresample(orig_x, orig_y, intermed_freq, final_freq)

        assert result is not None
        # Final length should be approximately (duration * final_freq)
        duration = orig_x[-1] - orig_x[0]
        expected_len = int(duration * final_freq)
        assert len(result) == pytest.approx(expected_len, rel=0.1)

    def test_dotwostepresample_with_antialias(self):
        """Test two-step resampling with antialiasing."""
        # Use longer data to satisfy padding requirements
        orig_x = np.linspace(0, 100, 2000)
        orig_y = np.sin(2 * np.pi * 0.05 * orig_x)
        intermed_freq = 50.0
        final_freq = 20.0

        result = tide_resample.dotwostepresample(
            orig_x, orig_y, intermed_freq, final_freq, antialias=True
        )

        assert result is not None

    def test_dotwostepresample_without_antialias(self):
        """Test two-step resampling without antialiasing."""
        orig_x = np.linspace(0, 10, 100)
        orig_y = np.sin(2 * np.pi * 0.5 * orig_x)
        intermed_freq = 50.0
        final_freq = 20.0

        result = tide_resample.dotwostepresample(
            orig_x, orig_y, intermed_freq, final_freq, antialias=False
        )

        assert result is not None


# ============================================================================
# Tests for calcsliceoffset function
# ============================================================================


class TestCalcsliceoffset:
    """Tests for the calcsliceoffset function."""

    def test_sotype_0_none(self):
        """Test slice timing type 0 (none)."""
        result = tide_resample.calcsliceoffset(0, 5, 32, 2.0)
        assert result == 0.0

    def test_sotype_2_regular_down(self):
        """Test slice timing type 2 (regular down)."""
        numslices = 32
        tr = 2.0
        slicenum = 5

        result = tide_resample.calcsliceoffset(2, slicenum, numslices, tr)

        expected = (numslices - slicenum - 1) * (tr / numslices)
        assert result == pytest.approx(expected)

    def test_sotype_3_unsupported(self):
        """Test slice timing type 3 (slice order file - not supported)."""
        result = tide_resample.calcsliceoffset(3, 5, 32, 2.0)
        assert result == 0.0

    def test_sotype_4_unsupported(self):
        """Test slice timing type 4 (slice timings file - not supported)."""
        result = tide_resample.calcsliceoffset(4, 5, 32, 2.0)
        assert result == 0.0

    def test_sotype_5_standard_interleaved_even_slice(self):
        """Test slice timing type 5 (standard interleaved) with even slice."""
        numslices = 16
        tr = 1.5
        slicenum = 4  # even slice

        result = tide_resample.calcsliceoffset(5, slicenum, numslices, tr)

        # For even slice: (tr / numslices) * (slicenum / 2)
        expected = (tr / numslices) * (slicenum / 2)
        assert result == pytest.approx(expected)

    def test_sotype_5_standard_interleaved_odd_slice(self):
        """Test slice timing type 5 (standard interleaved) with odd slice."""
        numslices = 16
        tr = 1.5
        slicenum = 3  # odd slice

        result = tide_resample.calcsliceoffset(5, slicenum, numslices, tr)

        # For odd slice: (tr / numslices) * ((numslices + 1) / 2 + (slicenum - 1) / 2)
        expected = (tr / numslices) * ((numslices + 1) / 2 + (slicenum - 1) / 2)
        assert result == pytest.approx(expected)

    def test_sotype_6_siemens_interleaved_odd_numslices(self):
        """Test slice timing type 6 (Siemens interleaved) with odd number of slices."""
        numslices = 31  # odd
        tr = 2.0
        slicenum = 4

        result = tide_resample.calcsliceoffset(6, slicenum, numslices, tr)

        # Odd numslices, even slicenum: (tr / numslices) * (slicenum / 2)
        expected = (tr / numslices) * (slicenum / 2)
        assert result == pytest.approx(expected)

    def test_sotype_6_siemens_interleaved_even_numslices(self):
        """Test slice timing type 6 (Siemens interleaved) with even number of slices."""
        numslices = 32  # even
        tr = 2.0
        slicenum = 5  # odd

        result = tide_resample.calcsliceoffset(6, slicenum, numslices, tr)

        # Even numslices, odd slicenum: (tr / numslices) * ((slicenum - 1) / 2)
        expected = (tr / numslices) * ((slicenum - 1) / 2)
        assert result == pytest.approx(expected)

    def test_sotype_7_multiband(self):
        """Test slice timing type 7 (Siemens multiband interleaved)."""
        numslices = 32
        tr = 2.0
        multiband = 2
        slicenum = 5

        result = tide_resample.calcsliceoffset(7, slicenum, numslices, tr, multiband)

        assert result is not None
        assert isinstance(result, float)

    def test_calcsliceoffset_all_slices(self):
        """Test that all slices get valid offsets."""
        numslices = 16
        tr = 2.0

        for sotype in [0, 2, 5, 6]:
            offsets = [
                tide_resample.calcsliceoffset(sotype, i, numslices, tr) for i in range(numslices)
            ]
            # All offsets should be non-negative and less than TR
            for offset in offsets:
                assert 0.0 <= offset < tr, f"Invalid offset {offset} for sotype {sotype}"


# ============================================================================
# Tests for timeshift function
# ============================================================================


class TestTimeshift:
    """Tests for the timeshift function."""

    def test_timeshift_basic(self):
        """Test basic time shifting."""
        inputtc = np.sin(np.linspace(0, 4 * np.pi, 100))
        shifttrs = 2.0
        padtrs = 10

        shifted_y, shifted_weights, full_shifted, full_weights = tide_resample.timeshift(
            inputtc, shifttrs, padtrs
        )

        assert shifted_y is not None
        assert len(shifted_y) == len(inputtc)
        assert len(shifted_weights) == len(inputtc)

    def test_timeshift_positive_shift(self):
        """Test positive time shift (delay)."""
        inputtc = np.sin(np.linspace(0, 4 * np.pi, 100))
        shifttrs = 5.0
        padtrs = 10

        shifted_y, shifted_weights, full_shifted, full_weights = tide_resample.timeshift(
            inputtc, shifttrs, padtrs
        )

        assert shifted_y is not None

    def test_timeshift_negative_shift(self):
        """Test negative time shift (advance)."""
        inputtc = np.sin(np.linspace(0, 4 * np.pi, 100))
        shifttrs = -5.0
        padtrs = 10

        shifted_y, shifted_weights, full_shifted, full_weights = tide_resample.timeshift(
            inputtc, shifttrs, padtrs
        )

        assert shifted_y is not None

    def test_timeshift_zero_shift(self):
        """Test zero time shift (no change)."""
        inputtc = np.sin(np.linspace(0, 4 * np.pi, 100))
        shifttrs = 0.0
        padtrs = 10

        shifted_y, shifted_weights, full_shifted, full_weights = tide_resample.timeshift(
            inputtc, shifttrs, padtrs
        )

        # With zero shift, output should closely match input
        assert np.allclose(shifted_y, inputtc, atol=1e-10)

    def test_timeshift_preserves_length(self):
        """Test that timeshift preserves the signal length."""
        inputtc = np.random.rand(100)
        shifttrs = 3.0
        padtrs = 20

        shifted_y, shifted_weights, full_shifted, full_weights = tide_resample.timeshift(
            inputtc, shifttrs, padtrs
        )

        assert len(shifted_y) == len(inputtc)

    def test_timeshift_returns_full_padded(self):
        """Test that timeshift returns full padded arrays."""
        inputtc = np.sin(np.linspace(0, 4 * np.pi, 100))
        shifttrs = 2.0
        padtrs = 10

        shifted_y, shifted_weights, full_shifted, full_weights = tide_resample.timeshift(
            inputtc, shifttrs, padtrs
        )

        # Full arrays should be longer due to padding
        expected_padded_len = len(inputtc) + 2 * padtrs
        assert len(full_shifted) == expected_padded_len
        assert len(full_weights) == expected_padded_len


# ============================================================================
# Tests for timewarp function
# ============================================================================


class TestTimewarp:
    """Tests for the timewarp function."""

    def test_timewarp_basic(self):
        """Test basic time warping."""
        orig_x = np.linspace(0, 10, 100)
        orig_y = np.sin(2 * np.pi * 0.5 * orig_x)
        timeoffset = np.random.normal(0, 0.1, 100)

        result = tide_resample.timewarp(orig_x, orig_y, timeoffset)

        assert result is not None
        assert len(result) == len(orig_x)

    def test_timewarp_with_demean(self):
        """Test time warping with demeaning."""
        orig_x = np.linspace(0, 10, 100)
        orig_y = np.sin(2 * np.pi * 0.5 * orig_x)
        timeoffset = np.random.normal(0.5, 0.1, 100)  # Non-zero mean offset

        result = tide_resample.timewarp(orig_x, orig_y, timeoffset, demean=True)

        assert result is not None

    def test_timewarp_without_demean(self):
        """Test time warping without demeaning."""
        orig_x = np.linspace(0, 10, 100)
        orig_y = np.sin(2 * np.pi * 0.5 * orig_x)
        timeoffset = np.random.normal(0, 0.1, 100)

        result = tide_resample.timewarp(orig_x, orig_y, timeoffset, demean=False)

        assert result is not None

    def test_timewarp_zero_offset(self):
        """Test time warping with zero offset."""
        orig_x = np.linspace(0, 10, 100)
        orig_y = np.sin(2 * np.pi * 0.5 * orig_x)
        timeoffset = np.zeros(100)

        result = tide_resample.timewarp(orig_x, orig_y, timeoffset)

        # With zero offset, result should match original
        assert np.allclose(result, orig_y, atol=0.01)

    def test_timewarp_constant_offset(self):
        """Test time warping with constant offset."""
        orig_x = np.linspace(0, 10, 100)
        orig_y = np.sin(2 * np.pi * 0.5 * orig_x)
        timeoffset = np.ones(100) * 0.1

        result = tide_resample.timewarp(orig_x, orig_y, timeoffset)

        assert result is not None
        assert len(result) == len(orig_x)

    def test_timewarp_different_methods(self):
        """Test time warping with different interpolation methods."""
        orig_x = np.linspace(0, 10, 100)
        orig_y = np.sin(2 * np.pi * 0.5 * orig_x)
        timeoffset = np.random.normal(0, 0.05, 100)

        for method in ["univariate", "cubic", "quadratic"]:
            result = tide_resample.timewarp(orig_x, orig_y, timeoffset, method=method)
            assert result is not None, f"Failed for method {method}"
            assert len(result) == len(orig_x), f"Wrong length for method {method}"


# ============================================================================
# Integration tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple resampling operations."""

    def test_resample_chain(self):
        """Test chaining multiple resampling operations."""
        # Create original signal - use longer data to satisfy antialiasing filter padding
        orig_x = np.linspace(0, 100, 2000)
        orig_y = np.sin(2 * np.pi * 0.05 * orig_x)

        # Upsample
        upsampled = tide_resample.upsample(orig_y, 20.0, 80.0)

        # Then downsample back
        downsampled = tide_resample.arbresample(
            upsampled, 80.0, 20.0, decimate=True, antialias=True
        )

        # Result should be similar to original
        assert len(downsampled) == pytest.approx(len(orig_y), rel=0.1)

    def test_fast_resampler_workflow(self):
        """Test complete FastResampler workflow."""
        # Create signal
        timeaxis = np.linspace(0, 10, 100)
        timecourse = np.sin(2 * np.pi * 0.5 * timeaxis)

        # Create resampler
        resampler = tide_resample.FastResampler(
            timeaxis, timecourse, padtime=5.0, upsampleratio=100
        )

        # Get data at different time points
        new_times = np.linspace(1, 9, 200)
        result = resampler.yfromx(new_times)

        # Verify result quality
        expected = np.sin(2 * np.pi * 0.5 * new_times)
        correlation = np.corrcoef(result, expected)[0, 1]
        assert correlation > 0.99

    def test_timeshift_then_warp(self):
        """Test time shifting followed by warping."""
        inputtc = np.sin(np.linspace(0, 4 * np.pi, 100))

        # First shift
        shifted_y, _, _, _ = tide_resample.timeshift(inputtc, 2.0, 10)

        # Then warp
        orig_x = np.linspace(0, 10, 100)
        timeoffset = np.random.normal(0, 0.05, 100)
        warped = tide_resample.timewarp(orig_x, shifted_y, timeoffset)

        assert warped is not None
        assert len(warped) == len(inputtc)


class TestCongridEdgeCases:
    """Edge cases, error handling, and reporting paths in congrid."""

    def test_illegal_kernel_exits(self):
        """An unrecognized kernel name is fatal rather than silently ignored."""
        xaxis = np.linspace(0.0, 10.0, 101)
        with pytest.raises(SystemExit):
            tide_resample.congrid(xaxis, 5.0, 1.0, 2.5, kernel="notakernel", cache=False)

    def test_illegal_width_exits(self):
        """Widths outside the supported half-integral range are fatal."""
        xaxis = np.linspace(0.0, 10.0, 101)
        for badwidth in [1.0, 5.5, 2.3]:
            with pytest.raises(SystemExit):
                tide_resample.congrid(xaxis, 5.0, 1.0, badwidth, kernel="kaiser", cache=False)

    def test_out_of_range_location_noncyclic(self, capsys):
        """A location off the end of a noncyclic axis is reported."""
        xaxis = np.linspace(0.0, 10.0, 101)
        tide_resample.congrid(xaxis, 50.0, 1.0, 2.5, kernel="kaiser", cyclic=False, cache=False)
        assert "not in range" in capsys.readouterr().out

    def test_kernel_change_reinitializes_cache(self, capsys):
        """Switching kernel or width discards the cached kernel values."""
        xaxis = np.linspace(0.0, 10.0, 101)
        tide_resample.congrid(xaxis, 5.0, 1.0, 2.5, kernel="kaiser", debug=True)
        tide_resample.congrid(
            xaxis, 5.0, 1.0, 3.5, kernel="gauss", debug=True, onlykeynotices=False
        )
        theoutput = capsys.readouterr().out
        assert "(re)initializing congridyvals" in theoutput

    def test_cyclic_wraparound_both_ends(self):
        """A location past either end of a cyclic axis wraps around to the other end.

        The offset has to exceed half a grid step for the wrap to trigger, since below that
        the nearest grid point is still the end point itself.
        """
        xaxis = np.linspace(0.0, 10.0, 101)
        xstep = xaxis[1] - xaxis[0]
        for loc in [xaxis[-1] + 0.8 * xstep, xaxis[0] - 0.8 * xstep]:
            vals, weights, indices = tide_resample.congrid(
                xaxis, loc, 1.0, 2.5, kernel="kaiser", cyclic=True, cache=False
            )
            # indices stay inside the axis because they are taken modulo its length
            assert np.all(indices >= 0)
            assert np.all(indices < len(xaxis))

    def test_offset_out_of_range_noncyclic_exits(self):
        """A noncyclic location off the axis end by a partial step is unusable and exits.

        The offset is taken modulo one grid step, so only a location whose distance past
        the end has a fractional part above half a step lands outside the legal range -
        being merely far away is not enough.
        """
        xaxis = np.linspace(0.0, 10.0, 101)
        # 0.7 of a grid step past the final sample, which no wrap can bring back in range
        with pytest.raises(SystemExit):
            tide_resample.congrid(
                xaxis, 10.07, 1.0, 2.5, kernel="kaiser", cyclic=False, cache=False
            )

    def test_old_kernel_debug_reporting(self, capsys):
        """The legacy kernel path also reports its indices and weights when verbose."""
        xaxis = np.linspace(0.0, 10.0, 101)
        tide_resample.congrid(
            xaxis, 5.3, 1.0, 2.0, kernel="old", cache=False, debug=True, onlykeynotices=False
        )
        assert "center, offset, indices, yvals" in capsys.readouterr().out

    def test_gauss_and_kaiser_kernels_agree_on_total_weight(self):
        """Both supported kernels return one weight per contributing index."""
        xaxis = np.linspace(0.0, 10.0, 101)
        for kernel in ["kaiser", "gauss"]:
            vals, weights, indices = tide_resample.congrid(
                xaxis, 5.13, 2.0, 3.0, kernel=kernel, cache=False
            )
            assert len(vals) == len(weights) == len(indices)
            np.testing.assert_allclose(vals, 2.0 * weights)

    def test_old_kernel_path(self):
        """The legacy 'old' kernel bypasses the half-integral width check."""
        xaxis = np.linspace(0.0, 10.0, 101)
        vals, weights, indices = tide_resample.congrid(
            xaxis, 5.0, 1.0, 1.0, kernel="old", cache=False, debug=True
        )
        assert len(vals) == len(weights) == len(indices)

    def test_debug_reporting(self, capsys):
        """The verbose reporting path prints the computed indices and weights."""
        xaxis = np.linspace(0.0, 10.0, 101)
        tide_resample.congrid(
            xaxis, 5.27, 1.0, 2.5, kernel="kaiser", cache=False, debug=True, onlykeynotices=False
        )
        assert "center, offset, indices, yvals" in capsys.readouterr().out


class TestFastResamplerEdgeCases:
    """Reporting and failure paths in FastResampler."""

    def test_constructor_debug(self, capsys):
        """The constructor reports its axis configuration when asked."""
        timeaxis = np.linspace(0.0, 10.0, 101)
        data = np.sin(timeaxis)
        tide_resample.FastResampler(timeaxis, data, debug=True)
        assert "FastResampler __init__:" in capsys.readouterr().out

    def test_yfromx_debug(self, capsys):
        """yfromx reports the axis limits it is working between."""
        timeaxis = np.linspace(0.0, 10.0, 101)
        resampler = tide_resample.FastResampler(timeaxis, np.sin(timeaxis))
        resampler.yfromx(np.linspace(1.0, 9.0, 50), debug=True)
        assert "yfromx called with following parameters" in capsys.readouterr().out

    def test_yfromx_out_of_bounds_exits(self):
        """Requesting times beyond the padded range is fatal rather than silently wrong."""
        timeaxis = np.linspace(0.0, 10.0, 101)
        resampler = tide_resample.FastResampler(timeaxis, np.sin(timeaxis), padtime=1.0)
        with pytest.raises(SystemExit):
            resampler.yfromx(np.linspace(0.0, 1.0e6, 20))


class TestFastResamplerFromFileErrors:
    """Input validation in FastResamplerFromFile."""

    def test_multicolumn_file_rejected(self, tmp_path):
        """A file with more than one column is ambiguous and is rejected."""
        filepath = str(tmp_path / "twocol")
        samplerate, npoints = 10.0, 100
        t = np.arange(npoints) / samplerate
        data = np.vstack([np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)])
        tide_io.writebidstsv(
            filepath, data, samplerate, starttime=0.0, columns=["first", "second"]
        )
        actual_file = filepath + ".tsv.gz"
        if not os.path.exists(actual_file):
            actual_file = filepath + ".tsv"
        with pytest.raises(ValueError, match="Multiple columns"):
            tide_resample.FastResamplerFromFile(actual_file)

    def test_missing_column_names_rejected(self, tmp_path):
        """A file the reader returns no column names for is rejected.

        The reader synthesizes an index when a sidecar simply omits the column list, so this
        guard is exercised by stubbing the reader rather than by writing a malformed file.
        """
        filepath = str(tmp_path / "nocols")
        create_test_timecourse_file(filepath, length=100, samplerate=10.0)
        actual_file = filepath + ".tsv.gz"
        if not os.path.exists(actual_file):
            actual_file = filepath + ".tsv"

        def _nocolumns(*args, **kwargs):
            return (10.0, 0.0, None, np.zeros((1, 100)), False, None, None)

        with patch.object(tide_io, "readbidstsv", side_effect=_nocolumns):
            with pytest.raises(ValueError, match="No column names"):
                tide_resample.FastResamplerFromFile(actual_file)

    def test_single_column_file_debug(self, tmp_path, capsys):
        """A single column file loads, and the debug path reports what it read."""
        filepath = str(tmp_path / "onecol")
        create_test_timecourse_file(filepath, length=100, samplerate=10.0)
        actual_file = filepath + ".tsv.gz"
        if not os.path.exists(actual_file):
            actual_file = filepath + ".tsv"
        resampler = tide_resample.FastResamplerFromFile(actual_file, debug=True)
        assert "FastResamplerFromFile:" in capsys.readouterr().out
        assert resampler is not None


class TestArbresampleEdgeCases:
    """Method dispatch, up/downsampling, and reporting in arbresample."""

    def test_invalid_method_rejected(self):
        """An unknown interpolation method is rejected up front."""
        data = np.sin(np.linspace(0.0, 6.0, 100))
        with pytest.raises(ValueError, match="invalid interpolation method"):
            tide_resample.arbresample(data, 10.0, 5.0, method="notamethod")

    def test_downsample_reports_stages(self, capsys):
        """Downsampling reports each stage when run verbosely."""
        # long enough that the antialiasing filter's padding fits inside the data
        data = np.sin(np.linspace(0.0, 40 * np.pi, 2000))
        result = tide_resample.arbresample(data, 20.0, 5.0, debug=True)
        theoutput = capsys.readouterr().out
        assert "arbresample - initial points:" in theoutput
        assert len(result) > 0

    def test_upsample_reports_stages(self, capsys):
        """Upsampling reports its point counts when run verbosely."""
        data = np.sin(np.linspace(0.0, 40 * np.pi, 2000))
        result = tide_resample.arbresample(data, 5.0, 20.0, debug=True)
        assert "arbresample" in capsys.readouterr().out
        assert len(result) > 0

    def test_decimate_method(self):
        """The decimate path produces a shorter output at the requested ratio."""
        data = np.sin(np.linspace(0.0, 40 * np.pi, 2000))
        result = tide_resample.arbresample(data, 20.0, 5.0, decimate=True)
        assert len(result) < len(data)

    def test_decimate_upsample_only(self, capsys):
        """With decimate set and a higher target frequency, only upsampling happens."""
        data = np.sin(np.linspace(0.0, 40 * np.pi, 2000))
        result = tide_resample.arbresample(data, 5.0, 20.0, decimate=True, debug=True)
        assert len(result) > len(data)
        assert "arbresample - upsampled points:" in capsys.readouterr().out

    def test_decimate_same_frequency_passes_through(self, capsys):
        """Resampling to the same frequency returns the input untouched."""
        data = np.sin(np.linspace(0.0, 40 * np.pi, 2000))
        result = tide_resample.arbresample(data, 10.0, 10.0, decimate=True, debug=True)
        assert result is data
        assert "arbresample - final points:" in capsys.readouterr().out

    def test_decimate_downsample_with_antialias(self, capsys):
        """The antialiased decimation path reports its intermediate frequency."""
        data = np.sin(np.linspace(0.0, 40 * np.pi, 2000))
        result = tide_resample.arbresample(
            data, 20.0, 6.0, decimate=True, antialias=True, debug=True
        )
        theoutput = capsys.readouterr().out
        assert "Hz, then decimating by," in theoutput
        assert "arbresample - downsampled points:" in theoutput
        assert len(result) < len(data)

    def test_decimate_downsample_without_antialias(self):
        """The interpolation based decimation path also shortens the data."""
        data = np.sin(np.linspace(0.0, 40 * np.pi, 2000))
        result = tide_resample.arbresample(data, 20.0, 6.0, decimate=True, antialias=False)
        assert len(result) < len(data)


class TestUpsampleErrors:
    """Argument validation and reporting in upsample."""

    def test_target_frequency_must_be_higher(self):
        """Upsampling to a lower frequency is a usage error."""
        data = np.sin(np.linspace(0.0, 6.0, 100))
        with pytest.raises(SystemExit):
            tide_resample.upsample(data, 10.0, 5.0)

    def test_equal_frequency_rejected(self):
        """Upsampling to the same frequency is also rejected."""
        data = np.sin(np.linspace(0.0, 6.0, 100))
        with pytest.raises(SystemExit):
            tide_resample.upsample(data, 10.0, 10.0)

    def test_debug_reports_timing(self, capsys):
        """The verbose path reports how long the resampling took."""
        data = np.sin(np.linspace(0.0, 40 * np.pi, 2000))
        tide_resample.upsample(data, 10.0, 20.0, debug=True)
        assert "upsampling took" in capsys.readouterr().out


class TestDotwostepresampleErrors:
    """Argument validation and reporting in dotwostepresample."""

    def test_intermediate_frequency_must_be_higher(self):
        """The intermediate frequency has to exceed the final one."""
        orig_x = np.linspace(0.0, 10.0, 100)
        orig_y = np.sin(orig_x)
        with pytest.raises(SystemExit):
            tide_resample.dotwostepresample(orig_x, orig_y, 5.0, 10.0)

    def test_debug_reports_timing(self, capsys):
        """The verbose path reports the antialiasing and downsampling times."""
        orig_x = np.linspace(0.0, 100.0, 2000)
        orig_y = np.sin(2 * np.pi * 0.05 * orig_x)
        tide_resample.dotwostepresample(orig_x, orig_y, 50.0, 20.0, debug=True)
        theoutput = capsys.readouterr().out
        assert "downsampling took" in theoutput


class TestCalcsliceoffsetRegularUp:
    """Slice timing for ascending acquisition order (sotype 1)."""

    def test_regular_up_ramps_with_slice_number(self):
        """Ascending order gives each slice a time proportional to its index.

        This branch tested ``type == 1`` - the Python builtin - instead of ``sotype == 1``,
        so it never fired and ascending order silently returned no correction at all.
        """
        numslices, tr = 8, 2.0
        offsets = [
            tide_resample.calcsliceoffset(1, slicenum, numslices, tr)
            for slicenum in range(numslices)
        ]
        expected = [slicenum * (tr / numslices) for slicenum in range(numslices)]
        np.testing.assert_allclose(offsets, expected)
        # and it must differ from "no correction at all"
        assert any(offset != 0.0 for offset in offsets)

    def test_regular_up_matches_documented_example(self):
        """The value quoted in the docstring is the value produced."""
        assert tide_resample.calcsliceoffset(1, 5, 32, 2.0) == pytest.approx(0.3125)

    def test_regular_up_is_the_mirror_of_regular_down(self):
        """Ascending and descending order are reverses of one another."""
        numslices, tr = 8, 2.0
        up = [tide_resample.calcsliceoffset(1, s, numslices, tr) for s in range(numslices)]
        down = [tide_resample.calcsliceoffset(2, s, numslices, tr) for s in range(numslices)]
        np.testing.assert_allclose(up, down[::-1])

    def test_multiband_interleaved_odd_shots(self):
        """The odd-shot-count branch of Siemens multiband ordering is reachable."""
        numslices, tr, multiband = 15, 3.0, 3
        offsets = [
            tide_resample.calcsliceoffset(7, s, numslices, tr, multiband) for s in range(numslices)
        ]
        assert len(offsets) == numslices
        assert all(0.0 <= offset < tr for offset in offsets)

    def test_multiband_interleaved_even_shots(self):
        """The even-shot-count branch of Siemens multiband ordering is reachable."""
        numslices, tr, multiband = 16, 3.0, 2
        offsets = [
            tide_resample.calcsliceoffset(7, s, numslices, tr, multiband) for s in range(numslices)
        ]
        assert len(offsets) == numslices
        assert all(np.isfinite(offset) for offset in offsets)

    def test_siemens_interleaved_odd_slicecount(self):
        """Siemens interleaved ordering with an odd slice count uses its own formula."""
        numslices, tr = 15, 3.0
        offsets = [tide_resample.calcsliceoffset(6, s, numslices, tr) for s in range(numslices)]
        assert len(offsets) == numslices
        # odd slice counts start with the even-numbered slices, so slice 1 comes late
        assert offsets[1] > offsets[0]
        assert all(np.isfinite(offset) for offset in offsets)


class TestResampleReportingPaths:
    """Verbose reporting paths in the remaining resample routines."""

    def test_doresample_debug(self, capsys):
        """doresample reports its padding when run verbosely."""
        orig_x = np.linspace(0.0, 10.0, 100)
        orig_y = np.sin(orig_x)
        new_x = np.linspace(1.0, 9.0, 50)
        result = tide_resample.doresample(orig_x, orig_y, new_x, debug=True)
        assert len(result) == 50
        assert "padlen=" in capsys.readouterr().out

    def test_timeshift_debug(self, capsys):
        """timeshift reports its padded length when run verbosely."""
        data = np.sin(np.linspace(0.0, 6.0, 64))
        shifted, weights, paddedshifted, paddedweights = tide_resample.timeshift(
            data, 3.0, 10, debug=True
        )
        assert len(shifted) == len(data)
        assert "timesshift:" in capsys.readouterr().out

    def test_timeshift_odd_length_input(self):
        """An odd number of points still round trips through the FFT based shifter."""
        data = np.sin(np.linspace(0.0, 6.0, 65))
        shifted, weights, paddedshifted, paddedweights = tide_resample.timeshift(data, 2.0, 8)
        assert len(shifted) == len(data)
        assert np.all(np.isfinite(shifted))

    def test_timeshift_phase_vector_overshoot_is_trimmed(self):
        """A padded length where the phase ramp overshoots by one sample is trimmed.

        The phase modulation vector is built with ``np.arange``, which for some lengths
        emits one extra sample because of floating point rounding; 59 points padded by one
        TR on each side gives such a length (61).
        """
        thelen, padtrs = 59, 1
        fftlen = thelen + 2 * padtrs
        # confirm this length really is one of the overshooting cases
        assert len(np.arange(0.0, 2.0 * np.pi, 2.0 * np.pi / float(fftlen))) > fftlen
        data = np.sin(np.linspace(0.0, 6.0, thelen))
        shifted, weights, paddedshifted, paddedweights = tide_resample.timeshift(data, 2.0, padtrs)
        assert len(shifted) == thelen
        assert np.all(np.isfinite(shifted))

    def test_fastresampler_init_plot(self):
        """The constructor's diagnostic plot path runs under a headless backend."""
        timeaxis = np.linspace(0.0, 10.0, 101)
        resampler = tide_resample.FastResampler(timeaxis, np.sin(timeaxis), doplot=True)
        assert resampler.hires_y is not None
        plt.close("all")

    def test_fastresampler_yfromx_plot(self):
        """The yfromx diagnostic plot path runs under a headless backend."""
        timeaxis = np.linspace(0.0, 10.0, 101)
        resampler = tide_resample.FastResampler(timeaxis, np.sin(timeaxis))
        out = resampler.yfromx(np.linspace(1.0, 9.0, 50), doplot=True)
        assert len(out) == 50
        plt.close("all")

    def test_timeshift_plot(self):
        """The timeshift diagnostic plot path runs under a headless backend."""
        data = np.sin(np.linspace(0.0, 6.0, 64))
        shifted, weights, paddedshifted, paddedweights = tide_resample.timeshift(
            data, 3.0, 10, doplot=True
        )
        assert len(shifted) == len(data)
        plt.close("all")

    def test_timewarp_debug(self, capsys):
        """timewarp reports the mean delay it removed and the peak deviation."""
        orig_x = np.linspace(0.0, 10.0, 100)
        orig_y = np.sin(orig_x)
        timeoffset = np.linspace(0.2, 0.4, 100)
        warped = tide_resample.timewarp(orig_x, orig_y, timeoffset, debug=True)
        theoutput = capsys.readouterr().out
        assert len(warped) == len(orig_y)
        assert "maximum deviation in samples:" in theoutput


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
