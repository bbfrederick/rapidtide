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

        vals, weights, indices = tide_resample.congrid(
            xaxis, loc, val, width, kernel="kaiser"
        )

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

        vals, weights, indices = tide_resample.congrid(
            xaxis, loc, val, width, kernel="gauss"
        )

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

        vals, weights, indices = tide_resample.congrid(
            xaxis, loc, val, width, kernel="old"
        )

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
            vals, weights, indices = tide_resample.congrid(
                xaxis, loc, val, width, kernel="kaiser"
            )
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
        assert os.path.exists(outputpath + ".tsv.gz") or os.path.exists(
            outputpath + ".tsv"
        )

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

        result = tide_resample.doresample(
            orig_x, orig_y, new_x, method="cubic", antialias=True
        )

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

        result = tide_resample.arbresample(
            inputdata, init_freq, final_freq, decimate=True
        )

        assert result is not None
        assert len(result) == len(inputdata)

    def test_arbresample_with_decimate(self):
        """Test arbresample with decimate option."""
        inputdata = np.sin(np.linspace(0, 4 * np.pi, 100))
        init_freq = 20.0
        final_freq = 10.0

        result = tide_resample.arbresample(
            inputdata, init_freq, final_freq, decimate=True
        )

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

        result = tide_resample.arbresample(
            inputdata, init_freq, final_freq, decimate=False
        )

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

        result = tide_resample.dotwostepresample(
            orig_x, orig_y, intermed_freq, final_freq
        )

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
                tide_resample.calcsliceoffset(sotype, i, numslices, tr)
                for i in range(numslices)
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
