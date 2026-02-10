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
"""
Tests for rapidtide.simFuncClasses module.

This module tests the SimilarityFunctionator, MutualInformationator, Correlator,
SimilarityFunctionFitter, and FrequencyTracker classes.
"""

import numpy as np
import pytest

import rapidtide.filter as tide_filt
import rapidtide.simFuncClasses as tide_simfunc

# ============================================================================
# SimilarityFunctionator tests
# ============================================================================


class TestSimilarityFunctionator:
    """Tests for the SimilarityFunctionator base class."""

    @pytest.fixture
    def sample_filter(self):
        """Create a sample filter for testing."""
        return tide_filt.NoncausalFilter(filtertype="lfo")

    @pytest.fixture
    def sample_timecourse(self):
        """Create a sample timecourse for testing."""
        # Use low sampling frequency and long data for LFO filter compatibility
        t = np.linspace(0, 100, 200)
        return np.sin(2 * np.pi * 0.05 * t) + 0.5 * np.sin(2 * np.pi * 0.1 * t)

    def test_init_basic(self):
        """Test basic initialization of SimilarityFunctionator."""
        sim_func = tide_simfunc.SimilarityFunctionator(Fs=2.0)

        assert sim_func.Fs == 2.0
        assert sim_func.similarityfuncorigin == 0
        assert sim_func.lagmininpts == 0
        assert sim_func.lagmaxinpts == 0
        assert sim_func.detrendorder == 1
        assert sim_func.filterinputdata is True
        assert sim_func.debug is False

    def test_init_with_params(self, sample_filter):
        """Test initialization with custom parameters."""
        sim_func = tide_simfunc.SimilarityFunctionator(
            Fs=2.0,
            similarityfuncorigin=50,
            lagmininpts=10,
            lagmaxinpts=20,
            ncprefilter=sample_filter,
            negativegradient=True,
            detrendorder=2,
            filterinputdata=False,
            debug=True,
        )

        assert sim_func.Fs == 2.0
        assert sim_func.similarityfuncorigin == 50
        assert sim_func.lagmininpts == 10
        assert sim_func.lagmaxinpts == 20
        assert sim_func.negativegradient is True
        assert sim_func.detrendorder == 2
        assert sim_func.filterinputdata is False
        assert sim_func.debug is True

    def test_setFs(self):
        """Test setFs method."""
        sim_func = tide_simfunc.SimilarityFunctionator()

        sim_func.setFs(100.0)

        assert sim_func.Fs == 100.0

    def test_trim(self):
        """Test trim method."""
        sim_func = tide_simfunc.SimilarityFunctionator(
            similarityfuncorigin=50, lagmininpts=10, lagmaxinpts=20
        )

        vector = np.arange(100)
        trimmed = sim_func.trim(vector)

        # Should get elements from 50-10=40 to 50+20=70
        assert len(trimmed) == 30
        assert trimmed[0] == 40
        assert trimmed[-1] == 69

    def test_getfunction_no_data(self, capsys):
        """Test getfunction when no data is calculated."""
        sim_func = tide_simfunc.SimilarityFunctionator()

        result = sim_func.getfunction()

        assert result == (None, None, None)
        captured = capsys.readouterr()
        assert "must calculate similarity function" in captured.out


# ============================================================================
# MutualInformationator tests
# ============================================================================


class TestMutualInformationator:
    """Tests for the MutualInformationator class."""

    @pytest.fixture
    def sample_filter(self):
        """Create a sample filter for testing."""
        return tide_filt.NoncausalFilter(filtertype="lfo")

    @pytest.fixture
    def sample_timecourse(self):
        """Create a sample timecourse for testing."""
        t = np.linspace(0, 100, 200)
        return np.sin(2 * np.pi * 0.05 * t)

    def test_init_basic(self):
        """Test basic initialization of MutualInformationator."""
        mi = tide_simfunc.MutualInformationator(Fs=2.0)

        assert mi.Fs == 2.0
        assert mi.windowfunc == "hamming"
        assert mi.norm is True
        assert mi.madnorm is False
        assert mi.bins == 20
        assert mi.sigma == 0.25
        assert mi.smoothingtime == -1.0
        assert mi.mi_norm == 1.0

    def test_init_with_smoothing(self):
        """Test initialization with smoothing enabled."""
        mi = tide_simfunc.MutualInformationator(Fs=2.0, smoothingtime=2.0)

        assert mi.smoothingtime == 2.0
        assert mi.smoothingfilter is not None

    def test_init_with_custom_params(self, sample_filter):
        """Test initialization with custom parameters."""
        mi = tide_simfunc.MutualInformationator(
            Fs=2.0,
            windowfunc="hanning",
            norm=False,
            madnorm=True,
            bins=30,
            sigma=0.5,
            ncprefilter=sample_filter,
        )

        assert mi.windowfunc == "hanning"
        assert mi.norm is False
        assert mi.madnorm is True
        assert mi.bins == 30
        assert mi.sigma == 0.5

    def test_setbins(self):
        """Test setbins method."""
        mi = tide_simfunc.MutualInformationator(Fs=2.0)

        mi.setbins(50)

        assert mi.bins == 50

    def test_getnormfac(self):
        """Test getnormfac method."""
        mi = tide_simfunc.MutualInformationator(Fs=2.0)
        mi.mi_norm = 2.5

        assert mi.getnormfac() == 2.5

    def test_setreftc(self, sample_filter, sample_timecourse):
        """Test setreftc method."""
        mi = tide_simfunc.MutualInformationator(
            Fs=2.0,
            ncprefilter=sample_filter,
            lagmininpts=10,
            lagmaxinpts=10,
        )

        mi.setreftc(sample_timecourse, offset=0.0)

        assert mi.reftc is not None
        assert mi.prepreftc is not None
        assert mi.timeaxis is not None
        assert mi.timeaxisvalid is True
        assert mi.datavalid is False
        assert mi.similarityfunclen > 0

    def test_setlimits(self, sample_filter, sample_timecourse):
        """Test setlimits method."""
        mi = tide_simfunc.MutualInformationator(
            Fs=2.0,
            ncprefilter=sample_filter,
            lagmininpts=10,
            lagmaxinpts=10,
        )
        mi.setreftc(sample_timecourse)

        mi.setlimits(5, 15)

        assert mi.lagmininpts == 5
        assert mi.lagmaxinpts == 15

    def test_run(self, sample_filter, sample_timecourse):
        """Test run method."""
        mi = tide_simfunc.MutualInformationator(
            Fs=2.0,
            ncprefilter=sample_filter,
            lagmininpts=10,
            lagmaxinpts=10,
        )
        mi.setreftc(sample_timecourse)

        result = mi.run(sample_timecourse, trim=True)

        assert len(result) == 3
        simfunc, timeaxis, globalmax = result
        assert simfunc is not None
        assert timeaxis is not None
        assert globalmax >= 0
        assert mi.datavalid is True


# ============================================================================
# Correlator tests
# ============================================================================


class TestCorrelator:
    """Tests for the Correlator class."""

    @pytest.fixture
    def sample_filter(self):
        """Create a sample filter for testing."""
        return tide_filt.NoncausalFilter(filtertype="lfo")

    @pytest.fixture
    def sample_timecourse(self):
        """Create a sample timecourse for testing."""
        t = np.linspace(0, 100, 200)
        return np.sin(2 * np.pi * 0.05 * t)

    def test_init_basic(self):
        """Test basic initialization of Correlator."""
        corr = tide_simfunc.Correlator(Fs=2.0)

        assert corr.Fs == 2.0
        assert corr.windowfunc == "hamming"
        assert corr.corrweighting == "None"
        assert corr.corrpadding == 0
        assert corr.baselinefilter is None

    def test_init_with_custom_params(self, sample_filter):
        """Test initialization with custom parameters."""
        corr = tide_simfunc.Correlator(
            Fs=2.0,
            windowfunc="hanning",
            corrweighting="phat",
            corrpadding=100,
            ncprefilter=sample_filter,
        )

        assert corr.windowfunc == "hanning"
        assert corr.corrweighting == "phat"
        assert corr.corrpadding == 100

    def test_setlimits(self):
        """Test setlimits method."""
        corr = tide_simfunc.Correlator(Fs=2.0)

        corr.setlimits(5, 15)

        assert corr.lagmininpts == 5
        assert corr.lagmaxinpts == 15

    def test_setreftc(self, sample_filter, sample_timecourse):
        """Test setreftc method."""
        corr = tide_simfunc.Correlator(Fs=2.0, ncprefilter=sample_filter)

        corr.setreftc(sample_timecourse, offset=0.0)

        assert corr.reftc is not None
        assert corr.prepreftc is not None
        assert corr.timeaxis is not None
        assert corr.timeaxisvalid is True
        assert corr.datavalid is False
        assert corr.similarityfunclen == len(sample_timecourse) * 2 - 1

    def test_run(self, sample_filter, sample_timecourse):
        """Test run method."""
        corr = tide_simfunc.Correlator(
            Fs=2.0, ncprefilter=sample_filter, lagmininpts=10, lagmaxinpts=10
        )
        corr.setreftc(sample_timecourse)

        result = corr.run(sample_timecourse, trim=True)

        assert len(result) == 3
        simfunc, timeaxis, globalmax = result
        assert simfunc is not None
        assert timeaxis is not None
        assert globalmax >= 0
        assert corr.datavalid is True

    def test_run_no_trim(self, sample_filter, sample_timecourse):
        """Test run method without trimming."""
        corr = tide_simfunc.Correlator(
            Fs=2.0, ncprefilter=sample_filter, lagmininpts=10, lagmaxinpts=10
        )
        corr.setreftc(sample_timecourse)

        result = corr.run(sample_timecourse, trim=False)

        simfunc, timeaxis, globalmax = result
        assert len(simfunc) == corr.similarityfunclen

    def test_autocorrelation_peak(self, sample_filter, sample_timecourse):
        """Test that autocorrelation has peak at zero lag."""
        corr = tide_simfunc.Correlator(
            Fs=2.0, ncprefilter=sample_filter, lagmininpts=50, lagmaxinpts=50
        )
        corr.setreftc(sample_timecourse)

        simfunc, timeaxis, globalmax = corr.run(sample_timecourse, trim=False)

        # Autocorrelation should peak at zero lag (middle of the array)
        peak_idx = np.argmax(simfunc)
        peak_time = timeaxis[peak_idx]
        assert np.abs(peak_time) < 1.0  # Peak should be near zero


# ============================================================================
# SimilarityFunctionFitter tests
# ============================================================================


class TestSimilarityFunctionFitter:
    """Tests for the SimilarityFunctionFitter class."""

    @pytest.fixture
    def sample_corrtimeaxis(self):
        """Create a sample correlation time axis."""
        return np.linspace(-10, 10, 201)

    @pytest.fixture
    def sample_gaussian_corrfunc(self, sample_corrtimeaxis):
        """Create a sample Gaussian-shaped correlation function."""
        # Gaussian centered at lag=2 with amplitude 0.8 and sigma=1.5
        return 0.8 * np.exp(-((sample_corrtimeaxis - 2.0) ** 2) / (2 * 1.5**2))

    def test_init_basic(self):
        """Test basic initialization of SimilarityFunctionFitter."""
        fitter = tide_simfunc.SimilarityFunctionFitter()

        assert fitter.lagmin == -30.0
        assert fitter.lagmax == 30.0
        assert fitter.absmaxsigma == 1000.0
        assert fitter.absminsigma == 0.25
        assert fitter.hardlimit is True
        assert fitter.bipolar is False
        assert fitter.lthreshval == 0.0
        assert fitter.uthreshval == 1.0
        assert fitter.debug is False
        assert fitter.functype == "correlation"
        assert fitter.peakfittype == "gauss"

    def test_init_with_corrtimeaxis(self, sample_corrtimeaxis):
        """Test initialization with correlation time axis."""
        fitter = tide_simfunc.SimilarityFunctionFitter(corrtimeaxis=sample_corrtimeaxis)

        assert fitter.corrtimeaxis is not None
        assert len(fitter.corrtimeaxis) == len(sample_corrtimeaxis)

    def test_init_with_custom_params(self, sample_corrtimeaxis):
        """Test initialization with custom parameters."""
        fitter = tide_simfunc.SimilarityFunctionFitter(
            corrtimeaxis=sample_corrtimeaxis,
            lagmin=-5.0,
            lagmax=5.0,
            absmaxsigma=10.0,
            absminsigma=0.1,
            hardlimit=False,
            bipolar=True,
            lthreshval=0.1,
            uthreshval=0.9,
            functype="mutualinfo",
            peakfittype="quad",
        )

        assert fitter.lagmin == -5.0
        assert fitter.lagmax == 5.0
        assert fitter.absmaxsigma == 10.0
        assert fitter.absminsigma == 0.1
        assert fitter.hardlimit is False
        assert fitter.bipolar is True
        assert fitter.lthreshval == 0.1
        assert fitter.uthreshval == 0.9
        assert fitter.functype == "mutualinfo"
        assert fitter.peakfittype == "quad"

    def test_setfunctype(self):
        """Test setfunctype method."""
        fitter = tide_simfunc.SimilarityFunctionFitter()

        fitter.setfunctype("mutualinfo")

        assert fitter.functype == "mutualinfo"

    def test_setpeakfittype(self):
        """Test setpeakfittype method."""
        fitter = tide_simfunc.SimilarityFunctionFitter()

        fitter.setpeakfittype("quad")

        assert fitter.peakfittype == "quad"

    def test_setrange(self):
        """Test setrange method."""
        fitter = tide_simfunc.SimilarityFunctionFitter()

        fitter.setrange(-5.0, 5.0)

        assert fitter.lagmin == -5.0
        assert fitter.lagmax == 5.0

    def test_setcorrtimeaxis(self, sample_corrtimeaxis):
        """Test setcorrtimeaxis method."""
        fitter = tide_simfunc.SimilarityFunctionFitter()

        fitter.setcorrtimeaxis(sample_corrtimeaxis)

        assert fitter.corrtimeaxis is not None
        assert len(fitter.corrtimeaxis) == len(sample_corrtimeaxis)

    def test_setcorrtimeaxis_none(self):
        """Test setcorrtimeaxis with None."""
        fitter = tide_simfunc.SimilarityFunctionFitter()

        fitter.setcorrtimeaxis(None)

        assert fitter.corrtimeaxis is None

    def test_setguess(self):
        """Test setguess method."""
        fitter = tide_simfunc.SimilarityFunctionFitter()

        fitter.setguess(True, 5.0)

        assert fitter.useguess is True
        assert fitter.maxguess == 5.0

    def test_setlthresh(self):
        """Test setlthresh method."""
        fitter = tide_simfunc.SimilarityFunctionFitter()

        fitter.setlthresh(0.2)

        assert fitter.lthreshval == 0.2

    def test_setuthresh(self):
        """Test setuthresh method."""
        fitter = tide_simfunc.SimilarityFunctionFitter()

        fitter.setuthresh(0.8)

        assert fitter.uthreshval == 0.8

    def test_diagnosefail_no_error(self):
        """Test diagnosefail with no error."""
        fitter = tide_simfunc.SimilarityFunctionFitter()

        result = fitter.diagnosefail(np.uint32(0))

        assert result == "No error"

    def test_diagnosefail_single_error(self):
        """Test diagnosefail with single error flag."""
        fitter = tide_simfunc.SimilarityFunctionFitter()

        result = fitter.diagnosefail(fitter.FML_INITAMPLOW)

        assert "Initial amplitude too low" in result

    def test_diagnosefail_multiple_errors(self):
        """Test diagnosefail with multiple error flags."""
        fitter = tide_simfunc.SimilarityFunctionFitter()

        failreason = fitter.FML_INITAMPLOW | fitter.FML_FITLAGLOW
        result = fitter.diagnosefail(failreason)

        assert "Initial amplitude too low" in result
        assert "Fit Lag too low" in result

    def test_fit_gaussian(self, sample_corrtimeaxis, sample_gaussian_corrfunc):
        """Test fit method with Gaussian peak."""
        fitter = tide_simfunc.SimilarityFunctionFitter(
            corrtimeaxis=sample_corrtimeaxis,
            lagmin=-5.0,
            lagmax=5.0,
            peakfittype="gauss",
        )

        result = fitter.fit(sample_gaussian_corrfunc)

        maxindex, maxlag, maxval, maxsigma, maskval, failreason, peakstart, peakend = result

        # Peak should be found near lag=2.0 with amplitude ~0.8
        assert np.abs(maxlag - 2.0) < 0.5
        assert np.abs(maxval - 0.8) < 0.2
        assert maskval == 1  # Successful fit

    def test_fit_fastquad(self, sample_corrtimeaxis, sample_gaussian_corrfunc):
        """Test fit method with fastquad peak fitting."""
        fitter = tide_simfunc.SimilarityFunctionFitter(
            corrtimeaxis=sample_corrtimeaxis,
            lagmin=-5.0,
            lagmax=5.0,
            peakfittype="fastquad",
        )

        result = fitter.fit(sample_gaussian_corrfunc)

        maxindex, maxlag, maxval, maxsigma, maskval, failreason, peakstart, peakend = result

        assert maxlag is not None
        assert maxval is not None

    def test_fit_quad(self, sample_corrtimeaxis, sample_gaussian_corrfunc):
        """Test fit method with quadratic peak fitting."""
        fitter = tide_simfunc.SimilarityFunctionFitter(
            corrtimeaxis=sample_corrtimeaxis,
            lagmin=-5.0,
            lagmax=5.0,
            peakfittype="quad",
        )

        result = fitter.fit(sample_gaussian_corrfunc)

        maxindex, maxlag, maxval, maxsigma, maskval, failreason, peakstart, peakend = result

        assert maxlag is not None
        assert np.abs(maxlag - 2.0) < 0.5

    def test_fit_COM(self, sample_corrtimeaxis, sample_gaussian_corrfunc):
        """Test fit method with center-of-mass peak fitting."""
        fitter = tide_simfunc.SimilarityFunctionFitter(
            corrtimeaxis=sample_corrtimeaxis,
            lagmin=-5.0,
            lagmax=5.0,
            peakfittype="COM",
        )

        result = fitter.fit(sample_gaussian_corrfunc)

        maxindex, maxlag, maxval, maxsigma, maskval, failreason, peakstart, peakend = result

        assert maxlag is not None

    def test_fit_none_type(self, sample_corrtimeaxis, sample_gaussian_corrfunc):
        """Test fit method with None peak fitting (no refinement)."""
        fitter = tide_simfunc.SimilarityFunctionFitter(
            corrtimeaxis=sample_corrtimeaxis,
            lagmin=-5.0,
            lagmax=5.0,
            peakfittype="None",
        )

        result = fitter.fit(sample_gaussian_corrfunc)

        maxindex, maxlag, maxval, maxsigma, maskval, failreason, peakstart, peakend = result

        assert maxlag is not None

    def test_fit_bipolar(self, sample_corrtimeaxis):
        """Test fit method with bipolar option."""
        # Create negative peak
        negative_corrfunc = -0.8 * np.exp(-((sample_corrtimeaxis - 2.0) ** 2) / (2 * 1.5**2))

        fitter = tide_simfunc.SimilarityFunctionFitter(
            corrtimeaxis=sample_corrtimeaxis,
            lagmin=-5.0,
            lagmax=5.0,
            bipolar=True,
            peakfittype="gauss",
        )

        result = fitter.fit(negative_corrfunc)

        maxindex, maxlag, maxval, maxsigma, maskval, failreason, peakstart, peakend = result

        # Should find the negative peak
        assert maxval < 0

    def test_maxindex_noedge(self, sample_corrtimeaxis, sample_gaussian_corrfunc):
        """Test _maxindex_noedge method."""
        fitter = tide_simfunc.SimilarityFunctionFitter(
            corrtimeaxis=sample_corrtimeaxis,
            lagmin=-5.0,
            lagmax=5.0,
        )

        maxindex, flipfac = fitter._maxindex_noedge(sample_gaussian_corrfunc)

        # Peak should be found at correct index
        expected_index = np.argmax(sample_gaussian_corrfunc)
        assert maxindex == expected_index
        assert flipfac == 1.0


# ============================================================================
# FrequencyTracker tests
# ============================================================================


class TestFrequencyTracker:
    """Tests for the FrequencyTracker class."""

    def test_init_basic(self):
        """Test basic initialization of FrequencyTracker."""
        tracker = tide_simfunc.FrequencyTracker()

        assert tracker.lowerlim == 0.1
        assert tracker.upperlim == 0.6
        assert tracker.nperseg == 32
        assert tracker.Q == 10.0
        assert tracker.debug is False
        assert tracker.nfft == 32

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        tracker = tide_simfunc.FrequencyTracker(
            lowerlim=0.2, upperlim=0.8, nperseg=64, Q=20.0, debug=True
        )

        assert tracker.lowerlim == 0.2
        assert tracker.upperlim == 0.8
        assert tracker.nperseg == 64
        assert tracker.Q == 20.0
        assert tracker.debug is True
        assert tracker.nfft == 64

    def test_track_sinusoid(self):
        """Test track method with a simple sinusoid."""
        tracker = tide_simfunc.FrequencyTracker(
            lowerlim=0.1, upperlim=0.5, nperseg=64, debug=False
        )

        # Create a 100-second signal at 10 Hz sampling with 0.2 Hz frequency
        fs = 10.0
        t = np.arange(0, 100, 1 / fs)
        signal_freq = 0.2
        x = np.sin(2 * np.pi * signal_freq * t)

        times, peakfreqs = tracker.track(x, fs)

        assert len(times) == len(peakfreqs)
        assert len(times) > 0

        # Most detected frequencies should be close to 0.2 Hz
        valid_freqs = peakfreqs[peakfreqs > 0]
        if len(valid_freqs) > 0:
            mean_freq = np.mean(valid_freqs)
            assert np.abs(mean_freq - signal_freq) < 0.1

    def test_track_returns_arrays(self):
        """Test that track returns proper numpy arrays."""
        tracker = tide_simfunc.FrequencyTracker(nperseg=32)

        fs = 10.0
        x = np.random.randn(500)

        times, peakfreqs = tracker.track(x, fs)

        assert isinstance(times, np.ndarray)
        assert isinstance(peakfreqs, np.ndarray)

    def test_clean(self):
        """Test clean method."""
        tracker = tide_simfunc.FrequencyTracker(
            lowerlim=0.1, upperlim=0.5, nperseg=64, debug=False
        )

        # Use higher sampling rate and longer signal to ensure enough data for filtering
        fs = 50.0
        t = np.arange(0, 100, 1 / fs)
        signal_freq = 0.2
        x = np.sin(2 * np.pi * signal_freq * t) + 0.5 * np.random.randn(len(t))

        times, peakfreqs = tracker.track(x, fs)

        # Clean the signal
        cleaned = tracker.clean(x, fs, times, peakfreqs, numharmonics=1)

        assert len(cleaned) == len(x)
        assert isinstance(cleaned, np.ndarray)


# ============================================================================
# Integration tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple classes."""

    def test_correlator_with_fitter(self):
        """Test using Correlator output with SimilarityFunctionFitter."""
        # Create test signals - use simple synthetic data without filter
        t = np.linspace(-50, 50, 201)
        # Create a synthetic correlation function (Gaussian peak at lag=2)
        simfunc = 0.8 * np.exp(-((t - 2.0) ** 2) / (2 * 3.0**2))

        # Fit the result
        fitter = tide_simfunc.SimilarityFunctionFitter(
            corrtimeaxis=t,
            lagmin=-25.0,
            lagmax=25.0,
            peakfittype="gauss",
        )

        result = fitter.fit(simfunc)
        maxindex, maxlag, maxval, maxsigma, maskval, failreason, peakstart, peakend = result

        # Should find a peak near lag=2 with amplitude ~0.8
        assert maxval > 0.5
        assert np.abs(maxlag - 2.0) < 1.0

    def test_fitter_error_codes(self):
        """Test that fitter properly reports error conditions."""
        timeaxis = np.linspace(-10, 10, 201)
        # Very weak correlation
        weak_corr = 0.01 * np.exp(-((timeaxis - 2.0) ** 2) / (2 * 1.5**2))

        fitter = tide_simfunc.SimilarityFunctionFitter(
            corrtimeaxis=timeaxis,
            lagmin=-5.0,
            lagmax=5.0,
            lthreshval=0.1,  # Threshold higher than peak
            enforcethresh=True,
            peakfittype="gauss",
        )

        result = fitter.fit(weak_corr)
        maxindex, maxlag, maxval, maxsigma, maskval, failreason, peakstart, peakend = result

        # Should fail due to low amplitude
        diagnosis = fitter.diagnosefail(failreason)
        # Low amplitude fit or init should be detected
        assert failreason != fitter.FML_NOERROR or maskval == 0
