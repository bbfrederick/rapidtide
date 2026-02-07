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
"""
Tests for rapidtide.helper_classes module.

This module tests the fMRIDataset, ProbeRegressor, and Coherer classes.
"""

import numpy as np
import pytest

import rapidtide.filter as tide_filt
import rapidtide.helper_classes as tide_classes

# ============================================================================
# fMRIDataset tests
# ============================================================================


class TestFMRIDataset:
    """Tests for the fMRIDataset class."""

    def test_init_basic(self):
        """Test basic initialization of fMRIDataset."""
        # Create 4D test data (x, y, slices, timepoints)
        data = np.random.rand(10, 12, 8, 100).astype(np.float32)

        dataset = tide_classes.fMRIDataset(data)

        assert dataset.thedata is not None
        assert dataset.xsize == 10
        assert dataset.ysize == 12
        assert dataset.numslices == 8
        assert dataset.realtimepoints == 100
        assert dataset.timepoints == 100
        assert dataset.numskip == 0

    def test_init_zerodata(self):
        """Test initialization with zerodata=True."""
        data = np.random.rand(10, 12, 8, 100).astype(np.float32)

        dataset = tide_classes.fMRIDataset(data, zerodata=True)

        # Data should be all zeros
        assert np.allclose(dataset.thedata, 0.0)
        assert dataset.thedata.shape == data.shape

    def test_init_copydata(self):
        """Test initialization with copydata=True."""
        data = np.random.rand(10, 12, 8, 100).astype(np.float32)

        dataset = tide_classes.fMRIDataset(data, copydata=True)

        # Modifying original should not affect dataset
        original_val = dataset.thedata[0, 0, 0, 0]
        data[0, 0, 0, 0] = 999.0
        assert dataset.thedata[0, 0, 0, 0] == original_val

    def test_init_no_copydata(self):
        """Test initialization without copying data (default)."""
        data = np.random.rand(10, 12, 8, 100).astype(np.float32)

        dataset = tide_classes.fMRIDataset(data, copydata=False)

        # Modifying original should affect dataset (same reference)
        data[0, 0, 0, 0] = 999.0
        assert dataset.thedata[0, 0, 0, 0] == 999.0

    def test_init_with_numskip(self):
        """Test initialization with numskip parameter."""
        data = np.random.rand(10, 12, 8, 100).astype(np.float32)

        dataset = tide_classes.fMRIDataset(data, numskip=5)

        assert dataset.numskip == 5
        assert dataset.realtimepoints == 100
        assert dataset.timepoints == 95

    def test_getsizes(self):
        """Test getsizes method."""
        data = np.random.rand(8, 10, 5, 50).astype(np.float32)

        dataset = tide_classes.fMRIDataset(data)

        assert dataset.theshape == (8, 10, 5, 50)
        assert dataset.xsize == 8
        assert dataset.ysize == 10
        assert dataset.numslices == 5
        assert dataset.realtimepoints == 50
        assert dataset.slicesize == 80  # 8 * 10
        assert dataset.numvox == 400  # 80 * 5

    def test_setnumskip(self):
        """Test setnumskip method."""
        data = np.random.rand(10, 12, 8, 100).astype(np.float32)
        dataset = tide_classes.fMRIDataset(data)

        assert dataset.numskip == 0
        assert dataset.timepoints == 100

        dataset.setnumskip(10)

        assert dataset.numskip == 10
        assert dataset.timepoints == 90

    def test_setvalid(self):
        """Test setvalid method."""
        data = np.random.rand(10, 12, 8, 100).astype(np.float32)
        dataset = tide_classes.fMRIDataset(data)

        valid_voxels = np.array([0, 1, 5, 10, 20])
        dataset.setvalid(valid_voxels)

        assert np.array_equal(dataset.validvoxels, valid_voxels)

    def test_byslice(self):
        """Test byslice method."""
        data = np.random.rand(10, 12, 8, 100).astype(np.float32)
        dataset = tide_classes.fMRIDataset(data, numskip=5)

        result = dataset.byslice()

        # Expected shape: (slicesize, numslices, timepoints)
        # slicesize = 10 * 12 = 120
        # numslices = 8
        # timepoints = 100 - 5 = 95
        assert result.shape == (120, 8, 95)

    def test_byvol(self):
        """Test byvol method."""
        data = np.random.rand(10, 12, 8, 100).astype(np.float32)
        dataset = tide_classes.fMRIDataset(data, numskip=5)

        result = dataset.byvol()

        # Expected shape: (numvox, timepoints)
        # numvox = 10 * 12 * 8 = 960
        # timepoints = 100 - 5 = 95
        assert result.shape == (960, 95)

    def test_byvox(self):
        """Test byvox method."""
        data = np.random.rand(10, 12, 8, 100).astype(np.float32)
        dataset = tide_classes.fMRIDataset(data, numskip=5)

        result = dataset.byvox()

        # Expected shape: (10, 12, 8, 95)
        assert result.shape == (10, 12, 8, 95)

    def test_byslice_no_skip(self):
        """Test byslice with no skip."""
        data = np.random.rand(5, 6, 4, 20).astype(np.float32)
        dataset = tide_classes.fMRIDataset(data)

        result = dataset.byslice()

        assert result.shape == (30, 4, 20)  # 5*6=30 slicesize

    def test_byvol_no_skip(self):
        """Test byvol with no skip."""
        data = np.random.rand(5, 6, 4, 20).astype(np.float32)
        dataset = tide_classes.fMRIDataset(data)

        result = dataset.byvol()

        assert result.shape == (120, 20)  # 5*6*4=120 numvox

    def test_byvox_no_skip(self):
        """Test byvox with no skip."""
        data = np.random.rand(5, 6, 4, 20).astype(np.float32)
        dataset = tide_classes.fMRIDataset(data)

        result = dataset.byvox()

        assert result.shape == (5, 6, 4, 20)


# ============================================================================
# ProbeRegressor tests
# ============================================================================


class TestProbeRegressor:
    """Tests for the ProbeRegressor class."""

    def test_init_basic(self):
        """Test basic initialization of ProbeRegressor."""
        inputvec = np.sin(np.linspace(0, 10, 1000))
        inputfreq = 100.0
        targetperiod = 0.5
        targetpoints = 100
        targetstartpoint = 0

        regressor = tide_classes.ProbeRegressor(
            inputvec, inputfreq, targetperiod, targetpoints, targetstartpoint
        )

        assert regressor.inputvec is not None
        assert regressor.inputfreq == inputfreq
        assert regressor.targetperiod == targetperiod
        assert regressor.targetpoints == targetpoints
        assert regressor.targetstartpoint == targetstartpoint
        assert regressor.inputtimeaxis is not None

    def test_init_with_offsets(self):
        """Test initialization with custom offsets."""
        inputvec = np.sin(np.linspace(0, 10, 1000))

        regressor = tide_classes.ProbeRegressor(
            inputvec,
            inputfreq=100.0,
            targetperiod=0.5,
            targetpoints=100,
            targetstartpoint=0,
            inputstart=1.0,
            inputoffset=0.5,
        )

        assert regressor.inputstart == 1.0
        assert regressor.inputoffset == 0.5

    def test_setinputvec(self):
        """Test setinputvec method directly."""
        inputvec = np.sin(np.linspace(0, 10, 1000))

        # Create a minimal object to test setinputvec
        regressor = object.__new__(tide_classes.ProbeRegressor)
        regressor.setinputvec(inputvec, 100.0, inputstart=0.5)

        assert np.array_equal(regressor.inputvec, inputvec)
        assert regressor.inputfreq == 100.0
        assert regressor.inputstart == 0.5

    def test_makeinputtimeaxis(self):
        """Test makeinputtimeaxis method."""
        regressor = object.__new__(tide_classes.ProbeRegressor)
        regressor.inputvec = np.zeros(100)
        regressor.inputfreq = 10.0
        regressor.inputstart = 0.0
        regressor.inputoffset = 0.0

        regressor.makeinputtimeaxis()

        assert regressor.inputtimeaxis is not None
        # np.linspace default is 50 points from 0 to len(inputvec)
        assert len(regressor.inputtimeaxis) == 50
        assert regressor.inputtimeaxis[0] == pytest.approx(0.0)
        # Last value should be len(inputvec)/inputfreq = 100/10 = 10
        assert regressor.inputtimeaxis[-1] == pytest.approx(10.0)

    def test_maketargettimeaxis(self):
        """Test maketargettimeaxis method."""
        # Create a minimal object
        regressor = object.__new__(tide_classes.ProbeRegressor)
        regressor.targetperiod = 0.1
        regressor.targetstartpoint = 0
        regressor.targetpoints = 10

        regressor.maketargettimeaxis()

        assert regressor.targettimeaxis is not None
        assert len(regressor.targettimeaxis) == 10
        assert regressor.targettimeaxis[0] == pytest.approx(0.0)
        # endpoint=True means last value is targetperiod * (targetstartpoint + targetpoints)
        assert regressor.targettimeaxis[-1] == pytest.approx(1.0)

    def test_maketargettimeaxis_with_offset(self):
        """Test maketargettimeaxis with non-zero start point."""
        regressor = object.__new__(tide_classes.ProbeRegressor)
        regressor.targetperiod = 0.1
        regressor.targetstartpoint = 5
        regressor.targetpoints = 10

        regressor.maketargettimeaxis()

        assert regressor.targettimeaxis[0] == pytest.approx(0.5)  # 0.1 * 5
        assert len(regressor.targettimeaxis) == 10


# ============================================================================
# Coherer tests
# ============================================================================


class TestCoherer:
    """Tests for the Coherer class."""

    @pytest.fixture
    def sample_filter(self):
        """Create a sample filter for testing."""
        # Use LFO filter - requires low sampling frequency to avoid frequency limit issues
        return tide_filt.NoncausalFilter(filtertype="lfo")

    @pytest.fixture
    def sample_timecourse(self):
        """Create a sample timecourse for testing."""
        # Use Fs=2.0 which is compatible with LFO filter
        t = np.linspace(0, 100, 200)  # 100 seconds at 2 Hz
        return np.sin(2 * np.pi * 0.05 * t) + 0.5 * np.sin(2 * np.pi * 0.1 * t)

    @pytest.fixture
    def sample_fs(self):
        """Sample frequency for testing."""
        return 2.0

    def test_init_basic(self):
        """Test basic initialization of Coherer."""
        coherer = tide_classes.Coherer(Fs=10.0)

        assert coherer.Fs == 10.0
        assert coherer.detrendorder == 1
        assert coherer.windowfunc == "hamming"
        assert coherer.debug is False
        assert coherer.freqmin is None
        assert coherer.freqmax is None

    def test_init_with_freq_limits(self):
        """Test initialization with frequency limits."""
        coherer = tide_classes.Coherer(Fs=10.0, freqmin=0.01, freqmax=0.5)

        assert coherer.freqmin == 0.01
        assert coherer.freqmax == 0.5

    def test_init_with_detrendorder(self):
        """Test initialization with custom detrend order."""
        coherer = tide_classes.Coherer(Fs=10.0, detrendorder=2)

        assert coherer.detrendorder == 2

    def test_init_with_windowfunc(self):
        """Test initialization with custom window function."""
        coherer = tide_classes.Coherer(Fs=10.0, windowfunc="hanning")

        assert coherer.windowfunc == "hanning"

    def test_init_with_reftc(self, sample_filter, sample_timecourse, sample_fs):
        """Test initialization with reference timecourse."""
        coherer = tide_classes.Coherer(
            Fs=sample_fs, ncprefilter=sample_filter, reftc=sample_timecourse
        )

        assert coherer.reftc is not None
        assert coherer.freqaxisvalid is True

    def test_setlimits(self, sample_filter, sample_timecourse, sample_fs):
        """Test setlimits method."""
        coherer = tide_classes.Coherer(Fs=sample_fs, ncprefilter=sample_filter)
        coherer.setreftc(sample_timecourse)

        coherer.setlimits(0.02, 0.14)

        assert coherer.freqmin == 0.02
        assert coherer.freqmax == 0.14
        assert hasattr(coherer, "freqmininpts")
        assert hasattr(coherer, "freqmaxinpts")

    def test_setreftc(self, sample_filter, sample_timecourse, sample_fs):
        """Test setreftc method."""
        coherer = tide_classes.Coherer(Fs=sample_fs, ncprefilter=sample_filter)

        coherer.setreftc(sample_timecourse)

        assert coherer.reftc is not None
        assert coherer.prepreftc is not None
        assert coherer.freqaxis is not None
        assert coherer.thecoherence is not None
        assert coherer.freqaxisvalid is True
        assert coherer.similarityfunclen > 0

    def test_preptc(self, sample_filter, sample_timecourse, sample_fs):
        """Test preptc method."""
        coherer = tide_classes.Coherer(Fs=sample_fs, ncprefilter=sample_filter)

        result = coherer.preptc(sample_timecourse)

        assert result is not None
        assert len(result) == len(sample_timecourse)

    def test_getaxisinfo(self, sample_filter, sample_timecourse, sample_fs):
        """Test getaxisinfo method."""
        coherer = tide_classes.Coherer(
            Fs=sample_fs, ncprefilter=sample_filter, freqmin=0.02, freqmax=0.14
        )
        coherer.setreftc(sample_timecourse)

        freqmin, freqmax, freqstep, numpts = coherer.getaxisinfo()

        assert freqmin >= 0
        assert freqmax > freqmin
        assert freqstep > 0
        assert numpts > 0

    def test_trim(self, sample_filter, sample_timecourse, sample_fs):
        """Test trim method."""
        coherer = tide_classes.Coherer(
            Fs=sample_fs, ncprefilter=sample_filter, freqmin=0.02, freqmax=0.14
        )
        coherer.setreftc(sample_timecourse)

        # Create a test vector
        test_vector = np.arange(len(coherer.freqaxis))

        trimmed = coherer.trim(test_vector)

        expected_len = coherer.freqmaxinpts - coherer.freqmininpts
        assert len(trimmed) == expected_len

    def test_run_basic(self, sample_filter, sample_timecourse, sample_fs):
        """Test run method with basic parameters."""
        coherer = tide_classes.Coherer(
            Fs=sample_fs, ncprefilter=sample_filter, freqmin=0.02, freqmax=0.14
        )
        coherer.setreftc(sample_timecourse)

        # Create a slightly different test timecourse with same length
        t = np.linspace(0, 100, 200)
        testtc = np.sin(2 * np.pi * 0.05 * t) + 0.3 * np.sin(2 * np.pi * 0.1 * t)

        coherence, freqaxis, themax = coherer.run(testtc, trim=True)

        assert coherence is not None
        assert freqaxis is not None
        assert themax >= 0
        assert len(coherence) == len(freqaxis)
        assert coherer.datavalid is True

    def test_run_no_trim(self, sample_filter, sample_timecourse, sample_fs):
        """Test run method without trimming."""
        coherer = tide_classes.Coherer(
            Fs=sample_fs, ncprefilter=sample_filter, freqmin=0.02, freqmax=0.14
        )
        coherer.setreftc(sample_timecourse)

        t = np.linspace(0, 100, 200)
        testtc = np.sin(2 * np.pi * 0.05 * t)

        coherence, freqaxis, themax = coherer.run(testtc, trim=False)

        assert coherence is not None
        assert len(coherence) == coherer.similarityfunclen

    def test_run_with_alt(self, sample_filter, sample_timecourse, sample_fs):
        """Test run method with alt=True (CSD-based coherence)."""
        coherer = tide_classes.Coherer(
            Fs=sample_fs, ncprefilter=sample_filter, freqmin=0.02, freqmax=0.14
        )
        coherer.setreftc(sample_timecourse)

        t = np.linspace(0, 100, 200)
        testtc = np.sin(2 * np.pi * 0.05 * t)

        result = coherer.run(testtc, trim=True, alt=True)

        # With alt=True, returns 6 values
        assert len(result) == 6
        coherence, freqaxis, themax, csdxx, csdyy, csdxy = result
        assert coherence is not None
        assert csdxx is not None
        assert csdyy is not None
        assert csdxy is not None

    def test_run_with_alt_no_trim(self, sample_filter, sample_timecourse, sample_fs):
        """Test run method with alt=True and trim=False."""
        coherer = tide_classes.Coherer(
            Fs=sample_fs, ncprefilter=sample_filter, freqmin=0.02, freqmax=0.14
        )
        coherer.setreftc(sample_timecourse)

        t = np.linspace(0, 100, 200)
        testtc = np.sin(2 * np.pi * 0.05 * t)

        result = coherer.run(testtc, trim=False, alt=True)

        assert len(result) == 6
        coherence, freqaxis, themax, csdxx, csdyy, csdxy = result
        assert len(coherence) == coherer.similarityfunclen

    def test_run_identical_signals(self, sample_filter, sample_timecourse, sample_fs):
        """Test run with identical reference and test signals."""
        coherer = tide_classes.Coherer(
            Fs=sample_fs, ncprefilter=sample_filter, freqmin=0.02, freqmax=0.14
        )
        coherer.setreftc(sample_timecourse)

        # Use the same signal as test
        coherence, freqaxis, themax = coherer.run(sample_timecourse, trim=True)

        # Coherence should be high (close to 1) for identical signals
        assert np.max(coherence) >= 0.9

    def test_coherence_values_in_range(self, sample_filter, sample_timecourse, sample_fs):
        """Test that coherence values are between 0 and 1."""
        coherer = tide_classes.Coherer(
            Fs=sample_fs, ncprefilter=sample_filter, freqmin=0.02, freqmax=0.14
        )
        coherer.setreftc(sample_timecourse)

        t = np.linspace(0, 100, 200)
        testtc = np.sin(2 * np.pi * 0.05 * t) + np.random.randn(200) * 0.1

        coherence, freqaxis, themax = coherer.run(testtc, trim=False)

        # Coherence should be between 0 and 1 (with small tolerance for numerical precision)
        assert np.all(coherence >= -1e-10)
        assert np.all(coherence <= 1 + 1e-10)


# ============================================================================
# Integration tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple classes."""

    def test_fmri_dataset_reshape_consistency(self):
        """Test that different reshape methods produce consistent data."""
        data = np.random.rand(8, 10, 4, 50).astype(np.float32)
        dataset = tide_classes.fMRIDataset(data, numskip=5)

        byslice = dataset.byslice()
        byvol = dataset.byvol()
        byvox = dataset.byvox()

        # All should have the same total number of elements
        assert byslice.size == byvol.size == byvox.size

        # Specific values should be accessible in all formats
        # Check first timepoint after skip at first voxel
        assert byvox[0, 0, 0, 0] == data[0, 0, 0, 5]

    def test_coherer_workflow(self):
        """Test complete Coherer workflow."""
        # Create sample data with low sampling frequency for LFO filter compatibility
        Fs = 2.0
        t = np.linspace(0, 100, int(100 * Fs))
        signal_freq = 0.05

        # Reference signal
        reftc = np.sin(2 * np.pi * signal_freq * t)

        # Test signal with same frequency but slightly different phase
        testtc = np.sin(2 * np.pi * signal_freq * t + 0.5)

        # Create filter and coherer
        ncprefilter = tide_filt.NoncausalFilter(filtertype="lfo")
        coherer = tide_classes.Coherer(
            Fs=Fs, ncprefilter=ncprefilter, freqmin=0.02, freqmax=0.14
        )

        # Set reference and run coherence
        coherer.setreftc(reftc)
        coherence, freqaxis, themax = coherer.run(testtc)

        # Should find high coherence at the signal frequency
        assert coherer.datavalid is True
        assert len(coherence) > 0

    def test_fmri_dataset_memory_efficiency(self):
        """Test that copydata=False doesn't duplicate memory."""
        data = np.random.rand(10, 10, 5, 100).astype(np.float32)
        dataset = tide_classes.fMRIDataset(data, copydata=False)

        # Should share memory
        assert np.shares_memory(data, dataset.thedata)

        dataset_copy = tide_classes.fMRIDataset(data, copydata=True)

        # Should not share memory
        assert not np.shares_memory(data, dataset_copy.thedata)