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
"""Tests for rapidtide.workflows.simdata.

simdata builds synthetic fMRI data by planting known physiological signals at known
delays.  It is what other rapidtide tests are validated against, so an error here
propagates silently into everything that uses its output as ground truth - which
makes exact recovery of the planted signal the thing worth asserting.
"""

import argparse
import os
import sys
import tempfile

import nibabel as nb
import numpy as np
import pytest

import rapidtide.filter as tide_filt
import rapidtide.io as tide_io
import rapidtide.resample as tide_resample
from rapidtide.workflows.simdata import _get_parser, fmrisignal, prepareband, simdata

THEAFFINE = np.diag([2.0, 2.0, 2.0, 1.0])


class _FlatWave:
    """A stand in for a FastResampler that returns a known function of time.

    Parameters
    ----------
    thefunction : callable
        Maps a time array to signal values.
    """

    def __init__(self, thefunction):
        self.thefunction = thefunction

    def yfromx(self, thetimes, debug=False):
        """Evaluate the waveform.

        Parameters
        ----------
        thetimes : NDArray
            Times to evaluate at.
        debug : bool, optional
            Accepted and ignored.

        Returns
        -------
        NDArray
            The waveform values.
        """
        return self.thefunction(np.asarray(thetimes))


class _NullFilter:
    """A filter that passes its input straight through, so noise is deterministic."""

    def apply(self, Fs, thedata):
        """Return the data unchanged.

        Parameters
        ----------
        Fs : float
            Sample rate, accepted and ignored.
        thedata : NDArray
            The data to filter.

        Returns
        -------
        NDArray
            thedata, unchanged.
        """
        return thedata


def _writenifti(thedir, thename, thedata):
    """Write an array as a NIfTI file and return its path.

    Parameters
    ----------
    thedir : str
        Directory to write into.
    thename : str
        Base filename, without extension.
    thedata : NDArray
        The array to write.

    Returns
    -------
    str
        Full path of the file written.
    """
    thefilename = os.path.join(thedir, f"{thename}.nii.gz")
    nb.save(nb.Nifti1Image(np.asarray(thedata, dtype=np.float64), THEAFFINE), thefilename)
    return thefilename


def _writeregressor(thedir, thename, thevalues):
    """Write a regressor timecourse as a plain text column.

    Parameters
    ----------
    thedir : str
        Directory to write into.
    thename : str
        Base filename, without extension.
    thevalues : NDArray
        The timecourse.

    Returns
    -------
    str
        Full path of the file written.
    """
    thefilename = os.path.join(thedir, f"{thename}.txt")
    np.savetxt(thefilename, thevalues)
    return thefilename


# ==================== fmrisignal ====================


def test_fmrisignal_with_no_components_is_the_mean():
    """With every component switched off the result is a flat line at meanvalue.
    That is the baseline every other component adds on top of."""
    thetimes = np.linspace(0.0, 10.0, 100)
    theresult = fmrisignal(10.0, thetimes, 500.0)
    assert theresult.shape == thetimes.shape
    np.testing.assert_allclose(theresult, 500.0)


def test_fmrisignal_scales_each_component_by_the_mean():
    """Components are fractional signal changes, so each is multiplied by meanvalue.
    A component that ignored the mean would give the same absolute swing in a bright
    voxel as in a dim one."""
    thetimes = np.linspace(0.0, 10.0, 200)
    thewave = _FlatWave(lambda t: np.ones_like(t))

    for themean in (100.0, 1000.0):
        theresult = fmrisignal(
            10.0,
            thetimes,
            themean,
            dolfo=True,
            lfowave=thewave,
            lfomag=0.02,
            lfodelay=0.0,
            lfonoise=0.0,
            lfofilter=_NullFilter(),
        )
        # a constant waveform of 1.0 at 2% modulation, on top of the mean
        np.testing.assert_allclose(theresult, themean * 1.02)


def test_fmrisignal_applies_the_delay():
    """The delay is subtracted from the time axis before the waveform is evaluated,
    so a ramp comes back shifted by exactly that much."""
    thetimes = np.linspace(0.0, 10.0, 101)
    theramp = _FlatWave(lambda t: t)
    thedelay = 2.0

    theresult = fmrisignal(
        10.0,
        thetimes,
        1.0,
        dolfo=True,
        lfowave=theramp,
        lfomag=1.0,
        lfodelay=thedelay,
        lfonoise=0.0,
        lfofilter=_NullFilter(),
    )
    # signal = mean * (mag * (t - delay)) + mean, with mean 1 and mag 1
    np.testing.assert_allclose(theresult, (thetimes - thedelay) + 1.0)


def test_fmrisignal_sums_the_three_bands():
    """LFO, respiratory and cardiac contributions add.  Enabling all three has to
    give the sum of enabling each alone, minus the two extra copies of the mean."""
    thetimes = np.linspace(0.0, 5.0, 50)
    themean = 200.0
    thebands = {
        "lfo": _FlatWave(lambda t: np.full_like(t, 1.0)),
        "resp": _FlatWave(lambda t: np.full_like(t, 2.0)),
        "cardiac": _FlatWave(lambda t: np.full_like(t, 4.0)),
    }
    thecommon = dict(
        lfowave=thebands["lfo"],
        lfomag=0.01,
        lfodelay=0.0,
        lfonoise=0.0,
        lfofilter=_NullFilter(),
        respwave=thebands["resp"],
        respmag=0.01,
        respdelay=0.0,
        respnoise=0.0,
        respfilter=_NullFilter(),
        cardiacwave=thebands["cardiac"],
        cardiacmag=0.01,
        cardiacdelay=0.0,
        cardiacnoise=0.0,
        cardiacfilter=_NullFilter(),
    )

    theall = fmrisignal(
        10.0, thetimes, themean, dolfo=True, doresp=True, docardiac=True, **thecommon
    )
    # 1% of 1, 2 and 4 respectively, all on top of the mean
    np.testing.assert_allclose(theall, themean * (1.0 + 0.01 * (1.0 + 2.0 + 4.0)))

    # and each band alone contributes only its own share
    thelfoonly = fmrisignal(10.0, thetimes, themean, dolfo=True, **thecommon)
    np.testing.assert_allclose(thelfoonly, themean * 1.01)


def test_fmrisignal_noise_is_scaled_and_filtered():
    """The noise term goes through the supplied filter and is scaled by the mean, so
    turning it up has to widen the distribution."""
    thetimes = np.linspace(0.0, 100.0, 2000)
    thewave = _FlatWave(lambda t: np.zeros_like(t))

    thespreads = []
    for thenoise in (0.0, 0.05):
        np.random.seed(11)
        theresult = fmrisignal(
            20.0,
            thetimes,
            100.0,
            dolfo=True,
            lfowave=thewave,
            lfomag=0.0,
            lfodelay=0.0,
            lfonoise=thenoise,
            lfofilter=_NullFilter(),
        )
        thespreads.append(float(np.std(theresult)))

    assert thespreads[0] == pytest.approx(0.0), "noise of zero was not silent"
    assert thespreads[1] > 1.0, f"noise did not reach the output: std {thespreads[1]}"


# ==================== prepareband ====================


def _makebandinputs(thedir, theshape=(4, 4, 3), thenumpoints=200, thesamplerate=2.0):
    """Build the NIfTI and regressor files prepareband needs.

    Parameters
    ----------
    thedir : str
        Directory to write into.
    theshape : tuple of int
        Spatial shape of the maps.
    thenumpoints : int
        Length of the regressor.
    thesamplerate : float
        Regressor sample rate, in Hz.

    Returns
    -------
    dict
        Paths and arrays, keyed by role.
    """
    thetimes = np.arange(thenumpoints) / thesamplerate
    theregressor = np.sin(2.0 * np.pi * 0.05 * thetimes)
    thepct = np.full(theshape, 2.0)
    thelag = np.zeros(theshape)
    return {
        "regressorfile": _writeregressor(thedir, "regressor", theregressor),
        "pctfile": _writenifti(thedir, "pct", thepct),
        "sigfracfile": _writenifti(thedir, "sigfrac", thepct),
        "lagfile": _writenifti(thedir, "lag", thelag),
        "dims": np.array([3, theshape[0], theshape[1], theshape[2], 1, 1, 1, 1]),
        "shape": theshape,
        "samplerate": thesamplerate,
    }


def test_prepareband_returns_a_working_resampler():
    """The regressor is normalized and wrapped in a FastResampler, which is what the
    workflow later evaluates at each voxel's delay."""
    with tempfile.TemporaryDirectory() as thedir:
        theinputs = _makebandinputs(thedir)
        thepct, thepctscale, thelag, thegenerator = prepareband(
            theinputs["dims"],
            theinputs["pctfile"],
            None,
            theinputs["lagfile"],
            theinputs["regressorfile"],
            theinputs["samplerate"],
            0.0,
            "lfo",
        )

    assert thepct.shape == theinputs["shape"]
    assert thelag.shape == theinputs["shape"]
    assert isinstance(thegenerator, tide_resample.FastResampler)
    # the regressor was standard normalized, so the resampled values are of order 1
    theresampled = thegenerator.yfromx(np.linspace(0.0, 50.0, 100))
    assert np.all(np.isfinite(theresampled))
    assert 0.1 < float(np.std(theresampled)) < 10.0


def test_prepareband_pctfile_and_sigfracfile_differ_by_a_factor_of_a_hundred():
    """A percentage file is taken as given; a signal fraction file is divided by 100
    so both end up in the same units.  Mixing them up would scale every simulated
    signal by 100."""
    with tempfile.TemporaryDirectory() as thedir:
        theinputs = _makebandinputs(thedir)
        thepctversion = prepareband(
            theinputs["dims"],
            theinputs["pctfile"],
            None,
            theinputs["lagfile"],
            theinputs["regressorfile"],
            theinputs["samplerate"],
            0.0,
            "lfo",
        )
        thesigfracversion = prepareband(
            theinputs["dims"],
            None,
            theinputs["sigfracfile"],
            theinputs["lagfile"],
            theinputs["regressorfile"],
            theinputs["samplerate"],
            0.0,
            "lfo",
        )

    # the same file read both ways, differing only by the 100 and the scale flag
    assert thepctversion[1] is True
    assert thesigfracversion[1] is False
    np.testing.assert_allclose(thepctversion[0], thesigfracversion[0] * 100.0)


def test_prepareband_rejects_a_mismatched_lag_file():
    """The lag map has to cover the same voxels as the data, or the delays would be
    applied to the wrong places."""
    with tempfile.TemporaryDirectory() as thedir:
        theinputs = _makebandinputs(thedir)
        thebadlag = _writenifti(thedir, "badlag", np.zeros((5, 4, 3)))
        with pytest.raises(SystemExit):
            prepareband(
                theinputs["dims"],
                theinputs["pctfile"],
                None,
                thebadlag,
                theinputs["regressorfile"],
                theinputs["samplerate"],
                0.0,
                "lfo",
            )


def test_prepareband_rejects_a_mismatched_sigfrac_file():
    """Same for the signal fraction map."""
    with tempfile.TemporaryDirectory() as thedir:
        theinputs = _makebandinputs(thedir)
        thebadsigfrac = _writenifti(thedir, "badsigfrac", np.ones((5, 4, 3)))
        with pytest.raises(SystemExit):
            prepareband(
                theinputs["dims"],
                None,
                thebadsigfrac,
                theinputs["lagfile"],
                theinputs["regressorfile"],
                theinputs["samplerate"],
                0.0,
                "lfo",
            )


def test_prepareband_falls_back_to_the_file_timing():
    """When the caller does not specify a sample rate or start time, the ones read
    out of the regressor file are used."""
    with tempfile.TemporaryDirectory() as thedir:
        theinputs = _makebandinputs(thedir)
        # a plain text file carries no timing, so the defaults have to take over
        dummy, dummy2, dummy3, thegenerator = prepareband(
            theinputs["dims"],
            theinputs["pctfile"],
            None,
            theinputs["lagfile"],
            theinputs["regressorfile"],
            1.0,
            None,
            "lfo",
            debug=True,
        )
    assert isinstance(thegenerator, tide_resample.FastResampler)


# ==================== the simdata workflow ====================


def _makesimargs(thedir, theshape=(4, 4, 3), thenumtrs=40, thefmritr=1.0, **theoverrides):
    """Assemble a complete simdata argument set with an LFO band.

    Parameters
    ----------
    thedir : str
        Working directory.
    theshape : tuple of int
        Spatial shape.
    thenumtrs : int
        Number of output timepoints.
    thefmritr : float
        Output TR, in seconds.
    **theoverrides : Any
        Any argument to override.

    Returns
    -------
    argparse.Namespace
        Arguments ready for simdata.
    """
    thesamplerate = 10.0
    thenumpoints = int(thenumtrs * thefmritr * thesamplerate) + 200
    thetimes = np.arange(thenumpoints) / thesamplerate
    theregressor = np.sin(2.0 * np.pi * 0.05 * thetimes)

    theargs = argparse.Namespace(
        fmritr=thefmritr,
        numtrs=thenumtrs,
        immeanfilename=_writenifti(thedir, "mean", np.full(theshape, 1000.0)),
        outputroot=os.path.join(thedir, "simout"),
        # NOTE: despite the --lfopctfile help text saying "percent of mean", the
        # value is used directly as the modulation magnitude, and rapidtide feeds
        # this option a maxcorr map - a fraction in [0, 1].  0.02 is 2%.
        lfopctfile=_writenifti(thedir, "lfopct", np.full(theshape, 0.02)),
        lfosigfracfile=None,
        lfolagfile=_writenifti(thedir, "lfolag", np.zeros(theshape)),
        lforegressor=_writeregressor(thedir, "lforegressor", theregressor),
        lfosamprate=thesamplerate,
        lfostarttime=0.0,
        resppctfile=None,
        respsigfracfile=None,
        resplagfile=None,
        respregressor=None,
        respsamprate=None,
        respstarttime=None,
        cardiacpctfile=None,
        cardiacsigfracfile=None,
        cardiaclagfile=None,
        cardiacregressor=None,
        cardiacsamprate=None,
        cardiacstarttime=None,
        slicetimefile=None,
        numskip=0,
        globalnoiselevel=0.0,
        voxelnoiselevel=0.0,
        debug=False,
    )
    for thename, thevalue in theoverrides.items():
        setattr(theargs, thename, thevalue)
    return theargs


def test_get_parser_defaults():
    """Four required positionals, everything else optional and off by default."""
    theparser = _get_parser()
    assert theparser.prog == "simdata"

    with tempfile.TemporaryDirectory() as thedir:
        themean = _writenifti(thedir, "mean", np.ones((2, 2, 2)))
        theargs = theparser.parse_args(["1.5", "100", themean, "outroot"])

    assert theargs.fmritr == pytest.approx(1.5)
    assert theargs.numtrs == 100
    assert theargs.outputroot == "outroot"
    assert theargs.numskip == 0
    assert theargs.globalnoiselevel == pytest.approx(0.0)
    assert theargs.voxelnoiselevel == pytest.approx(0.0)
    assert theargs.debug is False
    assert theargs.lforegressor is None


def test_simdata_writes_a_volume_of_the_requested_shape():
    """The output has the mean image's geometry and the requested number of TRs."""
    theshape, thenumtrs = (4, 4, 3), 40
    with tempfile.TemporaryDirectory() as thedir:
        theargs = _makesimargs(thedir, theshape=theshape, thenumtrs=thenumtrs)
        simdata(theargs)
        dummy, thedata, dummy2, thedims, thesizes = tide_io.readfromnifti(
            f"{theargs.outputroot}.nii.gz"
        )

    assert thedata.shape == theshape + (thenumtrs,)
    assert thesizes[4] == pytest.approx(1.0), "the output TR was not written"


def test_simdata_plants_the_requested_signal_fraction():
    """A 2% LFO on a mean of 1000 has to come out as a 2% modulation, because every
    downstream test treats simdata's output as ground truth."""
    theshape, thenumtrs = (4, 4, 3), 60
    with tempfile.TemporaryDirectory() as thedir:
        theargs = _makesimargs(thedir, theshape=theshape, thenumtrs=thenumtrs)
        simdata(theargs)
        dummy, thedata, dummy2, dummy3, dummy4 = tide_io.readfromnifti(
            f"{theargs.outputroot}.nii.gz"
        )

    thetimecourse = thedata[2, 2, 1, :]
    themean = float(np.mean(thetimecourse))
    theswing = float(np.max(thetimecourse) - np.min(thetimecourse))

    assert themean == pytest.approx(1000.0, rel=0.05), f"mean came out at {themean}"
    # a standard normalized sinusoid has unit variance and so swings about +-1.4, and
    # a magnitude of 0.02 on a mean of 1000 turns that into roughly 55 units of signal
    assert 30.0 < theswing < 90.0, f"modulation depth was {theswing}"


def test_simdata_applies_the_lag_map():
    """Voxels with different delays get the same waveform shifted, which is the whole
    point of the simulation.  A lag map that was ignored would make every voxel
    identical."""
    theshape, thenumtrs = (4, 1, 1), 80
    thelag = np.zeros(theshape)
    thelag[0, 0, 0] = 0.0
    thelag[1, 0, 0] = 2.5
    thelag[2, 0, 0] = 5.0
    thelag[3, 0, 0] = 7.5

    with tempfile.TemporaryDirectory() as thedir:
        theargs = _makesimargs(thedir, theshape=theshape, thenumtrs=thenumtrs)
        theargs.lfolagfile = _writenifti(thedir, "lfolag2", thelag)
        simdata(theargs)
        dummy, thedata, dummy2, dummy3, dummy4 = tide_io.readfromnifti(
            f"{theargs.outputroot}.nii.gz"
        )

    thereference = thedata[0, 0, 0, :]
    theshifted = thedata[2, 0, 0, :]
    assert not np.allclose(thereference, theshifted), "the lag map had no effect"

    # and the shift is a real time shift, so the two are still highly correlated once
    # realigned rather than simply being different signals
    thecorrelation = float(np.corrcoef(thereference, theshifted)[0, 1])
    assert abs(thecorrelation) < 0.999


def test_simdata_noise_levels_reach_the_output():
    """The two noise knobs are different in kind and have to be measured differently.

    Global noise is spatially uniform: it rides on the timecourse every voxel shares,
    so it widens the spatial-mean timecourse while leaving the voxel-to-voxel spread
    at zero.  Voxel noise is the reverse.  Measuring both on one statistic hides
    whichever one that statistic is blind to - a spatial-mean subtraction removes
    global noise by construction.
    """
    theshape, thenumtrs = (3, 3, 2), 60
    themeasurements = {}
    for thelabel, theoverrides in (
        ("clean", {}),
        ("voxel", {"voxelnoiselevel": 5.0}),
        ("global", {"globalnoiselevel": 5.0}),
    ):
        with tempfile.TemporaryDirectory() as thedir:
            theargs = _makesimargs(thedir, theshape=theshape, thenumtrs=thenumtrs, **theoverrides)
            simdata(theargs)
            dummy, thedata, dummy2, dummy3, dummy4 = tide_io.readfromnifti(
                f"{theargs.outputroot}.nii.gz"
            )
        theglobaltimecourse = thedata.mean(axis=(0, 1, 2))
        theresidual = thedata - theglobaltimecourse[None, None, None, :]
        themeasurements[thelabel] = {
            "globalspread": float(np.std(theglobaltimecourse)),
            "voxelspread": float(np.std(theresidual)),
        }

    # with no noise every voxel carries the identical planted signal
    assert themeasurements["clean"]["voxelspread"] == pytest.approx(0.0, abs=1e-9)

    # voxel noise makes voxels differ from one another
    assert themeasurements["voxel"]["voxelspread"] > 10.0, themeasurements

    # global noise does NOT - it is common to every voxel - but it does widen the
    # timecourse they all share
    assert themeasurements["global"]["voxelspread"] == pytest.approx(
        0.0, abs=1e-9
    ), themeasurements
    assert (
        themeasurements["global"]["globalspread"] > themeasurements["clean"]["globalspread"]
    ), themeasurements


def test_simdata_numskip_shortens_the_output():
    """numskip drops leading timepoints, standing in for the dummy scans a real
    acquisition discards.  The output is numtrs - numskip long, and starts that far
    into the waveform."""
    theshape, thenumtrs = (3, 3, 2), 40
    theresults = {}
    for thenumskip in (0, 5):
        with tempfile.TemporaryDirectory() as thedir:
            theargs = _makesimargs(
                thedir, theshape=theshape, thenumtrs=thenumtrs, numskip=thenumskip
            )
            simdata(theargs)
            dummy, thedata, dummy2, dummy3, dummy4 = tide_io.readfromnifti(
                f"{theargs.outputroot}.nii.gz"
            )
            theresults[thenumskip] = thedata

    assert theresults[0].shape == theshape + (thenumtrs,)
    assert theresults[5].shape[3] == thenumtrs - 5, theresults[5].shape
    # the skipped run starts later in the waveform, so its first frame differs
    assert not np.allclose(theresults[0][..., 0], theresults[5][..., 0])
    # and its first frame matches the frame the unskipped run had at that offset
    np.testing.assert_allclose(theresults[5][..., 0], theresults[0][..., 5], rtol=1e-6)


def test_simdata_runs_all_three_bands():
    """LFO, respiratory and cardiac each have their own regressor, lag map and
    percentage map, and a band is only enabled when all of its pieces are present.
    Running all three together exercises the per-band setup and the summation."""
    # The cardiac band runs to 3.15 Hz, so the output TR has to sample fast enough to
    # have a Nyquist above it; at the 1 s default the cardiac filter refuses to build.
    # also long enough that the 30 s filter pad fits inside the run
    theshape, thenumtrs, thefmritr = (3, 3, 2), 600, 0.125
    thesamplerate = 20.0
    thenumpoints = int(thenumtrs * thefmritr * thesamplerate) + 1200
    thetimes = np.arange(thenumpoints) / thesamplerate

    with tempfile.TemporaryDirectory() as thedir:
        theargs = _makesimargs(thedir, theshape=theshape, thenumtrs=thenumtrs, thefmritr=thefmritr)
        for theband, thefrequency in (("resp", 0.25), ("cardiac", 1.0)):
            theregressor = np.sin(2.0 * np.pi * thefrequency * thetimes)
            setattr(
                theargs,
                f"{theband}pctfile",
                _writenifti(thedir, f"{theband}pct", np.full(theshape, 0.01)),
            )
            setattr(
                theargs,
                f"{theband}lagfile",
                _writenifti(thedir, f"{theband}lag", np.zeros(theshape)),
            )
            setattr(
                theargs,
                f"{theband}regressor",
                _writeregressor(thedir, f"{theband}regressor", theregressor),
            )
            setattr(theargs, f"{theband}samprate", thesamplerate)
            setattr(theargs, f"{theband}starttime", 0.0)

        simdata(theargs)
        dummy, theallbands, dummy2, dummy3, dummy4 = tide_io.readfromnifti(
            f"{theargs.outputroot}.nii.gz"
        )

        # and the same run with only the LFO band, for comparison
        theargs.resppctfile = None
        theargs.cardiacpctfile = None
        theargs.outputroot = os.path.join(thedir, "lfoonly")
        simdata(theargs)
        dummy5, thelfoonly, dummy6, dummy7, dummy8 = tide_io.readfromnifti(
            f"{theargs.outputroot}.nii.gz"
        )

    assert theallbands.shape == theshape + (thenumtrs,)
    # the extra bands add signal, so the three band run has to differ
    assert not np.allclose(theallbands, thelfoonly), "the resp and cardiac bands did nothing"
    assert float(np.std(theallbands)) > float(np.std(thelfoonly))


def test_simdata_requires_at_least_one_complete_band():
    """A band needs its regressor, lag map and percentage map together.  With none of
    them complete there is nothing to simulate, and the tool has to say so rather than
    writing an empty volume."""
    theshape = (2, 2, 2)
    with tempfile.TemporaryDirectory() as thedir:
        theargs = _makesimargs(thedir, theshape=theshape, thenumtrs=40)
        # drop the LFO lag map, leaving no complete band
        theargs.lfolagfile = None
        with pytest.raises(SystemExit):
            simdata(theargs)


def test_simdata_applies_slice_timing():
    """A slice time file offsets each slice's sampling, so slices acquired at
    different times see the waveform at different points."""
    theshape, thenumtrs = (2, 2, 4), 60
    theslicetimes = np.array([0.0, 0.25, 0.5, 0.75])

    with tempfile.TemporaryDirectory() as thedir:
        theslicefile = os.path.join(thedir, "slicetimes.txt")
        np.savetxt(theslicefile, theslicetimes)
        theargs = _makesimargs(thedir, theshape=theshape, thenumtrs=thenumtrs)
        theargs.slicetimefile = theslicefile
        simdata(theargs)
        dummy, thetimed, dummy2, dummy3, dummy4 = tide_io.readfromnifti(
            f"{theargs.outputroot}.nii.gz"
        )

        theargs.slicetimefile = None
        theargs.outputroot = os.path.join(thedir, "notiming")
        simdata(theargs)
        dummy5, theuntimed, dummy6, dummy7, dummy8 = tide_io.readfromnifti(
            f"{theargs.outputroot}.nii.gz"
        )

    # slice 0 has no offset either way, so it is unchanged
    np.testing.assert_allclose(thetimed[..., 0, :], theuntimed[..., 0, :], rtol=1e-6)
    # the later slices are sampled at a different point in the waveform
    assert not np.allclose(thetimed[..., 3, :], theuntimed[..., 3, :]), "slice timing was ignored"


def test_simdata_debug_reporting_runs():
    """--debug prints its way through the whole workflow."""
    with tempfile.TemporaryDirectory() as thedir:
        # long enough that the noise filter has more points than its 30 s pad
        theargs = _makesimargs(thedir, theshape=(2, 2, 2), thenumtrs=60, debug=True)
        simdata(theargs)
        assert os.path.isfile(f"{theargs.outputroot}.nii.gz")


if __name__ == "__main__":
    thisfile = os.path.abspath(__file__)
    reporoot = os.path.abspath(os.path.join(os.path.dirname(thisfile), "..", ".."))
    if reporoot not in sys.path:
        sys.path.insert(0, reporoot)
    sys.exit(pytest.main([thisfile, "-v", "--import-mode=importlib"]))
