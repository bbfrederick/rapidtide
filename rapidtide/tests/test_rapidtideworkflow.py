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
"""Tests for rapidtide.workflows.rapidtide.

The full-run tests drive rapidtide_main end to end on the example dataset and check the
maps it produces.  They are slow, and because they only exercise runs that succeed, they
never reach the input validation that rejects a bad run.  This file covers the two things
they leave out: the module's small helper routines, tested directly, and the guard
clauses in rapidtide_main, driven with a tiny synthetic dataset so that a run which is
supposed to fail costs a couple of seconds rather than a minute and a half.
"""

import os
import sys

import nibabel as nb
import numpy as np
import pytest

import rapidtide.io as tide_io
import rapidtide.resample as tide_resample
import rapidtide.workflows.rapidtide_parser as rapidtide_parser
from rapidtide.tests.utils import run_rapidtide
from rapidtide.workflows.rapidtide import checkforzeromean, echocancel, setpassoptions

# ==================== checkforzeromean ====================


@pytest.mark.unit
def test_checkforzeromean_separates_demeaned_data_from_bold():
    """The routine decides whether a dataset has already been demeaned, which is how
    rapidtide knows a percentage change conversion would divide by roughly zero.

    Real BOLD sits on a large positive baseline, so its means dwarf its standard
    deviations; demeaned data has means near zero and therefore smaller than its
    standard deviations.  Both directions are checked, since a routine that always
    returned one answer would satisfy either case alone.
    """
    thedemeaned = np.random.RandomState(0).normal(0.0, 1.0, (10, 200))
    thebold = 1000.0 + np.random.RandomState(1).normal(0.0, 5.0, (10, 200))

    assert checkforzeromean(thedemeaned) is True
    assert checkforzeromean(thebold) is False


@pytest.mark.unit
def test_checkforzeromean_compares_means_not_individual_voxels():
    """The comparison is between the mean std and the mean mean over all voxels, so a
    single loud voxel does not flip the answer for an otherwise unambiguous dataset."""
    therng = np.random.RandomState(2)
    thedata = 1000.0 + therng.normal(0.0, 5.0, (20, 200))
    # one voxel with a huge standard deviation, but not enough to move the mean of
    # the standard deviations above the mean of the means
    thedata[0, :] = 1000.0 + therng.normal(0.0, 500.0, 200)

    assert checkforzeromean(thedata) is False


@pytest.mark.unit
def test_checkforzeromean_matches_its_own_docstring_example():
    """The docstring promises False for this input; a reader will believe it."""
    assert checkforzeromean(np.array([[1, 2, 3], [4, 5, 6]])) is False


# ==================== setpassoptions ====================


@pytest.mark.unit
def test_setpassoptions_copies_into_the_second_argument():
    """The pass dictionary is the source and the option dictionary is the destination.

    Getting the direction backwards would silently discard the per-pass settings, so
    both dictionaries are checked: one gains the keys, the other is left alone.
    """
    thepassdict = {"passnumber": 2, "refinetype": "pca"}
    theoptiondict = {"debug": False}

    theresult = setpassoptions(thepassdict, theoptiondict)

    assert theresult is None, "the update is in place, so nothing is returned"
    assert theoptiondict == {"debug": False, "passnumber": 2, "refinetype": "pca"}
    assert thepassdict == {"passnumber": 2, "refinetype": "pca"}, "the source was modified"


@pytest.mark.unit
def test_setpassoptions_overwrites_existing_keys():
    """A later pass has to be able to change a setting an earlier pass established."""
    thepassdict = {"passnumber": 3}
    theoptiondict = {"passnumber": 1, "untouched": "yes"}

    setpassoptions(thepassdict, theoptiondict)

    assert theoptiondict["passnumber"] == 3
    assert theoptiondict["untouched"] == "yes"


@pytest.mark.unit
def test_setpassoptions_with_nothing_to_pass_is_a_no_op():
    """An empty pass dictionary must not clear the options."""
    theoptiondict = {"kept": 1}
    setpassoptions({}, theoptiondict)
    assert theoptiondict == {"kept": 1}


# ==================== echocancel ====================


def _makelfo(numpoints, samplerate, seed=5, numfreqs=30):
    """Build a deterministic broadband signal in the LFO band.

    Parameters
    ----------
    numpoints : int
        Length of the signal.
    samplerate : float
        Sample rate in Hz.
    seed : int, optional
        Seed for the frequency and phase draws.
    numfreqs : int, optional
        Number of sinusoids to sum.

    Returns
    -------
    NDArray
        The generated signal.
    """
    therng = np.random.RandomState(seed)
    thetimes = np.arange(numpoints) / samplerate
    thesignal = np.zeros(numpoints, dtype=np.float64)
    for dummy in range(numfreqs):
        thefreq = therng.uniform(0.01, 0.2)
        thephase = therng.uniform(0.0, 2.0 * np.pi)
        thesignal += np.sin(2.0 * np.pi * thefreq * (thetimes - 0.0) + thephase)
    return thesignal


def _makeechoed(thebase, theoffset, thetimestep, thestrength):
    """Add a delayed copy of a signal to itself.

    Parameters
    ----------
    thebase : NDArray
        The clean signal.
    theoffset : float
        Echo delay in seconds.
    thetimestep : float
        Sample spacing in seconds.
    thestrength : float
        Amplitude of the echo relative to the signal.

    Returns
    -------
    NDArray
        The signal with an echo added.
    """
    theshift = int(round(theoffset / thetimestep))
    theecho = np.roll(thebase, theshift)
    theecho[:theshift] = 0.0
    return thebase + thestrength * theecho


def test_echocancel_subtracts_the_fitted_echo(tmp_path):
    """The returned timecourse must be the input minus the fitted echo.

    This is the relationship the docstring describes, and it is checked exactly rather
    than approximately: the echo is rebuilt here with the same resampler the routine
    uses, so the arithmetic has to agree to floating point.
    """
    thetimestep, thenumpoints = 1.0, 256
    theoffset = 4.0
    theinput = _makeechoed(_makelfo(thenumpoints, 1.0 / thetimestep), theoffset, thetimestep, 0.4)

    theoutput, thefit, theR2 = echocancel(
        theinput, theoffset, thetimestep, str(tmp_path / "ec"), 20
    )

    # rebuild the echo the way the routine does
    theechotc, dummy, dummy2, dummy3 = tide_resample.timeshift(
        theinput, theoffset / thetimestep, 20
    )
    theechotc[0 : int(np.ceil(theoffset / thetimestep))] = 0.0

    np.testing.assert_allclose(theoutput, theinput - thefit[0, 1] * theechotc, atol=1e-10)


def test_echocancel_leaves_a_residual_orthogonal_to_the_echo(tmp_path):
    """Least squares removal means the leftover signal carries no more of the echo.

    If the fit used the wrong coefficient - or the wrong sign - some echo would survive
    and this correlation would move away from zero.
    """
    thetimestep, thenumpoints = 1.0, 256
    theoffset = 4.0
    theinput = _makeechoed(_makelfo(thenumpoints, 1.0 / thetimestep), theoffset, thetimestep, 0.4)

    theoutput, thefit, theR2 = echocancel(
        theinput, theoffset, thetimestep, str(tmp_path / "ec"), 20
    )
    theechotc, dummy, dummy2, dummy3 = tide_resample.timeshift(
        theinput, theoffset / thetimestep, 20
    )
    theechotc[0 : int(np.ceil(theoffset / thetimestep))] = 0.0

    thecorrelation = np.corrcoef(theoutput, theechotc)[0, 1]
    assert abs(thecorrelation) < 1e-8, f"echo survives in the output, r={thecorrelation}"


def test_echocancel_scales_its_estimate_with_the_echo_strength(tmp_path):
    """A stronger echo has to produce a larger fitted coefficient.

    A routine returning a constant, or ignoring its input entirely, would still satisfy
    a single-value check; requiring the estimate to track the injected strength does
    not let that pass.
    """
    thetimestep, thenumpoints = 1.0, 256
    theoffset = 4.0
    thebase = _makelfo(thenumpoints, 1.0 / thetimestep)

    thecoefficients = []
    thestrengths = [0.0, 0.2, 0.4, 0.6]
    for thestrength in thestrengths:
        theinput = _makeechoed(thebase, theoffset, thetimestep, thestrength)
        dummy, thefit, dummy2 = echocancel(
            theinput, theoffset, thetimestep, str(tmp_path / f"ec{thestrength}"), 20
        )
        thecoefficients.append(thefit[0, 1])

    assert thecoefficients == sorted(thecoefficients), f"not monotonic: {thecoefficients}"
    # with no echo present there is nothing to remove
    assert abs(thecoefficients[0]) < 0.1, f"found an echo that was not there: {thecoefficients[0]}"
    # and the strongest echo is clearly detected
    assert thecoefficients[-1] > 0.2


def test_echocancel_writes_all_three_timecourses(tmp_path):
    """The output file documents what was removed, so it has to hold the input, the
    echo that was estimated, and the result - in that order, appended to one file."""
    thetimestep, thenumpoints = 1.0, 256
    theoffset = 4.0
    theinput = _makeechoed(_makelfo(thenumpoints, 1.0 / thetimestep), theoffset, thetimestep, 0.4)

    theroot = str(tmp_path / "ec")
    theoutput, thefit, theR2 = echocancel(theinput, theoffset, thetimestep, theroot, 20)

    (
        thesamplerate,
        thestarttime,
        thecolumns,
        thedata,
        dummy,
        dummy2,
        dummy3,
    ) = tide_io.readbidstsv(f"{theroot}_desc-echocancellation_timeseries")

    assert thecolumns == ["original", "echo", "filtered"]
    assert thesamplerate == pytest.approx(1.0 / thetimestep)
    np.testing.assert_allclose(thedata[0, :], theinput, atol=1e-10)
    np.testing.assert_allclose(thedata[2, :], theoutput, atol=1e-10)


def test_echocancel_blanks_the_wrapped_in_samples(tmp_path):
    """The first samples of the shifted copy have no real data behind them, so they are
    zeroed before fitting.  Leaving them in would let wrapped-around signal drive the
    regression."""
    thetimestep, thenumpoints = 1.0, 256
    theoffset = 6.0
    theinput = _makeechoed(_makelfo(thenumpoints, 1.0 / thetimestep), theoffset, thetimestep, 0.4)

    theroot = str(tmp_path / "ec")
    echocancel(theinput, theoffset, thetimestep, theroot, 20)

    dummy, dummy2, thecolumns, thedata, dummy3, dummy4, dummy5 = tide_io.readbidstsv(
        f"{theroot}_desc-echocancellation_timeseries"
    )
    theecho = thedata[thecolumns.index("echo"), :]
    thenumblanked = int(np.ceil(theoffset / thetimestep))

    np.testing.assert_allclose(theecho[:thenumblanked], 0.0, atol=1e-12)
    assert np.any(theecho[thenumblanked:] != 0.0), "the whole echo was blanked"


# ==================== rapidtide_main guard clauses ====================
#
# These drive the real workflow on a 6x6x2 synthetic dataset.  A run that is rejected
# costs a fraction of a second because the guard fires before any correlation is done;
# the few tests that run to completion take a handful of seconds.

THEAFFINE = np.diag([3.0, 3.0, 3.0, 1.0])


def _savevolume(thepath, thedata, thetr=None):
    """Write a NIfTI volume on the shared test grid.

    Parameters
    ----------
    thepath : str
        Where to write.
    thedata : NDArray
        3D or 4D array to save.
    thetr : float, optional
        If given, written into pixdim[4].

    Returns
    -------
    str
        The path written.
    """
    theimage = nb.Nifti1Image(thedata, THEAFFINE)
    theimage.header.set_xyzt_units("mm", "sec")
    if thetr is not None:
        theimage.header["pixdim"][4] = thetr
    nb.save(theimage, thepath)
    return thepath


@pytest.fixture(scope="module")
def thetinydataset(tmp_path_factory):
    """Build a small delayed-sinusoid dataset and the masks the guard tests need.

    The volume is deliberately tiny so that a full run takes seconds rather than the
    minute and a half the example dataset needs.  Module scoped because none of the
    tests modify it.

    Returns
    -------
    dict
        Paths keyed by role, plus the containing directory under "dir".
    """
    thedir = tmp_path_factory.mktemp("rapidtideguards")
    thexsize, theysize, thenumslices, thetimepoints, thetr = 6, 6, 2, 100, 1.5
    thetimes = np.arange(thetimepoints) * thetr
    therng = np.random.RandomState(0)
    thedata = np.zeros((thexsize, theysize, thenumslices, thetimepoints), dtype=np.float32)
    for thex in range(thexsize):
        for they in range(theysize):
            for thez in range(thenumslices):
                thedata[thex, they, thez, :] = (
                    100.0
                    + 10.0 * np.sin(2.0 * np.pi * 0.05 * (thetimes - 0.3 * (thex + they)))
                    + therng.normal(0.0, 0.5, thetimepoints)
                )

    theshape = (thexsize, theysize, thenumslices)
    theleft = np.zeros(theshape, dtype=np.float32)
    theleft[:3] = 1.0
    theright = np.zeros(theshape, dtype=np.float32)
    theright[3:] = 1.0
    theterritories = np.zeros(theshape, dtype=np.float32)
    theterritories[:3] = 1.0
    theterritories[3:] = 2.0

    thepaths = {"dir": str(thedir), "tr": thetr, "shape": theshape}
    thepaths["fmri"] = _savevolume(str(thedir / "tiny.nii.gz"), thedata, thetr=thetr)
    thepaths["fullmask"] = _savevolume(
        str(thedir / "full.nii.gz"), np.ones(theshape, dtype=np.float32)
    )
    thepaths["emptymask"] = _savevolume(
        str(thedir / "empty.nii.gz"), np.zeros(theshape, dtype=np.float32)
    )
    thepaths["leftmask"] = _savevolume(str(thedir / "left.nii.gz"), theleft)
    thepaths["rightmask"] = _savevolume(str(thedir / "right.nii.gz"), theright)
    thepaths["territories"] = _savevolume(str(thedir / "terr.nii.gz"), theterritories)
    # deliberately on a different grid than the fmri data
    thepaths["wronggrid"] = _savevolume(
        str(thedir / "wronggrid.nii.gz"), np.ones((4, 4, 2), dtype=np.float32)
    )
    thepaths["constantdelay"] = _savevolume(
        str(thedir / "delay.nii.gz"), np.full(theshape, 1.5, dtype=np.float32)
    )
    thelabelfile = thedir / "onelabel.txt"
    thelabelfile.write_text("only_one_label\n")
    thepaths["onelabel"] = str(thelabelfile)
    return thepaths


def _runtiny(thedataset, theoutputname, theextraargs):
    """Run rapidtide on the tiny dataset with a single pass.

    Parameters
    ----------
    thedataset : dict
        The fixture dictionary.
    theoutputname : str
        Basename for this run's outputs.
    theextraargs : list of str
        Arguments appended to the standard set.

    Returns
    -------
    str
        The output root, so the caller can read results back.
    """
    theoutputroot = os.path.join(thedataset["dir"], theoutputname)
    run_rapidtide(
        [thedataset["fmri"], theoutputroot, "--passes", "1", "--nprocs", "1", "--nodenoise"]
        + theextraargs
    )
    return theoutputroot


def test_a_lagmin_longer_than_the_run_is_rejected(thetinydataset):
    """A search range wider than half the acquisition cannot be satisfied, and saying
    so beats correlating against lags that do not exist."""
    with pytest.raises(ValueError, match="magnitude of lagmin exceeds"):
        _runtiny(
            thetinydataset,
            "lagmin",
            ["--corrmask", thetinydataset["fullmask"], "--searchrange", "-500", "10"],
        )


def test_a_lagmax_longer_than_the_run_is_rejected(thetinydataset):
    """The upper end of the search range is checked the same way as the lower end."""
    with pytest.raises(ValueError, match="magnitude of lagmax exceeds"):
        _runtiny(
            thetinydataset,
            "lagmax",
            ["--corrmask", thetinydataset["fullmask"], "--searchrange", "-10", "500"],
        )


def test_an_empty_correlation_mask_is_rejected(thetinydataset):
    """With no voxels to process there is nothing to compute; the run has to stop
    rather than produce empty maps."""
    with pytest.raises(ValueError, match="no voxels in the correlation mask"):
        _runtiny(thetinydataset, "emptymask", ["--corrmask", thetinydataset["emptymask"]])


def test_a_refine_include_mask_disjoint_from_the_corrmask_is_rejected(thetinydataset):
    """Both masks have voxels, but they do not overlap.

    An earlier check in maskutil only rejects a refine mask that is empty outright, so
    this condition - two perfectly valid masks that share no voxels - is caught here or
    not at all.  Refinement would otherwise proceed with nothing to refine from.
    """
    with pytest.raises(ValueError, match="refine include mask does not leave any voxels"):
        _runtiny(
            thetinydataset,
            "incdisjoint",
            [
                "--corrmask",
                thetinydataset["leftmask"],
                "--refineinclude",
                thetinydataset["rightmask"],
            ],
        )


def test_a_refine_exclude_mask_covering_the_corrmask_is_rejected(thetinydataset):
    """The exclude mask leaves plenty of voxels in the volume, just none of the ones
    being processed.  maskutil only complains when the exclude mask covers everything."""
    with pytest.raises(ValueError, match="refine exclude mask does not leave any voxels"):
        _runtiny(
            thetinydataset,
            "exccovers",
            [
                "--corrmask",
                thetinydataset["leftmask"],
                "--refineexclude",
                thetinydataset["leftmask"],
            ],
        )


def test_refine_include_and_exclude_masks_are_checked_together(thetinydataset):
    """When both are supplied it is their combination that has to leave voxels; each
    one alone is satisfiable here."""
    with pytest.raises(ValueError, match="refine include and exclude masks"):
        _runtiny(
            thetinydataset,
            "bothmasks",
            [
                "--corrmask",
                thetinydataset["leftmask"],
                "--refineinclude",
                thetinydataset["fullmask"],
                "--refineexclude",
                thetinydataset["leftmask"],
            ],
        )


def test_a_territory_map_on_the_wrong_grid_is_rejected(thetinydataset):
    """Region averaging indexes the fmri data with the template, so a mismatched grid
    would silently average the wrong voxels together."""
    with pytest.raises(SystemExit):
        _runtiny(
            thetinydataset,
            "terrgrid",
            [
                "--corrmask",
                thetinydataset["fullmask"],
                "--territorymap",
                thetinydataset["wronggrid"],
            ],
        )


def test_a_territory_label_count_mismatch_is_rejected(thetinydataset):
    """The template here has two regions but the label file names one, so the labels
    cannot be trusted to identify the columns that would be written."""
    with pytest.raises(SystemExit):
        _runtiny(
            thetinydataset,
            "terrlabels",
            [
                "--corrmask",
                thetinydataset["fullmask"],
                "--territorymap",
                thetinydataset["territories"],
                "--territorylabels",
                thetinydataset["onelabel"],
            ],
        )


def test_territory_timecourses_are_written_and_named(thetinydataset):
    """The happy path for the territory option.

    The two regions are the two halves of the volume, which carry different delays, so
    the extracted timecourses have to differ - averaging the whole volume for both
    regions would produce identical columns and still look like a working feature.
    """
    theoutputroot = _runtiny(
        thetinydataset,
        "terrok",
        [
            "--corrmask",
            thetinydataset["fullmask"],
            "--territorymap",
            thetinydataset["territories"],
        ],
    )

    (
        thesamplerate,
        thestarttime,
        thecolumns,
        thedata,
        dummy,
        dummy2,
        dummy3,
    ) = tide_io.readbidstsv(f"{theoutputroot}_desc-territory_timeseries")

    # two regions in the template, so two default labels, zero padded to one digit
    assert thecolumns == ["region_1", "region_2"]
    assert thedata.shape[0] == 2
    assert thesamplerate == pytest.approx(1.0 / thetinydataset["tr"])
    assert not np.allclose(thedata[0, :], thedata[1, :]), "both regions gave the same timecourse"


def test_a_fixed_delay_map_on_the_wrong_grid_is_rejected(thetinydataset):
    """A delay map is applied voxel by voxel, so its grid has to match the data."""
    with pytest.raises(ValueError, match="fixed delay map dimensions do not match"):
        _runtiny(
            thetinydataset,
            "delaygrid",
            [
                "--corrmask",
                thetinydataset["fullmask"],
                "--nodelayfit",
                "--initialdelay",
                thetinydataset["wronggrid"],
            ],
        )


def test_a_fixed_delay_map_is_actually_applied(thetinydataset):
    """Reading the map is not enough - the delays in it have to end up in the output.

    The map here is a constant 1.5 seconds, and with fitting disabled every voxel's
    reported delay must be exactly that.  Without this check the map could be read and
    discarded and the dimension test above would still pass.
    """
    theoutputroot = _runtiny(
        thetinydataset,
        "delayok",
        [
            "--corrmask",
            thetinydataset["fullmask"],
            "--nodelayfit",
            "--initialdelay",
            thetinydataset["constantdelay"],
        ],
    )

    dummy, thelagtimes, dummy2, dummy3, dummy4 = tide_io.readfromnifti(
        f"{theoutputroot}_desc-maxtime_map.nii.gz"
    )

    np.testing.assert_allclose(thelagtimes, 1.5, atol=1e-5)


def test_initialdelay_is_discarded_without_nodelayfit(thetinydataset):
    """--initialdelay on its own is silently dropped by the parser.

    rapidtide_parser sets initialdelayvalue back to None whenever fixdelay is false, and
    fixdelay is only set by --nodelayfit.  Nothing in the help for --initialdelay says
    so, and no warning is issued, so a run asking for an initial delay map quietly gets
    an ordinary unconstrained fit instead.

    Pinned rather than fixed: the sensible repair could be either to have --initialdelay
    imply --nodelayfit or to honour an initial delay while still fitting, and those are
    different features.  Note that two of the three configurations in
    test_fullrunrapidtide_v1 pass --initialdelay without --nodelayfit and are therefore
    not testing the delay handling they appear to.
    """
    thebaseargs = [
        thetinydataset["fmri"],
        os.path.join(thetinydataset["dir"], "parseronly"),
        "--initialdelay",
        thetinydataset["constantdelay"],
    ]

    thedropped, dummy = rapidtide_parser.process_args(inputargs=thebaseargs)
    thekept, dummy2 = rapidtide_parser.process_args(inputargs=thebaseargs + ["--nodelayfit"])

    assert thedropped["fixdelay"] is False
    assert thedropped["initialdelayvalue"] is None, "the option appears to be honoured now"
    assert thekept["fixdelay"] is True
    assert thekept["initialdelayvalue"] == thetinydataset["constantdelay"]


# ==================== the preparation pass ====================


@pytest.fixture(scope="module")
def thepreppassdataset(tmp_path_factory):
    """Build a dataset large enough for the preparation pass to rebuild the regressor.

    The rebuild only happens when at least fifty voxels survive selection, so this
    volume is bigger than the one the guard tests use.  The delays are drawn from a
    small set of values so that a modal lag exists, and the noise is low so that the
    fits are good enough to pass the R2 threshold.

    Returns
    -------
    dict
        Paths keyed by role, plus the containing directory under "dir".
    """
    thedir = tmp_path_factory.mktemp("preppass")
    thexsize, theysize, thenumslices, thetimepoints, thetr = 10, 10, 2, 100, 1.5
    thetimes = np.arange(thetimepoints) * thetr
    therng = np.random.RandomState(7)

    thesignal = np.zeros(thetimepoints, dtype=np.float64)
    for dummy in range(30):
        thefreq = therng.uniform(0.01, 0.15)
        thephase = therng.uniform(0.0, 2.0 * np.pi)
        thesignal += np.sin(2.0 * np.pi * thefreq * thetimes + thephase)
    thesignal /= np.std(thesignal)

    theshape = (thexsize, theysize, thenumslices)
    thedata = np.zeros(theshape + (thetimepoints,), dtype=np.float32)
    for thex in range(thexsize):
        for they in range(theysize):
            for thez in range(thenumslices):
                thedelay = 0.5 * ((thex + they) % 5)
                theshifted = np.interp(
                    thetimes - thedelay,
                    thetimes,
                    thesignal,
                    left=thesignal[0],
                    right=thesignal[-1],
                )
                thedata[thex, they, thez, :] = (
                    100.0 + 8.0 * theshifted + therng.normal(0.0, 0.3, thetimepoints)
                )

    thepaths = {"dir": str(thedir)}
    thepaths["fmri"] = _savevolume(str(thedir / "prep.nii.gz"), thedata, thetr=thetr)
    thepaths["fullmask"] = _savevolume(
        str(thedir / "prepmask.nii.gz"), np.ones(theshape, dtype=np.float32)
    )
    return thepaths


def _readregressorcolumns(theoutputroot):
    """Return the column names of the oversampled moving regressor file.

    Parameters
    ----------
    theoutputroot : str
        Output root of a completed run.

    Returns
    -------
    thecolumns : list of str
        The column names, one per regressor entry written.
    thedata : NDArray
        The regressor data, one row per column.
    """
    (
        dummy,
        dummy2,
        thecolumns,
        thedata,
        dummy3,
        dummy4,
        dummy5,
    ) = tide_io.readbidstsv(f"{theoutputroot}_desc-oversampledmovingregressor_timeseries")
    return thecolumns, thedata


@pytest.mark.slow
def test_preppass_rebuilds_the_regressor_from_clean_voxels(thepreppassdataset):
    """The whole point of --preppass is to replace the starting regressor with one
    built from voxels that fitted cleanly at a short delay.

    The rebuilt regressor is written as an extra entry in the moving regressor file, so
    its presence is what distinguishes a preparation pass that did something from one
    that ran and gave up.  It also has to differ from the regressor it replaced, which
    rules out the rebuild simply copying what was already there.
    """
    theoutputroot = os.path.join(thepreppassdataset["dir"], "prepdone")
    run_rapidtide(
        [
            thepreppassdataset["fmri"],
            theoutputroot,
            "--passes",
            "1",
            "--nprocs",
            "1",
            "--nodenoise",
            "--corrmask",
            thepreppassdataset["fullmask"],
            "--preppass",
            "--preppass-lag-window",
            "100",
            "--preppass-r2-threshold",
            "0.0",
        ]
    )

    thecolumns, thedata = _readregressorcolumns(theoutputroot)

    assert "pass1_preppass" in thecolumns, f"no rebuilt regressor was written: {thecolumns}"
    theoriginal = thedata[thecolumns.index("pass1"), :]
    therebuilt = thedata[thecolumns.index("pass1_preppass"), :]
    assert not np.allclose(theoriginal, therebuilt), "the rebuild reproduced the input regressor"


def _readgoodvoxelcount(theoutputroot):
    """Pull the number of voxels the preparation pass selected out of the run log.

    Parameters
    ----------
    theoutputroot : str
        Output root of a completed run.

    Returns
    -------
    int
        The count the preparation pass reported.

    Raises
    ------
    AssertionError
        If the log does not contain the line the preparation pass writes.
    """
    with open(f"{theoutputroot}_log.txt") as thefile:
        for theline in thefile:
            if "good voxels=" in theline:
                return int(theline.strip().split("good voxels=")[1].split()[0])
    raise AssertionError("the preparation pass did not report a voxel count")


@pytest.fixture(scope="module")
def thebimodaldataset(tmp_path_factory):
    """Build a dataset with two well separated delay populations.

    Half the voxels carry no delay and half are delayed by six seconds, which is far
    outside the selection window.  This makes the effect of the upper lag bound
    visible: only the early population may be selected.

    Returns
    -------
    dict
        Paths keyed by role, the voxel counts, and the containing directory.
    """
    thedir = tmp_path_factory.mktemp("bimodal")
    thexsize, theysize, thenumslices, thetimepoints, thetr = 10, 10, 2, 100, 1.5
    thetimes = np.arange(thetimepoints) * thetr
    therng = np.random.RandomState(7)

    thesignal = np.zeros(thetimepoints, dtype=np.float64)
    for dummy in range(30):
        thefreq = therng.uniform(0.01, 0.15)
        thephase = therng.uniform(0.0, 2.0 * np.pi)
        thesignal += np.sin(2.0 * np.pi * thefreq * thetimes + thephase)
    thesignal /= np.std(thesignal)

    theshape = (thexsize, theysize, thenumslices)
    thedata = np.zeros(theshape + (thetimepoints,), dtype=np.float32)
    thenumlate = 0
    for thex in range(thexsize):
        for they in range(theysize):
            for thez in range(thenumslices):
                theislate = thex >= thexsize // 2
                thenumlate += 1 if theislate else 0
                thedelay = 6.0 if theislate else 0.0
                theshifted = np.interp(
                    thetimes - thedelay,
                    thetimes,
                    thesignal,
                    left=thesignal[0],
                    right=thesignal[-1],
                )
                thedata[thex, they, thez, :] = (
                    100.0 + 8.0 * theshifted + therng.normal(0.0, 0.3, thetimepoints)
                )

    thepaths = {
        "dir": str(thedir),
        "numvoxels": thexsize * theysize * thenumslices,
        "numlate": thenumlate,
    }
    thepaths["fmri"] = _savevolume(str(thedir / "bimodal.nii.gz"), thedata, thetr=thetr)
    thepaths["fullmask"] = _savevolume(
        str(thedir / "bimodalmask.nii.gz"), np.ones(theshape, dtype=np.float32)
    )
    return thepaths


@pytest.mark.slow
def test_preppass_excludes_voxels_delayed_past_the_modal_lag(thebimodaldataset):
    """Selection keeps voxels at or before the modal lag, never after it.

    That upper bound is what makes the rebuilt regressor an early, arterial one rather
    than an average over the whole vascular tree.  Half of this dataset is delayed six
    seconds, well past the mode, so those voxels must not be counted - with the bound
    removed the count would be the whole volume.  The lag window and R2 threshold are
    opened up so that the upper bound is the only thing doing the rejecting.
    """
    theoutputroot = os.path.join(thebimodaldataset["dir"], "bimodal")
    run_rapidtide(
        [
            thebimodaldataset["fmri"],
            theoutputroot,
            "--passes",
            "1",
            "--nprocs",
            "1",
            "--nodenoise",
            "--corrmask",
            thebimodaldataset["fullmask"],
            "--preppass",
            "--preppass-lag-window",
            "100",
            "--preppass-r2-threshold",
            "0.0",
        ]
    )

    thegoodcount = _readgoodvoxelcount(theoutputroot)
    theearlycount = thebimodaldataset["numvoxels"] - thebimodaldataset["numlate"]

    assert thegoodcount > 0, "nothing was selected at all"
    assert thegoodcount <= theearlycount, (
        f"{thegoodcount} voxels selected out of {thebimodaldataset['numvoxels']}, but only "
        f"{theearlycount} are at or before the modal lag - late voxels are being included"
    )


@pytest.mark.slow
def test_preppass_skips_the_rebuild_when_no_voxels_qualify(thepreppassdataset):
    """An R2 threshold no voxel can meet leaves nothing to rebuild from.

    The run has to continue with the regressor it already had rather than fail or write
    an empty one, so the extra entry must be absent while the run still completes.
    """
    theoutputroot = os.path.join(thepreppassdataset["dir"], "prepskip")
    run_rapidtide(
        [
            thepreppassdataset["fmri"],
            theoutputroot,
            "--passes",
            "1",
            "--nprocs",
            "1",
            "--nodenoise",
            "--corrmask",
            thepreppassdataset["fullmask"],
            "--preppass",
            "--preppass-r2-threshold",
            "1.5",
        ]
    )

    thecolumns, dummy = _readregressorcolumns(theoutputroot)

    assert thecolumns == ["pass1"], f"a regressor was rebuilt after all: {thecolumns}"
    # the run still produced its usual output
    assert os.path.exists(f"{theoutputroot}_desc-maxtime_map.nii.gz")


if __name__ == "__main__":
    thisfile = os.path.abspath(__file__)
    reporoot = os.path.abspath(os.path.join(os.path.dirname(thisfile), "..", ".."))
    if reporoot not in sys.path:
        sys.path.insert(0, reporoot)
    sys.exit(pytest.main([thisfile, "-v", "--import-mode=importlib"]))
