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

import matplotlib as mpl
import numpy as np

import pytest

import rapidtide.correlate as tide_corr
import rapidtide.filter as tide_filt
import rapidtide.miscmath as tide_math
import rapidtide.resample as tide_resample
import rapidtide.simFuncClasses as tide_simFuncClasses
import rapidtide.workflows.cleanregressor as tide_cleanregressor
from rapidtide.tests.utils import get_examples_path, get_test_temp_path, mse

# ==================== _compute_acf and sharpen_regressor ====================


def _makeechoedregressor(samplerate=10.0, numpoints=3000, echodelay=8.0, echoamp=0.7, seed=0):
    """A broadband regressor plus a delayed copy of itself.

    An echo in the regressor puts a sidelobe in its autocorrelation at the echo
    delay, which is exactly the structure sharpen_regressor exists to remove.  The
    plain version is returned alongside so tests can check that a regressor without
    an echo is left alone.

    Parameters
    ----------
    samplerate : float
        Samples per second.
    numpoints : int
        Length of the regressor.
    echodelay : float
        Echo delay, in seconds.
    echoamp : float
        Echo amplitude relative to the original.
    seed : int
        Seed for the component phases and amplitudes.

    Returns
    -------
    theplain, theechoed : NDArray
        The regressor without and with the echo.
    """
    thetimes = np.arange(numpoints) / samplerate
    therng = np.random.RandomState(seed)
    theplain = np.zeros(numpoints)
    for thefreq, thephase, theamp in zip(
        np.linspace(0.02, 0.15, 25),
        therng.uniform(0, 2 * np.pi, 25),
        therng.uniform(0.5, 1.5, 25),
    ):
        theplain += theamp * np.sin(2 * np.pi * thefreq * thetimes + thephase)
    theechoed = theplain + echoamp * np.roll(theplain, int(echodelay * samplerate))
    return theplain, theechoed


def compute_acf_tests(debug=False):
    """The ACF is the input to every sidelobe decision, so its scaling and axis have
    to be right.  corrnormalize already normalizes the energy, so a second division by
    n would leave the values around 1/n and no sidelobe would ever clear a threshold.
    """
    if debug:
        print("compute_acf_tests")

    thesamplerate, thenumpoints = 10.0, 2000
    theoversamptr = 1.0 / thesamplerate
    theplain, dummy = _makeechoedregressor(thesamplerate, thenumpoints)

    thelags, theacf = tide_cleanregressor._compute_acf(theplain, theoversamptr, 20.0)

    # zero lag is the signal's energy, which corrnormalize has already set to 1
    thezeroindex = int(np.argmin(np.abs(thelags)))
    assert theacf[thezeroindex] == pytest.approx(1.0, abs=0.02), theacf[thezeroindex]
    assert theacf[thezeroindex] == pytest.approx(theacf.max())

    # an autocorrelation is symmetric about zero lag
    np.testing.assert_allclose(theacf, theacf[::-1], atol=1e-9)

    # the lag axis is in seconds and spans the requested range
    assert thelags[thezeroindex] == pytest.approx(0.0)
    np.testing.assert_allclose(np.diff(thelags), theoversamptr)
    assert thelags[-1] == pytest.approx(20.0, abs=theoversamptr)

    # a shorter lagmax returns a shorter function
    theshortlags, theshortacf = tide_cleanregressor._compute_acf(theplain, theoversamptr, 5.0)
    assert len(theshortacf) < len(theacf)
    assert theshortlags[-1] == pytest.approx(5.0, abs=theoversamptr)


def sharpen_regressor_finds_and_removes_sidelobes(testtemproot, debug=False):
    """A regressor carrying an echo has a sidelobe at the echo delay.  Sharpening has
    to shrink it - that is the entire purpose of the routine."""
    if debug:
        print("sharpen_regressor_finds_and_removes_sidelobes")

    thesamplerate, thenumpoints, theechodelay = 10.0, 3000, 8.0
    theoversamptr = 1.0 / thesamplerate
    dummy, theechoed = _makeechoedregressor(thesamplerate, thenumpoints, theechodelay)

    thelags, theacf = tide_cleanregressor._compute_acf(theechoed, theoversamptr, 20.0)
    thesidelobesbefore = tide_corr.find_all_acf_sidelobes(thelags, theacf, ampthresh=0.2)
    assert thesidelobesbefore, "the fixture has no sidelobe to remove"
    # the sidelobe sits at the planted echo delay
    thepositions = sorted(abs(thelag) for thelag, dummy2 in thesidelobesbefore)
    assert thepositions[0] == pytest.approx(theechodelay, abs=0.5), thepositions

    thesharpened = tide_cleanregressor.sharpen_regressor(
        theechoed,
        theoversamptr,
        thesamplerate,
        0,
        thenumpoints - 1,
        20.0,
        100,
        os.path.join(testtemproot, "sharpentest"),
        debug=debug,
    )

    assert thesharpened.shape == theechoed.shape
    dummy3, theacfafter = tide_cleanregressor._compute_acf(thesharpened, theoversamptr, 20.0)
    thesidelobesafter = tide_corr.find_all_acf_sidelobes(thelags, theacfafter, ampthresh=0.2)

    thebeforemax = max(abs(theamp) for dummy4, theamp in thesidelobesbefore)
    theaftermax = max((abs(theamp) for dummy5, theamp in thesidelobesafter), default=0.0)
    if debug:
        print(f"  sidelobe amplitude {thebeforemax:.3f} -> {theaftermax:.3f}")
    assert theaftermax < thebeforemax, "sharpening did not reduce the sidelobe"

    # the output timecourse is written for inspection
    assert os.path.isfile(
        os.path.join(testtemproot, "sharpentest_desc-sharpenedregressor_timeseries.json")
    )


def sharpen_regressor_leaves_a_clean_regressor_alone(testtemproot, debug=False):
    """With nothing to sharpen the input is returned untouched.  Running the
    deconvolution anyway would distort a perfectly good regressor."""
    if debug:
        print("sharpen_regressor_leaves_a_clean_regressor_alone")

    # 3000 points: a shorter record leaves enough residual structure in the ACF of a
    # finite sum of sinusoids to trip the sidelobe finder on its own
    thesamplerate, thenumpoints = 10.0, 3000
    theoversamptr = 1.0 / thesamplerate
    theplain, dummy = _makeechoedregressor(thesamplerate, thenumpoints)

    thelags, theacf = tide_cleanregressor._compute_acf(theplain, theoversamptr, 20.0)
    assert not tide_corr.find_all_acf_sidelobes(
        thelags, theacf, ampthresh=0.2
    ), "the clean fixture already has sidelobes"

    theresult = tide_cleanregressor.sharpen_regressor(
        theplain,
        theoversamptr,
        thesamplerate,
        0,
        thenumpoints - 1,
        20.0,
        100,
        os.path.join(testtemproot, "sharpenclean"),
        debug=debug,
    )
    np.testing.assert_array_equal(theresult, theplain)


def sharpen_regressor_falls_back_when_wiener_does_not_help(testtemproot, debug=False):
    """Wiener deconvolution is tried first and only kept if it cuts the sidelobe by at
    least 20 percent; otherwise an iterative echo subtraction runs instead.  Forcing
    the acceptance threshold to be unreachable exercises that fallback."""
    if debug:
        print("sharpen_regressor_falls_back_when_wiener_does_not_help")

    thesamplerate, thenumpoints = 10.0, 3000
    theoversamptr = 1.0 / thesamplerate
    dummy, theechoed = _makeechoedregressor(thesamplerate, thenumpoints, 8.0)

    # a very low amplitude threshold makes find_all_acf_sidelobes report structure
    # that the Wiener step cannot clear, so the fallback has to take over
    theresult = tide_cleanregressor.sharpen_regressor(
        theechoed,
        theoversamptr,
        thesamplerate,
        0,
        thenumpoints - 1,
        20.0,
        100,
        os.path.join(testtemproot, "sharpenfallback"),
        ampthresh=0.05,
        max_iters=2,
        debug=debug,
    )
    assert theresult.shape == theechoed.shape
    assert np.all(np.isfinite(theresult))
    assert not np.allclose(theresult, theechoed), "the fallback changed nothing"


def test_cleanregressor(debug=False, local=False, displayplots=False):
    # set input and output directories
    exampleroot = get_examples_path(local)
    testtemproot = get_test_temp_path(local)

    outputname = os.path.join(testtemproot, "cleanregressortest")
    thepass = 1
    padtrs = 30
    fmrifreq = 1.0
    oversampfac = 2
    oversampfreq = oversampfac * fmrifreq
    theprefilter = tide_filt.NoncausalFilter("lfo")
    lagmin = -30
    lagmax = 30
    lagmininpts = int((lagmin * oversampfreq) - 0.5)
    lagmaxinpts = int((lagmax * oversampfreq) + 0.5)
    lagmod = 1000.0
    noiseamp = 0.25
    detrendorder = 3
    windowfunc = "hamming"

    tclen = 500
    osvalidsimcalcstart = 0
    osvalidsimcalcend = tclen * oversampfac

    theCorrelator = tide_simFuncClasses.Correlator(
        Fs=oversampfreq,
        ncprefilter=theprefilter,
        detrendorder=1,
        windowfunc="hamming",
        corrweighting="phat",
    )
    theFitter = tide_simFuncClasses.SimilarityFunctionFitter(
        lagmod=lagmod,
        lagmin=lagmin,
        lagmax=lagmax,
        debug=debug,
        enforcethresh=False,
        zerooutbadfit=False,
    )

    # make a reference timecourse
    rng = np.random.default_rng(seed=1234)
    basewave = theprefilter.apply(fmrifreq, rng.normal(loc=0.0, scale=1.0, size=tclen))
    noisewave = rng.normal(loc=0.0, scale=noiseamp, size=tclen)
    theparamsets = [
        [2.0, 0.0, False, 0],
        [2.5, 0.8, True, 0],
        [5.0, 0.5, True, 0],
        [7.5, 0.25, True, 100],
        [10.0, 0.1, True, 0],
    ]
    for paramset in theparamsets:
        echotime = paramset[0]
        echoamp = paramset[1]
        check_autocorrelation = paramset[2]
        osvalidsimcalcstart = paramset[3]
        if debug:
            print(
                "**********Start******************************************************************"
            )
            print(f"{echotime=}, {echoamp=}, {check_autocorrelation=}, {osvalidsimcalcstart=}")
            print(
                "*********************************************************************************"
            )
        theechotc, dummy, dummy, dummy = tide_resample.timeshift(
            basewave, echotime * oversampfreq, padtrs, doplot=displayplots, debug=debug
        )
        resampnonosref_y = basewave + echoamp * theechotc + noisewave
        resampref_y = tide_resample.upsample(resampnonosref_y, fmrifreq, oversampfreq)
        theCorrelator.setreftc(resampnonosref_y)
        referencetc = tide_math.corrnormalize(
            resampref_y[osvalidsimcalcstart:],
            detrendorder=detrendorder,
            windowfunc=windowfunc,
        )

        resampref_y = tide_resample.upsample(resampnonosref_y, fmrifreq, oversampfreq)

        (
            cleaned_resampref_y,
            cleaned_referencetc,
            cleaned_nonosreferencetc,
            despeckle_thresh,
            sidelobeamp,
            sidelobetime,
            lagmod,
            acwidth,
            absmaxsigma,
        ) = tide_cleanregressor.cleanregressor(
            outputname,
            thepass,
            referencetc,
            resampref_y,
            resampnonosref_y,
            fmrifreq,
            oversampfreq,
            osvalidsimcalcstart,
            osvalidsimcalcend,
            lagmininpts,
            lagmaxinpts,
            theFitter,
            theCorrelator,
            lagmin,
            lagmax,
            LGR=None,
            check_autocorrelation=check_autocorrelation,
            fix_autocorrelation=True,
            despeckle_thresh=5.0,
            lthreshval=0.0,
            fixdelay=False,
            detrendorder=detrendorder,
            windowfunc=windowfunc,
            respdelete=False,
            displayplots=displayplots,
            debug=debug,
            rt_floattype=np.float64,
        )
        print(f"\t{len(referencetc)=}")
        print(f"\t{len(resampref_y)=}")
        print(f"\t{len(resampnonosref_y)=}")
        print(f"\t{len(cleaned_resampref_y)=}")
        print(f"\t{len(cleaned_referencetc)=}")
        print(f"\t{len(cleaned_nonosreferencetc)=}")
        print(f"\t{check_autocorrelation=}")
        print(f"\t{despeckle_thresh=}")
        print(f"\t{sidelobeamp=}")
        print(f"\t{sidelobetime=}")
        print(f"\t{lagmod=}")
        print(f"\t{acwidth=}")
        print(f"\t{absmaxsigma=}")
        assert len(referencetc) == len(cleaned_referencetc)
        assert len(resampref_y) == len(cleaned_resampref_y)
        assert len(resampnonosref_y) == len(cleaned_nonosreferencetc)

        if debug:
            print(
                "*********************************************************************************"
            )
            print(f"{echotime=}, {echoamp=}, {check_autocorrelation=}, {osvalidsimcalcstart=}")
            print(
                "**************End****************************************************************"
            )


def _runwithsidelobe(autodespecklethresh, local=False, debug=False):
    """Run cleanregressor on a strongly periodic regressor, which puts a sidelobe in its
    autocorrelation at the driving period, and report the despeckle_thresh that comes back.

    A periodic driver rather than a delayed echo, because the autocorrelation of
    lfo-filtered noise already has its first sidelobe near 9 s from the passband alone -
    that is below 2 * 5.0, so the threshold raise would be a no-op and the test would
    not discriminate no matter where the echo was put."""
    outputname = os.path.join(get_test_temp_path(local), "autodespecklethreshtest")
    fmrifreq = 1.0
    oversampfreq = 2.0
    theprefilter = tide_filt.NoncausalFilter("lfo")
    lagmin, lagmax = -30, 30
    detrendorder, windowfunc = 3, "hamming"
    # the period has to be long enough that period / 2 clears the requested
    # despeckle_thresh of 5.0 s, or the raise would be a no-op and prove nothing
    tclen, theperiod = 500, 20.0

    theCorrelator = tide_simFuncClasses.Correlator(
        Fs=oversampfreq, ncprefilter=theprefilter, detrendorder=1, windowfunc=windowfunc
    )
    theFitter = tide_simFuncClasses.SimilarityFunctionFitter(
        lagmod=1000.0, lagmin=lagmin, lagmax=lagmax, enforcethresh=False, zerooutbadfit=False
    )

    rng = np.random.default_rng(seed=1234)
    thetimeaxis = np.arange(tclen) / fmrifreq
    resampnonosref_y = theprefilter.apply(
        fmrifreq,
        np.sin(2.0 * np.pi * thetimeaxis / theperiod)
        + rng.normal(loc=0.0, scale=0.25, size=tclen),
    )
    resampref_y = tide_resample.upsample(resampnonosref_y, fmrifreq, oversampfreq)
    theCorrelator.setreftc(resampnonosref_y)
    referencetc = tide_math.corrnormalize(
        resampref_y, detrendorder=detrendorder, windowfunc=windowfunc
    )

    thereturn = tide_cleanregressor.cleanregressor(
        outputname,
        1,
        referencetc,
        resampref_y,
        resampnonosref_y,
        fmrifreq,
        oversampfreq,
        0,
        tclen * 2,
        int((lagmin * oversampfreq) - 0.5),
        int((lagmax * oversampfreq) + 0.5),
        theFitter,
        theCorrelator,
        lagmin,
        lagmax,
        LGR=None,
        check_autocorrelation=True,
        fix_autocorrelation=True,
        despeckle_thresh=5.0,
        autodespecklethresh=autodespecklethresh,
        lthreshval=0.0,
        fixdelay=False,
        detrendorder=detrendorder,
        windowfunc=windowfunc,
        respdelete=False,
        displayplots=False,
        debug=debug,
        rt_floattype=np.float64,
    )
    despeckle_thresh, sidelobeamp, sidelobetime = thereturn[3], thereturn[4], thereturn[5]
    return despeckle_thresh, sidelobeamp, sidelobetime


def test_sharpenregressor(debug=False, local=False):
    """Entry point for the ACF sharpening tests."""
    testtemproot = get_test_temp_path(local)
    compute_acf_tests(debug=debug)
    sharpen_regressor_finds_and_removes_sidelobes(testtemproot, debug=debug)
    sharpen_regressor_leaves_a_clean_regressor_alone(testtemproot, debug=debug)
    sharpen_regressor_falls_back_when_wiener_does_not_help(testtemproot, debug=debug)


def test_autodespecklethresh(local=False, debug=False):
    """--noautodespecklethresh must hold despeckle_thresh at the requested value even
    when a sidelobe is found.  Without it the threshold is raised to sidelobetime / 2,
    which makes the effective threshold vary from run to run."""
    onthresh, onamp, ontime = _runwithsidelobe(True, local=local, debug=debug)
    offthresh, offamp, offtime = _runwithsidelobe(False, local=local, debug=debug)

    # the test is only meaningful if the sidelobe was actually detected, and only
    # discriminating if the raise would have moved the threshold
    assert (
        ontime is not None
    ), "no sidelobe detected - the test regressor is not exercising the path"
    assert ontime / 2.0 > 5.0, f"sidelobe at {ontime} s would not have raised the threshold"

    # detection itself must be untouched: the flag governs the response, not the detector
    assert ontime == offtime
    assert onamp == offamp

    assert onthresh == np.max([5.0, ontime / 2.0])
    assert offthresh == 5.0


if __name__ == "__main__":
    mpl.use("TkAgg")
    test_cleanregressor(debug=True, local=True, displayplots=True)
    test_autodespecklethresh(local=True, debug=True)
