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
from argparse import Namespace
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import pytest

import rapidtide.io as tide_io

from rapidtide.workflows.showxcorrx import (
    DEFAULT_SIGMAMAX,
    DEFAULT_SIGMAMIN,
    _get_parser,
    printthresholds,
    showxcorrx,
)

# ==================== Helpers ====================

SAMPLERATE = 10.0
DURATION = 200.0
NPOINTS = int(SAMPLERATE * DURATION)
DELAY = 2.0


def _make_broadband_signals(samplerate=SAMPLERATE, duration=DURATION, delay=DELAY, noise=3.0):
    """Generate broadband LFO signals with a known time delay.

    Uses sum of sinusoids at LFO frequencies (0.01-0.2 Hz) with random
    phases to produce a single Gaussian-shaped cross-correlation peak.
    """
    npoints = int(samplerate * duration)
    t = np.arange(npoints) / samplerate
    rng_signal = np.random.RandomState(42)
    freqs = np.linspace(0.01, 0.2, 30)
    phases = rng_signal.uniform(0, 2 * np.pi, len(freqs))
    amps = rng_signal.uniform(0.5, 1.5, len(freqs))
    signal1 = np.zeros(npoints)
    signal2 = np.zeros(npoints)
    for freq, phase, amp in zip(freqs, phases, amps):
        signal1 += amp * np.sin(2 * np.pi * freq * t + phase)
        signal2 += amp * np.sin(2 * np.pi * freq * (t - delay) + phase)
    rng_noise = np.random.RandomState(99)
    signal1 += rng_noise.randn(npoints) * noise
    signal2 += rng_noise.randn(npoints) * noise
    return signal1, signal2


def _write_test_file(filepath, data):
    """Write a 1D signal array to a text file."""
    np.savetxt(filepath, data)


def _make_default_args(tmpdir, signal1=None, signal2=None, **overrides):
    """Create a default args Namespace for showxcorrx.

    Writes signal data to temp files and constructs an args namespace
    with all required attributes.
    """
    if signal1 is None or signal2 is None:
        signal1, signal2 = _make_broadband_signals()

    f1 = os.path.join(tmpdir, "signal1.txt")
    f2 = os.path.join(tmpdir, "signal2.txt")
    _write_test_file(f1, signal1)
    _write_test_file(f2, signal2)

    args = Namespace(
        infilename1=f1,
        infilename2=f2,
        samplerate=SAMPLERATE,
        display=False,
        # Search range
        lag_extrema=(-15.0, 15.0),
        initialdelayvalue=None,
        # Time range
        timerange=(-1, -1),
        # Window options
        windowfunc="hamming",
        zeropadding=0,
        # Filter options
        filterband="None",
        passvec=None,
        stopvec=None,
        filtertype="trapezoidal",
        filtorder=6,
        padseconds=30.0,
        ncfiltpadtype="reflect",
        # Preprocessing
        detrendorder=1,
        trimdata=False,
        corrweighting="None",
        invert=False,
        label="None",
        controlvariablefile=None,
        # Additional calculations
        cepstral=False,
        calccsd=False,
        calccoherence=False,
        # Permutation
        numestreps=0,
        showprogressbar=False,
        permutationmethod="shuffle",
        nprocs=1,
        # Similarity function
        similaritymetric="correlation",
        absmaxsigma=DEFAULT_SIGMAMAX,
        absminsigma=DEFAULT_SIGMAMIN,
        smoothingtime=3.0,
        minorm=True,
        # Output
        resoutputfile=None,
        corroutputfile=None,
        summarymode=False,
        labelline=False,
        # Plot options
        colors=None,
        linewidths=None,
        legendloc=2,
        legends=None,
        dolegend=True,
        thetitle=None,
        showxax=True,
        showyax=True,
        xlabel=None,
        ylabel=None,
        outputfile=None,
        saveres=1000,
        fontscalefac=1.0,
        # Misc
        debug=False,
        verbose=False,
    )
    args.__dict__.update(overrides)
    return args


@contextmanager
def _showxcorrx_run(**overrides):
    """Build args in a temporary directory, execute showxcorrx, and yield context.

    The run happens with the temporary directory as the working directory.  That is
    not cosmetic: with --debug, showxcorrx sets dumpfiltered and writes
    filtereddata1.txt, filtereddata2.txt, correlator_filtereddata*.txt and
    MI_filtereddata*.txt using bare relative names, so they land wherever pytest
    happened to be invoked from and are left behind.  Running from the temporary
    directory means they are cleaned up with it.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        run_overrides = dict(overrides)
        for key in ("resoutputfile", "corroutputfile", "outputfile"):
            outpath = run_overrides.get(key)
            if isinstance(outpath, str) and not os.path.isabs(outpath):
                run_overrides[key] = os.path.join(tmpdir, outpath)
        args = _make_default_args(tmpdir, **run_overrides)
        theolddirectory = os.getcwd()
        try:
            os.chdir(tmpdir)
            showxcorrx(args)
        finally:
            os.chdir(theolddirectory)
        yield tmpdir, args


# ==================== Parser tests ====================


def parser_defaults(debug=False):
    """Test _get_parser returns parser with correct defaults."""
    if debug:
        print("parser_defaults")
    parser = _get_parser()
    args = parser.parse_args(["file1.txt", "file2.txt"])
    assert args.infilename1 == "file1.txt"
    assert args.infilename2 == "file2.txt"
    assert args.samplerate == "auto"
    assert args.detrendorder == 1
    assert args.trimdata is False
    assert args.corrweighting == "None"
    assert args.invert is False
    assert args.label == "None"
    assert args.cepstral is False
    assert args.calccsd is False
    assert args.calccoherence is False
    assert args.similaritymetric == "correlation"
    assert args.absmaxsigma == DEFAULT_SIGMAMAX
    assert args.absminsigma == DEFAULT_SIGMAMIN
    assert args.display is True
    assert args.debug is False
    assert args.verbose is False
    assert args.nprocs == 1
    assert args.minorm is True


def parser_samplerate(debug=False):
    """Test parser with explicit samplerate."""
    if debug:
        print("parser_samplerate")
    parser = _get_parser()
    args = parser.parse_args(["file1.txt", "file2.txt", "--samplerate", "10.0"])
    assert args.samplerate == pytest.approx(10.0)


def parser_sampletime(debug=False):
    """Test parser with sampletime (inverted to samplerate)."""
    if debug:
        print("parser_sampletime")
    parser = _get_parser()
    args = parser.parse_args(["file1.txt", "file2.txt", "--sampletime", "0.1"])
    assert args.samplerate == pytest.approx(10.0)


def parser_searchrange(debug=False):
    """Test parser with explicit search range."""
    if debug:
        print("parser_searchrange")
    parser = _get_parser()
    args = parser.parse_args(["file1.txt", "file2.txt", "--searchrange", "-5.0", "5.0"])
    assert args.lag_extrema[0] == pytest.approx(-5.0)
    assert args.lag_extrema[1] == pytest.approx(5.0)


def parser_detrendorder(debug=False):
    """Test parser with detrendorder option."""
    if debug:
        print("parser_detrendorder")
    parser = _get_parser()
    args = parser.parse_args(["file1.txt", "file2.txt", "--detrendorder", "3"])
    assert args.detrendorder == 3


def parser_corrweighting(debug=False):
    """Test parser with correlation weighting options."""
    if debug:
        print("parser_corrweighting")
    parser = _get_parser()
    for weight in ["None", "phat", "liang", "eckart"]:
        args = parser.parse_args(["file1.txt", "file2.txt", "--corrweighting", weight])
        assert args.corrweighting == weight


def parser_similaritymetric(debug=False):
    """Test parser with similarity metric options."""
    if debug:
        print("parser_similaritymetric")
    parser = _get_parser()
    for metric in ["correlation", "mutualinfo", "hybrid"]:
        args = parser.parse_args(["file1.txt", "file2.txt", "--similaritymetric", metric])
        assert args.similaritymetric == metric


def parser_sigma_limits(debug=False):
    """Test parser with sigmamax and sigmamin options."""
    if debug:
        print("parser_sigma_limits")
    parser = _get_parser()
    args = parser.parse_args(
        ["file1.txt", "file2.txt", "--sigmamax", "500.0", "--sigmamin", "0.5"]
    )
    assert args.absmaxsigma == pytest.approx(500.0)
    assert args.absminsigma == pytest.approx(0.5)


def parser_output_options(debug=False):
    """Test parser output-related options."""
    if debug:
        print("parser_output_options")
    parser = _get_parser()
    args = parser.parse_args(
        [
            "file1.txt",
            "file2.txt",
            "--outputfile",
            "results.txt",
            "--corroutputfile",
            "corr.txt",
            "--summarymode",
            "--labelline",
        ]
    )
    assert args.resoutputfile == "results.txt"
    assert args.corroutputfile == "corr.txt"
    assert args.summarymode is True
    assert args.labelline is True


def parser_preprocessing(debug=False):
    """Test parser preprocessing options."""
    if debug:
        print("parser_preprocessing")
    parser = _get_parser()
    args = parser.parse_args(
        [
            "file1.txt",
            "file2.txt",
            "--invert",
            "--trimdata",
            "--label",
            "test_label",
        ]
    )
    assert args.invert is True
    assert args.trimdata is True
    assert args.label == "test_label"


def parser_additional_calcs(debug=False):
    """Test parser additional calculation options."""
    if debug:
        print("parser_additional_calcs")
    parser = _get_parser()
    args = parser.parse_args(
        ["file1.txt", "file2.txt", "--cepstral", "--calccsd", "--calccoherence"]
    )
    assert args.cepstral is True
    assert args.calccsd is True
    assert args.calccoherence is True


def parser_nodisplay(debug=False):
    """Test parser --nodisplay option."""
    if debug:
        print("parser_nodisplay")
    parser = _get_parser()
    args = parser.parse_args(["file1.txt", "file2.txt", "--nodisplay"])
    assert args.display is False


def parser_nprocs(debug=False):
    """Test parser --nprocs option."""
    if debug:
        print("parser_nprocs")
    parser = _get_parser()
    args = parser.parse_args(["file1.txt", "file2.txt", "--nprocs", "4"])
    assert args.nprocs == 4


# ==================== printthresholds tests ====================


def printthresholds_basic(debug=False):
    """Test printthresholds prints formatted output."""
    if debug:
        print("printthresholds_basic")
    pcts = [0.5, 0.6, 0.7]
    thepercentiles = [0.95, 0.99, 0.995]
    # Just verify it runs without error
    printthresholds(pcts, thepercentiles, "Test thresholds:")


def printthresholds_single(debug=False):
    """Test printthresholds with single entry."""
    if debug:
        print("printthresholds_single")
    printthresholds([0.42], [0.95], "Single threshold:")


def printthresholds_empty(debug=False):
    """Test printthresholds with empty lists."""
    if debug:
        print("printthresholds_empty")
    printthresholds([], [], "Empty thresholds:")


# ==================== showxcorrx workflow tests ====================


def showxcorrx_correlation_default(debug=False):
    """Test showxcorrx basic correlation workflow with default settings."""
    if debug:
        print("showxcorrx_correlation_default")
    with _showxcorrx_run():
        pass


def showxcorrx_finds_correct_delay(debug=False):
    """Test that showxcorrx finds approximately correct delay."""
    if debug:
        print("showxcorrx_finds_correct_delay")
    with _showxcorrx_run(summarymode=True, resoutputfile="results.txt") as (tmpdir, args):
        # Read the results file
        with open(args.resoutputfile, "r") as f:
            content = f.read().strip()
        # Parse out the maxdelay value (last tab-separated field)
        fields = content.split("\t")
        maxdelay = float(fields[-1])
        # The delay should be close to DELAY (2.0 seconds)
        # The file already contains -maxdelay (the actual delay), so compare directly
        assert abs(maxdelay - DELAY) < 1.0, f"Expected delay ~{DELAY}, got {maxdelay}"


def showxcorrx_correlation_summarymode(debug=False):
    """Test showxcorrx with summarymode output."""
    if debug:
        print("showxcorrx_correlation_summarymode")
    with _showxcorrx_run(summarymode=True, resoutputfile="summary.txt") as (_tmpdir, args):
        assert os.path.exists(args.resoutputfile)
        with open(args.resoutputfile, "r") as f:
            content = f.read().strip()
        # Should contain tab-separated values
        assert "\t" in content


def showxcorrx_correlation_labelline(debug=False):
    """Test showxcorrx with label line output."""
    if debug:
        print("showxcorrx_correlation_labelline")
    with _showxcorrx_run(
        summarymode=True,
        labelline=True,
        label="test_run",
        resoutputfile="labeled.txt",
    ) as (_tmpdir, args):
        with open(args.resoutputfile, "r") as f:
            content = f.read().strip()
        lines = content.split("\n")
        # With labelline=True, should have header + data line
        assert len(lines) == 2
        assert "thelabel" in lines[0]
        assert "test_run" in lines[1]


def showxcorrx_correlation_invert(debug=False):
    """Test showxcorrx with inverted second timecourse."""
    if debug:
        print("showxcorrx_correlation_invert")
    with _showxcorrx_run(invert=True):
        pass


def showxcorrx_correlation_trimdata(debug=False):
    """Test showxcorrx with trimdata option for unequal length signals."""
    if debug:
        print("showxcorrx_correlation_trimdata")
    sig1, sig2 = _make_broadband_signals()
    # Make sig2 shorter
    sig2_short = sig2[:1500]
    with _showxcorrx_run(signal1=sig1, signal2=sig2_short, trimdata=True):
        pass


def showxcorrx_auto_samplerate(debug=False):
    """Test showxcorrx with auto samplerate (defaults to 1.0)."""
    if debug:
        print("showxcorrx_auto_samplerate")
    # Generate signals for samplerate=1.0
    sig1, sig2 = _make_broadband_signals(samplerate=1.0, duration=500.0, delay=2.0, noise=3.0)
    with _showxcorrx_run(signal1=sig1, signal2=sig2, samplerate="auto"):
        pass


def showxcorrx_corroutputfile(debug=False):
    """Test showxcorrx saves correlation function to file."""
    if debug:
        print("showxcorrx_corroutputfile")
    with _showxcorrx_run(corroutputfile="corrfunc.txt") as (_tmpdir, args):
        assert os.path.exists(args.corroutputfile)


def showxcorrx_detrendorder_zero(debug=False):
    """Test showxcorrx with no detrending."""
    if debug:
        print("showxcorrx_detrendorder_zero")
    with _showxcorrx_run(detrendorder=0):
        pass


def showxcorrx_detrendorder_high(debug=False):
    """Test showxcorrx with higher order detrending."""
    if debug:
        print("showxcorrx_detrendorder_high")
    with _showxcorrx_run(detrendorder=3):
        pass


def showxcorrx_hann_window(debug=False):
    """Test showxcorrx with hann window function."""
    if debug:
        print("showxcorrx_hann_window")
    with _showxcorrx_run(windowfunc="hann"):
        pass


def showxcorrx_no_window(debug=False):
    """Test showxcorrx with no windowing."""
    if debug:
        print("showxcorrx_no_window")
    with _showxcorrx_run(windowfunc="None"):
        pass


def showxcorrx_phat_weighting(debug=False):
    """Test showxcorrx with PHAT cross-correlation weighting."""
    if debug:
        print("showxcorrx_phat_weighting")
    with _showxcorrx_run(corrweighting="phat"):
        pass


def showxcorrx_liang_weighting(debug=False):
    """Test showxcorrx with Liang cross-correlation weighting."""
    if debug:
        print("showxcorrx_liang_weighting")
    with _showxcorrx_run(corrweighting="liang"):
        pass


def showxcorrx_eckart_weighting(debug=False):
    """Test showxcorrx with Eckart cross-correlation weighting."""
    if debug:
        print("showxcorrx_eckart_weighting")
    with _showxcorrx_run(corrweighting="eckart"):
        pass


def showxcorrx_zero_delay(debug=False):
    """Test showxcorrx with zero delay between signals."""
    if debug:
        print("showxcorrx_zero_delay")
    sig1, sig2 = _make_broadband_signals(delay=0.0)
    with _showxcorrx_run(
        signal1=sig1,
        signal2=sig2,
        summarymode=True,
        resoutputfile="results.txt",
    ) as (_tmpdir, args):
        with open(args.resoutputfile, "r") as f:
            content = f.read().strip()
        fields = content.split("\t")
        maxdelay = float(fields[-1])
        # With zero delay, should find near-zero delay
        assert abs(maxdelay) < 1.0, f"Expected delay ~0, got {maxdelay}"


def showxcorrx_narrow_search_range(debug=False):
    """Test showxcorrx with narrow search range centered on true delay."""
    if debug:
        print("showxcorrx_narrow_search_range")
    with _showxcorrx_run(lag_extrema=(-5.0, 5.0)):
        pass


def showxcorrx_cepstral(debug=False):
    """Test showxcorrx with cepstral delay estimation."""
    if debug:
        print("showxcorrx_cepstral")
    with _showxcorrx_run(cepstral=True):
        pass


def showxcorrx_calccoherence(debug=False):
    """Test showxcorrx with coherence calculation (no display)."""
    if debug:
        print("showxcorrx_calccoherence")
    with _showxcorrx_run(calccoherence=True):
        pass


def showxcorrx_calccsd(debug=False):
    """Test showxcorrx with cross-spectral density calculation."""
    if debug:
        print("showxcorrx_calccsd")
    with _showxcorrx_run(calccsd=True):
        pass


def showxcorrx_mutualinfo(debug=False):
    """Test showxcorrx with mutual information metric."""
    if debug:
        print("showxcorrx_mutualinfo")
    with _showxcorrx_run(similaritymetric="mutualinfo"):
        pass


def showxcorrx_mutualinfo_summarymode(debug=False):
    """Test showxcorrx with mutual info in summarymode."""
    if debug:
        print("showxcorrx_mutualinfo_summarymode")
    with _showxcorrx_run(
        similaritymetric="mutualinfo",
        summarymode=True,
        resoutputfile="mi_results.txt",
    ) as (_tmpdir, args):
        assert os.path.exists(args.resoutputfile)


def showxcorrx_hybrid(debug=False):
    """Test showxcorrx with hybrid similarity metric."""
    if debug:
        print("showxcorrx_hybrid")
    with _showxcorrx_run(similaritymetric="hybrid"):
        pass


def showxcorrx_with_lfo_filter(debug=False):
    """Test showxcorrx with LFO bandpass filtering."""
    if debug:
        print("showxcorrx_with_lfo_filter")
    with _showxcorrx_run(filterband="lfo"):
        pass


def showxcorrx_timerange(debug=False):
    """Test showxcorrx with explicit time range."""
    if debug:
        print("showxcorrx_timerange")
    # Use first 1500 samples (0 to 1499)
    with _showxcorrx_run(timerange=(0, 1500)):
        pass


def showxcorrx_sigma_limits(debug=False):
    """Test showxcorrx with custom sigma limits."""
    if debug:
        print("showxcorrx_sigma_limits")
    with _showxcorrx_run(absmaxsigma=500.0, absminsigma=0.5):
        pass


def showxcorrx_zeropadding(debug=False):
    """Test showxcorrx with zero padding enabled."""
    if debug:
        print("showxcorrx_zeropadding")
    with _showxcorrx_run(zeropadding=100):
        pass


def showxcorrx_butterworth_filter(debug=False):
    """Test showxcorrx with butterworth filter type."""
    if debug:
        print("showxcorrx_butterworth_filter")
    with _showxcorrx_run(filterband="lfo", filtertype="butterworth"):
        pass


# ==================== Main test function ====================


# ==================== Partial correlation ====================


def _make_confounded_signals(sharedseed=3, confoundseed=9, thegain=3.0):
    """Two signals with a real shared component that a confound is masking.

    The confound enters the two signals with OPPOSITE sign, so it drives the raw
    correlation strongly negative and hides the relationship that is really there.
    Partialling it out of BOTH signals recovers that relationship.

    The opposite sign matters.  With a confound shared in the same sense, removing it
    from either signal alone is enough to kill the spurious correlation, so a test
    that only checks "the correlation went down" cannot tell a fix that cleans both
    timecourses from one that cleans a single one, or from one that subtracts the
    intercept instead of the slope.  Making the confound mask a real relationship
    means only a fully correct implementation recovers it.

    Parameters
    ----------
    sharedseed : int
        Seed for the genuine shared component.
    confoundseed : int
        Seed for the masking confound.
    thegain : float
        How strongly the confound enters, relative to the shared component.

    Returns
    -------
    theconfound, thesignal1, thesignal2 : NDArray
        The confound and the two signals built from it.
    """

    def thebroadband(theseed):
        therng = np.random.RandomState(theseed)
        thetimes = np.arange(NPOINTS) / SAMPLERATE
        theresult = np.zeros(NPOINTS)
        for thefreq, thephase, theamp in zip(
            np.linspace(0.01, 0.2, 30),
            therng.uniform(0, 2 * np.pi, 30),
            therng.uniform(0.5, 1.5, 30),
        ):
            theresult += theamp * np.sin(2 * np.pi * thefreq * thetimes + thephase)
        return theresult

    theshared = thebroadband(sharedseed)
    theconfound = thebroadband(confoundseed)
    thesignal1 = theshared + thegain * theconfound
    thesignal2 = theshared - thegain * theconfound
    return theconfound, thesignal1, thesignal2


def _readsummary(tmpdir, signal1, signal2, controlvariable=None):
    """Run showxcorrx in summary mode and return its results as a dict.

    Parameters
    ----------
    tmpdir : str
        Working directory.
    signal1, signal2 : NDArray
        The two input timecourses.
    controlvariable : NDArray or None
        Written out and passed as --partialcorr when not None.

    Returns
    -------
    dict
        The summary line, keyed by its column headings.
    """
    theresultfile = os.path.join(tmpdir, "res.txt")
    thecontrolfile = None
    if controlvariable is not None:
        thecontrolfile = os.path.join(tmpdir, "controlvars.txt")
        np.savetxt(thecontrolfile, controlvariable)
    theargs = _make_default_args(
        tmpdir,
        signal1,
        signal2,
        summarymode=True,
        labelline=True,
        resoutputfile=theresultfile,
        controlvariablefile=thecontrolfile,
    )
    showxcorrx(theargs)
    thelines = open(theresultfile).read().strip().split("\n")
    return dict(zip(thelines[0].split("\t"), thelines[1].split("\t")))


def showxcorrx_partialcorr_removes_the_control_variable(debug=False):
    """--partialcorr has to actually partial the control variable out.

    Two things used to stop that happening.  The control variable file was read with
    tide_io.readnpvecs, which does not exist, so the option raised AttributeError
    before reading anything.  And the regression against it was computed and then
    discarded, so even reaching it left the data untouched and the reported
    correlation was an ordinary one.
    """
    if debug:
        print("showxcorrx_partialcorr_removes_the_control_variable")

    theconfound, thesignal1, thesignal2 = _make_confounded_signals()

    with tempfile.TemporaryDirectory() as tmpdir:
        theplain = _readsummary(tmpdir, thesignal1, thesignal2)
    with tempfile.TemporaryDirectory() as tmpdir:
        thepartial = _readsummary(tmpdir, thesignal1, thesignal2, controlvariable=theconfound)

    theplainr = float(theplain["pearson_R"])
    thepartialr = float(thepartial["pearson_R"])
    if debug:
        print(f"  pearson_R plain {theplainr:.4f}, partial {thepartialr:.4f}")

    # the confound masks the real relationship, driving the raw correlation negative
    assert theplainr < -0.5, f"the fixture is not actually masked: {theplainr}"
    # removing it from BOTH signals recovers the relationship underneath.  Cleaning
    # only one of them, or subtracting the intercept rather than the slope, lands
    # around 0.3 instead.
    assert thepartialr > 0.9, f"the control variable was not fully removed: {thepartialr}"

    # The cross correlation has to be partialled too, not just the Pearson value.  It
    # used to be computed from the untouched trimmed data, so --partialcorr left the
    # tool's headline number and its delay estimate alone.
    theplainxcorr = float(theplain["xcorr_R"])
    thepartialxcorr = float(thepartial["xcorr_R"])
    if debug:
        print(f"  xcorr_R plain {theplainxcorr:.4f}, partial {thepartialxcorr:.4f}")
    assert thepartialxcorr > 0.9, f"xcorr_R was only {thepartialxcorr}"
    # and it is still a correlation coefficient: partialling happens before the
    # normalization, so the residual's reduced amplitude does not drag it down
    assert thepartialxcorr > theplainxcorr

    # with the masking confound gone the two signals are the same, so zero delay
    thepartialdelay = float(thepartial["xcorr_maxdelay"])
    assert abs(thepartialdelay) < 0.5, f"partialled delay came out at {thepartialdelay}"


def showxcorrx_partialcorr_leaves_an_unrelated_control_alone(debug=False):
    """Partialling out something the signals do not contain must barely move the
    answer.  Without this a fix that simply zeroed the data would pass the test
    above."""
    if debug:
        print("showxcorrx_partialcorr_leaves_an_unrelated_control_alone")

    dummy, thesignal1, thesignal2 = _make_confounded_signals()
    theunrelated = np.random.RandomState(12345).normal(size=NPOINTS)

    with tempfile.TemporaryDirectory() as tmpdir:
        theplain = _readsummary(tmpdir, thesignal1, thesignal2)
    with tempfile.TemporaryDirectory() as tmpdir:
        thepartial = _readsummary(tmpdir, thesignal1, thesignal2, controlvariable=theunrelated)

    theplainr = float(theplain["pearson_R"])
    thepartialr = float(thepartial["pearson_R"])
    if debug:
        print(f"  pearson_R plain {theplainr:.4f}, unrelated control {thepartialr:.4f}")
    assert abs(thepartialr - theplainr) < 0.1, (
        f"an unrelated control variable changed the correlation from {theplainr} "
        f"to {thepartialr}"
    )


def showxcorrx_partialcorr_accepts_several_control_variables(debug=False):
    """The option takes the columns of a file, so more than one control variable has
    to be handled - each with its own coefficient."""
    if debug:
        print("showxcorrx_partialcorr_accepts_several_control_variables")

    theconfound, thesignal1, thesignal2 = _make_confounded_signals()
    # a second masking confound, also entering with opposite signs, so BOTH controls
    # have to be removed from BOTH signals before the relationship reappears
    thesecondconfound = np.random.RandomState(555).normal(size=NPOINTS)
    thesignal1 = thesignal1 + 3.0 * thesecondconfound
    thesignal2 = thesignal2 - 3.0 * thesecondconfound
    thecontrolvars = np.vstack((theconfound, thesecondconfound))

    with tempfile.TemporaryDirectory() as tmpdir:
        theplain = _readsummary(tmpdir, thesignal1, thesignal2)
    with tempfile.TemporaryDirectory() as tmpdir:
        thepartial = _readsummary(tmpdir, thesignal1, thesignal2, controlvariable=thecontrolvars.T)

    theplainr = float(theplain["pearson_R"])
    thepartialr = float(thepartial["pearson_R"])
    if debug:
        print(f"  pearson_R plain {theplainr:.4f}, two controls {thepartialr:.4f}")
    assert theplainr < -0.3, f"the fixture is not actually masked: {theplainr}"
    assert thepartialr > 0.9, f"two control variables were not both removed: {thepartialr}"


def showxcorrx_partialcorr_rejects_a_short_control_file(debug=False):
    """The control variables have to span the data.  A file that is too short would
    otherwise be silently broadcast or truncated against the timecourses, which is a
    quietly wrong answer rather than an error."""
    if debug:
        print("showxcorrx_partialcorr_rejects_a_short_control_file")

    dummy, thesignal1, thesignal2 = _make_confounded_signals()
    theshortcontrol = np.random.RandomState(1).normal(size=NPOINTS // 2)

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            _readsummary(tmpdir, thesignal1, thesignal2, controlvariable=theshortcontrol)
        except SystemExit:
            theoutcome = "SystemExit"
        else:
            theoutcome = "accepted"
    if debug:
        print(f"  short control file outcome: {theoutcome}")
    assert theoutcome == "SystemExit", "a control file shorter than the data was accepted"


# ==================== null distributions, reporting and plotting ====================


def showxcorrx_null_distribution_sets_thresholds(debug=False):
    """--numestreps builds a null distribution by permutation and reports the
    correlation values that clear it.  Those thresholds are what turn a correlation
    into a significance statement, so they have to appear and be ordered."""
    if debug:
        print("showxcorrx_null_distribution_sets_thresholds")

    with _showxcorrx_run(
        numestreps=100,
        summarymode=True,
        labelline=True,
        resoutputfile="res.txt",
    ) as (tmpdir, args):
        thelines = open(args.resoutputfile).read().strip().split("\n")

    theheadings = thelines[0].split("\t")
    thevalues = thelines[1].split("\t")
    if debug:
        print(f"  headings {theheadings}")
    assert len(theheadings) == len(thevalues), f"{theheadings} vs {thevalues}"

    # the null distribution adds significance columns that a plain run does not have
    thesignificancecolumns = [thename for thename in theheadings if "p=" in thename.lower()]
    assert thesignificancecolumns, f"no significance thresholds reported: {theheadings}"
    # both the Pearson and the cross correlation get their own threshold
    assert any("pearson" in thename.lower() for thename in thesignificancecolumns)
    assert any("xcorr" in thename.lower() for thename in thesignificancecolumns)

    # and the threshold is a real number that the measured correlation is compared to
    thevaluesbyname = dict(zip(theheadings, thevalues))
    for thename in thesignificancecolumns:
        assert np.isfinite(float(thevaluesbyname[thename])), thename


def showxcorrx_mutualinfo_with_numestreps_is_unsupported(debug=False):
    """--similaritymetric mutualinfo combined with --numestreps currently crashes.

    The null distribution block references thexsimfuncfitter, which is only built in
    the non-mutualinfo branch, and it feeds theCorrelator - which the mutualinfo path
    never runs, so thexcorr and xcorr_x do not exist either.  Making this work needs a
    decision about what a null distribution means for mutual information, so the
    combination is pinned as unsupported rather than guessed at.  If this test starts
    failing, the combination has been implemented and the test should assert on the
    thresholds instead.
    """
    if debug:
        print("showxcorrx_mutualinfo_with_numestreps_is_unsupported")

    try:
        with _showxcorrx_run(
            similaritymetric="mutualinfo",
            numestreps=50,
            summarymode=True,
            resoutputfile="res.txt",
        ) as (tmpdir, args):
            pass
    except UnboundLocalError as theerror:
        assert "thexsimfuncfitter" in str(theerror), theerror
    else:
        raise AssertionError(
            "mutualinfo with numestreps now runs - update this test to check the "
            "significance thresholds it produces"
        )


def showxcorrx_mismatched_sample_rates_are_rejected(debug=False):
    """Two timecourses sampled at different rates cannot be correlated point for
    point, so the run has to stop rather than silently comparing mismatched axes."""
    if debug:
        print("showxcorrx_mismatched_sample_rates_are_rejected")

    thesignal1, thesignal2 = _make_broadband_signals()
    with tempfile.TemporaryDirectory() as tmpdir:
        # write file 2 as a BIDS style tsv carrying its own, different sample rate
        thefile1 = os.path.join(tmpdir, "one.txt")
        _write_test_file(thefile1, thesignal1)
        thefile2 = os.path.join(tmpdir, "two_desc-x_timeseries")
        tide_io.writebidstsv(thefile2, thesignal2, SAMPLERATE / 2.0, columns=["x"])

        thefile1bids = os.path.join(tmpdir, "one_desc-y_timeseries")
        tide_io.writebidstsv(thefile1bids, thesignal1, SAMPLERATE, columns=["y"])

        theargs = _make_default_args(tmpdir)
        theargs.infilename1 = f"{thefile1bids}.json:y"
        theargs.infilename2 = f"{thefile2}.json:x"
        theargs.samplerate = "auto"

        with pytest.raises(SystemExit):
            showxcorrx(theargs)


def showxcorrx_verbose_and_debug_reporting(debug=False):
    """The verbose and debug branches run inside the workflow and print derived
    quantities; a stale f-string in one of them takes the whole run down."""
    if debug:
        print("showxcorrx_verbose_and_debug_reporting")

    with _showxcorrx_run(verbose=True, debug=True) as (tmpdir, args):
        pass


def showxcorrx_dumpfiltered_writes_the_preprocessed_timecourses(debug=False):
    """--debug turns on dumpfiltered, which writes the preprocessed timecourses out
    for inspection.  They are written with bare relative names, so the harness runs
    from a temporary directory and they go away with it."""
    if debug:
        print("showxcorrx_dumpfiltered_writes_the_preprocessed_timecourses")

    with _showxcorrx_run(debug=True) as (tmpdir, args):
        thedumped = sorted(thename for thename in os.listdir(tmpdir) if "filtereddata" in thename)

    if debug:
        print(f"  dumped {thedumped}")
    assert "filtereddata1.txt" in thedumped
    assert "filtereddata2.txt" in thedumped
    assert "correlator_filtereddata1.txt" in thedumped


def showxcorrx_does_not_hardcode_a_backend(debug=False):
    """The module must not pin an interactive matplotlib backend.

    It used to call mpl.use("TkAgg") inside the display block, which made --display
    fail outright on any headless machine - a cluster, a container, CI.  Choosing the
    backend belongs to the caller's environment, not to the tool.
    """
    if debug:
        print("showxcorrx_does_not_hardcode_a_backend")

    import inspect

    import rapidtide.workflows.showxcorrx as theworkflow

    thesource = inspect.getsource(theworkflow)
    for thebackend in ("TkAgg", "Qt5Agg", "QtAgg", "MacOSX", "WXAgg"):
        assert (
            f'use("{thebackend}")' not in thesource
        ), f"showxcorrx pins the {thebackend} backend again"


def showxcorrx_display_paths_run(debug=False):
    """--display draws the similarity function and, optionally, a styled plot.  The
    drawing code is a large block that never runs in a headless test otherwise, and
    it is where a mismatched colour or legend list shows up."""
    if debug:
        print("showxcorrx_display_paths_run")

    import matplotlib

    # Agg so nothing tries to open a window.  showxcorrx binds show() into its own
    # namespace with "from matplotlib.pyplot import ... show", so patching
    # matplotlib.pyplot.show would NOT reach it - the name to patch is the one on the
    # workflow module itself.
    matplotlib.use("Agg")

    with patch("rapidtide.workflows.showxcorrx.show") as mock_show:
        _rundisplaycases()
        assert mock_show.call_count > 0, "the display path never called show()"


def _rundisplaycases():
    """Run showxcorrx with plotting on, plain and with explicit styling.

    The styled case matters beyond coverage: supplying --legends used to raise
    "'list' object attribute 'append' is read-only", and even with that repaired the
    plotting call sat inside the else branch, so a supplied legend produced an empty
    figure.  Saving the figure and checking it is not blank catches both.
    """
    with _showxcorrx_run(display=True) as (tmpdir, args):
        pass
    # with explicit styling, which walks the colour and legend handling
    with _showxcorrx_run(
        display=True,
        colors="red",
        linewidths="2",
        legends="thelegend",
        thetitle="a title",
        xlabel="time",
        ylabel="correlation",
        outputfile="theplot.png",
    ) as (tmpdir, args):
        # a legend was supplied, so the data still has to have been drawn
        assert os.path.isfile(args.outputfile), "no figure was written"
        assert os.path.getsize(args.outputfile) > 5000, "the figure looks empty"


def test_showxcorrx(debug=False):
    # Parser tests
    if debug:
        print("Running parser tests")
    parser_defaults(debug=debug)
    parser_samplerate(debug=debug)
    parser_sampletime(debug=debug)
    parser_searchrange(debug=debug)
    parser_detrendorder(debug=debug)
    parser_corrweighting(debug=debug)
    parser_similaritymetric(debug=debug)
    parser_sigma_limits(debug=debug)
    parser_output_options(debug=debug)
    parser_preprocessing(debug=debug)
    parser_additional_calcs(debug=debug)
    parser_nodisplay(debug=debug)
    parser_nprocs(debug=debug)

    # printthresholds tests
    if debug:
        print("Running printthresholds tests")
    printthresholds_basic(debug=debug)
    printthresholds_single(debug=debug)
    printthresholds_empty(debug=debug)

    # showxcorrx workflow tests
    if debug:
        print("Running showxcorrx workflow tests")
    showxcorrx_correlation_default(debug=debug)
    showxcorrx_finds_correct_delay(debug=debug)
    showxcorrx_correlation_summarymode(debug=debug)
    showxcorrx_correlation_labelline(debug=debug)
    showxcorrx_correlation_invert(debug=debug)
    showxcorrx_correlation_trimdata(debug=debug)
    showxcorrx_auto_samplerate(debug=debug)
    showxcorrx_corroutputfile(debug=debug)
    showxcorrx_detrendorder_zero(debug=debug)
    showxcorrx_detrendorder_high(debug=debug)
    showxcorrx_hann_window(debug=debug)
    showxcorrx_no_window(debug=debug)
    showxcorrx_phat_weighting(debug=debug)
    showxcorrx_liang_weighting(debug=debug)
    showxcorrx_eckart_weighting(debug=debug)
    showxcorrx_zero_delay(debug=debug)
    showxcorrx_narrow_search_range(debug=debug)
    showxcorrx_cepstral(debug=debug)
    showxcorrx_calccoherence(debug=debug)
    showxcorrx_calccsd(debug=debug)
    showxcorrx_mutualinfo(debug=debug)
    showxcorrx_mutualinfo_summarymode(debug=debug)
    showxcorrx_hybrid(debug=debug)
    showxcorrx_with_lfo_filter(debug=debug)
    showxcorrx_timerange(debug=debug)
    showxcorrx_sigma_limits(debug=debug)
    showxcorrx_zeropadding(debug=debug)
    showxcorrx_butterworth_filter(debug=debug)

    # partial correlation tests
    if debug:
        print("Running partial correlation tests")
    showxcorrx_partialcorr_removes_the_control_variable(debug=debug)
    showxcorrx_partialcorr_leaves_an_unrelated_control_alone(debug=debug)
    showxcorrx_partialcorr_accepts_several_control_variables(debug=debug)
    showxcorrx_partialcorr_rejects_a_short_control_file(debug=debug)

    # null distributions, reporting and plotting
    if debug:
        print("Running null distribution and plotting tests")
    showxcorrx_null_distribution_sets_thresholds(debug=debug)
    showxcorrx_mutualinfo_with_numestreps_is_unsupported(debug=debug)
    showxcorrx_mismatched_sample_rates_are_rejected(debug=debug)
    showxcorrx_verbose_and_debug_reporting(debug=debug)
    showxcorrx_dumpfiltered_writes_the_preprocessed_timecourses(debug=debug)
    showxcorrx_does_not_hardcode_a_backend(debug=debug)
    showxcorrx_display_paths_run(debug=debug)


if __name__ == "__main__":
    test_showxcorrx(debug=True)
