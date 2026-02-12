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
from io import StringIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

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
    with tempfile.TemporaryDirectory() as tmpdir:
        args = _make_default_args(tmpdir)
        showxcorrx(args)


def showxcorrx_finds_correct_delay(debug=False):
    """Test that showxcorrx finds approximately correct delay."""
    if debug:
        print("showxcorrx_finds_correct_delay")
    with tempfile.TemporaryDirectory() as tmpdir:
        resfile = os.path.join(tmpdir, "results.txt")
        args = _make_default_args(
            tmpdir,
            summarymode=True,
            resoutputfile=resfile,
        )
        showxcorrx(args)
        # Read the results file
        with open(resfile, "r") as f:
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
    with tempfile.TemporaryDirectory() as tmpdir:
        resfile = os.path.join(tmpdir, "summary.txt")
        args = _make_default_args(
            tmpdir,
            summarymode=True,
            resoutputfile=resfile,
        )
        showxcorrx(args)
        assert os.path.exists(resfile)
        with open(resfile, "r") as f:
            content = f.read().strip()
        # Should contain tab-separated values
        assert "\t" in content


def showxcorrx_correlation_labelline(debug=False):
    """Test showxcorrx with label line output."""
    if debug:
        print("showxcorrx_correlation_labelline")
    with tempfile.TemporaryDirectory() as tmpdir:
        resfile = os.path.join(tmpdir, "labeled.txt")
        args = _make_default_args(
            tmpdir,
            summarymode=True,
            labelline=True,
            label="test_run",
            resoutputfile=resfile,
        )
        showxcorrx(args)
        with open(resfile, "r") as f:
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
    with tempfile.TemporaryDirectory() as tmpdir:
        args = _make_default_args(tmpdir, invert=True)
        showxcorrx(args)


def showxcorrx_correlation_trimdata(debug=False):
    """Test showxcorrx with trimdata option for unequal length signals."""
    if debug:
        print("showxcorrx_correlation_trimdata")
    sig1, sig2 = _make_broadband_signals()
    # Make sig2 shorter
    sig2_short = sig2[:1500]
    with tempfile.TemporaryDirectory() as tmpdir:
        args = _make_default_args(tmpdir, signal1=sig1, signal2=sig2_short, trimdata=True)
        showxcorrx(args)


def showxcorrx_auto_samplerate(debug=False):
    """Test showxcorrx with auto samplerate (defaults to 1.0)."""
    if debug:
        print("showxcorrx_auto_samplerate")
    # Generate signals for samplerate=1.0
    sig1, sig2 = _make_broadband_signals(samplerate=1.0, duration=500.0, delay=2.0, noise=3.0)
    with tempfile.TemporaryDirectory() as tmpdir:
        args = _make_default_args(tmpdir, signal1=sig1, signal2=sig2, samplerate="auto")
        showxcorrx(args)


def showxcorrx_corroutputfile(debug=False):
    """Test showxcorrx saves correlation function to file."""
    if debug:
        print("showxcorrx_corroutputfile")
    with tempfile.TemporaryDirectory() as tmpdir:
        corrfile = os.path.join(tmpdir, "corrfunc.txt")
        args = _make_default_args(tmpdir, corroutputfile=corrfile)
        showxcorrx(args)
        assert os.path.exists(corrfile)


def showxcorrx_detrendorder_zero(debug=False):
    """Test showxcorrx with no detrending."""
    if debug:
        print("showxcorrx_detrendorder_zero")
    with tempfile.TemporaryDirectory() as tmpdir:
        args = _make_default_args(tmpdir, detrendorder=0)
        showxcorrx(args)


def showxcorrx_detrendorder_high(debug=False):
    """Test showxcorrx with higher order detrending."""
    if debug:
        print("showxcorrx_detrendorder_high")
    with tempfile.TemporaryDirectory() as tmpdir:
        args = _make_default_args(tmpdir, detrendorder=3)
        showxcorrx(args)


def showxcorrx_hann_window(debug=False):
    """Test showxcorrx with hann window function."""
    if debug:
        print("showxcorrx_hann_window")
    with tempfile.TemporaryDirectory() as tmpdir:
        args = _make_default_args(tmpdir, windowfunc="hann")
        showxcorrx(args)


def showxcorrx_no_window(debug=False):
    """Test showxcorrx with no windowing."""
    if debug:
        print("showxcorrx_no_window")
    with tempfile.TemporaryDirectory() as tmpdir:
        args = _make_default_args(tmpdir, windowfunc="None")
        showxcorrx(args)


def showxcorrx_phat_weighting(debug=False):
    """Test showxcorrx with PHAT cross-correlation weighting."""
    if debug:
        print("showxcorrx_phat_weighting")
    with tempfile.TemporaryDirectory() as tmpdir:
        args = _make_default_args(tmpdir, corrweighting="phat")
        showxcorrx(args)


def showxcorrx_liang_weighting(debug=False):
    """Test showxcorrx with Liang cross-correlation weighting."""
    if debug:
        print("showxcorrx_liang_weighting")
    with tempfile.TemporaryDirectory() as tmpdir:
        args = _make_default_args(tmpdir, corrweighting="liang")
        showxcorrx(args)


def showxcorrx_eckart_weighting(debug=False):
    """Test showxcorrx with Eckart cross-correlation weighting."""
    if debug:
        print("showxcorrx_eckart_weighting")
    with tempfile.TemporaryDirectory() as tmpdir:
        args = _make_default_args(tmpdir, corrweighting="eckart")
        showxcorrx(args)


def showxcorrx_zero_delay(debug=False):
    """Test showxcorrx with zero delay between signals."""
    if debug:
        print("showxcorrx_zero_delay")
    sig1, sig2 = _make_broadband_signals(delay=0.0)
    with tempfile.TemporaryDirectory() as tmpdir:
        resfile = os.path.join(tmpdir, "results.txt")
        args = _make_default_args(
            tmpdir,
            signal1=sig1,
            signal2=sig2,
            summarymode=True,
            resoutputfile=resfile,
        )
        showxcorrx(args)
        with open(resfile, "r") as f:
            content = f.read().strip()
        fields = content.split("\t")
        maxdelay = float(fields[-1])
        # With zero delay, should find near-zero delay
        assert abs(maxdelay) < 1.0, f"Expected delay ~0, got {maxdelay}"


def showxcorrx_narrow_search_range(debug=False):
    """Test showxcorrx with narrow search range centered on true delay."""
    if debug:
        print("showxcorrx_narrow_search_range")
    with tempfile.TemporaryDirectory() as tmpdir:
        args = _make_default_args(tmpdir, lag_extrema=(-5.0, 5.0))
        showxcorrx(args)


def showxcorrx_cepstral(debug=False):
    """Test showxcorrx with cepstral delay estimation."""
    if debug:
        print("showxcorrx_cepstral")
    with tempfile.TemporaryDirectory() as tmpdir:
        args = _make_default_args(tmpdir, cepstral=True)
        showxcorrx(args)


def showxcorrx_calccoherence(debug=False):
    """Test showxcorrx with coherence calculation (no display)."""
    if debug:
        print("showxcorrx_calccoherence")
    with tempfile.TemporaryDirectory() as tmpdir:
        args = _make_default_args(tmpdir, calccoherence=True)
        showxcorrx(args)


def showxcorrx_calccsd(debug=False):
    """Test showxcorrx with cross-spectral density calculation."""
    if debug:
        print("showxcorrx_calccsd")
    with tempfile.TemporaryDirectory() as tmpdir:
        args = _make_default_args(tmpdir, calccsd=True)
        showxcorrx(args)


def showxcorrx_mutualinfo(debug=False):
    """Test showxcorrx with mutual information metric."""
    if debug:
        print("showxcorrx_mutualinfo")
    with tempfile.TemporaryDirectory() as tmpdir:
        args = _make_default_args(tmpdir, similaritymetric="mutualinfo")
        showxcorrx(args)


def showxcorrx_mutualinfo_summarymode(debug=False):
    """Test showxcorrx with mutual info in summarymode."""
    if debug:
        print("showxcorrx_mutualinfo_summarymode")
    with tempfile.TemporaryDirectory() as tmpdir:
        resfile = os.path.join(tmpdir, "mi_results.txt")
        args = _make_default_args(
            tmpdir,
            similaritymetric="mutualinfo",
            summarymode=True,
            resoutputfile=resfile,
        )
        showxcorrx(args)
        assert os.path.exists(resfile)


def showxcorrx_hybrid(debug=False):
    """Test showxcorrx with hybrid similarity metric."""
    if debug:
        print("showxcorrx_hybrid")
    with tempfile.TemporaryDirectory() as tmpdir:
        args = _make_default_args(tmpdir, similaritymetric="hybrid")
        showxcorrx(args)


def showxcorrx_with_lfo_filter(debug=False):
    """Test showxcorrx with LFO bandpass filtering."""
    if debug:
        print("showxcorrx_with_lfo_filter")
    with tempfile.TemporaryDirectory() as tmpdir:
        args = _make_default_args(tmpdir, filterband="lfo")
        showxcorrx(args)


def showxcorrx_timerange(debug=False):
    """Test showxcorrx with explicit time range."""
    if debug:
        print("showxcorrx_timerange")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use first 1500 samples (0 to 1499)
        args = _make_default_args(tmpdir, timerange=(0, 1500))
        showxcorrx(args)


def showxcorrx_sigma_limits(debug=False):
    """Test showxcorrx with custom sigma limits."""
    if debug:
        print("showxcorrx_sigma_limits")
    with tempfile.TemporaryDirectory() as tmpdir:
        args = _make_default_args(tmpdir, absmaxsigma=500.0, absminsigma=0.5)
        showxcorrx(args)


def showxcorrx_zeropadding(debug=False):
    """Test showxcorrx with zero padding enabled."""
    if debug:
        print("showxcorrx_zeropadding")
    with tempfile.TemporaryDirectory() as tmpdir:
        args = _make_default_args(tmpdir, zeropadding=100)
        showxcorrx(args)


def showxcorrx_butterworth_filter(debug=False):
    """Test showxcorrx with butterworth filter type."""
    if debug:
        print("showxcorrx_butterworth_filter")
    with tempfile.TemporaryDirectory() as tmpdir:
        args = _make_default_args(tmpdir, filterband="lfo", filtertype="butterworth")
        showxcorrx(args)


# ==================== Main test function ====================


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


if __name__ == "__main__":
    test_showxcorrx(debug=True)
