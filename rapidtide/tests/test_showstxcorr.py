#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2026-2026 Blaise Frederick
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
import argparse
import io
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rapidtide.workflows.showstxcorr import _get_parser, printthresholds, showstxcorr

# ==================== Helpers ====================


def _make_test_timecourses(samplerate=10.0, duration=200.0, delay=0.5, noise=3.0):
    """Create two synthetic broadband timecourses with a known delay relationship.

    Uses a sum of many sinusoids at different LFO frequencies with random phases
    to produce broadband signals whose cross-correlation has a single dominant
    peak that findmaxlag_gauss can fit (unlike pure sinusoids which produce
    oscillating cross-correlations).

    The noise level must be high enough (>= 3.0) so that after LFO bandpass
    filtering, the cross-correlation peak stays below 1.0. If the peak reaches
    ~1.0, the Gaussian fit overshoots slightly above 1.0 and findmaxlag_gauss
    rejects the fit.
    """
    npoints = int(samplerate * duration)
    t = np.arange(npoints) / samplerate

    # Create broadband signal by summing sinusoids at many frequencies in LFO band
    rng_signal = np.random.RandomState(42)
    freqs = np.linspace(0.01, 0.2, 30)
    phases = rng_signal.uniform(0, 2 * np.pi, len(freqs))
    amps = rng_signal.uniform(0.5, 1.5, len(freqs))

    signal1 = np.zeros(npoints)
    signal2 = np.zeros(npoints)
    for freq, phase, amp in zip(freqs, phases, amps):
        signal1 += amp * np.sin(2 * np.pi * freq * t + phase)
        signal2 += amp * np.sin(2 * np.pi * freq * (t - delay) + phase)

    # Add independent noise to each signal
    rng_noise = np.random.RandomState(99)
    signal1 += rng_noise.randn(npoints) * noise
    signal2 += rng_noise.randn(npoints) * noise

    return signal1, signal2


def _make_default_args(outfilename="/tmp/test_showstxcorr_out"):
    """Create a default args Namespace with all required attributes."""
    args = argparse.Namespace(
        infilename1="dummy_in1.txt",
        infilename2="dummy_in2.txt",
        outfilename=outfilename,
        samplerate=10.0,
        starttime=0.0,
        duration=1000000.0,
        stepsize=25.0,
        windowwidth=50.0,
        corrthresh=0.5,
        corrweighting="None",
        detrendorder=1,
        invert=False,
        display=False,
        debug=False,
        verbose=False,
        label="None",
        # Filter options (set by addfilteropts / postprocessfilteropts)
        filterband="lfo",
        filtertype="trapezoidal",
        filtorder=6,
        padseconds=30.0,
        passvec=None,
        stopvec=None,
        # Search range options (set by addsearchrangeopts / postprocesssearchrangeopts)
        lag_extrema=(-15.0, 15.0),
        lag_extrema_nondefault=False,
        initialdelayvalue=None,
        # Time range options (set by addtimerangeopts / postprocesstimerangeopts)
        timerange=(-1, -1),
        # Window options (set by addwindowopts)
        windowfunc="hamming",
        zeropadding=0,
    )
    return args


def _run_showstxcorr(signal1, signal2, args):
    """Helper to run showstxcorr with mocked IO. Returns saved text files dict."""
    saved_text = {}

    def mock_readvec(fname, **kwargs):
        if "in1" in fname:
            return signal1.copy()
        return signal2.copy()

    def mock_writenpvecs(data, fname, **kwargs):
        saved_text[fname] = np.array(data).copy()

    with (
        patch("rapidtide.workflows.showstxcorr.tide_io.readvec", side_effect=mock_readvec),
        patch("rapidtide.workflows.showstxcorr.tide_io.writenpvecs", side_effect=mock_writenpvecs),
    ):

        showstxcorr(args)

    return saved_text


# ==================== _get_parser tests ====================


def parser_basic(debug=False):
    """Test that parser creates successfully."""
    if debug:
        print("parser_basic")
    parser = _get_parser()
    assert parser is not None
    assert parser.prog == "showstxcorr"


def parser_required_args(debug=False):
    """Test parser has required positional arguments."""
    if debug:
        print("parser_required_args")
    parser = _get_parser()
    actions = {a.dest: a for a in parser._actions}
    assert "infilename1" in actions
    assert "infilename2" in actions
    assert "outfilename" in actions


def parser_defaults(debug=False):
    """Test default values for optional arguments."""
    if debug:
        print("parser_defaults")
    parser = _get_parser()
    with (
        tempfile.NamedTemporaryFile(suffix=".txt") as f1,
        tempfile.NamedTemporaryFile(suffix=".txt") as f2,
    ):
        args = parser.parse_args([f1.name, f2.name, "outroot"])
    assert args.samplerate == "auto"
    assert args.corrthresh == 0.5
    assert args.windowwidth == 50.0
    assert args.stepsize == 25.0
    assert args.starttime == 0.0
    assert args.duration == 1000000.0
    assert args.display is True
    assert args.debug is False
    assert args.verbose is False
    assert args.detrendorder == 1
    assert args.corrweighting == "None"
    assert args.invert is False
    assert args.label == "None"


def parser_samplerate(debug=False):
    """Test --samplerate option."""
    if debug:
        print("parser_samplerate")
    parser = _get_parser()
    with (
        tempfile.NamedTemporaryFile(suffix=".txt") as f1,
        tempfile.NamedTemporaryFile(suffix=".txt") as f2,
    ):
        args = parser.parse_args([f1.name, f2.name, "out", "--samplerate", "25.0"])
    assert args.samplerate == 25.0


def parser_sampletime(debug=False):
    """Test --sampletime option (inverts to samplerate)."""
    if debug:
        print("parser_sampletime")
    parser = _get_parser()
    with (
        tempfile.NamedTemporaryFile(suffix=".txt") as f1,
        tempfile.NamedTemporaryFile(suffix=".txt") as f2,
    ):
        args = parser.parse_args([f1.name, f2.name, "out", "--sampletime", "0.5"])
    assert args.samplerate == pytest.approx(2.0)


def parser_boolean_flags(debug=False):
    """Test boolean flag options."""
    if debug:
        print("parser_boolean_flags")
    parser = _get_parser()
    with (
        tempfile.NamedTemporaryFile(suffix=".txt") as f1,
        tempfile.NamedTemporaryFile(suffix=".txt") as f2,
    ):
        args = parser.parse_args(
            [
                f1.name,
                f2.name,
                "out",
                "--nodisplay",
                "--debug",
                "--verbose",
                "--invert",
            ]
        )
    assert args.display is False
    assert args.debug is True
    assert args.verbose is True
    assert args.invert is True


def parser_corr_options(debug=False):
    """Test correlation-related options."""
    if debug:
        print("parser_corr_options")
    parser = _get_parser()
    with (
        tempfile.NamedTemporaryFile(suffix=".txt") as f1,
        tempfile.NamedTemporaryFile(suffix=".txt") as f2,
    ):
        args = parser.parse_args(
            [
                f1.name,
                f2.name,
                "out",
                "--corrthresh",
                "0.3",
                "--windowwidth",
                "30.0",
                "--stepsize",
                "10.0",
                "--corrweighting",
                "phat",
                "--detrendorder",
                "2",
            ]
        )
    assert args.corrthresh == 0.3
    assert args.windowwidth == 30.0
    assert args.stepsize == 10.0
    assert args.corrweighting == "phat"
    assert args.detrendorder == 2


def parser_samplerate_sampletime_mutual_exclusion(debug=False):
    """Test that --samplerate and --sampletime are mutually exclusive."""
    if debug:
        print("parser_samplerate_sampletime_mutual_exclusion")
    parser = _get_parser()
    with (
        tempfile.NamedTemporaryFile(suffix=".txt") as f1,
        tempfile.NamedTemporaryFile(suffix=".txt") as f2,
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    f1.name,
                    f2.name,
                    "out",
                    "--samplerate",
                    "10.0",
                    "--sampletime",
                    "0.1",
                ]
            )


# ==================== printthresholds tests ====================


def printthresholds_basic(debug=False):
    """Test printthresholds outputs correct format."""
    if debug:
        print("printthresholds_basic")
    pcts = [0.5, 0.3, 0.1]
    thepercentiles = [0.95, 0.99, 0.999]
    labeltext = "Test thresholds:"

    captured = io.StringIO()
    with patch("sys.stdout", captured):
        printthresholds(pcts, thepercentiles, labeltext)

    output = captured.getvalue()
    assert "Test thresholds:" in output
    assert "p < 0.05" in output or "p < 0.050" in output
    assert "p < 0.01" in output or "p < 0.010" in output


def printthresholds_empty(debug=False):
    """Test printthresholds with empty lists."""
    if debug:
        print("printthresholds_empty")
    captured = io.StringIO()
    with patch("sys.stdout", captured):
        printthresholds([], [], "Empty:")

    output = captured.getvalue()
    assert "Empty:" in output


def printthresholds_single(debug=False):
    """Test printthresholds with a single threshold."""
    if debug:
        print("printthresholds_single")
    captured = io.StringIO()
    with patch("sys.stdout", captured):
        printthresholds([0.42], [0.95], "Single:")

    output = captured.getvalue()
    assert "Single:" in output
    assert "0.42" in output


# ==================== showstxcorr tests ====================


def showstxcorr_auto_samplerate_exit(debug=False):
    """Test that samplerate='auto' causes exit."""
    if debug:
        print("showstxcorr_auto_samplerate_exit")
    args = _make_default_args()
    args.samplerate = "auto"

    with pytest.raises(SystemExit):
        showstxcorr(args)


def showstxcorr_basic(debug=False):
    """Test basic showstxcorr workflow produces output files."""
    if debug:
        print("showstxcorr_basic")
    signal1, signal2 = _make_test_timecourses(samplerate=10.0, duration=200.0)
    args = _make_default_args()
    args.corrthresh = 0.1

    saved = _run_showstxcorr(signal1, signal2, args)

    outroot = "/tmp/test_showstxcorr_out"
    assert f"{outroot}_pearson.txt" in saved
    assert f"{outroot}_pvalue.txt" in saved
    assert f"{outroot}_Rvalue.txt" in saved
    assert f"{outroot}_delay.txt" in saved
    assert f"{outroot}_mask.txt" in saved
    assert f"{outroot}_timewarped.txt" in saved
    assert f"{outroot}_hiresdelayvals.txt" in saved


def showstxcorr_output_lengths(debug=False):
    """Test that output arrays have consistent lengths."""
    if debug:
        print("showstxcorr_output_lengths")
    signal1, signal2 = _make_test_timecourses(samplerate=10.0, duration=200.0)
    args = _make_default_args()
    args.corrthresh = 0.1

    saved = _run_showstxcorr(signal1, signal2, args)

    outroot = "/tmp/test_showstxcorr_out"
    pearson = saved[f"{outroot}_pearson.txt"]
    pvalue = saved[f"{outroot}_pvalue.txt"]
    rvalue = saved[f"{outroot}_Rvalue.txt"]
    delay = saved[f"{outroot}_delay.txt"]
    mask = saved[f"{outroot}_mask.txt"]

    # Pearson and pvalue come from shorttermcorr_1D, should have same length
    assert len(pearson) == len(pvalue)
    # Rvalue, delay, mask come from shorttermcorr_2D, should have same length
    assert len(rvalue) == len(delay) == len(mask)


def showstxcorr_correlated_signals(debug=False):
    """Test that correlated signals produce high correlation values."""
    if debug:
        print("showstxcorr_correlated_signals")
    signal1, signal2 = _make_test_timecourses(
        samplerate=10.0,
        duration=200.0,
        delay=0.5,
    )
    args = _make_default_args()
    args.corrthresh = 0.1

    saved = _run_showstxcorr(signal1, signal2, args)

    outroot = "/tmp/test_showstxcorr_out"
    pearson = saved[f"{outroot}_pearson.txt"]
    rvalue = saved[f"{outroot}_Rvalue.txt"]

    # Correlated signals should have high Pearson R values
    assert (
        np.mean(np.abs(pearson)) > 0.3
    ), f"Mean |Pearson R| = {np.mean(np.abs(pearson))}, expected > 0.3"
    # Max cross-correlation should be even higher
    assert (
        np.mean(np.abs(rvalue)) > 0.3
    ), f"Mean |Rvalue| = {np.mean(np.abs(rvalue))}, expected > 0.3"


def showstxcorr_invert(debug=False):
    """Test that --invert negates the second signal and flips correlation sign."""
    if debug:
        print("showstxcorr_invert")
    signal1, signal2 = _make_test_timecourses(
        samplerate=10.0,
        duration=200.0,
        delay=0.5,
    )

    # Run without invert
    args1 = _make_default_args(outfilename="/tmp/test_showstxcorr_noinv")
    args1.corrthresh = 0.1
    saved1 = _run_showstxcorr(signal1, signal2, args1)
    pearson_noinv = saved1["/tmp/test_showstxcorr_noinv_pearson.txt"]

    # Run with invert
    args2 = _make_default_args(outfilename="/tmp/test_showstxcorr_inv")
    args2.invert = True
    args2.corrthresh = 0.1
    saved2 = _run_showstxcorr(signal1, signal2, args2)
    pearson_inv = saved2["/tmp/test_showstxcorr_inv_pearson.txt"]

    # Inverting should flip the sign of correlations
    assert (
        np.mean(pearson_inv) * np.mean(pearson_noinv) < 0
    ), "Inversion should flip correlation sign"


def showstxcorr_zero_delay(debug=False):
    """Test showstxcorr with zero delay between signals."""
    if debug:
        print("showstxcorr_zero_delay")
    signal1, signal2 = _make_test_timecourses(
        samplerate=10.0,
        duration=200.0,
        delay=0.0,
    )
    args = _make_default_args(outfilename="/tmp/test_showstxcorr_zerodelay")
    args.corrthresh = 0.1

    saved = _run_showstxcorr(signal1, signal2, args)

    outroot = "/tmp/test_showstxcorr_zerodelay"
    delay = saved[f"{outroot}_delay.txt"]
    mask = saved[f"{outroot}_mask.txt"]

    # With zero delay, valid delay values should be near zero
    valid_delays = delay[np.where(mask > 0)]
    if len(valid_delays) > 0:
        assert (
            np.mean(np.abs(valid_delays)) < 1.0
        ), f"Mean |delay| = {np.mean(np.abs(valid_delays))}, expected < 1.0 for zero-delay signals"


def showstxcorr_starttime_duration(debug=False):
    """Test starttime and duration trim the data correctly."""
    if debug:
        print("showstxcorr_starttime_duration")
    signal1, signal2 = _make_test_timecourses(samplerate=10.0, duration=200.0)
    args_full = _make_default_args(outfilename="/tmp/test_showstxcorr_full")
    args_full.corrthresh = 0.1

    saved_full = _run_showstxcorr(signal1, signal2, args_full)

    # Now run with starttime=10s and duration=100s (subset of data)
    args_trim = _make_default_args(outfilename="/tmp/test_showstxcorr_trim")
    args_trim.starttime = 10.0
    args_trim.duration = 100.0
    args_trim.corrthresh = 0.1

    saved_trim = _run_showstxcorr(signal1, signal2, args_trim)

    outroot_full = "/tmp/test_showstxcorr_full"
    outroot_trim = "/tmp/test_showstxcorr_trim"
    assert f"{outroot_trim}_pearson.txt" in saved_trim
    assert f"{outroot_trim}_Rvalue.txt" in saved_trim

    # Trimmed run should produce fewer windows than full run
    pearson_full = saved_full[f"{outroot_full}_pearson.txt"]
    pearson_trim = saved_trim[f"{outroot_trim}_pearson.txt"]
    assert len(pearson_trim) < len(
        pearson_full
    ), f"Trimmed ({len(pearson_trim)}) should have fewer windows than full ({len(pearson_full)})"


def showstxcorr_timewarped_output(debug=False):
    """Test that timewarped output has same length as hiresdelayvals."""
    if debug:
        print("showstxcorr_timewarped_output")
    signal1, signal2 = _make_test_timecourses(
        samplerate=10.0,
        duration=200.0,
        delay=0.5,
    )
    args = _make_default_args()
    args.corrthresh = 0.1

    saved = _run_showstxcorr(signal1, signal2, args)

    outroot = "/tmp/test_showstxcorr_out"
    timewarped = saved[f"{outroot}_timewarped.txt"]
    hiresdelay = saved[f"{outroot}_hiresdelayvals.txt"]

    # Both should have the same length
    assert len(timewarped) == len(hiresdelay)


def showstxcorr_custom_corrweighting(debug=False):
    """Test showstxcorr with phat correlation weighting."""
    if debug:
        print("showstxcorr_custom_corrweighting")
    signal1, signal2 = _make_test_timecourses(samplerate=10.0, duration=200.0)
    args = _make_default_args(outfilename="/tmp/test_showstxcorr_phat")
    args.corrweighting = "phat"
    args.corrthresh = 0.1

    saved = _run_showstxcorr(signal1, signal2, args)

    outroot = "/tmp/test_showstxcorr_phat"
    assert f"{outroot}_pearson.txt" in saved
    assert f"{outroot}_Rvalue.txt" in saved


def showstxcorr_detrendorder_zero(debug=False):
    """Test showstxcorr with detrending disabled."""
    if debug:
        print("showstxcorr_detrendorder_zero")
    signal1, signal2 = _make_test_timecourses(samplerate=10.0, duration=200.0)
    args = _make_default_args(outfilename="/tmp/test_showstxcorr_nodt")
    args.detrendorder = 0
    args.corrthresh = 0.1

    saved = _run_showstxcorr(signal1, signal2, args)

    outroot = "/tmp/test_showstxcorr_nodt"
    assert f"{outroot}_pearson.txt" in saved


def showstxcorr_high_corrthresh(debug=False):
    """Test showstxcorr with very high corrthresh (may result in no valid points)."""
    if debug:
        print("showstxcorr_high_corrthresh")
    rng = np.random.RandomState(99)
    # Uncorrelated signals - low R values expected
    signal1 = rng.randn(2000).astype(np.float64)
    signal2 = rng.randn(2000).astype(np.float64)
    args = _make_default_args(outfilename="/tmp/test_showstxcorr_hithresh")
    args.corrthresh = 0.99  # Very high threshold

    # This may fail during polynomial fitting if no points pass threshold
    # Just verify it doesn't crash unexpectedly or produces output
    try:
        saved = _run_showstxcorr(signal1, signal2, args)
        # If it succeeds, check outputs exist
        outroot = "/tmp/test_showstxcorr_hithresh"
        assert f"{outroot}_pearson.txt" in saved
    except (np.linalg.LinAlgError, ValueError):
        # Polynomial fit may fail with too few valid points - that's acceptable
        pass


# ==================== Main test function ====================


def test_showstxcorr(debug=False):
    # _get_parser tests
    if debug:
        print("Running parser tests")
    parser_basic(debug=debug)
    parser_required_args(debug=debug)
    parser_defaults(debug=debug)
    parser_samplerate(debug=debug)
    parser_sampletime(debug=debug)
    parser_boolean_flags(debug=debug)
    parser_corr_options(debug=debug)
    parser_samplerate_sampletime_mutual_exclusion(debug=debug)

    # printthresholds tests
    if debug:
        print("Running printthresholds tests")
    printthresholds_basic(debug=debug)
    printthresholds_empty(debug=debug)
    printthresholds_single(debug=debug)

    # showstxcorr workflow tests
    if debug:
        print("Running showstxcorr workflow tests")
    showstxcorr_auto_samplerate_exit(debug=debug)
    showstxcorr_basic(debug=debug)
    showstxcorr_output_lengths(debug=debug)
    showstxcorr_correlated_signals(debug=debug)
    showstxcorr_invert(debug=debug)
    showstxcorr_zero_delay(debug=debug)
    showstxcorr_starttime_duration(debug=debug)
    showstxcorr_timewarped_output(debug=debug)
    showstxcorr_custom_corrweighting(debug=debug)
    showstxcorr_detrendorder_zero(debug=debug)
    showstxcorr_high_corrthresh(debug=debug)


if __name__ == "__main__":
    test_showstxcorr(debug=True)
