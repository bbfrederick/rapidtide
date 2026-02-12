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
import io
import os
import sys
import tempfile

import numpy as np

import rapidtide.workflows.showarbcorr as sac

# ==================== Helpers ====================

SAMPLERATE = 10.0  # Hz
NPTS = 500


def _make_broadband_signal(npts, Fs, delay=0.0, seed=42):
    """Generate a broadband signal as a sum of sinusoids."""
    rng = np.random.RandomState(seed)
    t = np.arange(npts) / Fs
    signal = np.zeros(npts, dtype=float)
    for _ in range(30):
        freq = rng.uniform(0.01, 0.15)
        phase = rng.uniform(0, 2 * np.pi)
        signal += np.sin(2 * np.pi * freq * (t - delay) + phase)
    return signal


def _write_plain_signal_file(tmpdir, signal, name="signal.txt"):
    """Write a signal to a plain text file (no sidecar)."""
    filepath = os.path.join(tmpdir, name)
    np.savetxt(filepath, signal)
    return filepath


def _make_test_args(tmpdir, file1, file2, Fs1=SAMPLERATE, Fs2=SAMPLERATE, **kwargs):
    """Create args via parser, override with kwargs."""
    parser = sac._get_parser()
    args = parser.parse_args([file1, file2])
    args.samplerate1 = Fs1
    args.samplerate2 = Fs2
    args.display = False
    args.debug = kwargs.get("debug", False)
    args.verbose = kwargs.get("verbose", False)
    args.summarymode = kwargs.get("summarymode", False)
    args.labelline = kwargs.get("labelline", False)
    args.bipolar = kwargs.get("bipolar", False)
    args.invert = kwargs.get("invert", False)
    args.trimdata = kwargs.get("trimdata", False)
    args.label = kwargs.get("label", "None")
    args.outputfile = kwargs.get("outputfile", None)
    args.corroutputfile = kwargs.get("corroutputfile", None)
    args.graphfile = kwargs.get("graphfile", None)
    args.showprogressbar = False
    return args


# ==================== _get_parser tests ====================


def get_parser_returns_parser(debug=False):
    """Test _get_parser returns an ArgumentParser."""
    if debug:
        print("get_parser_returns_parser")
    import argparse

    parser = sac._get_parser()
    assert isinstance(parser, argparse.ArgumentParser)


def get_parser_required_args(debug=False):
    """Test _get_parser requires two input files."""
    if debug:
        print("get_parser_required_args")
    parser = sac._get_parser()
    try:
        parser.parse_args([])
        assert False, "Should have raised SystemExit"
    except SystemExit:
        pass


def get_parser_defaults(debug=False):
    """Test _get_parser default values."""
    if debug:
        print("get_parser_defaults")
    parser = sac._get_parser()
    with (
        tempfile.NamedTemporaryFile(suffix=".txt") as f1,
        tempfile.NamedTemporaryFile(suffix=".txt") as f2,
    ):
        args = parser.parse_args([f1.name, f2.name])
        assert args.samplerate1 is None
        assert args.samplerate2 is None
        assert args.display
        assert not args.debug
        assert not args.verbose
        assert args.detrendorder == 1
        assert args.corrweighting == "None"
        assert not args.invert
        assert args.label == "None"
        assert not args.bipolar
        assert args.outputfile is None
        assert args.corroutputfile is None
        assert args.graphfile is None
        assert args.showprogressbar
        assert args.nprocs == 1


def get_parser_samplerate1(debug=False):
    """Test _get_parser accepts --samplerate1."""
    if debug:
        print("get_parser_samplerate1")
    parser = sac._get_parser()
    with (
        tempfile.NamedTemporaryFile(suffix=".txt") as f1,
        tempfile.NamedTemporaryFile(suffix=".txt") as f2,
    ):
        args = parser.parse_args([f1.name, f2.name, "--samplerate1", "25.0"])
        assert args.samplerate1 == 25.0


def get_parser_samplerate2(debug=False):
    """Test _get_parser accepts --samplerate2."""
    if debug:
        print("get_parser_samplerate2")
    parser = sac._get_parser()
    with (
        tempfile.NamedTemporaryFile(suffix=".txt") as f1,
        tempfile.NamedTemporaryFile(suffix=".txt") as f2,
    ):
        args = parser.parse_args([f1.name, f2.name, "--samplerate2", "50.0"])
        assert args.samplerate2 == 50.0


def get_parser_nodisplay(debug=False):
    """Test _get_parser accepts --nodisplay."""
    if debug:
        print("get_parser_nodisplay")
    parser = sac._get_parser()
    with (
        tempfile.NamedTemporaryFile(suffix=".txt") as f1,
        tempfile.NamedTemporaryFile(suffix=".txt") as f2,
    ):
        args = parser.parse_args([f1.name, f2.name, "--nodisplay"])
        assert not args.display


def get_parser_debug(debug=False):
    """Test _get_parser accepts --debug."""
    if debug:
        print("get_parser_debug")
    parser = sac._get_parser()
    with (
        tempfile.NamedTemporaryFile(suffix=".txt") as f1,
        tempfile.NamedTemporaryFile(suffix=".txt") as f2,
    ):
        args = parser.parse_args([f1.name, f2.name, "--debug"])
        assert args.debug


def get_parser_verbose(debug=False):
    """Test _get_parser accepts --verbose."""
    if debug:
        print("get_parser_verbose")
    parser = sac._get_parser()
    with (
        tempfile.NamedTemporaryFile(suffix=".txt") as f1,
        tempfile.NamedTemporaryFile(suffix=".txt") as f2,
    ):
        args = parser.parse_args([f1.name, f2.name, "--verbose"])
        assert args.verbose


def get_parser_detrendorder(debug=False):
    """Test _get_parser accepts --detrendorder."""
    if debug:
        print("get_parser_detrendorder")
    parser = sac._get_parser()
    with (
        tempfile.NamedTemporaryFile(suffix=".txt") as f1,
        tempfile.NamedTemporaryFile(suffix=".txt") as f2,
    ):
        args = parser.parse_args([f1.name, f2.name, "--detrendorder", "3"])
        assert args.detrendorder == 3


def get_parser_corrweighting(debug=False):
    """Test _get_parser accepts --corrweighting."""
    if debug:
        print("get_parser_corrweighting")
    parser = sac._get_parser()
    with (
        tempfile.NamedTemporaryFile(suffix=".txt") as f1,
        tempfile.NamedTemporaryFile(suffix=".txt") as f2,
    ):
        for choice in ["None", "phat", "liang", "eckart"]:
            args = parser.parse_args([f1.name, f2.name, "--corrweighting", choice])
            assert args.corrweighting == choice


def get_parser_invert(debug=False):
    """Test _get_parser accepts --invert."""
    if debug:
        print("get_parser_invert")
    parser = sac._get_parser()
    with (
        tempfile.NamedTemporaryFile(suffix=".txt") as f1,
        tempfile.NamedTemporaryFile(suffix=".txt") as f2,
    ):
        args = parser.parse_args([f1.name, f2.name, "--invert"])
        assert args.invert


def get_parser_label(debug=False):
    """Test _get_parser accepts --label."""
    if debug:
        print("get_parser_label")
    parser = sac._get_parser()
    with (
        tempfile.NamedTemporaryFile(suffix=".txt") as f1,
        tempfile.NamedTemporaryFile(suffix=".txt") as f2,
    ):
        args = parser.parse_args([f1.name, f2.name, "--label", "test_label"])
        assert args.label == "test_label"


def get_parser_bipolar(debug=False):
    """Test _get_parser accepts --bipolar."""
    if debug:
        print("get_parser_bipolar")
    parser = sac._get_parser()
    with (
        tempfile.NamedTemporaryFile(suffix=".txt") as f1,
        tempfile.NamedTemporaryFile(suffix=".txt") as f2,
    ):
        args = parser.parse_args([f1.name, f2.name, "--bipolar"])
        assert args.bipolar


def get_parser_outputfile(debug=False):
    """Test _get_parser accepts --outputfile."""
    if debug:
        print("get_parser_outputfile")
    parser = sac._get_parser()
    with (
        tempfile.NamedTemporaryFile(suffix=".txt") as f1,
        tempfile.NamedTemporaryFile(suffix=".txt") as f2,
    ):
        args = parser.parse_args([f1.name, f2.name, "--outputfile", "results.txt"])
        assert args.outputfile == "results.txt"


def get_parser_corroutputfile(debug=False):
    """Test _get_parser accepts --corroutputfile."""
    if debug:
        print("get_parser_corroutputfile")
    parser = sac._get_parser()
    with (
        tempfile.NamedTemporaryFile(suffix=".txt") as f1,
        tempfile.NamedTemporaryFile(suffix=".txt") as f2,
    ):
        args = parser.parse_args([f1.name, f2.name, "--corroutputfile", "corr.txt"])
        assert args.corroutputfile == "corr.txt"


def get_parser_summarymode(debug=False):
    """Test _get_parser accepts --summarymode."""
    if debug:
        print("get_parser_summarymode")
    parser = sac._get_parser()
    with (
        tempfile.NamedTemporaryFile(suffix=".txt") as f1,
        tempfile.NamedTemporaryFile(suffix=".txt") as f2,
    ):
        args = parser.parse_args([f1.name, f2.name, "--summarymode"])
        assert args.summarymode


def get_parser_labelline(debug=False):
    """Test _get_parser accepts --labelline."""
    if debug:
        print("get_parser_labelline")
    parser = sac._get_parser()
    with (
        tempfile.NamedTemporaryFile(suffix=".txt") as f1,
        tempfile.NamedTemporaryFile(suffix=".txt") as f2,
    ):
        args = parser.parse_args([f1.name, f2.name, "--labelline"])
        assert args.labelline


def get_parser_noprogressbar(debug=False):
    """Test _get_parser accepts --noprogressbar."""
    if debug:
        print("get_parser_noprogressbar")
    parser = sac._get_parser()
    with (
        tempfile.NamedTemporaryFile(suffix=".txt") as f1,
        tempfile.NamedTemporaryFile(suffix=".txt") as f2,
    ):
        args = parser.parse_args([f1.name, f2.name, "--noprogressbar"])
        assert not args.showprogressbar


def get_parser_nprocs(debug=False):
    """Test _get_parser accepts --nprocs."""
    if debug:
        print("get_parser_nprocs")
    parser = sac._get_parser()
    with (
        tempfile.NamedTemporaryFile(suffix=".txt") as f1,
        tempfile.NamedTemporaryFile(suffix=".txt") as f2,
    ):
        args = parser.parse_args([f1.name, f2.name, "--nprocs", "4"])
        assert args.nprocs == 4


def get_parser_trimdata(debug=False):
    """Test _get_parser accepts --trimdata."""
    if debug:
        print("get_parser_trimdata")
    parser = sac._get_parser()
    with (
        tempfile.NamedTemporaryFile(suffix=".txt") as f1,
        tempfile.NamedTemporaryFile(suffix=".txt") as f2,
    ):
        args = parser.parse_args([f1.name, f2.name, "--trimdata"])
        assert args.trimdata


def get_parser_windowfunc(debug=False):
    """Test _get_parser accepts --windowfunc."""
    if debug:
        print("get_parser_windowfunc")
    parser = sac._get_parser()
    with (
        tempfile.NamedTemporaryFile(suffix=".txt") as f1,
        tempfile.NamedTemporaryFile(suffix=".txt") as f2,
    ):
        args = parser.parse_args([f1.name, f2.name, "--windowfunc", "hann"])
        assert args.windowfunc == "hann"


def get_parser_nonorm(debug=False):
    """Test _get_parser accepts --nonorm."""
    if debug:
        print("get_parser_nonorm")
    parser = sac._get_parser()
    with (
        tempfile.NamedTemporaryFile(suffix=".txt") as f1,
        tempfile.NamedTemporaryFile(suffix=".txt") as f2,
    ):
        args = parser.parse_args([f1.name, f2.name, "--nonorm"])
        assert not args.minorm


def get_parser_saveres(debug=False):
    """Test _get_parser accepts --saveres."""
    if debug:
        print("get_parser_saveres")
    parser = sac._get_parser()
    with (
        tempfile.NamedTemporaryFile(suffix=".txt") as f1,
        tempfile.NamedTemporaryFile(suffix=".txt") as f2,
    ):
        args = parser.parse_args([f1.name, f2.name, "--saveres", "300"])
        assert args.saveres == 300


# ==================== printthresholds tests ====================


def printthresholds_basic(debug=False):
    """Test printthresholds prints correct output."""
    if debug:
        print("printthresholds_basic")
    pcts = [1.96, 2.58, 3.29]
    thepercentiles = [0.95, 0.99, 0.999]
    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        sac.printthresholds(pcts, thepercentiles, "Test thresholds:")
    finally:
        sys.stdout = old_stdout
    output = captured.getvalue()
    assert "Test thresholds:" in output
    assert "0.050" in output
    assert "0.010" in output
    assert "0.001" in output
    assert "1.96" in output
    assert "2.58" in output
    assert "3.29" in output


def printthresholds_single(debug=False):
    """Test printthresholds with a single threshold."""
    if debug:
        print("printthresholds_single")
    pcts = [0.5]
    thepercentiles = [0.5]
    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        sac.printthresholds(pcts, thepercentiles, "Single:")
    finally:
        sys.stdout = old_stdout
    output = captured.getvalue()
    assert "Single:" in output
    assert "0.500" in output


def printthresholds_empty(debug=False):
    """Test printthresholds with empty lists."""
    if debug:
        print("printthresholds_empty")
    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        sac.printthresholds([], [], "Empty:")
    finally:
        sys.stdout = old_stdout
    output = captured.getvalue()
    assert "Empty:" in output
    # Only the label line, no threshold lines
    lines = output.strip().split("\n")
    assert len(lines) == 1


# ==================== showarbcorr tests ====================


def showarbcorr_identical_signals(debug=False):
    """Test showarbcorr with identical signals."""
    if debug:
        print("showarbcorr_identical_signals")
    with tempfile.TemporaryDirectory() as tmpdir:
        signal = _make_broadband_signal(NPTS, SAMPLERATE, delay=0.0, seed=42)
        file1 = _write_plain_signal_file(tmpdir, signal, name="sig1.txt")
        file2 = _write_plain_signal_file(tmpdir, signal, name="sig2.txt")

        corroutput = os.path.join(tmpdir, "corr.txt")
        args = _make_test_args(tmpdir, file1, file2, corroutputfile=corroutput, debug=debug)
        sac.showarbcorr(args)

        # Check that correlation output file was created
        assert os.path.exists(corroutput), "Correlation output file not created"
        # writenpvecs writes columns, so shape is (N, 2): col0=time, col1=corr
        data = np.loadtxt(corroutput)
        assert data.shape[1] == 2, f"Expected 2 columns, got {data.shape[1]}"
        # Peak correlation should be very high for identical signals
        assert np.max(data[:, 1]) > 0.9


def showarbcorr_delayed_signals(debug=False):
    """Test showarbcorr with delayed signals finds correct delay."""
    if debug:
        print("showarbcorr_delayed_signals")
    with tempfile.TemporaryDirectory() as tmpdir:
        delay = 1.0
        sig1 = _make_broadband_signal(NPTS, SAMPLERATE, delay=0.0, seed=42)
        sig2 = _make_broadband_signal(NPTS, SAMPLERATE, delay=delay, seed=42)
        file1 = _write_plain_signal_file(tmpdir, sig1, name="sig1.txt")
        file2 = _write_plain_signal_file(tmpdir, sig2, name="sig2.txt")

        outfile = os.path.join(tmpdir, "results.txt")
        args = _make_test_args(
            tmpdir, file1, file2, summarymode=True, outputfile=outfile, debug=debug
        )
        sac.showarbcorr(args)

        assert os.path.exists(outfile), "Output file not created"
        with open(outfile, "r") as f:
            content = f.read()
        if debug:
            print(f"  Output content: {content.strip()}")
        # Parse the tab-separated output
        parts = content.strip().split("\t")
        # parts: label(0), R(1), maxdelay(2), failreason(3)
        r_val = float(parts[1])
        maxdelay_val = float(parts[2])
        assert abs(r_val) > 0.8, f"Expected high R, got {r_val}"
        # maxdelay should be near the delay
        assert (
            abs(abs(maxdelay_val) - delay) < 0.5
        ), f"Expected delay near {delay}, got {maxdelay_val}"


def showarbcorr_summarymode_with_label(debug=False):
    """Test showarbcorr summary mode with label and labelline."""
    if debug:
        print("showarbcorr_summarymode_with_label")
    with tempfile.TemporaryDirectory() as tmpdir:
        signal = _make_broadband_signal(NPTS, SAMPLERATE, seed=42)
        file1 = _write_plain_signal_file(tmpdir, signal, name="sig1.txt")
        file2 = _write_plain_signal_file(tmpdir, signal, name="sig2.txt")

        outfile = os.path.join(tmpdir, "results.txt")
        args = _make_test_args(
            tmpdir,
            file1,
            file2,
            summarymode=True,
            labelline=True,
            label="test_run",
            outputfile=outfile,
            debug=debug,
        )
        sac.showarbcorr(args)

        with open(outfile, "r") as f:
            content = f.read()
        if debug:
            print(f"  Output content: {content.strip()}")
        lines = content.strip().split("\n")
        # Should have header line + data line
        assert len(lines) == 2
        header = lines[0]
        assert "thelabel" in header
        assert "xcorr_R" in header
        assert "xcorr_maxdelay" in header
        data_line = lines[1]
        assert "test_run" in data_line


def showarbcorr_summarymode_to_stdout(debug=False):
    """Test showarbcorr summary mode printing to stdout."""
    if debug:
        print("showarbcorr_summarymode_to_stdout")
    with tempfile.TemporaryDirectory() as tmpdir:
        signal = _make_broadband_signal(NPTS, SAMPLERATE, seed=42)
        file1 = _write_plain_signal_file(tmpdir, signal, name="sig1.txt")
        file2 = _write_plain_signal_file(tmpdir, signal, name="sig2.txt")

        args = _make_test_args(tmpdir, file1, file2, summarymode=True, debug=debug)
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            sac.showarbcorr(args)
        finally:
            sys.stdout = old_stdout
        output = captured.getvalue()
        # Should contain tab-separated values somewhere in the output
        # The last line should be the summary: label\tR\tdelay\tfailreason
        lines = output.strip().split("\n")
        last_line = lines[-1]
        parts = last_line.split("\t")
        assert len(parts) >= 4, f"Expected >=4 tab-separated parts, got {len(parts)}: {last_line}"


def showarbcorr_normal_mode(debug=False):
    """Test showarbcorr normal (non-summary) mode."""
    if debug:
        print("showarbcorr_normal_mode")
    with tempfile.TemporaryDirectory() as tmpdir:
        signal = _make_broadband_signal(NPTS, SAMPLERATE, seed=42)
        file1 = _write_plain_signal_file(tmpdir, signal, name="sig1.txt")
        file2 = _write_plain_signal_file(tmpdir, signal, name="sig2.txt")

        # Set label to None (Python None) to get the Crosscorrelation_Rmax output path
        args = _make_test_args(tmpdir, file1, file2, debug=debug)
        args.label = None
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            sac.showarbcorr(args)
        finally:
            sys.stdout = old_stdout
        output = captured.getvalue()
        if debug:
            print(f"  Normal mode output: {output}")
        assert "Pearson_R:" in output
        assert "Crosscorrelation_Rmax:" in output
        assert "Crosscorrelation_maxdelay:" in output


def showarbcorr_invert(debug=False):
    """Test showarbcorr with --invert."""
    if debug:
        print("showarbcorr_invert")
    with tempfile.TemporaryDirectory() as tmpdir:
        signal = _make_broadband_signal(NPTS, SAMPLERATE, seed=42)
        file1 = _write_plain_signal_file(tmpdir, signal, name="sig1.txt")
        file2 = _write_plain_signal_file(tmpdir, signal, name="sig2.txt")

        # Without invert
        outfile1 = os.path.join(tmpdir, "results_noinv.txt")
        args1 = _make_test_args(tmpdir, file1, file2, summarymode=True, outputfile=outfile1)
        sac.showarbcorr(args1)

        # With invert
        outfile2 = os.path.join(tmpdir, "results_inv.txt")
        args2 = _make_test_args(
            tmpdir, file1, file2, invert=True, summarymode=True, outputfile=outfile2
        )
        sac.showarbcorr(args2)

        # parts: label(0), R(1), delay(2), failreason(3)
        with open(outfile1) as f:
            r1 = float(f.read().strip().split("\t")[1])
        with open(outfile2) as f:
            r2 = float(f.read().strip().split("\t")[1])
        if debug:
            print(f"  R without invert: {r1}, with invert: {r2}")
        # Inverted correlation should flip sign
        assert r1 * r2 < 0, f"Expected opposite signs, got {r1} and {r2}"


def showarbcorr_corroutputfile(debug=False):
    """Test showarbcorr writes correlation function to file."""
    if debug:
        print("showarbcorr_corroutputfile")
    with tempfile.TemporaryDirectory() as tmpdir:
        signal = _make_broadband_signal(NPTS, SAMPLERATE, seed=42)
        file1 = _write_plain_signal_file(tmpdir, signal, name="sig1.txt")
        file2 = _write_plain_signal_file(tmpdir, signal, name="sig2.txt")

        corrfile = os.path.join(tmpdir, "corrfunc.txt")
        args = _make_test_args(tmpdir, file1, file2, corroutputfile=corrfile, debug=debug)
        sac.showarbcorr(args)

        assert os.path.exists(corrfile)
        # writenpvecs writes columns: shape is (N, 2) where col0=time, col1=corr
        data = np.loadtxt(corrfile)
        assert data.shape[1] == 2, f"Expected 2 columns, got {data.shape[1]}"
        assert data.shape[0] > 10  # should have many points
        # Time axis should be centered near zero
        timeaxis = data[:, 0]
        assert np.min(timeaxis) < 0
        assert np.max(timeaxis) > 0


def showarbcorr_trimdata(debug=False):
    """Test showarbcorr with --trimdata for unequal length signals."""
    if debug:
        print("showarbcorr_trimdata")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create two signals of different lengths
        sig1 = _make_broadband_signal(500, SAMPLERATE, seed=42)
        sig2 = _make_broadband_signal(300, SAMPLERATE, seed=42)
        file1 = _write_plain_signal_file(tmpdir, sig1, name="sig1.txt")
        file2 = _write_plain_signal_file(tmpdir, sig2, name="sig2.txt")

        outfile = os.path.join(tmpdir, "results.txt")
        args = _make_test_args(
            tmpdir, file1, file2, trimdata=True, summarymode=True, outputfile=outfile, debug=debug
        )
        sac.showarbcorr(args)

        assert os.path.exists(outfile)


def showarbcorr_different_samplerates(debug=False):
    """Test showarbcorr with different sample rates."""
    if debug:
        print("showarbcorr_different_samplerates")
    with tempfile.TemporaryDirectory() as tmpdir:
        fs1 = 10.0
        fs2 = 20.0
        sig1 = _make_broadband_signal(500, fs1, seed=42)
        sig2 = _make_broadband_signal(1000, fs2, seed=99)
        file1 = _write_plain_signal_file(tmpdir, sig1, name="sig1.txt")
        file2 = _write_plain_signal_file(tmpdir, sig2, name="sig2.txt")

        outfile = os.path.join(tmpdir, "results.txt")
        args = _make_test_args(
            tmpdir,
            file1,
            file2,
            Fs1=fs1,
            Fs2=fs2,
            summarymode=True,
            outputfile=outfile,
            debug=debug,
        )
        sac.showarbcorr(args)

        assert os.path.exists(outfile)
        with open(outfile) as f:
            content = f.read()
        # parts: label(0), R(1), delay(2), failreason(3)
        parts = content.strip().split("\t")
        r_val = float(parts[1])
        assert abs(r_val) < 1.1  # reasonable correlation value


def showarbcorr_bipolar(debug=False):
    """Test showarbcorr with --bipolar option."""
    if debug:
        print("showarbcorr_bipolar")
    with tempfile.TemporaryDirectory() as tmpdir:
        signal = _make_broadband_signal(NPTS, SAMPLERATE, seed=42)
        file1 = _write_plain_signal_file(tmpdir, signal, name="sig1.txt")
        # Negate to make anti-correlated
        file2 = _write_plain_signal_file(tmpdir, -signal, name="sig2.txt")

        outfile = os.path.join(tmpdir, "results.txt")
        args = _make_test_args(
            tmpdir, file1, file2, bipolar=True, summarymode=True, outputfile=outfile, debug=debug
        )
        sac.showarbcorr(args)

        assert os.path.exists(outfile)
        with open(outfile) as f:
            content = f.read()
        # parts: label(0), R(1), delay(2), failreason(3)
        parts = content.strip().split("\t")
        r_val = float(parts[1])
        if debug:
            print(f"  Bipolar R: {r_val}")
        # With bipolar and anti-correlated signals, should find high magnitude
        assert abs(r_val) > 0.9


def showarbcorr_with_label_no_summary(debug=False):
    """Test showarbcorr normal mode with a label."""
    if debug:
        print("showarbcorr_with_label_no_summary")
    with tempfile.TemporaryDirectory() as tmpdir:
        signal = _make_broadband_signal(NPTS, SAMPLERATE, seed=42)
        file1 = _write_plain_signal_file(tmpdir, signal, name="sig1.txt")
        file2 = _write_plain_signal_file(tmpdir, signal, name="sig2.txt")

        args = _make_test_args(tmpdir, file1, file2, label="delay_test", debug=debug)
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            sac.showarbcorr(args)
        finally:
            sys.stdout = old_stdout
        output = captured.getvalue()
        assert "delay_test" in output


# ==================== Main test function ====================


def test_showarbcorr(debug=False):
    # _get_parser tests
    if debug:
        print("Running _get_parser tests")
    get_parser_returns_parser(debug=debug)
    get_parser_required_args(debug=debug)
    get_parser_defaults(debug=debug)
    get_parser_samplerate1(debug=debug)
    get_parser_samplerate2(debug=debug)
    get_parser_nodisplay(debug=debug)
    get_parser_debug(debug=debug)
    get_parser_verbose(debug=debug)
    get_parser_detrendorder(debug=debug)
    get_parser_corrweighting(debug=debug)
    get_parser_invert(debug=debug)
    get_parser_label(debug=debug)
    get_parser_bipolar(debug=debug)
    get_parser_outputfile(debug=debug)
    get_parser_corroutputfile(debug=debug)
    get_parser_summarymode(debug=debug)
    get_parser_labelline(debug=debug)
    get_parser_noprogressbar(debug=debug)
    get_parser_nprocs(debug=debug)
    get_parser_trimdata(debug=debug)
    get_parser_windowfunc(debug=debug)
    get_parser_nonorm(debug=debug)
    get_parser_saveres(debug=debug)

    # printthresholds tests
    if debug:
        print("Running printthresholds tests")
    printthresholds_basic(debug=debug)
    printthresholds_single(debug=debug)
    printthresholds_empty(debug=debug)

    # showarbcorr tests
    if debug:
        print("Running showarbcorr tests")
    showarbcorr_identical_signals(debug=debug)
    showarbcorr_delayed_signals(debug=debug)
    showarbcorr_summarymode_with_label(debug=debug)
    showarbcorr_summarymode_to_stdout(debug=debug)
    showarbcorr_normal_mode(debug=debug)
    showarbcorr_invert(debug=debug)
    showarbcorr_corroutputfile(debug=debug)
    showarbcorr_trimdata(debug=debug)
    showarbcorr_different_samplerates(debug=debug)
    showarbcorr_bipolar(debug=debug)
    showarbcorr_with_label_no_summary(debug=debug)


if __name__ == "__main__":
    test_showarbcorr(debug=True)
