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

import numpy as np
import pytest

import rapidtide.workflows.showarbcorr as sac

# ==================== Helpers ====================

SAMPLERATE = 10.0  # Hz
NPTS = 500


def _write_plain_signal_file(tmpdir, signal, name="signal.txt"):
    """
    Write a signal to a plain text file (no sidecar).

    Parameters
    ----------
    tmpdir : py.path.local
        Temporary directory for files created by the test.
    signal : object
        Value supplied as ``signal`` by the test case.
    name : str, optional
        Value supplied as ``name`` by the test case.

    Returns
    -------
    str
        The ``filepath`` value supplied to the caller.
    """
    filepath = os.path.join(tmpdir, name)
    np.savetxt(filepath, signal)
    return filepath


def _make_test_args(tmpdir, file1, file2, Fs1=SAMPLERATE, Fs2=SAMPLERATE, **kwargs):
    """
    Create args via parser, override with kwargs.

    Parameters
    ----------
    tmpdir : py.path.local
        Temporary directory for files created by the test.
    file1 : object
        Value supplied as ``file1`` by the test case.
    file2 : object
        Value supplied as ``file2`` by the test case.
    Fs1 : object, optional
        Value supplied as ``Fs1`` by the test case.
    Fs2 : object, optional
        Value supplied as ``Fs2`` by the test case.
    **kwargs : object
        Additional keyword arguments accepted by the stub.

    Returns
    -------
    object
        The ``args`` value supplied to the caller.
    """
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
    """
    Test _get_parser returns an ArgumentParser.

    Parameters
    ----------
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("get_parser_returns_parser")
    import argparse

    parser = sac._get_parser()
    assert isinstance(parser, argparse.ArgumentParser)


def get_parser_required_args(debug=False):
    """
    Test _get_parser requires two input files.

    Parameters
    ----------
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("get_parser_required_args")
    parser = sac._get_parser()
    try:
        parser.parse_args([])
        assert False, "Should have raised SystemExit"
    except SystemExit:
        pass


def get_parser_defaults(parse_with_temp_inputs, debug=False):
    """
    Test _get_parser default values.

    Parameters
    ----------
    parse_with_temp_inputs : callable
        Parse arguments using temporary input files.
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("get_parser_defaults")
    args = parse_with_temp_inputs(sac._get_parser())
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


def get_parser_samplerate1(parse_with_temp_inputs, debug=False):
    """
    Test _get_parser accepts --samplerate1.

    Parameters
    ----------
    parse_with_temp_inputs : callable
        Parse arguments using temporary input files.
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("get_parser_samplerate1")
    args = parse_with_temp_inputs(sac._get_parser(), ["--samplerate1", "25.0"])
    assert args.samplerate1 == 25.0


def get_parser_samplerate2(parse_with_temp_inputs, debug=False):
    """
    Test _get_parser accepts --samplerate2.

    Parameters
    ----------
    parse_with_temp_inputs : callable
        Parse arguments using temporary input files.
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("get_parser_samplerate2")
    args = parse_with_temp_inputs(sac._get_parser(), ["--samplerate2", "50.0"])
    assert args.samplerate2 == 50.0


def get_parser_nodisplay(parse_with_temp_inputs, debug=False):
    """
    Test _get_parser accepts --nodisplay.

    Parameters
    ----------
    parse_with_temp_inputs : callable
        Parse arguments using temporary input files.
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("get_parser_nodisplay")
    args = parse_with_temp_inputs(sac._get_parser(), ["--nodisplay"])
    assert not args.display


def get_parser_debug(parse_with_temp_inputs, debug=False):
    """
    Test _get_parser accepts --debug.

    Parameters
    ----------
    parse_with_temp_inputs : callable
        Parse arguments using temporary input files.
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("get_parser_debug")
    args = parse_with_temp_inputs(sac._get_parser(), ["--debug"])
    assert args.debug


def get_parser_verbose(parse_with_temp_inputs, debug=False):
    """
    Test _get_parser accepts --verbose.

    Parameters
    ----------
    parse_with_temp_inputs : callable
        Parse arguments using temporary input files.
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("get_parser_verbose")
    args = parse_with_temp_inputs(sac._get_parser(), ["--verbose"])
    assert args.verbose


def get_parser_detrendorder(parse_with_temp_inputs, debug=False):
    """
    Test _get_parser accepts --detrendorder.

    Parameters
    ----------
    parse_with_temp_inputs : callable
        Parse arguments using temporary input files.
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("get_parser_detrendorder")
    args = parse_with_temp_inputs(sac._get_parser(), ["--detrendorder", "3"])
    assert args.detrendorder == 3


def get_parser_corrweighting(parse_with_temp_inputs, debug=False):
    """
    Test _get_parser accepts --corrweighting.

    Parameters
    ----------
    parse_with_temp_inputs : callable
        Parse arguments using temporary input files.
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("get_parser_corrweighting")
    for choice in ["None", "phat", "liang", "eckart"]:
        args = parse_with_temp_inputs(sac._get_parser(), ["--corrweighting", choice])
        assert args.corrweighting == choice


def get_parser_invert(parse_with_temp_inputs, debug=False):
    """
    Test _get_parser accepts --invert.

    Parameters
    ----------
    parse_with_temp_inputs : callable
        Parse arguments using temporary input files.
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("get_parser_invert")
    args = parse_with_temp_inputs(sac._get_parser(), ["--invert"])
    assert args.invert


def get_parser_label(parse_with_temp_inputs, debug=False):
    """
    Test _get_parser accepts --label.

    Parameters
    ----------
    parse_with_temp_inputs : callable
        Parse arguments using temporary input files.
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("get_parser_label")
    args = parse_with_temp_inputs(sac._get_parser(), ["--label", "test_label"])
    assert args.label == "test_label"


def get_parser_bipolar(parse_with_temp_inputs, debug=False):
    """
    Test _get_parser accepts --bipolar.

    Parameters
    ----------
    parse_with_temp_inputs : callable
        Parse arguments using temporary input files.
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("get_parser_bipolar")
    args = parse_with_temp_inputs(sac._get_parser(), ["--bipolar"])
    assert args.bipolar


def get_parser_outputfile(parse_with_temp_inputs, debug=False):
    """
    Test _get_parser accepts --outputfile.

    Parameters
    ----------
    parse_with_temp_inputs : callable
        Parse arguments using temporary input files.
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("get_parser_outputfile")
    args = parse_with_temp_inputs(sac._get_parser(), ["--outputfile", "results.txt"])
    assert args.outputfile == "results.txt"


def get_parser_corroutputfile(parse_with_temp_inputs, debug=False):
    """
    Test _get_parser accepts --corroutputfile.

    Parameters
    ----------
    parse_with_temp_inputs : callable
        Parse arguments using temporary input files.
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("get_parser_corroutputfile")
    args = parse_with_temp_inputs(sac._get_parser(), ["--corroutputfile", "corr.txt"])
    assert args.corroutputfile == "corr.txt"


def get_parser_summarymode(parse_with_temp_inputs, debug=False):
    """
    Test _get_parser accepts --summarymode.

    Parameters
    ----------
    parse_with_temp_inputs : callable
        Parse arguments using temporary input files.
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("get_parser_summarymode")
    args = parse_with_temp_inputs(sac._get_parser(), ["--summarymode"])
    assert args.summarymode


def get_parser_labelline(parse_with_temp_inputs, debug=False):
    """
    Test _get_parser accepts --labelline.

    Parameters
    ----------
    parse_with_temp_inputs : callable
        Parse arguments using temporary input files.
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("get_parser_labelline")
    args = parse_with_temp_inputs(sac._get_parser(), ["--labelline"])
    assert args.labelline


def get_parser_noprogressbar(parse_with_temp_inputs, debug=False):
    """
    Test _get_parser accepts --noprogressbar.

    Parameters
    ----------
    parse_with_temp_inputs : callable
        Parse arguments using temporary input files.
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("get_parser_noprogressbar")
    args = parse_with_temp_inputs(sac._get_parser(), ["--noprogressbar"])
    assert not args.showprogressbar


def get_parser_nprocs(parse_with_temp_inputs, debug=False):
    """
    Test _get_parser accepts --nprocs.

    Parameters
    ----------
    parse_with_temp_inputs : callable
        Parse arguments using temporary input files.
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("get_parser_nprocs")
    args = parse_with_temp_inputs(sac._get_parser(), ["--nprocs", "4"])
    assert args.nprocs == 4


def get_parser_trimdata(parse_with_temp_inputs, debug=False):
    """
    Test _get_parser accepts --trimdata.

    Parameters
    ----------
    parse_with_temp_inputs : callable
        Parse arguments using temporary input files.
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("get_parser_trimdata")
    args = parse_with_temp_inputs(sac._get_parser(), ["--trimdata"])
    assert args.trimdata


def get_parser_windowfunc(parse_with_temp_inputs, debug=False):
    """
    Test _get_parser accepts --windowfunc.

    Parameters
    ----------
    parse_with_temp_inputs : callable
        Parse arguments using temporary input files.
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("get_parser_windowfunc")
    args = parse_with_temp_inputs(sac._get_parser(), ["--windowfunc", "hann"])
    assert args.windowfunc == "hann"


def get_parser_nonorm(parse_with_temp_inputs, debug=False):
    """
    Test _get_parser accepts --nonorm.

    Parameters
    ----------
    parse_with_temp_inputs : callable
        Parse arguments using temporary input files.
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("get_parser_nonorm")
    args = parse_with_temp_inputs(sac._get_parser(), ["--nonorm"])
    assert not args.minorm


def get_parser_saveres(parse_with_temp_inputs, debug=False):
    """
    Test _get_parser accepts --saveres.

    Parameters
    ----------
    parse_with_temp_inputs : callable
        Parse arguments using temporary input files.
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("get_parser_saveres")
    args = parse_with_temp_inputs(sac._get_parser(), ["--saveres", "300"])
    assert args.saveres == 300


# ==================== printthresholds tests ====================


def printthresholds_basic(capture_stdout, debug=False):
    """
    Test printthresholds prints correct output.

    Parameters
    ----------
    capture_stdout : callable
        Run a callable and capture its standard output.
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("printthresholds_basic")
    pcts = [1.96, 2.58, 3.29]
    thepercentiles = [0.95, 0.99, 0.999]
    output = capture_stdout(sac.printthresholds, pcts, thepercentiles, "Test thresholds:")
    assert "Test thresholds:" in output
    assert "0.050" in output
    assert "0.010" in output
    assert "0.001" in output
    assert "1.96" in output
    assert "2.58" in output
    assert "3.29" in output


def printthresholds_single(capture_stdout, debug=False):
    """
    Test printthresholds with a single threshold.

    Parameters
    ----------
    capture_stdout : callable
        Run a callable and capture its standard output.
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("printthresholds_single")
    pcts = [0.5]
    thepercentiles = [0.5]
    output = capture_stdout(sac.printthresholds, pcts, thepercentiles, "Single:")
    assert "Single:" in output
    assert "0.500" in output


def printthresholds_empty(capture_stdout, debug=False):
    """
    Test printthresholds with empty lists.

    Parameters
    ----------
    capture_stdout : callable
        Run a callable and capture its standard output.
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("printthresholds_empty")
    output = capture_stdout(sac.printthresholds, [], [], "Empty:")
    assert "Empty:" in output
    # Only the label line, no threshold lines
    lines = output.strip().split("\n")
    assert len(lines) == 1


# ==================== showarbcorr tests ====================


def showarbcorr_identical_signals(broadband_signal_factory, debug=False):
    """
    Test showarbcorr with identical signals.

    Parameters
    ----------
    broadband_signal_factory : callable
        Generate a broadband signal for the test.
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("showarbcorr_identical_signals")
    with tempfile.TemporaryDirectory() as tmpdir:
        signal = broadband_signal_factory(NPTS, SAMPLERATE, delay=0.0, seed=42, freqmax=0.15)
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


def showarbcorr_delayed_signals(broadband_signal_factory, debug=False):
    """
    Test showarbcorr with delayed signals finds correct delay.

    Parameters
    ----------
    broadband_signal_factory : callable
        Generate a broadband signal for the test.
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("showarbcorr_delayed_signals")
    with tempfile.TemporaryDirectory() as tmpdir:
        delay = 1.0
        sig1 = broadband_signal_factory(NPTS, SAMPLERATE, delay=0.0, seed=42, freqmax=0.15)
        sig2 = broadband_signal_factory(NPTS, SAMPLERATE, delay=delay, seed=42, freqmax=0.15)
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


def showarbcorr_summarymode_with_label(broadband_signal_factory, debug=False):
    """
    Test showarbcorr summary mode with label and labelline.

    Parameters
    ----------
    broadband_signal_factory : callable
        Generate a broadband signal for the test.
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("showarbcorr_summarymode_with_label")
    with tempfile.TemporaryDirectory() as tmpdir:
        signal = broadband_signal_factory(NPTS, SAMPLERATE, seed=42, freqmax=0.15)
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


def showarbcorr_summarymode_to_stdout(broadband_signal_factory, capture_stdout, debug=False):
    """
    Test showarbcorr summary mode printing to stdout.

    Parameters
    ----------
    broadband_signal_factory : callable
        Generate a broadband signal for the test.
    capture_stdout : callable
        Run a callable and capture its standard output.
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("showarbcorr_summarymode_to_stdout")
    with tempfile.TemporaryDirectory() as tmpdir:
        signal = broadband_signal_factory(NPTS, SAMPLERATE, seed=42, freqmax=0.15)
        file1 = _write_plain_signal_file(tmpdir, signal, name="sig1.txt")
        file2 = _write_plain_signal_file(tmpdir, signal, name="sig2.txt")

        args = _make_test_args(tmpdir, file1, file2, summarymode=True, debug=debug)
        output = capture_stdout(sac.showarbcorr, args)
        # Should contain tab-separated values somewhere in the output
        # The last line should be the summary: label\tR\tdelay\tfailreason
        lines = output.strip().split("\n")
        last_line = lines[-1]
        parts = last_line.split("\t")
        assert len(parts) >= 4, f"Expected >=4 tab-separated parts, got {len(parts)}: {last_line}"


def showarbcorr_normal_mode(broadband_signal_factory, capture_stdout, debug=False):
    """
    Test showarbcorr normal (non-summary) mode.

    Parameters
    ----------
    broadband_signal_factory : callable
        Generate a broadband signal for the test.
    capture_stdout : callable
        Run a callable and capture its standard output.
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("showarbcorr_normal_mode")
    with tempfile.TemporaryDirectory() as tmpdir:
        signal = broadband_signal_factory(NPTS, SAMPLERATE, seed=42, freqmax=0.15)
        file1 = _write_plain_signal_file(tmpdir, signal, name="sig1.txt")
        file2 = _write_plain_signal_file(tmpdir, signal, name="sig2.txt")

        # Set label to None (Python None) to get the Crosscorrelation_Rmax output path
        args = _make_test_args(tmpdir, file1, file2, debug=debug)
        args.label = None
        output = capture_stdout(sac.showarbcorr, args)
        if debug:
            print(f"  Normal mode output: {output}")
        assert "Pearson_R:" in output
        assert "Crosscorrelation_Rmax:" in output
        assert "Crosscorrelation_maxdelay:" in output


def showarbcorr_invert(broadband_signal_factory, debug=False):
    """
    Test showarbcorr with --invert.

    Parameters
    ----------
    broadband_signal_factory : callable
        Generate a broadband signal for the test.
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("showarbcorr_invert")
    with tempfile.TemporaryDirectory() as tmpdir:
        signal = broadband_signal_factory(NPTS, SAMPLERATE, seed=42, freqmax=0.15)
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


def showarbcorr_corroutputfile(broadband_signal_factory, debug=False):
    """
    Test showarbcorr writes correlation function to file.

    Parameters
    ----------
    broadband_signal_factory : callable
        Generate a broadband signal for the test.
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("showarbcorr_corroutputfile")
    with tempfile.TemporaryDirectory() as tmpdir:
        signal = broadband_signal_factory(NPTS, SAMPLERATE, seed=42, freqmax=0.15)
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


def showarbcorr_trimdata(broadband_signal_factory, debug=False):
    """
    Test showarbcorr with --trimdata for unequal length signals.

    Parameters
    ----------
    broadband_signal_factory : callable
        Generate a broadband signal for the test.
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("showarbcorr_trimdata")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create two signals of different lengths
        sig1 = broadband_signal_factory(500, SAMPLERATE, seed=42, freqmax=0.15)
        sig2 = broadband_signal_factory(300, SAMPLERATE, seed=42, freqmax=0.15)
        file1 = _write_plain_signal_file(tmpdir, sig1, name="sig1.txt")
        file2 = _write_plain_signal_file(tmpdir, sig2, name="sig2.txt")

        outfile = os.path.join(tmpdir, "results.txt")
        args = _make_test_args(
            tmpdir, file1, file2, trimdata=True, summarymode=True, outputfile=outfile, debug=debug
        )
        sac.showarbcorr(args)

        assert os.path.exists(outfile)


def showarbcorr_different_samplerates(broadband_signal_factory, debug=False):
    """
    Test showarbcorr with different sample rates.

    Parameters
    ----------
    broadband_signal_factory : callable
        Generate a broadband signal for the test.
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("showarbcorr_different_samplerates")
    with tempfile.TemporaryDirectory() as tmpdir:
        fs1 = 10.0
        fs2 = 20.0
        sig1 = broadband_signal_factory(500, fs1, seed=42, freqmax=0.15)
        sig2 = broadband_signal_factory(1000, fs2, seed=99, freqmax=0.15)
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


def showarbcorr_bipolar(broadband_signal_factory, debug=False):
    """
    Test showarbcorr with --bipolar option.

    Parameters
    ----------
    broadband_signal_factory : callable
        Generate a broadband signal for the test.
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("showarbcorr_bipolar")
    with tempfile.TemporaryDirectory() as tmpdir:
        signal = broadband_signal_factory(NPTS, SAMPLERATE, seed=42, freqmax=0.15)
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


def showarbcorr_with_label_no_summary(broadband_signal_factory, capture_stdout, debug=False):
    """
    Test showarbcorr normal mode with a label.

    Parameters
    ----------
    broadband_signal_factory : callable
        Generate a broadband signal for the test.
    capture_stdout : callable
        Run a callable and capture its standard output.
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("showarbcorr_with_label_no_summary")
    with tempfile.TemporaryDirectory() as tmpdir:
        signal = broadband_signal_factory(NPTS, SAMPLERATE, seed=42, freqmax=0.15)
        file1 = _write_plain_signal_file(tmpdir, signal, name="sig1.txt")
        file2 = _write_plain_signal_file(tmpdir, signal, name="sig2.txt")

        args = _make_test_args(tmpdir, file1, file2, label="delay_test", debug=debug)
        output = capture_stdout(sac.showarbcorr, args)
        assert "delay_test" in output


# ==================== Main test function ====================


PARSER_CASES = [
    get_parser_returns_parser,
    get_parser_required_args,
    get_parser_defaults,
    get_parser_samplerate1,
    get_parser_samplerate2,
    get_parser_nodisplay,
    get_parser_debug,
    get_parser_verbose,
    get_parser_detrendorder,
    get_parser_corrweighting,
    get_parser_invert,
    get_parser_label,
    get_parser_bipolar,
    get_parser_outputfile,
    get_parser_corroutputfile,
    get_parser_summarymode,
    get_parser_labelline,
    get_parser_noprogressbar,
    get_parser_nprocs,
    get_parser_trimdata,
    get_parser_windowfunc,
    get_parser_nonorm,
    get_parser_saveres,
]


PRINTTHRESHOLD_CASES = [
    printthresholds_basic,
    printthresholds_single,
    printthresholds_empty,
]


WORKFLOW_CASES = [
    showarbcorr_identical_signals,
    showarbcorr_delayed_signals,
    showarbcorr_summarymode_with_label,
    showarbcorr_summarymode_to_stdout,
    showarbcorr_normal_mode,
    showarbcorr_invert,
    showarbcorr_corroutputfile,
    showarbcorr_trimdata,
    showarbcorr_different_samplerates,
    showarbcorr_bipolar,
    showarbcorr_with_label_no_summary,
]


@pytest.mark.parametrize("case_func", PARSER_CASES, ids=lambda func: func.__name__)
@pytest.mark.unit
def test_showarbcorr_parser_cases(case_runner, case_func):
    """
    Verify showarbcorr parser cases.

    Parameters
    ----------
    case_runner : callable
        Run the selected test case with the test fixtures.
    case_func : callable
        Test case to execute.
    """
    case_runner(case_func)


@pytest.mark.parametrize("case_func", PRINTTHRESHOLD_CASES, ids=lambda func: func.__name__)
@pytest.mark.unit
def test_showarbcorr_printthreshold_cases(case_runner, case_func):
    """
    Verify showarbcorr printthreshold cases.

    Parameters
    ----------
    case_runner : callable
        Run the selected test case with the test fixtures.
    case_func : callable
        Test case to execute.
    """
    case_runner(case_func)


@pytest.mark.parametrize("case_func", WORKFLOW_CASES, ids=lambda func: func.__name__)
@pytest.mark.slow
def test_showarbcorr_workflow_cases(case_runner, case_func):
    """
    Verify showarbcorr workflow cases.

    Parameters
    ----------
    case_runner : callable
        Run the selected test case with the test fixtures.
    case_func : callable
        Test case to execute.
    """
    case_runner(case_func)


if __name__ == "__main__":
    for case in PARSER_CASES + PRINTTHRESHOLD_CASES + WORKFLOW_CASES:
        case(debug=True)
