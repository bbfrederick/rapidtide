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
import argparse
import io
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rapidtide.workflows.endtidalproc import (
    _get_parser,
    endtidalproc,
    phase,
    process_args,
)

# ---- helpers ----


def _make_co2_signal(samplerate=10.0, duration=60.0):
    """Create a synthetic CO2-like signal with clear peaks.

    Generates a slow sinusoidal signal (period ~5s) with added noise,
    simulating a breathing CO2 trace with well-defined maxima.
    """
    npts = int(samplerate * duration)
    t = np.arange(npts) / samplerate
    # breathing at ~0.2 Hz (period 5s), amplitude 40, baseline 40
    signal = 40.0 + 10.0 * np.sin(2.0 * np.pi * 0.2 * t)
    # add a little noise
    rng = np.random.RandomState(42)
    signal += rng.normal(0, 0.3, npts)
    return signal, t


def _make_default_args(**overrides):
    """Create a default args Namespace for endtidalproc."""
    defaults = dict(
        infilename="dummy_in.txt",
        outfilename="dummy_out.txt",
        isoxygen=False,
        samplerate=10.0,
        thestarttime=-1000000.0,
        theendtime=1000000.0,
        thresh=1.0,
        debug=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ---- phase tests ----


def phase_scalar(debug=False):
    """Test phase() with scalar complex numbers."""
    # 1+1j => pi/4
    result = phase(complex(1, 1))
    assert np.isclose(result, np.pi / 4), f"Expected pi/4, got {result}"

    # -1+0j => pi
    result = phase(complex(-1, 0))
    assert np.isclose(result, np.pi), f"Expected pi, got {result}"

    # 0+1j => pi/2
    result = phase(complex(0, 1))
    assert np.isclose(result, np.pi / 2), f"Expected pi/2, got {result}"

    # 0-1j => -pi/2
    result = phase(complex(0, -1))
    assert np.isclose(result, -np.pi / 2), f"Expected -pi/2, got {result}"

    # 1+0j => 0
    result = phase(complex(1, 0))
    assert np.isclose(result, 0.0), f"Expected 0, got {result}"

    if debug:
        print("phase_scalar passed")


def phase_array(debug=False):
    """Test phase() with an array of complex numbers."""
    arr = np.array([1 + 1j, -1 - 1j, 1 + 0j, 0 + 1j])
    result = phase(arr)
    expected = np.array([np.pi / 4, -3 * np.pi / 4, 0.0, np.pi / 2])
    np.testing.assert_allclose(result, expected, atol=1e-10)
    assert result.shape == arr.shape

    if debug:
        print("phase_array passed")


def phase_zero(debug=False):
    """Test phase() with zero complex number."""
    result = phase(complex(0, 0))
    assert np.isclose(result, 0.0), f"Expected 0, got {result}"

    if debug:
        print("phase_zero passed")


# ---- _get_parser tests ----


def parser_basic(debug=False):
    """Test that _get_parser() returns a valid parser."""
    parser = _get_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    assert parser.prog == "endtidalproc"

    if debug:
        print("parser_basic passed")


def parser_required_args(debug=False):
    """Test that parser requires infilename and outfilename."""
    parser = _get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])

    with pytest.raises(SystemExit):
        parser.parse_args(["only_one_arg"])

    if debug:
        print("parser_required_args passed")


def parser_defaults(debug=False):
    """Test default values from the parser."""
    parser = _get_parser()
    args = parser.parse_args(["input.txt", "output.txt"])

    assert args.infilename == "input.txt"
    assert args.outfilename == "output.txt"
    assert args.isoxygen is False
    assert args.samplerate is None
    assert args.thestarttime == -1000000.0
    assert args.theendtime == 1000000.0
    assert args.thresh == 1.0
    assert args.debug is False

    if debug:
        print("parser_defaults passed")


def parser_isoxygen(debug=False):
    """Test --isoxygen flag."""
    parser = _get_parser()
    args = parser.parse_args(["input.txt", "output.txt", "--isoxygen"])
    assert args.isoxygen is True

    if debug:
        print("parser_isoxygen passed")


def parser_samplerate(debug=False):
    """Test --samplerate option."""
    parser = _get_parser()
    args = parser.parse_args(["input.txt", "output.txt", "--samplerate", "10.0"])
    assert np.isclose(args.samplerate, 10.0)

    if debug:
        print("parser_samplerate passed")


def parser_sampletime(debug=False):
    """Test --sampletime option (inverts to frequency)."""
    parser = _get_parser()
    args = parser.parse_args(["input.txt", "output.txt", "--sampletime", "0.1"])
    assert np.isclose(args.samplerate, 10.0)

    if debug:
        print("parser_sampletime passed")


def parser_samplerate_sampletime_mutual_exclusion(debug=False):
    """Test that --samplerate and --sampletime are mutually exclusive."""
    parser = _get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["input.txt", "output.txt", "--samplerate", "10.0", "--sampletime", "0.1"]
        )

    if debug:
        print("parser_samplerate_sampletime_mutual_exclusion passed")


def parser_starttime_endtime(debug=False):
    """Test --starttime and --endtime options."""
    parser = _get_parser()
    args = parser.parse_args(
        ["input.txt", "output.txt", "--starttime", "5.0", "--endtime", "50.0"]
    )
    assert np.isclose(args.thestarttime, 5.0)
    assert np.isclose(args.theendtime, 50.0)

    if debug:
        print("parser_starttime_endtime passed")


def parser_thresh(debug=False):
    """Test --thresh option."""
    parser = _get_parser()
    args = parser.parse_args(["input.txt", "output.txt", "--thresh", "2.5"])
    assert np.isclose(args.thresh, 2.5)

    if debug:
        print("parser_thresh passed")


def parser_debug(debug=False):
    """Test --debug flag."""
    parser = _get_parser()
    args = parser.parse_args(["input.txt", "output.txt", "--debug"])
    assert args.debug is True

    if debug:
        print("parser_debug passed")


# ---- process_args tests ----


def process_args_default_samplerate(debug=False):
    """Test that process_args sets samplerate to 1.0 when None."""
    args = argparse.Namespace(samplerate=None, debug=False)
    result = process_args(args)
    assert result.samplerate == 1.0

    if debug:
        print("process_args_default_samplerate passed")


def process_args_preserves_samplerate(debug=False):
    """Test that process_args keeps explicit samplerate."""
    args = argparse.Namespace(samplerate=25.0, debug=False)
    result = process_args(args)
    assert result.samplerate == 25.0

    if debug:
        print("process_args_preserves_samplerate passed")


def process_args_returns_args(debug=False):
    """Test that process_args returns the modified args object."""
    args = argparse.Namespace(samplerate=None, debug=False)
    result = process_args(args)
    assert result is args

    if debug:
        print("process_args_returns_args passed")


def process_args_debug_output(debug=False):
    """Test that process_args prints args when debug=True."""
    args = argparse.Namespace(samplerate=5.0, debug=True)
    captured = io.StringIO()
    with patch("sys.stdout", captured):
        process_args(args)
    output = captured.getvalue()
    assert "args:" in output

    if debug:
        print("process_args_debug_output passed")


def process_args_no_debug_output(debug=False):
    """Test that process_args is silent when debug=False."""
    args = argparse.Namespace(samplerate=5.0, debug=False)
    captured = io.StringIO()
    with patch("sys.stdout", captured):
        process_args(args)
    output = captured.getvalue()
    assert output == ""

    if debug:
        print("process_args_no_debug_output passed")


# ---- endtidalproc tests ----


def _run_endtidalproc_with_args(signal, args):
    """Run endtidalproc with mocked I/O and a fixed args Namespace.

    Returns the array written by writevec.
    """
    saved = {}

    def mock_readvec(fname, **kwargs):
        return signal.copy()

    def mock_writevec(data, fname, **kwargs):
        saved["data"] = np.array(data).copy()
        saved["fname"] = fname

    with (
        patch(
            "rapidtide.workflows.endtidalproc.process_args",
            return_value=args,
        ),
        patch(
            "rapidtide.workflows.endtidalproc._get_parser",
            return_value=MagicMock(parse_args=MagicMock(return_value=args)),
        ),
        patch("rapidtide.workflows.endtidalproc.tide_io.readvec", side_effect=mock_readvec),
        patch("rapidtide.workflows.endtidalproc.tide_io.writevec", side_effect=mock_writevec),
    ):
        endtidalproc()

    return saved


def endtidalproc_co2_basic(debug=False):
    """Test endtidalproc with a CO2 trace (fits maxima)."""
    samplerate = 10.0
    signal, t = _make_co2_signal(samplerate=samplerate, duration=60.0)
    args = _make_default_args(samplerate=samplerate, isoxygen=False)

    saved = _run_endtidalproc_with_args(signal, args)

    assert "data" in saved, "writevec was not called"
    assert saved["fname"] == "dummy_out.txt"
    assert len(saved["data"]) == len(signal)

    # endtidal CO2 should track the peaks (top envelope)
    # the output should be roughly near the peak values of the signal
    peak_value = np.max(signal)
    trough_value = np.min(signal)
    mean_output = np.mean(saved["data"])
    # mean of peak-interpolated should be closer to peak than trough
    assert mean_output > (peak_value + trough_value) / 2.0, (
        f"Peak interpolation mean {mean_output} should be above midline "
        f"{(peak_value + trough_value) / 2.0}"
    )

    if debug:
        print(f"endtidalproc_co2_basic: mean output = {mean_output:.2f}")
        print("endtidalproc_co2_basic passed")


def endtidalproc_oxygen_basic(debug=False):
    """Test endtidalproc with an oxygen trace (fits minima)."""
    samplerate = 10.0
    signal, t = _make_co2_signal(samplerate=samplerate, duration=60.0)
    args = _make_default_args(samplerate=samplerate, isoxygen=True)

    saved = _run_endtidalproc_with_args(signal, args)

    assert "data" in saved, "writevec was not called"
    assert len(saved["data"]) == len(signal)

    # endtidal O2 should track the troughs (bottom envelope)
    peak_value = np.max(signal)
    trough_value = np.min(signal)
    mean_output = np.mean(saved["data"])
    # mean of trough-interpolated should be closer to trough than peak
    assert mean_output < (peak_value + trough_value) / 2.0, (
        f"Trough interpolation mean {mean_output} should be below midline "
        f"{(peak_value + trough_value) / 2.0}"
    )

    if debug:
        print(f"endtidalproc_oxygen_basic: mean output = {mean_output:.2f}")
        print("endtidalproc_oxygen_basic passed")


def endtidalproc_output_length(debug=False):
    """Test that output has the same length as input."""
    samplerate = 10.0
    signal, t = _make_co2_signal(samplerate=samplerate, duration=30.0)
    args = _make_default_args(samplerate=samplerate)

    saved = _run_endtidalproc_with_args(signal, args)
    assert len(saved["data"]) == len(signal)

    if debug:
        print("endtidalproc_output_length passed")


def endtidalproc_output_smoothness(debug=False):
    """Test that peak-interpolated output is smoother than input (piecewise linear)."""
    samplerate = 10.0
    signal, t = _make_co2_signal(samplerate=samplerate, duration=60.0)
    args = _make_default_args(samplerate=samplerate)

    saved = _run_endtidalproc_with_args(signal, args)

    # piecewise linear interpolation should have smaller variance of second differences
    input_diff2 = np.diff(signal, n=2)
    output_diff2 = np.diff(saved["data"], n=2)
    assert np.std(output_diff2) < np.std(input_diff2), (
        "Peak-interpolated output should be smoother than input"
    )

    if debug:
        print("endtidalproc_output_smoothness passed")


def endtidalproc_time_range(debug=False):
    """Test endtidalproc with restricted start/end time."""
    samplerate = 10.0
    duration = 60.0
    signal, t = _make_co2_signal(samplerate=samplerate, duration=duration)
    args = _make_default_args(samplerate=samplerate, thestarttime=10.0, theendtime=50.0)

    saved = _run_endtidalproc_with_args(signal, args)

    # output should still be full length (starttime/endtime only clip bisect points)
    assert len(saved["data"]) == len(signal)

    if debug:
        print("endtidalproc_time_range passed")


def endtidalproc_invalid_time_range(debug=False):
    """Test that endtidalproc exits when starttime >= endtime."""
    args = _make_default_args(thestarttime=50.0, theendtime=10.0)

    with pytest.raises(SystemExit):
        with (
            patch(
                "rapidtide.workflows.endtidalproc.process_args",
                return_value=args,
            ),
            patch(
                "rapidtide.workflows.endtidalproc._get_parser",
                return_value=MagicMock(parse_args=MagicMock(return_value=args)),
            ),
        ):
            endtidalproc()

    if debug:
        print("endtidalproc_invalid_time_range passed")


def endtidalproc_equal_time_range(debug=False):
    """Test that endtidalproc exits when starttime == endtime."""
    args = _make_default_args(thestarttime=10.0, theendtime=10.0)

    with pytest.raises(SystemExit):
        with (
            patch(
                "rapidtide.workflows.endtidalproc.process_args",
                return_value=args,
            ),
            patch(
                "rapidtide.workflows.endtidalproc._get_parser",
                return_value=MagicMock(parse_args=MagicMock(return_value=args)),
            ),
        ):
            endtidalproc()

    if debug:
        print("endtidalproc_equal_time_range passed")


def endtidalproc_co2_prints_message(debug=False):
    """Test that CO2 mode prints the correct message."""
    samplerate = 10.0
    signal, t = _make_co2_signal(samplerate=samplerate, duration=30.0)
    args = _make_default_args(samplerate=samplerate, isoxygen=False)

    captured = io.StringIO()
    with patch("sys.stdout", captured):
        _run_endtidalproc_with_args(signal, args)

    output = captured.getvalue()
    assert "Fitting trace as CO2" in output

    if debug:
        print("endtidalproc_co2_prints_message passed")


def endtidalproc_oxygen_prints_message(debug=False):
    """Test that oxygen mode prints the correct message."""
    samplerate = 10.0
    signal, t = _make_co2_signal(samplerate=samplerate, duration=30.0)
    args = _make_default_args(samplerate=samplerate, isoxygen=True)

    captured = io.StringIO()
    with patch("sys.stdout", captured):
        _run_endtidalproc_with_args(signal, args)

    output = captured.getvalue()
    assert "Fitting trace as oxygen" in output

    if debug:
        print("endtidalproc_oxygen_prints_message passed")


def endtidalproc_high_samplerate(debug=False):
    """Test endtidalproc with a higher sample rate."""
    samplerate = 100.0
    npts = int(samplerate * 30.0)
    t = np.arange(npts) / samplerate
    signal = 40.0 + 10.0 * np.sin(2.0 * np.pi * 0.2 * t)
    args = _make_default_args(samplerate=samplerate)

    saved = _run_endtidalproc_with_args(signal, args)
    assert len(saved["data"]) == len(signal)

    if debug:
        print("endtidalproc_high_samplerate passed")


def endtidalproc_no_noise(debug=False):
    """Test endtidalproc with a clean sinusoidal signal (no noise)."""
    samplerate = 10.0
    npts = int(samplerate * 60.0)
    t = np.arange(npts) / samplerate
    signal = 40.0 + 10.0 * np.sin(2.0 * np.pi * 0.2 * t)
    args = _make_default_args(samplerate=samplerate)

    saved = _run_endtidalproc_with_args(signal, args)
    assert len(saved["data"]) == len(signal)
    # for a clean sine, peak interpolation should hover near 50 (peak value)
    assert np.mean(saved["data"]) > 45.0

    if debug:
        print("endtidalproc_no_noise passed")


# ---- regression test for thresh attribute fix ----


def endtidalproc_thresh_attribute_consistency(debug=False):
    """Verify parser and workflow agree on the thresh attribute name.

    Previously the parser defined dest="thresh" but the workflow referenced
    args.thethresh, causing an AttributeError at runtime. Fixed by changing
    the workflow to use args.thresh.
    """
    parser = _get_parser()
    args = parser.parse_args(["input.txt", "output.txt", "--thresh", "2.0"])

    assert hasattr(args, "thresh"), "Parser should produce args.thresh"
    assert args.thresh == 2.0

    if debug:
        print("endtidalproc_thresh_attribute_consistency passed")


# ---- main test function ----


def test_endtidalproc(debug=False):
    # phase tests
    phase_scalar(debug=debug)
    phase_array(debug=debug)
    phase_zero(debug=debug)

    # parser tests
    parser_basic(debug=debug)
    parser_required_args(debug=debug)
    parser_defaults(debug=debug)
    parser_isoxygen(debug=debug)
    parser_samplerate(debug=debug)
    parser_sampletime(debug=debug)
    parser_samplerate_sampletime_mutual_exclusion(debug=debug)
    parser_starttime_endtime(debug=debug)
    parser_thresh(debug=debug)
    parser_debug(debug=debug)

    # process_args tests
    process_args_default_samplerate(debug=debug)
    process_args_preserves_samplerate(debug=debug)
    process_args_returns_args(debug=debug)
    process_args_debug_output(debug=debug)
    process_args_no_debug_output(debug=debug)

    # endtidalproc workflow tests
    endtidalproc_co2_basic(debug=debug)
    endtidalproc_oxygen_basic(debug=debug)
    endtidalproc_output_length(debug=debug)
    endtidalproc_output_smoothness(debug=debug)
    endtidalproc_time_range(debug=debug)
    endtidalproc_invalid_time_range(debug=debug)
    endtidalproc_equal_time_range(debug=debug)
    endtidalproc_co2_prints_message(debug=debug)
    endtidalproc_oxygen_prints_message(debug=debug)
    endtidalproc_high_samplerate(debug=debug)
    endtidalproc_no_noise(debug=debug)

    # regression test for thresh attribute fix
    endtidalproc_thresh_attribute_consistency(debug=debug)


if __name__ == "__main__":
    test_endtidalproc(debug=True)
