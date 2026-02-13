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
from unittest.mock import patch

import numpy as np
import pytest

from rapidtide.workflows.physiofreq import _get_parser, getwavefreq, physiofreq

# ==================== Helpers ====================


def _make_args(
    textfilename="signal.txt",
    samplerate=100.0,
    lowestbpm=40.0,
    highestbpm=140.0,
    nosmooth=False,
    displayplots=False,
):
    return argparse.Namespace(
        textfilename=textfilename,
        samplerate=float(samplerate),
        lowestbpm=float(lowestbpm),
        highestbpm=float(highestbpm),
        nosmooth=bool(nosmooth),
        displayplots=bool(displayplots),
    )


def _make_sine(freq_hz=1.0, fs=100.0, duration_s=20.0, phase=0.0, noise=0.0, seed=0):
    t = np.arange(0.0, duration_s, 1.0 / fs)
    x = np.sin(2.0 * np.pi * freq_hz * t + phase)
    if noise > 0.0:
        rng = np.random.RandomState(seed)
        x = x + noise * rng.standard_normal(len(x))
    return x.astype(np.float64)


# ==================== _get_parser tests ====================


def parser_basic(debug=False):
    if debug:
        print("parser_basic")
    parser = _get_parser()
    assert parser is not None
    assert parser.prog == "physiofreq"


def parser_required_args(debug=False):
    if debug:
        print("parser_required_args")
    parser = _get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def parser_defaults(debug=False):
    if debug:
        print("parser_defaults")
    parser = _get_parser()
    args = parser.parse_args(["in.txt"])
    assert args.displayplots is False
    assert args.samplerate == 1.0
    assert args.lowestbpm == 6.0
    assert args.highestbpm == 20.0
    assert args.nosmooth is False


# ==================== getwavefreq tests ====================


def getwavefreq_estimates_frequency_smooth_on(debug=False):
    if debug:
        print("getwavefreq_estimates_frequency_smooth_on")
    fs = 100.0
    target_hz = 1.0  # 60 BPM
    x = _make_sine(freq_hz=target_hz, fs=fs, duration_s=30.0, noise=0.02)

    peak_hz = getwavefreq(
        x,
        fs,
        minpermin=40.0,
        maxpermin=140.0,
        smooth=True,
        displayplots=False,
        debug=False,
    )
    assert np.isfinite(peak_hz)
    assert abs(peak_hz - target_hz) < 0.1


def getwavefreq_estimates_frequency_smooth_off(debug=False):
    if debug:
        print("getwavefreq_estimates_frequency_smooth_off")
    fs = 100.0
    target_hz = 0.75  # 45 BPM
    x = _make_sine(freq_hz=target_hz, fs=fs, duration_s=30.0, noise=0.01)

    peak_hz = getwavefreq(
        x,
        fs,
        minpermin=30.0,
        maxpermin=200.0,
        smooth=False,
        displayplots=False,
        debug=False,
    )
    assert np.isfinite(peak_hz)
    assert abs(peak_hz - target_hz) < 0.1


def getwavefreq_handles_odd_length(debug=False):
    if debug:
        print("getwavefreq_handles_odd_length")
    fs = 50.0
    target_hz = 1.2
    x = _make_sine(freq_hz=target_hz, fs=fs, duration_s=20.0)
    # Force odd length
    x = x[:-1] if (len(x) % 2 == 0) else x
    assert len(x) % 2 == 1

    peak_hz = getwavefreq(
        x,
        fs,
        minpermin=40.0,
        maxpermin=140.0,
        smooth=True,
        displayplots=False,
        debug=False,
    )
    assert np.isfinite(peak_hz)
    assert abs(peak_hz - target_hz) < 0.15


# ==================== physiofreq tests ====================


def physiofreq_uses_readbidstsv_for_json(debug=False):
    if debug:
        print("physiofreq_uses_readbidstsv_for_json")

    fs = 100.0
    target_hz = 1.0
    x = _make_sine(freq_hz=target_hz, fs=fs, duration_s=20.0)

    args = _make_args(textfilename="sub-01_recording-ppg.json", displayplots=False, nosmooth=True)

    def mock_parsefilespec(spec):
        return ([spec], None)

    def mock_readbidstsv(_fname):
        return (fs, 0.0, ["ppg"], x, False, False, {})

    captured = io.StringIO()
    with (
        patch("rapidtide.workflows.physiofreq.tide_io.parsefilespec", side_effect=mock_parsefilespec),
        patch("rapidtide.workflows.physiofreq.tide_io.readbidstsv", side_effect=mock_readbidstsv),
        patch("rapidtide.workflows.physiofreq.tide_io.readvecs") as mocked_readvecs,
        patch("sys.stdout", captured),
    ):
        physiofreq(args)

    out = captured.getvalue()
    assert "sub-01_recording-ppg.json" in out
    assert "Hz" in out
    assert mocked_readvecs.call_count == 0


def physiofreq_uses_readvecs_for_text(debug=False):
    if debug:
        print("physiofreq_uses_readvecs_for_text")

    fs = 50.0
    target_hz = 0.8
    x = _make_sine(freq_hz=target_hz, fs=fs, duration_s=25.0)

    args = _make_args(textfilename="trace.txt", samplerate=fs, displayplots=False, nosmooth=False)

    def mock_parsefilespec(spec):
        return ([spec], None)

    def mock_readvecs(_fname):
        return [x]

    captured = io.StringIO()
    with (
        patch("rapidtide.workflows.physiofreq.tide_io.parsefilespec", side_effect=mock_parsefilespec),
        patch("rapidtide.workflows.physiofreq.tide_io.readvecs", side_effect=mock_readvecs),
        patch("rapidtide.workflows.physiofreq.tide_io.readbidstsv") as mocked_readbidstsv,
        patch("sys.stdout", captured),
    ):
        physiofreq(args)

    out = captured.getvalue()
    assert "trace.txt" in out
    assert "Hz" in out
    assert mocked_readbidstsv.call_count == 0


# ==================== Main test function ====================


def test_physiofreq(debug=False):
    # _get_parser tests
    parser_basic(debug=debug)
    parser_required_args(debug=debug)
    parser_defaults(debug=debug)

    # getwavefreq tests
    getwavefreq_estimates_frequency_smooth_on(debug=debug)
    getwavefreq_estimates_frequency_smooth_off(debug=debug)
    getwavefreq_handles_odd_length(debug=debug)

    # physiofreq tests
    physiofreq_uses_readbidstsv_for_json(debug=debug)
    physiofreq_uses_readvecs_for_text(debug=debug)


if __name__ == "__main__":
    test_physiofreq(debug=True)