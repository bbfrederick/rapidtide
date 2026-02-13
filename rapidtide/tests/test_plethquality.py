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

import rapidtide.workflows.plethquality as pq

# ==================== Helpers ====================


def _make_args(infilename="in.txt", outfilename="out.txt", samplerate=None, display=False):
    return argparse.Namespace(
        infilename=infilename,
        outfilename=outfilename,
        samplerate=samplerate,
        display=display,
    )


# ==================== _get_parser tests ====================


def parser_basic(debug=False):
    if debug:
        print("parser_basic")
    parser = pq._get_parser()
    assert parser is not None
    assert parser.prog == "plethquality"


def parser_required_args(debug=False):
    if debug:
        print("parser_required_args")
    parser = pq._get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def parser_defaults(debug=False):
    if debug:
        print("parser_defaults")
    parser = pq._get_parser()
    # Required args: infilename outfilename.  infilename is validated via parser_funcs,
    # so give an existing temp file.
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".txt") as tf:
        args = parser.parse_args([tf.name, "out.txt"])
    assert args.samplerate is None
    assert args.display is True


def parser_nodisplay_flag(debug=False):
    if debug:
        print("parser_nodisplay_flag")
    parser = pq._get_parser()
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".txt") as tf:
        args = parser.parse_args([tf.name, "out.txt", "--nodisplay"])
    assert args.display is False


# ==================== core plethquality(waveform, Fs) tests ====================


def waveform_quality_constant_is_zero(debug=False):
    if debug:
        print("waveform_quality_constant_is_zero")
    x = np.ones(101, dtype=float)
    mean_sqi, std_sqi, q = pq.plethquality(x, Fs=100.0, S_windowsecs=0.1, debug=False)
    assert q.shape == x.shape
    # Constant signal -> skewness should be 0 everywhere
    assert np.allclose(q, 0.0)
    assert abs(mean_sqi) < 1e-12
    assert abs(std_sqi) < 1e-12


def waveform_quality_windowpts_is_odd(debug=False):
    if debug:
        print("waveform_quality_windowpts_is_odd")

    # We patch the module's skew to capture the window length used.
    lengths = []

    def fake_skew(arr, nan_policy="omit"):
        lengths.append(len(arr))
        return 0.0

    x = np.arange(25, dtype=float)
    Fs = 10.0
    S_windowsecs = 0.2  # 2 pts -> should become 3 (odd)

    with patch("rapidtide.workflows.plethquality.skew", side_effect=fake_skew):
        mean_sqi, std_sqi, q = pq.plethquality(x, Fs=Fs, S_windowsecs=S_windowsecs, debug=False)

    assert q.shape == x.shape
    # ensure we saw only odd window lengths
    assert len(lengths) == len(x)
    assert all((l % 2) == 1 for l in lengths)
    assert max(lengths) == 3


def waveform_quality_nan_handling(debug=False):
    if debug:
        print("waveform_quality_nan_handling")
    x = np.random.RandomState(0).randn(200).astype(float)
    x[10:20] = np.nan
    mean_sqi, std_sqi, q = pq.plethquality(x, Fs=100.0, S_windowsecs=0.1, debug=False)
    assert np.isfinite(mean_sqi)
    assert np.isfinite(std_sqi)
    assert np.all(np.isfinite(q))


# ==================== CLI-style plethquality(args) tests ====================


def args_uses_samplerate_override(debug=False):
    if debug:
        print("args_uses_samplerate_override")

    # readvectorsfromtextfile returns Fs from file, but args.samplerate should override it
    file_Fs = 20.0
    override_Fs = 100.0
    plethdata = np.random.RandomState(1).randn(50).astype(float)

    captured = {}

    def mock_readvectorsfromtextfile(_fname, onecol=False):
        return (file_Fs, 0.0, None, plethdata, False, "text")

    def mock_core_plethquality(waveform, Fs):
        captured["Fs_used"] = Fs
        # return plausible outputs
        q = np.zeros_like(waveform)
        return 0.0, 0.0, q

    def mock_writevec(vec, outname):
        captured["written_vec"] = np.array(vec, copy=True)
        captured["outname"] = outname

    args = _make_args(infilename="in.txt", outfilename="out.txt", samplerate=override_Fs, display=False)

    with (
        patch(
            "rapidtide.workflows.plethquality.tide_io.readvectorsfromtextfile",
            side_effect=mock_readvectorsfromtextfile,
        ),
        patch("rapidtide.workflows.plethquality.plethquality", side_effect=mock_core_plethquality),
        patch("rapidtide.workflows.plethquality.tide_io.writevec", side_effect=mock_writevec),
    ):
        pq.plethquality(args)

    assert captured["Fs_used"] == override_Fs
    assert captured["outname"] == "out.txt"
    assert captured["written_vec"].shape == plethdata.shape


def args_missing_samplerate_exits(debug=False):
    if debug:
        print("args_missing_samplerate_exits")

    plethdata = np.random.RandomState(2).randn(25).astype(float)

    def mock_readvectorsfromtextfile(_fname, onecol=False):
        return (None, 0.0, None, plethdata, False, "text")

    args = _make_args(infilename="in.txt", outfilename="out.txt", samplerate=None, display=False)

    with patch(
        "rapidtide.workflows.plethquality.tide_io.readvectorsfromtextfile",
        side_effect=mock_readvectorsfromtextfile,
    ):
        with pytest.raises(SystemExit):
            pq.plethquality(args)


def args_prints_summary_line(debug=False):
    if debug:
        print("args_prints_summary_line")

    plethdata = np.random.RandomState(3).randn(40).astype(float)

    def mock_readvectorsfromtextfile(_fname, onecol=False):
        return (50.0, 0.0, None, plethdata, False, "text")

    def mock_writevec(vec, outname):
        pass

    # Use real core function, but capture stdout
    args = _make_args(infilename="in.txt", outfilename="out.txt", samplerate=None, display=False)

    buf = io.StringIO()
    with (
        patch(
            "rapidtide.workflows.plethquality.tide_io.readvectorsfromtextfile",
            side_effect=mock_readvectorsfromtextfile,
        ),
        patch("rapidtide.workflows.plethquality.tide_io.writevec", side_effect=mock_writevec),
        patch("sys.stdout", buf),
    ):
        pq.plethquality(args)

    out = buf.getvalue()
    assert "in.txt" in out
    assert "+/-" in out


# ==================== Main test function ====================


def test_plethquality(debug=False):
    # _get_parser tests
    parser_basic(debug=debug)
    parser_required_args(debug=debug)
    parser_defaults(debug=debug)
    parser_nodisplay_flag(debug=debug)

    # core plethquality(waveform, Fs) tests
    waveform_quality_constant_is_zero(debug=debug)
    waveform_quality_windowpts_is_odd(debug=debug)
    waveform_quality_nan_handling(debug=debug)

    # CLI-style plethquality(args) tests
    args_uses_samplerate_override(debug=debug)
    args_missing_samplerate_exits(debug=debug)
    args_prints_summary_line(debug=debug)


if __name__ == "__main__":
    test_plethquality(debug=True)