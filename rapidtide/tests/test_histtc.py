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
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rapidtide.workflows.histtc import _get_parser, histtc

# ============================================================================
# Helpers
# ============================================================================


def _make_default_args(**overrides):
    defaults = dict(
        inputfilename="input.txt",
        outputroot="output",
        histlen=101,
        minval=None,
        maxval=None,
        robustrange=False,
        nozero=False,
        nozerothresh=0.01,
        normhist=False,
        debug=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _make_mock_readvectors(inputdata_1d):
    """Return a mock readvectorsfromtextfile that returns inputdata as a 2D array (row vector)."""
    data_2d = np.atleast_2d(inputdata_1d)

    def _mock(filepath, onecol=False, debug=False):
        return (None, None, None, data_2d.copy(), None, "text")

    return _mock


# ============================================================================
# Parser tests
# ============================================================================


def parser_basic(debug=False):
    if debug:
        print("parser_basic")
    parser = _get_parser()
    assert parser is not None
    assert parser.prog == "histtc"


def parser_required_args(debug=False):
    if debug:
        print("parser_required_args")
    parser = _get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def parser_defaults(debug=False):
    if debug:
        print("parser_defaults")
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"1.0\n2.0\n3.0\n")
        tmpname = f.name
    parser = _get_parser()
    args = parser.parse_args([tmpname, "output"])
    assert args.outputroot == "output"
    assert args.histlen == 101
    assert args.minval is None
    assert args.maxval is None
    assert args.robustrange is False
    assert args.nozero is False
    assert args.nozerothresh == 0.01
    assert args.normhist is False
    assert args.debug is False


def parser_numbins(debug=False):
    if debug:
        print("parser_numbins")
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"1.0\n")
        tmpname = f.name
    parser = _get_parser()
    args = parser.parse_args([tmpname, "out", "--numbins", "50"])
    assert args.histlen == 50


def parser_minval_maxval(debug=False):
    if debug:
        print("parser_minval_maxval")
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"1.0\n")
        tmpname = f.name
    parser = _get_parser()
    args = parser.parse_args([tmpname, "out", "--minval", "-5.0", "--maxval", "10.0"])
    assert args.minval == -5.0
    assert args.maxval == 10.0


def parser_robustrange(debug=False):
    if debug:
        print("parser_robustrange")
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"1.0\n")
        tmpname = f.name
    parser = _get_parser()
    args = parser.parse_args([tmpname, "out", "--robustrange"])
    assert args.robustrange is True


def parser_nozero(debug=False):
    if debug:
        print("parser_nozero")
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"1.0\n")
        tmpname = f.name
    parser = _get_parser()
    args = parser.parse_args([tmpname, "out", "--nozero"])
    assert args.nozero is True


def parser_nozerothresh(debug=False):
    if debug:
        print("parser_nozerothresh")
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"1.0\n")
        tmpname = f.name
    parser = _get_parser()
    args = parser.parse_args([tmpname, "out", "--nozerothresh", "0.05"])
    assert args.nozerothresh == 0.05


def parser_normhist(debug=False):
    if debug:
        print("parser_normhist")
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"1.0\n")
        tmpname = f.name
    parser = _get_parser()
    args = parser.parse_args([tmpname, "out", "--normhist"])
    assert args.normhist is True


def parser_debug(debug=False):
    if debug:
        print("parser_debug")
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"1.0\n")
        tmpname = f.name
    parser = _get_parser()
    args = parser.parse_args([tmpname, "out", "--debug"])
    assert args.debug is True


def parser_all_flags(debug=False):
    if debug:
        print("parser_all_flags")
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"1.0\n")
        tmpname = f.name
    parser = _get_parser()
    args = parser.parse_args([
        tmpname, "out",
        "--numbins", "200",
        "--minval", "0.0",
        "--maxval", "100.0",
        "--robustrange",
        "--nozero",
        "--nozerothresh", "0.1",
        "--normhist",
        "--debug",
    ])
    assert args.histlen == 200
    assert args.minval == 0.0
    assert args.maxval == 100.0
    assert args.robustrange is True
    assert args.nozero is True
    assert args.nozerothresh == 0.1
    assert args.normhist is True
    assert args.debug is True


# ============================================================================
# histtc function tests
# ============================================================================


def histtc_basic(debug=False):
    """Test basic histogram generation with default settings."""
    if debug:
        print("histtc_basic")

    inputdata = np.random.default_rng(42).normal(0.0, 1.0, 1000)
    captured = {}

    def _mock_getfracvals(data, fracs, **kwargs):
        return [np.percentile(data, f * 100) for f in fracs]

    def _mock_makeandsavehistogram(indata, histlen, endtrim, outname, **kwargs):
        captured["indata"] = indata.copy()
        captured["histlen"] = histlen
        captured["endtrim"] = endtrim
        captured["outname"] = outname
        captured["therange"] = kwargs.get("therange")
        captured["normalize"] = kwargs.get("normalize")
        captured["debug"] = kwargs.get("debug")

    args = _make_default_args()

    with (
        patch(
            "rapidtide.workflows.histtc.tide_io.readvectorsfromtextfile",
            side_effect=_make_mock_readvectors(inputdata),
        ),
        patch(
            "rapidtide.workflows.histtc.tide_stats.getfracvals",
            side_effect=_mock_getfracvals,
        ),
        patch(
            "rapidtide.workflows.histtc.tide_stats.makeandsavehistogram",
            side_effect=_mock_makeandsavehistogram,
        ),
    ):
        histtc(args)

    np.testing.assert_array_equal(captured["indata"], inputdata)
    assert captured["histlen"] == 101
    assert captured["endtrim"] == 0
    assert captured["outname"] == "output"
    assert captured["normalize"] is False
    assert captured["debug"] is False
    # Default range: (min, max) of input data
    assert captured["therange"] == (np.min(inputdata), np.max(inputdata))


def histtc_nozero(debug=False):
    """Test that nozero filtering removes near-zero values."""
    if debug:
        print("histtc_nozero")

    inputdata = np.array([0.0, 0.005, -0.005, 1.0, 2.0, 3.0, -1.0, 0.02])
    captured = {}

    def _mock_getfracvals(data, fracs, **kwargs):
        return [np.percentile(data, f * 100) for f in fracs]

    def _mock_makeandsavehistogram(indata, histlen, endtrim, outname, **kwargs):
        captured["indata"] = indata.copy()

    args = _make_default_args(nozero=True, nozerothresh=0.01)

    with (
        patch(
            "rapidtide.workflows.histtc.tide_io.readvectorsfromtextfile",
            side_effect=_make_mock_readvectors(inputdata),
        ),
        patch(
            "rapidtide.workflows.histtc.tide_stats.getfracvals",
            side_effect=_mock_getfracvals,
        ),
        patch(
            "rapidtide.workflows.histtc.tide_stats.makeandsavehistogram",
            side_effect=_mock_makeandsavehistogram,
        ),
    ):
        histtc(args)

    # Values with |val| <= 0.01 should be removed: 0.0, 0.005, -0.005
    # Remaining: 1.0, 2.0, 3.0, -1.0, 0.02
    expected = np.array([1.0, 2.0, 3.0, -1.0, 0.02])
    np.testing.assert_array_equal(captured["indata"], expected)


def histtc_nozero_custom_thresh(debug=False):
    """Test nozero with a custom threshold."""
    if debug:
        print("histtc_nozero_custom_thresh")

    inputdata = np.array([0.0, 0.5, -0.5, 1.0, 2.0])
    captured = {}

    def _mock_getfracvals(data, fracs, **kwargs):
        return [np.percentile(data, f * 100) for f in fracs]

    def _mock_makeandsavehistogram(indata, histlen, endtrim, outname, **kwargs):
        captured["indata"] = indata.copy()

    args = _make_default_args(nozero=True, nozerothresh=0.5)

    with (
        patch(
            "rapidtide.workflows.histtc.tide_io.readvectorsfromtextfile",
            side_effect=_make_mock_readvectors(inputdata),
        ),
        patch(
            "rapidtide.workflows.histtc.tide_stats.getfracvals",
            side_effect=_mock_getfracvals,
        ),
        patch(
            "rapidtide.workflows.histtc.tide_stats.makeandsavehistogram",
            side_effect=_mock_makeandsavehistogram,
        ),
    ):
        histtc(args)

    # |val| > 0.5 → only 1.0 and 2.0 survive
    expected = np.array([1.0, 2.0])
    np.testing.assert_array_equal(captured["indata"], expected)


def histtc_robustrange(debug=False):
    """Test that robustrange uses 2nd and 98th percentiles for range."""
    if debug:
        print("histtc_robustrange")

    inputdata = np.arange(100, dtype=float)
    captured = {}

    call_count = {"n": 0}

    def _mock_getfracvals(data, fracs, **kwargs):
        call_count["n"] += 1
        return [np.percentile(data, f * 100) for f in fracs]

    def _mock_makeandsavehistogram(indata, histlen, endtrim, outname, **kwargs):
        captured["therange"] = kwargs.get("therange")

    args = _make_default_args(robustrange=True)

    with (
        patch(
            "rapidtide.workflows.histtc.tide_io.readvectorsfromtextfile",
            side_effect=_make_mock_readvectors(inputdata),
        ),
        patch(
            "rapidtide.workflows.histtc.tide_stats.getfracvals",
            side_effect=_mock_getfracvals,
        ),
        patch(
            "rapidtide.workflows.histtc.tide_stats.makeandsavehistogram",
            side_effect=_mock_makeandsavehistogram,
        ),
    ):
        histtc(args)

    # getfracvals called twice: once for thepercentiles, once for [0.02, 0.98]
    assert call_count["n"] == 2
    # Range should come from 2nd and 98th percentiles
    expected_min = np.percentile(inputdata, 2)
    expected_max = np.percentile(inputdata, 98)
    assert captured["therange"] == (expected_min, expected_max)


def histtc_minval_maxval(debug=False):
    """Test that explicit minval/maxval override data range."""
    if debug:
        print("histtc_minval_maxval")

    inputdata = np.arange(100, dtype=float)
    captured = {}

    def _mock_getfracvals(data, fracs, **kwargs):
        return [np.percentile(data, f * 100) for f in fracs]

    def _mock_makeandsavehistogram(indata, histlen, endtrim, outname, **kwargs):
        captured["therange"] = kwargs.get("therange")

    args = _make_default_args(minval=-10.0, maxval=200.0)

    with (
        patch(
            "rapidtide.workflows.histtc.tide_io.readvectorsfromtextfile",
            side_effect=_make_mock_readvectors(inputdata),
        ),
        patch(
            "rapidtide.workflows.histtc.tide_stats.getfracvals",
            side_effect=_mock_getfracvals,
        ),
        patch(
            "rapidtide.workflows.histtc.tide_stats.makeandsavehistogram",
            side_effect=_mock_makeandsavehistogram,
        ),
    ):
        histtc(args)

    assert captured["therange"] == (-10.0, 200.0)


def histtc_minval_only(debug=False):
    """Test that only minval is specified; maxval uses data max."""
    if debug:
        print("histtc_minval_only")

    inputdata = np.arange(50, dtype=float)
    captured = {}

    def _mock_getfracvals(data, fracs, **kwargs):
        return [np.percentile(data, f * 100) for f in fracs]

    def _mock_makeandsavehistogram(indata, histlen, endtrim, outname, **kwargs):
        captured["therange"] = kwargs.get("therange")

    args = _make_default_args(minval=-5.0, maxval=None)

    with (
        patch(
            "rapidtide.workflows.histtc.tide_io.readvectorsfromtextfile",
            side_effect=_make_mock_readvectors(inputdata),
        ),
        patch(
            "rapidtide.workflows.histtc.tide_stats.getfracvals",
            side_effect=_mock_getfracvals,
        ),
        patch(
            "rapidtide.workflows.histtc.tide_stats.makeandsavehistogram",
            side_effect=_mock_makeandsavehistogram,
        ),
    ):
        histtc(args)

    assert captured["therange"] == (-5.0, np.max(inputdata))


def histtc_maxval_only(debug=False):
    """Test that only maxval is specified; minval uses data min."""
    if debug:
        print("histtc_maxval_only")

    inputdata = np.arange(50, dtype=float)
    captured = {}

    def _mock_getfracvals(data, fracs, **kwargs):
        return [np.percentile(data, f * 100) for f in fracs]

    def _mock_makeandsavehistogram(indata, histlen, endtrim, outname, **kwargs):
        captured["therange"] = kwargs.get("therange")

    args = _make_default_args(minval=None, maxval=100.0)

    with (
        patch(
            "rapidtide.workflows.histtc.tide_io.readvectorsfromtextfile",
            side_effect=_make_mock_readvectors(inputdata),
        ),
        patch(
            "rapidtide.workflows.histtc.tide_stats.getfracvals",
            side_effect=_mock_getfracvals,
        ),
        patch(
            "rapidtide.workflows.histtc.tide_stats.makeandsavehistogram",
            side_effect=_mock_makeandsavehistogram,
        ),
    ):
        histtc(args)

    assert captured["therange"] == (np.min(inputdata), 100.0)


def histtc_normhist(debug=False):
    """Test that normhist is forwarded to makeandsavehistogram as normalize."""
    if debug:
        print("histtc_normhist")

    inputdata = np.ones(100)
    captured = {}

    def _mock_getfracvals(data, fracs, **kwargs):
        return [np.percentile(data, f * 100) for f in fracs]

    def _mock_makeandsavehistogram(indata, histlen, endtrim, outname, **kwargs):
        captured["normalize"] = kwargs.get("normalize")

    args = _make_default_args(normhist=True)

    with (
        patch(
            "rapidtide.workflows.histtc.tide_io.readvectorsfromtextfile",
            side_effect=_make_mock_readvectors(inputdata),
        ),
        patch(
            "rapidtide.workflows.histtc.tide_stats.getfracvals",
            side_effect=_mock_getfracvals,
        ),
        patch(
            "rapidtide.workflows.histtc.tide_stats.makeandsavehistogram",
            side_effect=_mock_makeandsavehistogram,
        ),
    ):
        histtc(args)

    assert captured["normalize"] is True


def histtc_debug_mode(debug=False):
    """Test that debug mode runs without error."""
    if debug:
        print("histtc_debug_mode")

    inputdata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    def _mock_getfracvals(data, fracs, **kwargs):
        return [np.percentile(data, f * 100) for f in fracs]

    args = _make_default_args(debug=True)

    captured = {}

    def _mock_makeandsavehistogram(indata, histlen, endtrim, outname, **kwargs):
        captured["debug"] = kwargs.get("debug")

    with (
        patch(
            "rapidtide.workflows.histtc.tide_io.readvectorsfromtextfile",
            side_effect=_make_mock_readvectors(inputdata),
        ),
        patch(
            "rapidtide.workflows.histtc.tide_stats.getfracvals",
            side_effect=_mock_getfracvals,
        ),
        patch(
            "rapidtide.workflows.histtc.tide_stats.makeandsavehistogram",
            side_effect=_mock_makeandsavehistogram,
        ),
    ):
        histtc(args)

    assert captured["debug"] is True


def histtc_percentiles_printed(debug=False):
    """Verify that percentile values are printed."""
    if debug:
        print("histtc_percentiles_printed")

    inputdata = np.arange(100, dtype=float)

    def _mock_getfracvals(data, fracs, **kwargs):
        return [np.percentile(data, f * 100) for f in fracs]

    args = _make_default_args()

    with (
        patch(
            "rapidtide.workflows.histtc.tide_io.readvectorsfromtextfile",
            side_effect=_make_mock_readvectors(inputdata),
        ),
        patch(
            "rapidtide.workflows.histtc.tide_stats.getfracvals",
            side_effect=_mock_getfracvals,
        ),
        patch("rapidtide.workflows.histtc.tide_stats.makeandsavehistogram"),
        patch("builtins.print") as mock_print,
    ):
        histtc(args)

    # Should print 5 percentile lines + 1 range line
    percentile_calls = [
        c for c in mock_print.call_args_list if "percentile" in str(c)
    ]
    assert len(percentile_calls) == 5


def histtc_inputfilename_forwarded(debug=False):
    """Verify the input filename is passed to readvectorsfromtextfile."""
    if debug:
        print("histtc_inputfilename_forwarded")

    inputdata = np.ones(10)
    captured = {}

    def _mock_readvectors(filepath, onecol=False, debug=False):
        captured["filepath"] = filepath
        return (None, None, None, np.atleast_2d(inputdata), None, "text")

    def _mock_getfracvals(data, fracs, **kwargs):
        return [np.percentile(data, f * 100) for f in fracs]

    args = _make_default_args(inputfilename="special_data.txt")

    with (
        patch(
            "rapidtide.workflows.histtc.tide_io.readvectorsfromtextfile",
            side_effect=_mock_readvectors,
        ),
        patch(
            "rapidtide.workflows.histtc.tide_stats.getfracvals",
            side_effect=_mock_getfracvals,
        ),
        patch("rapidtide.workflows.histtc.tide_stats.makeandsavehistogram"),
    ):
        histtc(args)

    assert captured["filepath"] == "special_data.txt"


def histtc_custom_histlen(debug=False):
    """Test that a custom histlen value is forwarded."""
    if debug:
        print("histtc_custom_histlen")

    inputdata = np.ones(100)
    captured = {}

    def _mock_getfracvals(data, fracs, **kwargs):
        return [np.percentile(data, f * 100) for f in fracs]

    def _mock_makeandsavehistogram(indata, histlen, endtrim, outname, **kwargs):
        captured["histlen"] = histlen

    args = _make_default_args(histlen=50)

    with (
        patch(
            "rapidtide.workflows.histtc.tide_io.readvectorsfromtextfile",
            side_effect=_make_mock_readvectors(inputdata),
        ),
        patch(
            "rapidtide.workflows.histtc.tide_stats.getfracvals",
            side_effect=_mock_getfracvals,
        ),
        patch(
            "rapidtide.workflows.histtc.tide_stats.makeandsavehistogram",
            side_effect=_mock_makeandsavehistogram,
        ),
    ):
        histtc(args)

    assert captured["histlen"] == 50


def histtc_extracts_first_row(debug=False):
    """Verify the function extracts row [0] from the 2D array returned by readvectors."""
    if debug:
        print("histtc_extracts_first_row")

    row0 = np.array([10.0, 20.0, 30.0])
    row1 = np.array([40.0, 50.0, 60.0])
    data_2d = np.vstack([row0, row1])
    captured = {}

    def _mock_readvectors(filepath, onecol=False, debug=False):
        return (None, None, None, data_2d.copy(), None, "text")

    def _mock_getfracvals(data, fracs, **kwargs):
        return [np.percentile(data, f * 100) for f in fracs]

    def _mock_makeandsavehistogram(indata, histlen, endtrim, outname, **kwargs):
        captured["indata"] = indata.copy()

    args = _make_default_args()

    with (
        patch(
            "rapidtide.workflows.histtc.tide_io.readvectorsfromtextfile",
            side_effect=_mock_readvectors,
        ),
        patch(
            "rapidtide.workflows.histtc.tide_stats.getfracvals",
            side_effect=_mock_getfracvals,
        ),
        patch(
            "rapidtide.workflows.histtc.tide_stats.makeandsavehistogram",
            side_effect=_mock_makeandsavehistogram,
        ),
    ):
        histtc(args)

    np.testing.assert_array_equal(captured["indata"], row0)


# ============================================================================
# Main test entry point
# ============================================================================


def test_histtc(debug=False):
    # Parser tests
    parser_basic(debug=debug)
    parser_required_args(debug=debug)
    parser_defaults(debug=debug)
    parser_numbins(debug=debug)
    parser_minval_maxval(debug=debug)
    parser_robustrange(debug=debug)
    parser_nozero(debug=debug)
    parser_nozerothresh(debug=debug)
    parser_normhist(debug=debug)
    parser_debug(debug=debug)
    parser_all_flags(debug=debug)

    # histtc function tests
    histtc_basic(debug=debug)
    histtc_nozero(debug=debug)
    histtc_nozero_custom_thresh(debug=debug)
    histtc_robustrange(debug=debug)
    histtc_minval_maxval(debug=debug)
    histtc_minval_only(debug=debug)
    histtc_maxval_only(debug=debug)
    histtc_normhist(debug=debug)
    histtc_debug_mode(debug=debug)
    histtc_percentiles_printed(debug=debug)
    histtc_inputfilename_forwarded(debug=debug)
    histtc_custom_histlen(debug=debug)
    histtc_extracts_first_row(debug=debug)


if __name__ == "__main__":
    test_histtc(debug=True)
