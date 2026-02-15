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
from unittest.mock import patch

import numpy as np
import pytest

from rapidtide.workflows.showhist import _get_parser, showhist

# ============================================================================
# Helpers
# ============================================================================


def _make_default_args(**overrides):
    defaults = dict(
        infilename="input.txt",
        thexlabel=None,
        theylabel=None,
        thetitle=None,
        outputfile=None,
        dobars=False,
        calcdist=False,
        debug=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ============================================================================
# Parser tests
# ============================================================================


def parser_basic(debug=False):
    if debug:
        print("parser_basic")
    parser = _get_parser()
    assert parser is not None
    assert parser.prog == "showhist"


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
    args = parser.parse_args(["data.txt"])
    assert args.infilename == "data.txt"
    assert args.thexlabel is None
    assert args.theylabel is None
    assert args.thetitle is None
    assert args.outputfile is None
    assert args.dobars is False
    assert args.calcdist is False
    assert args.debug is False


def parser_xlabel(debug=False):
    if debug:
        print("parser_xlabel")
    parser = _get_parser()
    args = parser.parse_args(["data.txt", "--xlabel", "Time (s)"])
    assert args.thexlabel == "Time (s)"


def parser_ylabel(debug=False):
    if debug:
        print("parser_ylabel")
    parser = _get_parser()
    args = parser.parse_args(["data.txt", "--ylabel", "Counts"])
    assert args.theylabel == "Counts"


def parser_title(debug=False):
    if debug:
        print("parser_title")
    parser = _get_parser()
    args = parser.parse_args(["data.txt", "--title", "My Histogram"])
    assert args.thetitle == "My Histogram"


def parser_outputfile(debug=False):
    if debug:
        print("parser_outputfile")
    parser = _get_parser()
    args = parser.parse_args(["data.txt", "--outputfile", "plot.png"])
    assert args.outputfile == "plot.png"


def parser_dobars(debug=False):
    if debug:
        print("parser_dobars")
    parser = _get_parser()
    args = parser.parse_args(["data.txt", "--dobars"])
    assert args.dobars is True


def parser_calcdist(debug=False):
    if debug:
        print("parser_calcdist")
    parser = _get_parser()
    args = parser.parse_args(["data.txt", "--calcdist"])
    assert args.calcdist is True


def parser_debug(debug=False):
    if debug:
        print("parser_debug")
    parser = _get_parser()
    args = parser.parse_args(["data.txt", "--debug"])
    assert args.debug is True


def parser_all_flags(debug=False):
    if debug:
        print("parser_all_flags")
    parser = _get_parser()
    args = parser.parse_args([
        "data.txt",
        "--xlabel", "X",
        "--ylabel", "Y",
        "--title", "T",
        "--outputfile", "out.png",
        "--dobars",
        "--calcdist",
        "--debug",
    ])
    assert args.thexlabel == "X"
    assert args.theylabel == "Y"
    assert args.thetitle == "T"
    assert args.outputfile == "out.png"
    assert args.dobars is True
    assert args.calcdist is True
    assert args.debug is True


# ============================================================================
# showhist function tests
# ============================================================================


def showhist_line_plot(debug=False):
    """Test basic line plot mode (no calcdist, no bars)."""
    if debug:
        print("showhist_line_plot")

    xvals = np.arange(10, dtype=float)
    yvals = np.arange(10, dtype=float) * 2.0
    indata = np.vstack([xvals, yvals])

    args = _make_default_args()

    with (
        patch("rapidtide.workflows.showhist.tide_io.readvecs", return_value=indata),
        patch("rapidtide.workflows.showhist.plot") as mock_plot,
        patch("rapidtide.workflows.showhist.show") as mock_show,
        patch("rapidtide.workflows.showhist.bar") as mock_bar,
    ):
        showhist(args)

    mock_plot.assert_called_once()
    call_args = mock_plot.call_args
    np.testing.assert_array_equal(call_args[0][0], xvals)
    np.testing.assert_array_equal(call_args[0][1], yvals)
    assert call_args[0][2] == "r"
    mock_bar.assert_not_called()
    mock_show.assert_called_once()


def showhist_bar_plot(debug=False):
    """Test bar plot mode (dobars=True)."""
    if debug:
        print("showhist_bar_plot")

    xvals = np.array([0.0, 1.0, 2.0, 3.0])
    yvals = np.array([5.0, 10.0, 15.0, 20.0])
    indata = np.vstack([xvals, yvals])

    args = _make_default_args(dobars=True)

    with (
        patch("rapidtide.workflows.showhist.tide_io.readvecs", return_value=indata),
        patch("rapidtide.workflows.showhist.bar") as mock_bar,
        patch("rapidtide.workflows.showhist.plot") as mock_plot,
        patch("rapidtide.workflows.showhist.show"),
    ):
        showhist(args)

    mock_bar.assert_called_once()
    call_args = mock_bar.call_args
    np.testing.assert_array_equal(call_args[0][0], xvals)
    np.testing.assert_array_equal(call_args[0][1], yvals)
    expected_width = 0.8 * (xvals[1] - xvals[0])
    assert call_args[1]["width"] == expected_width
    assert call_args[1]["color"] == "g"
    mock_plot.assert_not_called()


def showhist_calcdist(debug=False):
    """Test calcdist mode: reads raw data, computes histogram, then plots."""
    if debug:
        print("showhist_calcdist")

    raw_data = np.random.default_rng(42).normal(0.0, 1.0, (1, 500))
    histlen = 101
    # Mock histogram output: (counts, bin_edges)
    counts = np.ones(histlen, dtype=float)
    bin_edges = np.linspace(-3.0, 3.0, histlen + 1)
    mock_hist_tuple = (counts, bin_edges)

    captured = {}

    def _mock_makehistogram(indata, hl, **kwargs):
        captured["histlen"] = hl
        return (mock_hist_tuple, 1.0, 0.0, 1.0, 0.0, 50.0)

    args = _make_default_args(calcdist=True)

    with (
        patch("rapidtide.workflows.showhist.tide_io.readvecs", return_value=raw_data),
        patch(
            "rapidtide.workflows.showhist.tide_stats.makehistogram",
            side_effect=_mock_makehistogram,
        ),
        patch("rapidtide.workflows.showhist.plot") as mock_plot,
        patch("rapidtide.workflows.showhist.show"),
    ):
        showhist(args)

    assert captured["histlen"] == 101
    mock_plot.assert_called_once()
    call_args = mock_plot.call_args
    # xvecs should be bin centers: (bin_edges[1:] + bin_edges[:-1]) / 2
    expected_x = (bin_edges[1:] + bin_edges[:-1]) / 2.0
    np.testing.assert_allclose(call_args[0][0], expected_x)
    # yvecs should be counts[-histlen:]
    np.testing.assert_array_equal(call_args[0][1], counts[-histlen:])


def showhist_calcdist_with_bars(debug=False):
    """Test calcdist combined with dobars."""
    if debug:
        print("showhist_calcdist_with_bars")

    raw_data = np.ones((1, 100))
    histlen = 101
    counts = np.ones(histlen, dtype=float)
    bin_edges = np.linspace(0.0, 2.0, histlen + 1)
    mock_hist_tuple = (counts, bin_edges)

    args = _make_default_args(calcdist=True, dobars=True)

    with (
        patch("rapidtide.workflows.showhist.tide_io.readvecs", return_value=raw_data),
        patch(
            "rapidtide.workflows.showhist.tide_stats.makehistogram",
            return_value=(mock_hist_tuple, 1.0, 1.0, 1.0, 1.0, 50.0),
        ),
        patch("rapidtide.workflows.showhist.bar") as mock_bar,
        patch("rapidtide.workflows.showhist.plot") as mock_plot,
        patch("rapidtide.workflows.showhist.show"),
    ):
        showhist(args)

    mock_bar.assert_called_once()
    mock_plot.assert_not_called()


def showhist_with_title(debug=False):
    """Test that title is set when provided."""
    if debug:
        print("showhist_with_title")

    indata = np.zeros((2, 5))
    args = _make_default_args(thetitle="My Title")

    with (
        patch("rapidtide.workflows.showhist.tide_io.readvecs", return_value=indata),
        patch("rapidtide.workflows.showhist.plot"),
        patch("rapidtide.workflows.showhist.show"),
        patch("rapidtide.workflows.showhist.title") as mock_title,
    ):
        showhist(args)

    mock_title.assert_called_once_with("My Title")


def showhist_no_title(debug=False):
    """Test that title() is not called when thetitle is None."""
    if debug:
        print("showhist_no_title")

    indata = np.zeros((2, 5))
    args = _make_default_args(thetitle=None)

    with (
        patch("rapidtide.workflows.showhist.tide_io.readvecs", return_value=indata),
        patch("rapidtide.workflows.showhist.plot"),
        patch("rapidtide.workflows.showhist.show"),
        patch("rapidtide.workflows.showhist.title") as mock_title,
    ):
        showhist(args)

    mock_title.assert_not_called()


def showhist_with_xlabel(debug=False):
    """Test that xlabel is set when provided."""
    if debug:
        print("showhist_with_xlabel")

    indata = np.zeros((2, 5))
    args = _make_default_args(thexlabel="Time (s)")

    with (
        patch("rapidtide.workflows.showhist.tide_io.readvecs", return_value=indata),
        patch("rapidtide.workflows.showhist.plot"),
        patch("rapidtide.workflows.showhist.show"),
        patch("rapidtide.workflows.showhist.xlabel") as mock_xlabel,
    ):
        showhist(args)

    mock_xlabel.assert_called_once_with("Time (s)", fontsize=16, fontweight="bold")


def showhist_no_xlabel(debug=False):
    """Test that xlabel() is not called when thexlabel is None."""
    if debug:
        print("showhist_no_xlabel")

    indata = np.zeros((2, 5))
    args = _make_default_args(thexlabel=None)

    with (
        patch("rapidtide.workflows.showhist.tide_io.readvecs", return_value=indata),
        patch("rapidtide.workflows.showhist.plot"),
        patch("rapidtide.workflows.showhist.show"),
        patch("rapidtide.workflows.showhist.xlabel") as mock_xlabel,
    ):
        showhist(args)

    mock_xlabel.assert_not_called()


def showhist_with_ylabel(debug=False):
    """Test that ylabel is set when provided."""
    if debug:
        print("showhist_with_ylabel")

    indata = np.zeros((2, 5))
    args = _make_default_args(theylabel="Counts")

    with (
        patch("rapidtide.workflows.showhist.tide_io.readvecs", return_value=indata),
        patch("rapidtide.workflows.showhist.plot"),
        patch("rapidtide.workflows.showhist.show"),
        patch("rapidtide.workflows.showhist.ylabel") as mock_ylabel,
    ):
        showhist(args)

    mock_ylabel.assert_called_once_with("Counts", fontsize=16, fontweight="bold")


def showhist_no_ylabel(debug=False):
    """Test that ylabel() is not called when theylabel is None."""
    if debug:
        print("showhist_no_ylabel")

    indata = np.zeros((2, 5))
    args = _make_default_args(theylabel=None)

    with (
        patch("rapidtide.workflows.showhist.tide_io.readvecs", return_value=indata),
        patch("rapidtide.workflows.showhist.plot"),
        patch("rapidtide.workflows.showhist.show"),
        patch("rapidtide.workflows.showhist.ylabel") as mock_ylabel,
    ):
        showhist(args)

    mock_ylabel.assert_not_called()


def showhist_save_to_file(debug=False):
    """Test that savefig is called instead of show when outputfile is set."""
    if debug:
        print("showhist_save_to_file")

    indata = np.zeros((2, 5))
    args = _make_default_args(outputfile="output.png")

    with (
        patch("rapidtide.workflows.showhist.tide_io.readvecs", return_value=indata),
        patch("rapidtide.workflows.showhist.plot"),
        patch("rapidtide.workflows.showhist.show") as mock_show,
        patch("rapidtide.workflows.showhist.savefig") as mock_savefig,
    ):
        showhist(args)

    mock_savefig.assert_called_once_with("output.png", bbox_inches="tight")
    mock_show.assert_not_called()


def showhist_display_on_screen(debug=False):
    """Test that show() is called when outputfile is None."""
    if debug:
        print("showhist_display_on_screen")

    indata = np.zeros((2, 5))
    args = _make_default_args(outputfile=None)

    with (
        patch("rapidtide.workflows.showhist.tide_io.readvecs", return_value=indata),
        patch("rapidtide.workflows.showhist.plot"),
        patch("rapidtide.workflows.showhist.show") as mock_show,
        patch("rapidtide.workflows.showhist.savefig") as mock_savefig,
    ):
        showhist(args)

    mock_show.assert_called_once()
    mock_savefig.assert_not_called()


def showhist_debug_mode(debug=False):
    """Test that debug mode prints args."""
    if debug:
        print("showhist_debug_mode")

    indata = np.zeros((2, 5))
    args = _make_default_args(debug=True)

    with (
        patch("rapidtide.workflows.showhist.tide_io.readvecs", return_value=indata),
        patch("rapidtide.workflows.showhist.plot"),
        patch("rapidtide.workflows.showhist.show"),
        patch("builtins.print") as mock_print,
    ):
        showhist(args)

    # First print call should be the args object
    mock_print.assert_called_once_with(args)


def showhist_no_debug(debug=False):
    """Test that args are not printed when debug is False."""
    if debug:
        print("showhist_no_debug")

    indata = np.zeros((2, 5))
    args = _make_default_args(debug=False)

    with (
        patch("rapidtide.workflows.showhist.tide_io.readvecs", return_value=indata),
        patch("rapidtide.workflows.showhist.plot"),
        patch("rapidtide.workflows.showhist.show"),
        patch("builtins.print") as mock_print,
    ):
        showhist(args)

    mock_print.assert_not_called()


def showhist_all_labels_and_save(debug=False):
    """Test with all labels set and saving to file."""
    if debug:
        print("showhist_all_labels_and_save")

    indata = np.zeros((2, 5))
    args = _make_default_args(
        thetitle="Title",
        thexlabel="X",
        theylabel="Y",
        outputfile="out.pdf",
    )

    with (
        patch("rapidtide.workflows.showhist.tide_io.readvecs", return_value=indata),
        patch("rapidtide.workflows.showhist.plot"),
        patch("rapidtide.workflows.showhist.title") as mock_title,
        patch("rapidtide.workflows.showhist.xlabel") as mock_xlabel,
        patch("rapidtide.workflows.showhist.ylabel") as mock_ylabel,
        patch("rapidtide.workflows.showhist.savefig") as mock_savefig,
        patch("rapidtide.workflows.showhist.show") as mock_show,
    ):
        showhist(args)

    mock_title.assert_called_once_with("Title")
    mock_xlabel.assert_called_once_with("X", fontsize=16, fontweight="bold")
    mock_ylabel.assert_called_once_with("Y", fontsize=16, fontweight="bold")
    mock_savefig.assert_called_once_with("out.pdf", bbox_inches="tight")
    mock_show.assert_not_called()


def showhist_infilename_forwarded(debug=False):
    """Verify the input filename is passed to readvecs."""
    if debug:
        print("showhist_infilename_forwarded")

    indata = np.zeros((2, 5))
    args = _make_default_args(infilename="my_data.txt")

    mock_readvecs = patch(
        "rapidtide.workflows.showhist.tide_io.readvecs", return_value=indata
    )

    with (
        mock_readvecs as m_readvecs,
        patch("rapidtide.workflows.showhist.plot"),
        patch("rapidtide.workflows.showhist.show"),
    ):
        showhist(args)

    m_readvecs.assert_called_once_with("my_data.txt")


# ============================================================================
# Main test entry point
# ============================================================================


def test_showhist(debug=False):
    # Parser tests
    parser_basic(debug=debug)
    parser_required_args(debug=debug)
    parser_defaults(debug=debug)
    parser_xlabel(debug=debug)
    parser_ylabel(debug=debug)
    parser_title(debug=debug)
    parser_outputfile(debug=debug)
    parser_dobars(debug=debug)
    parser_calcdist(debug=debug)
    parser_debug(debug=debug)
    parser_all_flags(debug=debug)

    # showhist function tests
    showhist_line_plot(debug=debug)
    showhist_bar_plot(debug=debug)
    showhist_calcdist(debug=debug)
    showhist_calcdist_with_bars(debug=debug)
    showhist_with_title(debug=debug)
    showhist_no_title(debug=debug)
    showhist_with_xlabel(debug=debug)
    showhist_no_xlabel(debug=debug)
    showhist_with_ylabel(debug=debug)
    showhist_no_ylabel(debug=debug)
    showhist_save_to_file(debug=debug)
    showhist_display_on_screen(debug=debug)
    showhist_debug_mode(debug=debug)
    showhist_no_debug(debug=debug)
    showhist_all_labels_and_save(debug=debug)
    showhist_infilename_forwarded(debug=debug)


if __name__ == "__main__":
    test_showhist(debug=True)
