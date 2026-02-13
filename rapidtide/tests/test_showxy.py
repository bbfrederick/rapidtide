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
from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from rapidtide.workflows.showxy import (
    _get_parser,
    bland_altman_plot,
    showxy,
    stringtorange,
)

# ---- helpers ----


def _make_default_args(**overrides):
    """Create a default args Namespace for showxy."""
    defaults = dict(
        textfilenames=["data1.txt"],
        thexlabel=None,
        theylabel=None,
        xrange=None,
        yrange=None,
        thetitle=None,
        outputfile=None,
        fontscalefac=1.0,
        saveres=1000,
        legends=None,
        legendloc=2,
        colors=None,
        blandaltman=False,
        usex=False,
        doannotate=True,
        usepoints=False,
        dobars=False,
        debug=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _make_xy_data(nfiles=1, npoints=50):
    """Create synthetic xy data arrays for testing.

    Returns a dict mapping filenames to 2-row numpy arrays (x, y).
    """
    rng = np.random.RandomState(42)
    data = {}
    for i in range(nfiles):
        x = np.linspace(0, 10, npoints)
        y = 2.0 * x + 1.0 + rng.normal(0, 0.5, npoints)
        data[f"data{i + 1}.txt"] = np.vstack([x, y])
    return data


def _run_showxy_with_mocks(args, file_data):
    """Run showxy with mocked I/O and matplotlib.

    Parameters
    ----------
    args : Namespace
        The args to pass to showxy.
    file_data : dict
        Mapping from filename to 2-row numpy arrays.

    Returns a dict with captured matplotlib calls.
    """
    captured = {
        "plot_calls": [],
        "scatter_calls": [],
        "bar_calls": [],
        "savefig_calls": [],
        "show_calls": 0,
        "title_calls": [],
        "xlabel_calls": [],
        "ylabel_calls": [],
        "legend_calls": [],
        "xlim_calls": [],
        "ylim_calls": [],
        "axhline_calls": [],
        "annotate_calls": [],
    }

    def mock_readvecs(fname, **kwargs):
        if fname in file_data:
            return file_data[fname].copy()
        raise FileNotFoundError(f"No mock data for {fname}")

    def mock_plot(*a, **kw):
        captured["plot_calls"].append((a, kw))

    def mock_scatter(*a, **kw):
        captured["scatter_calls"].append((a, kw))

    def mock_bar(*a, **kw):
        captured["bar_calls"].append((a, kw))

    def mock_savefig(*a, **kw):
        captured["savefig_calls"].append((a, kw))

    def mock_show(*a, **kw):
        captured["show_calls"] += 1

    def mock_title(*a, **kw):
        captured["title_calls"].append((a, kw))

    def mock_xlabel(*a, **kw):
        captured["xlabel_calls"].append((a, kw))

    def mock_ylabel(*a, **kw):
        captured["ylabel_calls"].append((a, kw))

    def mock_legend(*a, **kw):
        captured["legend_calls"].append((a, kw))

    def mock_xlim(*a, **kw):
        captured["xlim_calls"].append((a, kw))

    def mock_ylim(*a, **kw):
        captured["ylim_calls"].append((a, kw))

    def mock_axhline(*a, **kw):
        captured["axhline_calls"].append((a, kw))

    def mock_annotate(*a, **kw):
        captured["annotate_calls"].append((a, kw))

    with (
        patch("rapidtide.workflows.showxy.tide_io.readvecs", side_effect=mock_readvecs),
        patch("rapidtide.workflows.showxy.plot", side_effect=mock_plot),
        patch("rapidtide.workflows.showxy.scatter", side_effect=mock_scatter),
        patch("rapidtide.workflows.showxy.bar", side_effect=mock_bar),
        patch("rapidtide.workflows.showxy.savefig", side_effect=mock_savefig),
        patch("rapidtide.workflows.showxy.show", side_effect=mock_show),
        patch("rapidtide.workflows.showxy.title", side_effect=mock_title),
        patch("rapidtide.workflows.showxy.xlabel", side_effect=mock_xlabel),
        patch("rapidtide.workflows.showxy.ylabel", side_effect=mock_ylabel),
        patch("rapidtide.workflows.showxy.legend", side_effect=mock_legend),
        patch("rapidtide.workflows.showxy.xlim", side_effect=mock_xlim),
        patch("rapidtide.workflows.showxy.ylim", side_effect=mock_ylim),
        patch("rapidtide.workflows.showxy.axhline", side_effect=mock_axhline),
        patch("rapidtide.workflows.showxy.annotate", side_effect=mock_annotate),
    ):
        showxy(args)

    return captured


# ---- _get_parser tests ----


def parser_basic(debug=False):
    """Test that _get_parser returns a valid parser."""
    parser = _get_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    assert parser.prog == "showxy"

    if debug:
        print("parser_basic passed")


def parser_required_args(debug=False):
    """Test that parser requires at least one textfilename."""
    parser = _get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])

    if debug:
        print("parser_required_args passed")


def parser_single_file(debug=False):
    """Test parser with a single file."""
    parser = _get_parser()
    args = parser.parse_args(["data.txt"])
    assert args.textfilenames == ["data.txt"]

    if debug:
        print("parser_single_file passed")


def parser_multiple_files(debug=False):
    """Test parser with multiple files."""
    parser = _get_parser()
    args = parser.parse_args(["data1.txt", "data2.txt", "data3.txt"])
    assert args.textfilenames == ["data1.txt", "data2.txt", "data3.txt"]

    if debug:
        print("parser_multiple_files passed")


def parser_defaults(debug=False):
    """Test default values from the parser."""
    parser = _get_parser()
    args = parser.parse_args(["data.txt"])

    assert args.thexlabel is None
    assert args.theylabel is None
    assert args.xrange is None
    assert args.yrange is None
    assert args.thetitle is None
    assert args.outputfile is None
    assert args.fontscalefac == 1.0
    assert args.saveres == 1000
    assert args.legends is None
    assert args.legendloc == 2
    assert args.colors is None
    assert args.blandaltman is False
    assert args.usex is False
    assert args.doannotate is True
    assert args.usepoints is False
    assert args.dobars is False
    assert args.debug is False

    if debug:
        print("parser_defaults passed")


def parser_xlabel_ylabel(debug=False):
    """Test --xlabel and --ylabel options."""
    parser = _get_parser()
    args = parser.parse_args(["data.txt", "--xlabel", "Time", "--ylabel", "Value"])
    assert args.thexlabel == "Time"
    assert args.theylabel == "Value"

    if debug:
        print("parser_xlabel_ylabel passed")


def parser_xrange_yrange(debug=False):
    """Test --xrange and --yrange options."""
    parser = _get_parser()
    args = parser.parse_args(["data.txt", "--xrange", "0", "10", "--yrange", "-5", "5"])
    assert args.xrange == [0.0, 10.0]
    assert args.yrange == [-5.0, 5.0]

    if debug:
        print("parser_xrange_yrange passed")


def parser_title(debug=False):
    """Test --title option."""
    parser = _get_parser()
    args = parser.parse_args(["data.txt", "--title", "My Plot"])
    assert args.thetitle == "My Plot"

    if debug:
        print("parser_title passed")


def parser_outputfile(debug=False):
    """Test --outputfile option."""
    parser = _get_parser()
    args = parser.parse_args(["data.txt", "--outputfile", "out.png"])
    assert args.outputfile == "out.png"

    if debug:
        print("parser_outputfile passed")


def parser_fontscalefac(debug=False):
    """Test --fontscalefac option."""
    parser = _get_parser()
    args = parser.parse_args(["data.txt", "--fontscalefac", "2.5"])
    assert np.isclose(args.fontscalefac, 2.5)

    if debug:
        print("parser_fontscalefac passed")


def parser_saveres(debug=False):
    """Test --saveres option."""
    parser = _get_parser()
    args = parser.parse_args(["data.txt", "--saveres", "300"])
    assert args.saveres == 300

    if debug:
        print("parser_saveres passed")


def parser_legends(debug=False):
    """Test --legends option."""
    parser = _get_parser()
    args = parser.parse_args(["data.txt", "--legends", "A,B,C"])
    assert args.legends == "A,B,C"

    if debug:
        print("parser_legends passed")


def parser_legendloc(debug=False):
    """Test --legendloc option."""
    parser = _get_parser()
    args = parser.parse_args(["data.txt", "--legendloc", "5"])
    assert args.legendloc == 5

    if debug:
        print("parser_legendloc passed")


def parser_colors(debug=False):
    """Test --colors option."""
    parser = _get_parser()
    args = parser.parse_args(["data.txt", "--colors", "red,blue,green"])
    assert args.colors == "red,blue,green"

    if debug:
        print("parser_colors passed")


def parser_boolean_flags(debug=False):
    """Test all boolean flags."""
    parser = _get_parser()
    args = parser.parse_args([
        "data.txt",
        "--blandaltman",
        "--usex",
        "--noannotate",
        "--usepoints",
        "--dobars",
        "--debug",
    ])
    assert args.blandaltman is True
    assert args.usex is True
    assert args.doannotate is False
    assert args.usepoints is True
    assert args.dobars is True
    assert args.debug is True

    if debug:
        print("parser_boolean_flags passed")


# ---- stringtorange tests ----


def stringtorange_basic(debug=False):
    """Test stringtorange with valid input."""
    result = stringtorange("1.5,10.0")
    assert result == (1.5, 10.0)

    if debug:
        print("stringtorange_basic passed")


def stringtorange_negative(debug=False):
    """Test stringtorange with negative values."""
    result = stringtorange("-5.0,5.0")
    assert result == (-5.0, 5.0)

    if debug:
        print("stringtorange_negative passed")


def stringtorange_zero(debug=False):
    """Test stringtorange with zero values."""
    result = stringtorange("0,0")
    assert result == (0.0, 0.0)

    if debug:
        print("stringtorange_zero passed")


def stringtorange_integers(debug=False):
    """Test stringtorange with integer string values."""
    result = stringtorange("3,7")
    assert result == (3.0, 7.0)

    if debug:
        print("stringtorange_integers passed")


def stringtorange_scientific(debug=False):
    """Test stringtorange with scientific notation."""
    result = stringtorange("1e-3,1e3")
    assert result == (0.001, 1000.0)

    if debug:
        print("stringtorange_scientific passed")


def stringtorange_too_few(debug=False):
    """Test stringtorange exits with only one value."""
    with pytest.raises(SystemExit):
        stringtorange("1.5")

    if debug:
        print("stringtorange_too_few passed")


def stringtorange_too_many(debug=False):
    """Test stringtorange exits with more than two values."""
    with pytest.raises(SystemExit):
        stringtorange("1.0,2.0,3.0")

    if debug:
        print("stringtorange_too_many passed")


def stringtorange_invalid_first(debug=False):
    """Test stringtorange exits with non-numeric first value."""
    with pytest.raises(SystemExit):
        stringtorange("abc,10.0")

    if debug:
        print("stringtorange_invalid_first passed")


def stringtorange_invalid_second(debug=False):
    """Test stringtorange exits with non-numeric second value."""
    with pytest.raises(SystemExit):
        stringtorange("1.0,xyz")

    if debug:
        print("stringtorange_invalid_second passed")


def stringtorange_empty(debug=False):
    """Test stringtorange exits with empty string."""
    with pytest.raises(SystemExit):
        stringtorange("")

    if debug:
        print("stringtorange_empty passed")


# ---- bland_altman_plot tests ----


def bland_altman_basic(debug=False):
    """Test bland_altman_plot runs with basic inputs."""
    data1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    data2 = np.array([1.1, 2.2, 2.8, 4.1, 4.9])

    captured_stdout = io.StringIO()
    with (
        patch("rapidtide.workflows.showxy.scatter"),
        patch("rapidtide.workflows.showxy.axhline"),
        patch("rapidtide.workflows.showxy.annotate"),
        patch("rapidtide.workflows.showxy.xlim"),
        patch("rapidtide.workflows.showxy.ylim"),
        patch("sys.stdout", captured_stdout),
    ):
        bland_altman_plot(data1, data2)

    output = captured_stdout.getvalue()
    assert "slope:" in output
    assert "r_value:" in output
    assert "mean difference:" in output
    assert "std difference:" in output

    if debug:
        print("bland_altman_basic passed")


def bland_altman_usex(debug=False):
    """Test bland_altman_plot with usex=True."""
    data1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    data2 = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

    captured_stdout = io.StringIO()
    with (
        patch("rapidtide.workflows.showxy.scatter"),
        patch("rapidtide.workflows.showxy.axhline"),
        patch("rapidtide.workflows.showxy.annotate"),
        patch("rapidtide.workflows.showxy.xlim"),
        patch("rapidtide.workflows.showxy.ylim"),
        patch("sys.stdout", captured_stdout),
    ):
        bland_altman_plot(data1, data2, usex=True, debug=True)

    output = captured_stdout.getvalue()
    assert "using X as the X axis" in output
    assert "diff_slope:" in output

    if debug:
        print("bland_altman_usex passed")


def bland_altman_default_mean_axis(debug=False):
    """Test bland_altman_plot uses (Y+X)/2 when usex=False."""
    data1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    data2 = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

    captured_stdout = io.StringIO()
    with (
        patch("rapidtide.workflows.showxy.scatter"),
        patch("rapidtide.workflows.showxy.axhline"),
        patch("rapidtide.workflows.showxy.annotate"),
        patch("rapidtide.workflows.showxy.xlim"),
        patch("rapidtide.workflows.showxy.ylim"),
        patch("sys.stdout", captured_stdout),
    ):
        bland_altman_plot(data1, data2, usex=False, debug=True)

    output = captured_stdout.getvalue()
    assert "using (Y + X)/2 as the X axis" in output

    if debug:
        print("bland_altman_default_mean_axis passed")


def bland_altman_identifier(debug=False):
    """Test bland_altman_plot prints identifier."""
    data1 = np.array([1.0, 2.0, 3.0])
    data2 = np.array([1.0, 2.0, 3.0])

    captured_stdout = io.StringIO()
    with (
        patch("rapidtide.workflows.showxy.scatter"),
        patch("rapidtide.workflows.showxy.axhline"),
        patch("rapidtide.workflows.showxy.annotate"),
        patch("rapidtide.workflows.showxy.xlim"),
        patch("rapidtide.workflows.showxy.ylim"),
        patch("sys.stdout", captured_stdout),
    ):
        bland_altman_plot(data1, data2, identifier="test_dataset")

    output = captured_stdout.getvalue()
    assert "id: test_dataset" in output

    if debug:
        print("bland_altman_identifier passed")


def bland_altman_no_identifier(debug=False):
    """Test bland_altman_plot without identifier."""
    data1 = np.array([1.0, 2.0, 3.0])
    data2 = np.array([1.0, 2.0, 3.0])

    captured_stdout = io.StringIO()
    with (
        patch("rapidtide.workflows.showxy.scatter"),
        patch("rapidtide.workflows.showxy.axhline"),
        patch("rapidtide.workflows.showxy.annotate"),
        patch("rapidtide.workflows.showxy.xlim"),
        patch("rapidtide.workflows.showxy.ylim"),
        patch("sys.stdout", captured_stdout),
    ):
        bland_altman_plot(data1, data2, identifier=None)

    output = captured_stdout.getvalue()
    assert "id:" not in output

    if debug:
        print("bland_altman_no_identifier passed")


def bland_altman_annotate(debug=False):
    """Test bland_altman_plot calls annotate when doannotate=True."""
    data1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    data2 = np.array([1.1, 1.9, 3.1, 3.9, 5.1])

    with (
        patch("rapidtide.workflows.showxy.scatter"),
        patch("rapidtide.workflows.showxy.axhline"),
        patch("rapidtide.workflows.showxy.annotate") as mock_ann,
        patch("rapidtide.workflows.showxy.xlim"),
        patch("rapidtide.workflows.showxy.ylim"),
    ):
        bland_altman_plot(data1, data2, doannotate=True)
        assert mock_ann.called, "annotate should be called when doannotate=True"

    if debug:
        print("bland_altman_annotate passed")


def bland_altman_no_annotate(debug=False):
    """Test bland_altman_plot skips annotate when doannotate=False."""
    data1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    data2 = np.array([1.1, 1.9, 3.1, 3.9, 5.1])

    with (
        patch("rapidtide.workflows.showxy.scatter"),
        patch("rapidtide.workflows.showxy.axhline"),
        patch("rapidtide.workflows.showxy.annotate") as mock_ann,
        patch("rapidtide.workflows.showxy.xlim"),
        patch("rapidtide.workflows.showxy.ylim"),
    ):
        bland_altman_plot(data1, data2, doannotate=False)
        assert not mock_ann.called, "annotate should not be called when doannotate=False"

    if debug:
        print("bland_altman_no_annotate passed")


def bland_altman_axhlines(debug=False):
    """Test bland_altman_plot draws three horizontal lines (mean, mean+2sd, mean-2sd)."""
    data1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    data2 = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

    with (
        patch("rapidtide.workflows.showxy.scatter"),
        patch("rapidtide.workflows.showxy.axhline") as mock_axhline,
        patch("rapidtide.workflows.showxy.annotate"),
        patch("rapidtide.workflows.showxy.xlim"),
        patch("rapidtide.workflows.showxy.ylim"),
    ):
        bland_altman_plot(data1, data2)
        assert mock_axhline.call_count == 3

    if debug:
        print("bland_altman_axhlines passed")


def bland_altman_xrange_yrange(debug=False):
    """Test bland_altman_plot applies xrange and yrange."""
    data1 = np.array([1.0, 2.0, 3.0])
    data2 = np.array([1.0, 2.0, 3.0])

    with (
        patch("rapidtide.workflows.showxy.scatter"),
        patch("rapidtide.workflows.showxy.axhline"),
        patch("rapidtide.workflows.showxy.annotate"),
        patch("rapidtide.workflows.showxy.xlim") as mock_xlim,
        patch("rapidtide.workflows.showxy.ylim") as mock_ylim,
    ):
        bland_altman_plot(data1, data2, xrange=[0, 5], yrange=[-2, 2])
        mock_xlim.assert_called_once_with([0, 5])
        mock_ylim.assert_called_once_with([-2, 2])

    if debug:
        print("bland_altman_xrange_yrange passed")


def bland_altman_no_range(debug=False):
    """Test bland_altman_plot does not set xlim/ylim when ranges are None."""
    data1 = np.array([1.0, 2.0, 3.0])
    data2 = np.array([1.0, 2.0, 3.0])

    with (
        patch("rapidtide.workflows.showxy.scatter"),
        patch("rapidtide.workflows.showxy.axhline"),
        patch("rapidtide.workflows.showxy.annotate"),
        patch("rapidtide.workflows.showxy.xlim") as mock_xlim,
        patch("rapidtide.workflows.showxy.ylim") as mock_ylim,
    ):
        bland_altman_plot(data1, data2, xrange=None, yrange=None)
        assert not mock_xlim.called
        assert not mock_ylim.called

    if debug:
        print("bland_altman_no_range passed")


def bland_altman_statistics(debug=False):
    """Test bland_altman_plot computes correct statistics for known data."""
    data1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    data2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # identical data

    captured_stdout = io.StringIO()
    with (
        patch("rapidtide.workflows.showxy.scatter"),
        patch("rapidtide.workflows.showxy.axhline"),
        patch("rapidtide.workflows.showxy.annotate"),
        patch("rapidtide.workflows.showxy.xlim"),
        patch("rapidtide.workflows.showxy.ylim"),
        patch("sys.stdout", captured_stdout),
    ):
        bland_altman_plot(data1, data2)

    output = captured_stdout.getvalue()
    # for identical data: slope=1.0, mean diff=0.0, std diff=0.0
    assert "slope: 1.0" in output
    assert "mean difference: 0.0" in output
    assert "std difference: 0.0" in output

    if debug:
        print("bland_altman_statistics passed")


def bland_altman_fontscalefac(debug=False):
    """Test bland_altman_plot passes fontscalefac to annotate."""
    data1 = np.array([1.0, 2.0, 3.0])
    data2 = np.array([1.5, 2.5, 3.5])

    with (
        patch("rapidtide.workflows.showxy.scatter"),
        patch("rapidtide.workflows.showxy.axhline"),
        patch("rapidtide.workflows.showxy.annotate") as mock_ann,
        patch("rapidtide.workflows.showxy.xlim"),
        patch("rapidtide.workflows.showxy.ylim"),
    ):
        bland_altman_plot(data1, data2, fontscalefac=2.0, doannotate=True)
        assert mock_ann.called
        call_kwargs = mock_ann.call_args[1]
        assert call_kwargs["fontsize"] == 2.0 * 16

    if debug:
        print("bland_altman_fontscalefac passed")


# ---- showxy tests ----


def showxy_basic_line_plot(debug=False):
    """Test showxy with basic line plot (default mode)."""
    file_data = _make_xy_data(nfiles=1)
    args = _make_default_args(textfilenames=["data1.txt"])

    captured = _run_showxy_with_mocks(args, file_data)

    assert len(captured["plot_calls"]) == 1
    assert captured["show_calls"] == 1
    assert len(captured["savefig_calls"]) == 0

    if debug:
        print("showxy_basic_line_plot passed")


def showxy_multiple_files(debug=False):
    """Test showxy with multiple input files."""
    file_data = _make_xy_data(nfiles=3)
    args = _make_default_args(
        textfilenames=["data1.txt", "data2.txt", "data3.txt"],
    )

    captured = _run_showxy_with_mocks(args, file_data)

    assert len(captured["plot_calls"]) == 3

    if debug:
        print("showxy_multiple_files passed")


def showxy_save_to_file(debug=False):
    """Test showxy saves to file instead of showing."""
    file_data = _make_xy_data(nfiles=1)
    args = _make_default_args(
        textfilenames=["data1.txt"],
        outputfile="output.png",
        saveres=300,
    )

    captured = _run_showxy_with_mocks(args, file_data)

    assert captured["show_calls"] == 0
    assert len(captured["savefig_calls"]) == 1
    savefig_args, savefig_kwargs = captured["savefig_calls"][0]
    assert savefig_args[0] == "output.png"
    assert savefig_kwargs["dpi"] == 300

    if debug:
        print("showxy_save_to_file passed")


def showxy_with_title(debug=False):
    """Test showxy sets title when provided."""
    file_data = _make_xy_data(nfiles=1)
    args = _make_default_args(
        textfilenames=["data1.txt"],
        thetitle="Test Title",
    )

    captured = _run_showxy_with_mocks(args, file_data)

    assert len(captured["title_calls"]) == 1
    assert captured["title_calls"][0][0][0] == "Test Title"

    if debug:
        print("showxy_with_title passed")


def showxy_no_title(debug=False):
    """Test showxy does not set title when None."""
    file_data = _make_xy_data(nfiles=1)
    args = _make_default_args(textfilenames=["data1.txt"], thetitle=None)

    captured = _run_showxy_with_mocks(args, file_data)

    assert len(captured["title_calls"]) == 0

    if debug:
        print("showxy_no_title passed")


def showxy_with_labels(debug=False):
    """Test showxy sets axis labels when provided."""
    file_data = _make_xy_data(nfiles=1)
    args = _make_default_args(
        textfilenames=["data1.txt"],
        thexlabel="X Axis",
        theylabel="Y Axis",
    )

    captured = _run_showxy_with_mocks(args, file_data)

    assert len(captured["xlabel_calls"]) == 1
    assert captured["xlabel_calls"][0][0][0] == "X Axis"
    assert len(captured["ylabel_calls"]) == 1
    assert captured["ylabel_calls"][0][0][0] == "Y Axis"

    if debug:
        print("showxy_with_labels passed")


def showxy_no_labels(debug=False):
    """Test showxy does not set labels when None."""
    file_data = _make_xy_data(nfiles=1)
    args = _make_default_args(textfilenames=["data1.txt"])

    captured = _run_showxy_with_mocks(args, file_data)

    assert len(captured["xlabel_calls"]) == 0
    assert len(captured["ylabel_calls"]) == 0

    if debug:
        print("showxy_no_labels passed")


def showxy_with_legends(debug=False):
    """Test showxy sets legend when provided."""
    file_data = _make_xy_data(nfiles=2)
    args = _make_default_args(
        textfilenames=["data1.txt", "data2.txt"],
        legends="Series A,Series B",
        legendloc=5,
    )

    captured = _run_showxy_with_mocks(args, file_data)

    assert len(captured["legend_calls"]) == 1
    legend_args = captured["legend_calls"][0][0][0]
    assert legend_args == ["Series A", "Series B"]
    legend_kwargs = captured["legend_calls"][0][1]
    assert legend_kwargs["loc"] == 5

    if debug:
        print("showxy_with_legends passed")


def showxy_no_legends(debug=False):
    """Test showxy does not set legend when None."""
    file_data = _make_xy_data(nfiles=1)
    args = _make_default_args(textfilenames=["data1.txt"], legends=None)

    captured = _run_showxy_with_mocks(args, file_data)

    assert len(captured["legend_calls"]) == 0

    if debug:
        print("showxy_no_legends passed")


def showxy_custom_colors(debug=False):
    """Test showxy uses specified colors."""
    file_data = _make_xy_data(nfiles=2)
    args = _make_default_args(
        textfilenames=["data1.txt", "data2.txt"],
        colors="red,blue",
    )

    captured = _run_showxy_with_mocks(args, file_data)

    assert len(captured["plot_calls"]) == 2
    assert captured["plot_calls"][0][1]["color"] == "red"
    assert captured["plot_calls"][1][1]["color"] == "blue"

    if debug:
        print("showxy_custom_colors passed")


def showxy_color_cycling(debug=False):
    """Test showxy cycles colors when fewer colors than files."""
    file_data = _make_xy_data(nfiles=3)
    args = _make_default_args(
        textfilenames=["data1.txt", "data2.txt", "data3.txt"],
        colors="red,blue",
    )

    captured = _run_showxy_with_mocks(args, file_data)

    assert len(captured["plot_calls"]) == 3
    assert captured["plot_calls"][0][1]["color"] == "red"
    assert captured["plot_calls"][1][1]["color"] == "blue"
    assert captured["plot_calls"][2][1]["color"] == "red"  # cycles back

    if debug:
        print("showxy_color_cycling passed")


def showxy_default_colormap(debug=False):
    """Test showxy uses nipy_spectral colormap when no colors specified."""
    file_data = _make_xy_data(nfiles=2)
    args = _make_default_args(
        textfilenames=["data1.txt", "data2.txt"],
        colors=None,
    )

    captured = _run_showxy_with_mocks(args, file_data)

    assert len(captured["plot_calls"]) == 2
    # colors should be tuples (RGBA) from colormap, not strings
    for call_args, call_kwargs in captured["plot_calls"]:
        color = call_kwargs["color"]
        assert isinstance(color, tuple) and len(color) == 4

    if debug:
        print("showxy_default_colormap passed")


def showxy_usepoints(debug=False):
    """Test showxy uses scatter-style points with usepoints=True."""
    file_data = _make_xy_data(nfiles=1)
    args = _make_default_args(textfilenames=["data1.txt"], usepoints=True)

    captured = _run_showxy_with_mocks(args, file_data)

    assert len(captured["plot_calls"]) == 1
    call_kwargs = captured["plot_calls"][0][1]
    assert call_kwargs["marker"] == "."
    assert call_kwargs["linestyle"] == "None"

    if debug:
        print("showxy_usepoints passed")


def showxy_dobars(debug=False):
    """Test showxy creates bar plots with dobars=True."""
    file_data = _make_xy_data(nfiles=1)
    args = _make_default_args(textfilenames=["data1.txt"], dobars=True)

    captured = _run_showxy_with_mocks(args, file_data)

    assert len(captured["bar_calls"]) == 1
    assert len(captured["plot_calls"]) == 0

    if debug:
        print("showxy_dobars passed")


def showxy_blandaltman(debug=False):
    """Test showxy creates Bland-Altman plot with blandaltman=True."""
    file_data = _make_xy_data(nfiles=1)
    args = _make_default_args(textfilenames=["data1.txt"], blandaltman=True)

    captured = _run_showxy_with_mocks(args, file_data)

    # bland_altman_plot calls scatter, not plot
    assert len(captured["scatter_calls"]) == 1
    assert len(captured["plot_calls"]) == 0
    # should draw 3 axhlines
    assert len(captured["axhline_calls"]) == 3
    # should annotate by default
    assert len(captured["annotate_calls"]) == 1

    if debug:
        print("showxy_blandaltman passed")


def showxy_blandaltman_noannotate(debug=False):
    """Test showxy Bland-Altman with doannotate=False."""
    file_data = _make_xy_data(nfiles=1)
    args = _make_default_args(
        textfilenames=["data1.txt"],
        blandaltman=True,
        doannotate=False,
    )

    captured = _run_showxy_with_mocks(args, file_data)

    assert len(captured["annotate_calls"]) == 0

    if debug:
        print("showxy_blandaltman_noannotate passed")


def showxy_blandaltman_usex(debug=False):
    """Test showxy Bland-Altman with usex=True."""
    file_data = _make_xy_data(nfiles=1)
    args = _make_default_args(
        textfilenames=["data1.txt"],
        blandaltman=True,
        usex=True,
    )

    captured = _run_showxy_with_mocks(args, file_data)

    assert len(captured["scatter_calls"]) == 1

    if debug:
        print("showxy_blandaltman_usex passed")


def showxy_xrange_yrange(debug=False):
    """Test showxy applies xrange and yrange to line plots."""
    file_data = _make_xy_data(nfiles=1)
    args = _make_default_args(
        textfilenames=["data1.txt"],
        xrange=[0.0, 5.0],
        yrange=[-1.0, 10.0],
    )

    captured = _run_showxy_with_mocks(args, file_data)

    assert len(captured["xlim_calls"]) == 1
    assert len(captured["ylim_calls"]) == 1

    if debug:
        print("showxy_xrange_yrange passed")


def showxy_bars_xrange_yrange(debug=False):
    """Test showxy applies xrange and yrange to bar plots."""
    file_data = _make_xy_data(nfiles=1)
    args = _make_default_args(
        textfilenames=["data1.txt"],
        dobars=True,
        xrange=[0.0, 5.0],
        yrange=[-1.0, 10.0],
    )

    captured = _run_showxy_with_mocks(args, file_data)

    assert len(captured["xlim_calls"]) == 1
    assert len(captured["ylim_calls"]) == 1

    if debug:
        print("showxy_bars_xrange_yrange passed")


def showxy_debug_output(debug=False):
    """Test showxy prints debug info when debug=True."""
    file_data = _make_xy_data(nfiles=1)
    args = _make_default_args(textfilenames=["data1.txt"], debug=True)

    captured_stdout = io.StringIO()
    with patch("sys.stdout", captured_stdout):
        _run_showxy_with_mocks(args, file_data)

    output = captured_stdout.getvalue()
    assert "reading data from" in output

    if debug:
        print("showxy_debug_output passed")


def showxy_fontscalefac_title(debug=False):
    """Test showxy applies fontscalefac to title."""
    file_data = _make_xy_data(nfiles=1)
    args = _make_default_args(
        textfilenames=["data1.txt"],
        thetitle="Scaled Title",
        fontscalefac=2.0,
    )

    captured = _run_showxy_with_mocks(args, file_data)

    assert len(captured["title_calls"]) == 1
    call_kwargs = captured["title_calls"][0][1]
    assert call_kwargs["fontsize"] == 2.0 * 18

    if debug:
        print("showxy_fontscalefac_title passed")


def showxy_fontscalefac_labels(debug=False):
    """Test showxy applies fontscalefac to axis labels."""
    file_data = _make_xy_data(nfiles=1)
    args = _make_default_args(
        textfilenames=["data1.txt"],
        thexlabel="X",
        theylabel="Y",
        fontscalefac=1.5,
    )

    captured = _run_showxy_with_mocks(args, file_data)

    xlabel_kwargs = captured["xlabel_calls"][0][1]
    ylabel_kwargs = captured["ylabel_calls"][0][1]
    assert xlabel_kwargs["fontsize"] == 1.5 * 16
    assert ylabel_kwargs["fontsize"] == 1.5 * 16

    if debug:
        print("showxy_fontscalefac_labels passed")


# ---- main test function ----


def test_showxy(debug=False):
    # parser tests
    parser_basic(debug=debug)
    parser_required_args(debug=debug)
    parser_single_file(debug=debug)
    parser_multiple_files(debug=debug)
    parser_defaults(debug=debug)
    parser_xlabel_ylabel(debug=debug)
    parser_xrange_yrange(debug=debug)
    parser_title(debug=debug)
    parser_outputfile(debug=debug)
    parser_fontscalefac(debug=debug)
    parser_saveres(debug=debug)
    parser_legends(debug=debug)
    parser_legendloc(debug=debug)
    parser_colors(debug=debug)
    parser_boolean_flags(debug=debug)

    # stringtorange tests
    stringtorange_basic(debug=debug)
    stringtorange_negative(debug=debug)
    stringtorange_zero(debug=debug)
    stringtorange_integers(debug=debug)
    stringtorange_scientific(debug=debug)
    stringtorange_too_few(debug=debug)
    stringtorange_too_many(debug=debug)
    stringtorange_invalid_first(debug=debug)
    stringtorange_invalid_second(debug=debug)
    stringtorange_empty(debug=debug)

    # bland_altman_plot tests
    bland_altman_basic(debug=debug)
    bland_altman_usex(debug=debug)
    bland_altman_default_mean_axis(debug=debug)
    bland_altman_identifier(debug=debug)
    bland_altman_no_identifier(debug=debug)
    bland_altman_annotate(debug=debug)
    bland_altman_no_annotate(debug=debug)
    bland_altman_axhlines(debug=debug)
    bland_altman_xrange_yrange(debug=debug)
    bland_altman_no_range(debug=debug)
    bland_altman_statistics(debug=debug)
    bland_altman_fontscalefac(debug=debug)

    # showxy workflow tests
    showxy_basic_line_plot(debug=debug)
    showxy_multiple_files(debug=debug)
    showxy_save_to_file(debug=debug)
    showxy_with_title(debug=debug)
    showxy_no_title(debug=debug)
    showxy_with_labels(debug=debug)
    showxy_no_labels(debug=debug)
    showxy_with_legends(debug=debug)
    showxy_no_legends(debug=debug)
    showxy_custom_colors(debug=debug)
    showxy_color_cycling(debug=debug)
    showxy_default_colormap(debug=debug)
    showxy_usepoints(debug=debug)
    showxy_dobars(debug=debug)
    showxy_blandaltman(debug=debug)
    showxy_blandaltman_noannotate(debug=debug)
    showxy_blandaltman_usex(debug=debug)
    showxy_xrange_yrange(debug=debug)
    showxy_bars_xrange_yrange(debug=debug)
    showxy_debug_output(debug=debug)
    showxy_fontscalefac_title(debug=debug)
    showxy_fontscalefac_labels(debug=debug)


if __name__ == "__main__":
    test_showxy(debug=True)
