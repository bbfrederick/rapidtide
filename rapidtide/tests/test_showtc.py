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
import os
import tempfile
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import rapidtide.io as tide_io
import rapidtide.workflows.showtc as showtc_module
from rapidtide.workflows.showtc import _get_parser, phase, showtc

# ==================== Helpers ====================


def _make_timecourses(numcolumns=2, numpoints=64, samplerate=1.0):
    """Build a deterministic multicolumn timecourse array.

    Parameters
    ----------
    numcolumns : int, optional
        Number of timecourses to generate.  Default is 2.
    numpoints : int, optional
        Number of timepoints per timecourse.  Default is 64.
    samplerate : float, optional
        Sample rate used to build the time axis.  Default is 1.0.

    Returns
    -------
    ndarray
        Array of shape (numcolumns, numpoints).
    """
    rng = np.random.RandomState(42)
    t = np.arange(numpoints, dtype=float) / samplerate
    data = np.zeros((numcolumns, numpoints), dtype=float)
    for col in range(numcolumns):
        data[col, :] = np.sin(2.0 * np.pi * 0.05 * (col + 1) * t) + 0.1 * rng.randn(numpoints)
    return data


def _write_plaintext(tmpdir, name, data):
    """Write a whitespace separated text file with one column per timecourse.

    Parameters
    ----------
    tmpdir : str
        Directory to write into.
    name : str
        Base file name.
    data : ndarray
        Array of shape (numcolumns, numpoints).

    Returns
    -------
    str
        Full path to the written file.
    """
    thepath = os.path.join(tmpdir, name)
    np.savetxt(thepath, np.transpose(data))
    return thepath


def _write_bids(tmpdir, name, data, samplerate=2.0, starttime=0.0, columns=None):
    """Write a BIDS continuous tsv plus json sidecar.

    Parameters
    ----------
    tmpdir : str
        Directory to write into.
    name : str
        File root (no extension).
    data : ndarray
        Array of shape (numcolumns, numpoints).
    samplerate : float, optional
        Sample rate to record in the sidecar.  Default is 2.0.
    starttime : float, optional
        Start time to record in the sidecar.  Default is 0.0.
    columns : list of str, optional
        Column names.  Default is None, which generates "col0", "col1", ...

    Returns
    -------
    str
        Full path to the written .tsv.gz file.
    """
    if columns is None:
        columns = [f"col{i}" for i in range(data.shape[0])]
    theroot = os.path.join(tmpdir, name)
    tide_io.writebidstsv(
        theroot,
        data,
        samplerate,
        columns=columns,
        starttime=starttime,
        compressed=True,
    )
    return theroot + ".tsv.gz"


def _run_showtc(tmpdir, arglist, outname="plot.png"):
    """Parse an argument list and run showtc, saving to a file rather than displaying.

    Parameters
    ----------
    tmpdir : str
        Directory to write the output image into.
    arglist : list of str
        Command line arguments, not including the output file specification.
    outname : str, optional
        Name of the output image file.  Default is "plot.png".

    Returns
    -------
    tuple of (str, argparse.Namespace)
        The path to the written image and the (showtc-mutated) argument namespace.
    """
    outputfile = os.path.join(tmpdir, outname)
    args = _get_parser().parse_args(arglist + ["--tofile", outputfile])
    try:
        showtc(args)
    finally:
        plt.close("all")
    return outputfile, args


def _assert_wrote_image(thepath):
    """Assert that a plot file was created and is not empty.

    Parameters
    ----------
    thepath : str
        Path that showtc was asked to write.

    Returns
    -------
    None
    """
    assert os.path.exists(thepath), f"{thepath} was not created"
    assert os.path.getsize(thepath) > 0, f"{thepath} is empty"


# ==================== phase tests ====================


def phase_cardinal_values(debug=False):
    """Test phase returns the expected angles for the four cardinal complex directions."""
    if debug:
        print("phase_cardinal_values")
    z = np.array([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j])
    expected = np.array([0.0, np.pi / 2.0, np.pi, -np.pi / 2.0])
    np.testing.assert_allclose(phase(z), expected, atol=1e-12)


def phase_diagonal_values(debug=False):
    """Test phase returns the expected angles in all four quadrants."""
    if debug:
        print("phase_diagonal_values")
    z = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j])
    expected = np.array([np.pi / 4.0, 3.0 * np.pi / 4.0, -3.0 * np.pi / 4.0, -np.pi / 4.0])
    np.testing.assert_allclose(phase(z), expected, atol=1e-12)


def phase_range_is_bounded(debug=False):
    """Test phase always returns values in [-pi, pi]."""
    if debug:
        print("phase_range_is_bounded")
    rng = np.random.RandomState(11)
    z = rng.randn(500) + 1j * rng.randn(500)
    result = phase(z)
    assert np.all(result >= -np.pi)
    assert np.all(result <= np.pi)


def phase_preserves_shape(debug=False):
    """Test phase preserves the shape of multidimensional input."""
    if debug:
        print("phase_preserves_shape")
    rng = np.random.RandomState(12)
    z = rng.randn(3, 4, 5) + 1j * rng.randn(3, 4, 5)
    assert phase(z).shape == (3, 4, 5)


def phase_origin_is_zero(debug=False):
    """Test phase of zero is zero (arctan2 handles the degenerate case without dividing)."""
    if debug:
        print("phase_origin_is_zero")
    assert phase(np.array([0 + 0j]))[0] == 0.0


# ==================== _get_parser tests ====================


def get_parser_returns_parser(debug=False):
    """Test _get_parser returns an ArgumentParser."""
    if debug:
        print("get_parser_returns_parser")
    assert isinstance(_get_parser(), argparse.ArgumentParser)


def get_parser_requires_a_filename(debug=False):
    """Test _get_parser requires at least one input file."""
    if debug:
        print("get_parser_requires_a_filename")
    try:
        _get_parser().parse_args([])
        assert False, "Should have raised SystemExit"
    except SystemExit:
        pass


def get_parser_accepts_multiple_filenames(debug=False):
    """Test _get_parser accepts a variable number of input files."""
    if debug:
        print("get_parser_accepts_multiple_filenames")
    args = _get_parser().parse_args(["a.txt", "b.txt", "c.txt"])
    assert args.textfilenames == ["a.txt", "b.txt", "c.txt"]


def get_parser_defaults(debug=False):
    """Test _get_parser default values."""
    if debug:
        print("get_parser_defaults")
    args = _get_parser().parse_args(["a.txt"])
    assert args.samplerate == "auto"
    assert args.displaymode == "time"
    assert args.plotformat == "overlaid"
    assert not args.dowaterfall
    assert args.voffset == 0.0
    assert not args.dotranspose
    assert not args.normall
    assert args.thestarttime is None
    assert args.theendtime is None
    assert not args.forcezerostart
    assert args.numskip == 0
    assert not args.fullxrange
    assert not args.debug
    # options contributed by pf.addplotopts
    assert args.thetitle is None
    assert args.xlabel is None
    assert args.ylabel is None
    assert args.legends is None
    assert args.legendloc == 2
    assert args.colors is None
    assert args.dolegend
    assert args.showxax
    assert args.showyax
    assert args.linewidths is None
    assert args.outputfile is None
    assert args.fontscalefac == 1.0
    assert args.saveres == 1000


def get_parser_samplerate(debug=False):
    """Test --samplerate is stored directly."""
    if debug:
        print("get_parser_samplerate")
    args = _get_parser().parse_args(["a.txt", "--samplerate", "4.0"])
    assert args.samplerate == 4.0


def get_parser_sampletime_is_inverted(debug=False):
    """Test --sampletime is inverted into a sample rate."""
    if debug:
        print("get_parser_sampletime_is_inverted")
    args = _get_parser().parse_args(["a.txt", "--sampletime", "0.25"])
    assert abs(args.samplerate - 4.0) < 1e-12


def get_parser_sampling_is_mutually_exclusive(debug=False):
    """Test --samplerate and --sampletime cannot both be given."""
    if debug:
        print("get_parser_sampling_is_mutually_exclusive")
    try:
        _get_parser().parse_args(["a.txt", "--samplerate", "4.0", "--sampletime", "0.25"])
        assert False, "Should have raised SystemExit"
    except SystemExit:
        pass


def get_parser_displaytype_choices(debug=False):
    """Test --displaytype accepts each valid choice."""
    if debug:
        print("get_parser_displaytype_choices")
    for choice in ["time", "power", "phase"]:
        args = _get_parser().parse_args(["a.txt", "--displaytype", choice])
        assert args.displaymode == choice


def get_parser_displaytype_rejects_bad_choice(debug=False):
    """Test --displaytype rejects an unknown choice."""
    if debug:
        print("get_parser_displaytype_rejects_bad_choice")
    try:
        _get_parser().parse_args(["a.txt", "--displaytype", "bogus"])
        assert False, "Should have raised SystemExit"
    except SystemExit:
        pass


def get_parser_format_choices(debug=False):
    """Test --format accepts each valid choice."""
    if debug:
        print("get_parser_format_choices")
    for choice in ["overlaid", "separate", "separatelinked"]:
        args = _get_parser().parse_args(["a.txt", "--format", choice])
        assert args.plotformat == choice


def get_parser_format_rejects_bad_choice(debug=False):
    """Test --format rejects an unknown choice."""
    if debug:
        print("get_parser_format_rejects_bad_choice")
    try:
        _get_parser().parse_args(["a.txt", "--format", "bogus"])
        assert False, "Should have raised SystemExit"
    except SystemExit:
        pass


def get_parser_boolean_flags(debug=False):
    """Test each store_true/store_false flag flips its destination."""
    if debug:
        print("get_parser_boolean_flags")
    for flag, dest, expected in [
        ("--waterfall", "dowaterfall", True),
        ("--transpose", "dotranspose", True),
        ("--normall", "normall", True),
        ("--forcezerostart", "forcezerostart", True),
        ("--fullxrange", "fullxrange", True),
        ("--debug", "debug", True),
        ("--nolegend", "dolegend", False),
        ("--noxax", "showxax", False),
        ("--noyax", "showyax", False),
    ]:
        args = _get_parser().parse_args(["a.txt", flag])
        assert getattr(args, dest) is expected, f"{flag} did not set {dest} to {expected}"


def get_parser_numeric_options(debug=False):
    """Test the numeric valued options are parsed with the right types."""
    if debug:
        print("get_parser_numeric_options")
    args = _get_parser().parse_args(
        [
            "a.txt",
            "--voffset",
            "-1.5",
            "--starttime",
            "2.5",
            "--endtime",
            "7.5",
            "--numskip",
            "3",
            "--legendloc",
            "4",
            "--fontscalefac",
            "1.5",
            "--saveres",
            "150",
        ]
    )
    assert args.voffset == -1.5
    assert args.thestarttime == 2.5
    assert args.theendtime == 7.5
    assert args.numskip == 3
    assert args.legendloc == 4
    assert args.fontscalefac == 1.5
    assert args.saveres == 150


def get_parser_string_options(debug=False):
    """Test the string valued plot appearance options are stored unparsed."""
    if debug:
        print("get_parser_string_options")
    args = _get_parser().parse_args(
        [
            "a.txt",
            "--title",
            "The Title",
            "--xlabel",
            "X",
            "--ylabel",
            "Y",
            "--legends",
            "one,two",
            "--colors",
            "red,blue",
            "--linewidth",
            "1.0,2.5",
            "--tofile",
            "out.png",
        ]
    )
    assert args.thetitle == "The Title"
    assert args.xlabel == "X"
    assert args.ylabel == "Y"
    assert args.legends == "one,two"
    assert args.colors == "red,blue"
    assert args.linewidths == "1.0,2.5"
    assert args.outputfile == "out.png"


def get_parser_rejects_abbreviations(debug=False):
    """Test the parser was built with allow_abbrev=False."""
    if debug:
        print("get_parser_rejects_abbreviations")
    try:
        _get_parser().parse_args(["a.txt", "--water"])
        assert False, "Should have raised SystemExit"
    except SystemExit:
        pass


# ==================== showtc display mode tests ====================


def showtc_time_domain(debug=False):
    """Test showtc plots a simple two column time series."""
    if debug:
        print("showtc_time_domain")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_plaintext(tmpdir, "data.txt", _make_timecourses())
        outputfile, args = _run_showtc(tmpdir, [thefile])
        _assert_wrote_image(outputfile)
        # in time mode the x label defaults to time
        assert args.xlabel == "Time (s)"


def showtc_power_spectrum(debug=False):
    """Test showtc plots a power spectrum and labels the axes accordingly."""
    if debug:
        print("showtc_power_spectrum")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_plaintext(tmpdir, "data.txt", _make_timecourses())
        outputfile, args = _run_showtc(tmpdir, [thefile, "--displaytype", "power"])
        _assert_wrote_image(outputfile)
        assert args.xlabel == "Frequency (Hz)"
        assert args.ylabel == "Signal power"


def showtc_phase_spectrum(debug=False):
    """Test showtc plots a phase spectrum and labels the axes accordingly."""
    if debug:
        print("showtc_phase_spectrum")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_plaintext(tmpdir, "data.txt", _make_timecourses())
        outputfile, args = _run_showtc(tmpdir, [thefile, "--displaytype", "phase"])
        _assert_wrote_image(outputfile)
        assert args.xlabel == "Frequency (Hz)"
        assert args.ylabel == "Signal phase"


def showtc_spectrum_of_odd_length_data(debug=False):
    """Test showtc drops the final point of odd length data before computing a spectrum."""
    if debug:
        print("showtc_spectrum_of_odd_length_data")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_plaintext(tmpdir, "odd.txt", _make_timecourses(numpoints=65))
        outputfile, _ = _run_showtc(tmpdir, [thefile, "--displaytype", "power"])
        _assert_wrote_image(outputfile)


def showtc_illegal_display_mode(debug=False):
    """Test showtc exits when handed a display mode the parser would have rejected."""
    if debug:
        print("showtc_illegal_display_mode")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_plaintext(tmpdir, "data.txt", _make_timecourses())
        args = _get_parser().parse_args([thefile])
        args.displaymode = "bogus"
        try:
            showtc(args)
            assert False, "Should have raised SystemExit"
        except SystemExit:
            pass
        finally:
            plt.close("all")


# ==================== showtc plot format tests ====================


def showtc_overlaid_format(debug=False):
    """Test showtc renders multiple timecourses overlaid in one axis."""
    if debug:
        print("showtc_overlaid_format")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_plaintext(tmpdir, "data.txt", _make_timecourses(numcolumns=3))
        outputfile, _ = _run_showtc(tmpdir, [thefile, "--format", "overlaid"])
        _assert_wrote_image(outputfile)


def showtc_separate_format(debug=False):
    """Test showtc renders each timecourse in its own independently scaled axis."""
    if debug:
        print("showtc_separate_format")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_plaintext(tmpdir, "data.txt", _make_timecourses(numcolumns=3))
        outputfile, _ = _run_showtc(tmpdir, [thefile, "--format", "separate"])
        _assert_wrote_image(outputfile)


def showtc_separatelinked_format(debug=False):
    """Test showtc renders each timecourse in its own axis with linked y scaling."""
    if debug:
        print("showtc_separatelinked_format")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_plaintext(tmpdir, "data.txt", _make_timecourses(numcolumns=3))
        outputfile, _ = _run_showtc(tmpdir, [thefile, "--format", "separatelinked"])
        _assert_wrote_image(outputfile)


def showtc_separate_format_single_vector(debug=False):
    """Test showtc handles the single subplot case, where fig.subplots returns a bare axis."""
    if debug:
        print("showtc_separate_format_single_vector")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_plaintext(tmpdir, "one.txt", _make_timecourses(numcolumns=1))
        outputfile, _ = _run_showtc(tmpdir, [thefile, "--format", "separate"])
        _assert_wrote_image(outputfile)


def showtc_illegal_plot_format(debug=False):
    """Test showtc exits when handed a plot format the parser would have rejected."""
    if debug:
        print("showtc_illegal_plot_format")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_plaintext(tmpdir, "data.txt", _make_timecourses())
        args = _get_parser().parse_args([thefile])
        args.plotformat = "bogus"
        try:
            showtc(args)
            assert False, "Should have raised SystemExit"
        except SystemExit:
            pass
        finally:
            plt.close("all")


# ==================== showtc input handling tests ====================


def showtc_multiple_files(debug=False):
    """Test showtc concatenates timecourses drawn from several files."""
    if debug:
        print("showtc_multiple_files")
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = _write_plaintext(tmpdir, "one.txt", _make_timecourses(numcolumns=2))
        file2 = _write_plaintext(tmpdir, "two.txt", _make_timecourses(numcolumns=1))
        outputfile, _ = _run_showtc(tmpdir, [file1, file2])
        _assert_wrote_image(outputfile)


def showtc_transpose(debug=False):
    """Test --transpose swaps the row and column interpretation of the input."""
    if debug:
        print("showtc_transpose")
    with tempfile.TemporaryDirectory() as tmpdir:
        # 3 columns of 40 points becomes 40 "columns" of 3 points when transposed
        thefile = _write_plaintext(
            tmpdir, "data.txt", _make_timecourses(numcolumns=3, numpoints=8)
        )
        outputfile, _ = _run_showtc(tmpdir, [thefile, "--transpose"])
        _assert_wrote_image(outputfile)


def showtc_explicit_samplerate(debug=False):
    """Test an explicit --samplerate sets the time axis spacing for a plain text file."""
    if debug:
        print("showtc_explicit_samplerate")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_plaintext(tmpdir, "data.txt", _make_timecourses(numpoints=32))
        outputfile, args = _run_showtc(tmpdir, [thefile, "--samplerate", "4.0"])
        _assert_wrote_image(outputfile)
        assert args.samplerate == 4.0
        # 32 points at 4 Hz spans (32 - 1) / 4 = 7.75 seconds
        assert abs(args.theendtime - 7.75) < 1e-9


def showtc_auto_samplerate_defaults_to_one(debug=False):
    """Test an unspecified sample rate becomes 1.0 for a file carrying no rate metadata."""
    if debug:
        print("showtc_auto_samplerate_defaults_to_one")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_plaintext(tmpdir, "data.txt", _make_timecourses(numpoints=16))
        outputfile, args = _run_showtc(tmpdir, [thefile])
        _assert_wrote_image(outputfile)
        assert args.samplerate == 1.0
        assert abs(args.theendtime - 15.0) < 1e-9


def showtc_bids_input_uses_embedded_metadata(debug=False):
    """Test showtc picks up sample rate and start time from a BIDS json sidecar."""
    if debug:
        print("showtc_bids_input_uses_embedded_metadata")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_bids(
            tmpdir,
            "sub-01_task-rest_physio",
            _make_timecourses(numcolumns=2, numpoints=32),
            samplerate=2.0,
            starttime=5.0,
        )
        outputfile, args = _run_showtc(tmpdir, [thefile])
        _assert_wrote_image(outputfile)
        # the plot should start at the offset recorded in the sidecar
        assert abs(args.thestarttime - 5.0) < 1e-9
        # 32 points at 2 Hz starting at 5.0 ends at 5.0 + 31/2
        assert abs(args.theendtime - 20.5) < 1e-9


def showtc_bids_column_selection(debug=False):
    """Test a column specification appended to a BIDS filename selects a single timecourse."""
    if debug:
        print("showtc_bids_column_selection")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_bids(
            tmpdir,
            "sub-01_task-rest_physio",
            _make_timecourses(numcolumns=3, numpoints=32),
            columns=["alpha", "beta", "gamma"],
        )
        outputfile, _ = _run_showtc(tmpdir, [f"{thefile}:beta"])
        _assert_wrote_image(outputfile)


def showtc_starttime_without_debug(debug=False):
    """Test --starttime works on a file with no embedded start offset.

    This pins the fix for a mis-indented assignment that left ``thestarttime`` unbound
    unless ``--debug`` happened to also be set.
    """
    if debug:
        print("showtc_starttime_without_debug")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_plaintext(tmpdir, "data.txt", _make_timecourses(numpoints=32))
        outputfile, args = _run_showtc(tmpdir, [thefile, "--starttime", "3.0"])
        _assert_wrote_image(outputfile)
        assert abs(args.thestarttime - 3.0) < 1e-9


def showtc_starttime_with_debug(debug=False):
    """Test --starttime behaves identically with --debug set."""
    if debug:
        print("showtc_starttime_with_debug")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_plaintext(tmpdir, "data.txt", _make_timecourses(numpoints=32))
        outputfile, args = _run_showtc(tmpdir, [thefile, "--starttime", "3.0", "--debug"])
        _assert_wrote_image(outputfile)
        assert abs(args.thestarttime - 3.0) < 1e-9


def showtc_starttime_overrides_bids_offset(debug=False):
    """Test an explicit --starttime overrides the offset recorded in a BIDS sidecar."""
    if debug:
        print("showtc_starttime_overrides_bids_offset")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_bids(
            tmpdir,
            "sub-01_task-rest_physio",
            _make_timecourses(numcolumns=1, numpoints=32),
            samplerate=2.0,
            starttime=5.0,
        )
        outputfile, args = _run_showtc(tmpdir, [thefile, "--starttime", "1.0"])
        _assert_wrote_image(outputfile)
        assert abs(args.thestarttime - 1.0) < 1e-9


def showtc_bids_offset_with_debug(debug=False):
    """Test the debug reporting path for a file that carries its own start offset."""
    if debug:
        print("showtc_bids_offset_with_debug")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_bids(
            tmpdir,
            "sub-01_task-rest_physio",
            _make_timecourses(numcolumns=1, numpoints=32),
            samplerate=2.0,
            starttime=5.0,
        )
        # no --starttime: the sidecar offset is adopted
        outputfile, args = _run_showtc(tmpdir, [thefile, "--debug"], outname="nostart.png")
        _assert_wrote_image(outputfile)
        assert abs(args.thestarttime - 5.0) < 1e-9
        # with --starttime: the command line wins
        outputfile, args = _run_showtc(
            tmpdir, [thefile, "--debug", "--starttime", "7.0"], outname="withstart.png"
        )
        _assert_wrote_image(outputfile)
        assert abs(args.thestarttime - 7.0) < 1e-9


def showtc_starttime_across_multiple_files(debug=False):
    """Test an explicit --starttime overrides the per file minimum across several files."""
    if debug:
        print("showtc_starttime_across_multiple_files")
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = _write_plaintext(tmpdir, "one.txt", _make_timecourses(numcolumns=1, numpoints=32))
        file2 = _write_plaintext(tmpdir, "two.txt", _make_timecourses(numcolumns=1, numpoints=32))
        outputfile, args = _run_showtc(tmpdir, [file1, file2, "--starttime", "4.0"])
        _assert_wrote_image(outputfile)
        assert abs(args.thestarttime - 4.0) < 1e-9


def showtc_without_seaborn(debug=False):
    """Test the plain matplotlib fallback used when seaborn is not installed."""
    if debug:
        print("showtc_without_seaborn")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_plaintext(tmpdir, "data.txt", _make_timecourses(numcolumns=2))
        outputfile = os.path.join(tmpdir, "noseaborn.png")
        args = _get_parser().parse_args([thefile, "--tofile", outputfile])
        with patch.object(showtc_module, "haveseaborn", False):
            try:
                showtc(args)
            finally:
                plt.close("all")
        _assert_wrote_image(outputfile)


def showtc_endtime_truncates_range(debug=False):
    """Test --endtime clips the plotted x range."""
    if debug:
        print("showtc_endtime_truncates_range")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_plaintext(tmpdir, "data.txt", _make_timecourses(numpoints=32))
        outputfile, args = _run_showtc(tmpdir, [thefile, "--endtime", "10.0"])
        _assert_wrote_image(outputfile)
        assert abs(args.theendtime - 10.0) < 1e-9


def showtc_endtime_before_starttime(debug=False):
    """Test showtc exits if the requested end time precedes the start time."""
    if debug:
        print("showtc_endtime_before_starttime")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_plaintext(tmpdir, "data.txt", _make_timecourses())
        args = _get_parser().parse_args([thefile, "--starttime", "10.0", "--endtime", "2.0"])
        try:
            showtc(args)
            assert False, "Should have raised SystemExit"
        except SystemExit:
            pass
        finally:
            plt.close("all")


def showtc_numskip_is_currently_ignored(debug=False):
    """Pin the fact that --numskip is parsed but never applied.

    The help text promises that NUM lines are skipped at the start of each file, but
    ``args.numskip`` is never read inside ``showtc``.  This test asserts the current (wrong)
    behaviour, so it will fail loudly if and when the option is actually implemented.
    """
    if debug:
        print("showtc_numskip_is_currently_ignored")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_plaintext(tmpdir, "data.txt", _make_timecourses(numpoints=32))
        _, plainargs = _run_showtc(tmpdir, [thefile], outname="plain.png")
        _, skipargs = _run_showtc(tmpdir, [thefile, "--numskip", "10"], outname="skipped.png")
        # skipping 10 of 32 lines would shorten the plotted range; it does not
        assert plainargs.thestarttime == skipargs.thestarttime
        assert plainargs.theendtime == skipargs.theendtime


# ==================== showtc appearance tests ====================


def showtc_normall(debug=False):
    """Test --normall renders normalized timecourses."""
    if debug:
        print("showtc_normall")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_plaintext(tmpdir, "data.txt", _make_timecourses(numcolumns=2))
        outputfile, _ = _run_showtc(tmpdir, [thefile, "--normall"])
        _assert_wrote_image(outputfile)


def showtc_normall_with_spectrum(debug=False):
    """Test --normall applies to spectral display as well as time series."""
    if debug:
        print("showtc_normall_with_spectrum")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_plaintext(tmpdir, "data.txt", _make_timecourses(numcolumns=2))
        outputfile, _ = _run_showtc(tmpdir, [thefile, "--normall", "--displaytype", "power"])
        _assert_wrote_image(outputfile)


def showtc_explicit_voffset(debug=False):
    """Test a positive --voffset separates overlaid traces."""
    if debug:
        print("showtc_explicit_voffset")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_plaintext(tmpdir, "data.txt", _make_timecourses(numcolumns=3))
        outputfile, args = _run_showtc(tmpdir, [thefile, "--voffset", "2.0"])
        _assert_wrote_image(outputfile)
        assert args.voffset == 2.0


def showtc_automatic_voffset(debug=False):
    """Test a negative --voffset is replaced by the full y range of the data."""
    if debug:
        print("showtc_automatic_voffset")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_plaintext(tmpdir, "data.txt", _make_timecourses(numcolumns=3))
        outputfile, args = _run_showtc(tmpdir, [thefile, "--voffset", "-1.0"])
        _assert_wrote_image(outputfile)
        # a negative offset is auto-set to the data range, so it comes back positive
        assert args.voffset > 0.0


def showtc_waterfall(debug=False):
    """Test --waterfall staggers the traces in both x and y."""
    if debug:
        print("showtc_waterfall")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_plaintext(tmpdir, "data.txt", _make_timecourses(numcolumns=3))
        outputfile, _ = _run_showtc(tmpdir, [thefile, "--waterfall"])
        _assert_wrote_image(outputfile)


def showtc_fullxrange(debug=False):
    """Test --fullxrange uses the union of all file time ranges."""
    if debug:
        print("showtc_fullxrange")
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = _write_plaintext(
            tmpdir, "short.txt", _make_timecourses(numcolumns=1, numpoints=16)
        )
        file2 = _write_plaintext(tmpdir, "long.txt", _make_timecourses(numcolumns=1, numpoints=48))
        outputfile, _ = _run_showtc(tmpdir, [file1, file2, "--fullxrange"])
        _assert_wrote_image(outputfile)


def showtc_forcezerostart(debug=False):
    """Test --forcezerostart shifts the displayed axis to begin at zero."""
    if debug:
        print("showtc_forcezerostart")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_bids(
            tmpdir,
            "sub-01_task-rest_physio",
            _make_timecourses(numcolumns=1, numpoints=32),
            samplerate=2.0,
            starttime=5.0,
        )
        outputfile, _ = _run_showtc(tmpdir, [thefile, "--forcezerostart"])
        _assert_wrote_image(outputfile)


def showtc_custom_colors_cycle(debug=False):
    """Test a --colors list shorter than the number of traces is cycled."""
    if debug:
        print("showtc_custom_colors_cycle")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_plaintext(tmpdir, "data.txt", _make_timecourses(numcolumns=4))
        outputfile, _ = _run_showtc(tmpdir, [thefile, "--colors", "red,blue"])
        _assert_wrote_image(outputfile)


def showtc_custom_legends_cycle(debug=False):
    """Test a --legends list shorter than the number of traces is cycled."""
    if debug:
        print("showtc_custom_legends_cycle")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_plaintext(tmpdir, "data.txt", _make_timecourses(numcolumns=3))
        outputfile, _ = _run_showtc(tmpdir, [thefile, "--legends", "first,second"])
        _assert_wrote_image(outputfile)


def showtc_custom_linewidths_cycle(debug=False):
    """Test a --linewidth list shorter than the number of traces is cycled."""
    if debug:
        print("showtc_custom_linewidths_cycle")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_plaintext(tmpdir, "data.txt", _make_timecourses(numcolumns=3))
        outputfile, _ = _run_showtc(tmpdir, [thefile, "--linewidth", "0.5,2.0"])
        _assert_wrote_image(outputfile)


def showtc_nolegend(debug=False):
    """Test --nolegend suppresses the legend."""
    if debug:
        print("showtc_nolegend")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_plaintext(tmpdir, "data.txt", _make_timecourses(numcolumns=2))
        outputfile, _ = _run_showtc(tmpdir, [thefile, "--nolegend"])
        _assert_wrote_image(outputfile)


def showtc_legendloc_bounds(debug=False):
    """Test the extremes of the legal legend location range are accepted."""
    if debug:
        print("showtc_legendloc_bounds")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_plaintext(tmpdir, "data.txt", _make_timecourses(numcolumns=2))
        for loc in ["0", "10"]:
            outputfile, _ = _run_showtc(
                tmpdir, [thefile, "--legendloc", loc], outname=f"loc{loc}.png"
            )
            _assert_wrote_image(outputfile)


def showtc_illegal_legendloc(debug=False):
    """Test showtc exits on an out of range legend location.

    The parser accepts any integer, so the range check only happens inside showtc.
    """
    if debug:
        print("showtc_illegal_legendloc")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_plaintext(tmpdir, "data.txt", _make_timecourses())
        for loc in ["11", "-1"]:
            args = _get_parser().parse_args([thefile, "--legendloc", loc])
            try:
                showtc(args)
                assert False, f"Should have raised SystemExit for legendloc {loc}"
            except SystemExit:
                pass
            finally:
                plt.close("all")


def showtc_hidden_axes(debug=False):
    """Test --noxax and --noyax hide the axes without erroring."""
    if debug:
        print("showtc_hidden_axes")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_plaintext(tmpdir, "data.txt", _make_timecourses(numcolumns=2))
        outputfile, _ = _run_showtc(tmpdir, [thefile, "--noxax", "--noyax"])
        _assert_wrote_image(outputfile)


def showtc_titles_and_labels(debug=False):
    """Test explicit title and axis labels are honoured rather than defaulted."""
    if debug:
        print("showtc_titles_and_labels")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_plaintext(tmpdir, "data.txt", _make_timecourses(numcolumns=2))
        outputfile, args = _run_showtc(
            tmpdir,
            [thefile, "--title", "My Plot", "--xlabel", "Xish", "--ylabel", "Yish"],
        )
        _assert_wrote_image(outputfile)
        # showtc only fills in defaults when the label is None
        assert args.xlabel == "Xish"
        assert args.ylabel == "Yish"


def showtc_separate_format_with_title(debug=False):
    """Test a title on a separate format plot becomes a figure suptitle."""
    if debug:
        print("showtc_separate_format_with_title")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_plaintext(tmpdir, "data.txt", _make_timecourses(numcolumns=2))
        outputfile, _ = _run_showtc(
            tmpdir, [thefile, "--format", "separate", "--title", "Stacked"]
        )
        _assert_wrote_image(outputfile)


def showtc_fontscalefac(debug=False):
    """Test a nondefault font scaling factor is applied without error."""
    if debug:
        print("showtc_fontscalefac")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_plaintext(tmpdir, "data.txt", _make_timecourses(numcolumns=2))
        outputfile, _ = _run_showtc(tmpdir, [thefile, "--fontscalefac", "2.0"])
        _assert_wrote_image(outputfile)


def showtc_figure_geometry(debug=False):
    """Test explicit figure width and aspect ratio are accepted in both plot formats."""
    if debug:
        print("showtc_figure_geometry")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_plaintext(tmpdir, "data.txt", _make_timecourses(numcolumns=2))
        for fmt, outname in [("overlaid", "wide.png"), ("separate", "stacked.png")]:
            outputfile, _ = _run_showtc(
                tmpdir,
                [thefile, "--figurewidth", "8.0", "--aspectratio", "2.0", "--format", fmt],
                outname=outname,
            )
            _assert_wrote_image(outputfile)


def showtc_debug_output(debug=False):
    """Test the debug path runs end to end and prints diagnostics."""
    if debug:
        print("showtc_debug_output")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_plaintext(tmpdir, "data.txt", _make_timecourses(numcolumns=2))
        outputfile, _ = _run_showtc(tmpdir, [thefile, "--debug"])
        _assert_wrote_image(outputfile)


def showtc_displays_when_no_output_file(debug=False):
    """Test showtc calls show() when no output file is requested."""
    if debug:
        print("showtc_displays_when_no_output_file")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_plaintext(tmpdir, "data.txt", _make_timecourses(numcolumns=2))
        args = _get_parser().parse_args([thefile])
        assert args.outputfile is None
        with patch.object(showtc_module, "show") as mockshow:
            try:
                showtc(args)
            finally:
                plt.close("all")
        assert mockshow.call_count == 1


def showtc_saves_at_requested_resolution(debug=False):
    """Test the --saveres value is passed through to savefig as the dpi."""
    if debug:
        print("showtc_saves_at_requested_resolution")
    with tempfile.TemporaryDirectory() as tmpdir:
        thefile = _write_plaintext(tmpdir, "data.txt", _make_timecourses(numcolumns=1))
        outputfile = os.path.join(tmpdir, "res.png")
        args = _get_parser().parse_args([thefile, "--tofile", outputfile, "--saveres", "72"])
        with patch.object(showtc_module, "savefig") as mocksavefig:
            try:
                showtc(args)
            finally:
                plt.close("all")
        assert mocksavefig.call_count == 1
        assert mocksavefig.call_args.kwargs["dpi"] == 72


# ==================== Main test function ====================


def test_showtc(debug=False):
    # phase tests
    if debug:
        print("Running phase tests")
    phase_cardinal_values(debug=debug)
    phase_diagonal_values(debug=debug)
    phase_range_is_bounded(debug=debug)
    phase_preserves_shape(debug=debug)
    phase_origin_is_zero(debug=debug)

    # _get_parser tests
    if debug:
        print("Running _get_parser tests")
    get_parser_returns_parser(debug=debug)
    get_parser_requires_a_filename(debug=debug)
    get_parser_accepts_multiple_filenames(debug=debug)
    get_parser_defaults(debug=debug)
    get_parser_samplerate(debug=debug)
    get_parser_sampletime_is_inverted(debug=debug)
    get_parser_sampling_is_mutually_exclusive(debug=debug)
    get_parser_displaytype_choices(debug=debug)
    get_parser_displaytype_rejects_bad_choice(debug=debug)
    get_parser_format_choices(debug=debug)
    get_parser_format_rejects_bad_choice(debug=debug)
    get_parser_boolean_flags(debug=debug)
    get_parser_numeric_options(debug=debug)
    get_parser_string_options(debug=debug)
    get_parser_rejects_abbreviations(debug=debug)

    # display mode tests
    if debug:
        print("Running showtc display mode tests")
    showtc_time_domain(debug=debug)
    showtc_power_spectrum(debug=debug)
    showtc_phase_spectrum(debug=debug)
    showtc_spectrum_of_odd_length_data(debug=debug)
    showtc_illegal_display_mode(debug=debug)

    # plot format tests
    if debug:
        print("Running showtc plot format tests")
    showtc_overlaid_format(debug=debug)
    showtc_separate_format(debug=debug)
    showtc_separatelinked_format(debug=debug)
    showtc_separate_format_single_vector(debug=debug)
    showtc_illegal_plot_format(debug=debug)

    # input handling tests
    if debug:
        print("Running showtc input handling tests")
    showtc_multiple_files(debug=debug)
    showtc_transpose(debug=debug)
    showtc_explicit_samplerate(debug=debug)
    showtc_auto_samplerate_defaults_to_one(debug=debug)
    showtc_bids_input_uses_embedded_metadata(debug=debug)
    showtc_bids_column_selection(debug=debug)
    showtc_starttime_without_debug(debug=debug)
    showtc_starttime_with_debug(debug=debug)
    showtc_starttime_overrides_bids_offset(debug=debug)
    showtc_bids_offset_with_debug(debug=debug)
    showtc_starttime_across_multiple_files(debug=debug)
    showtc_without_seaborn(debug=debug)
    showtc_endtime_truncates_range(debug=debug)
    showtc_endtime_before_starttime(debug=debug)
    showtc_numskip_is_currently_ignored(debug=debug)

    # appearance tests
    if debug:
        print("Running showtc appearance tests")
    showtc_normall(debug=debug)
    showtc_normall_with_spectrum(debug=debug)
    showtc_explicit_voffset(debug=debug)
    showtc_automatic_voffset(debug=debug)
    showtc_waterfall(debug=debug)
    showtc_fullxrange(debug=debug)
    showtc_forcezerostart(debug=debug)
    showtc_custom_colors_cycle(debug=debug)
    showtc_custom_legends_cycle(debug=debug)
    showtc_custom_linewidths_cycle(debug=debug)
    showtc_nolegend(debug=debug)
    showtc_legendloc_bounds(debug=debug)
    showtc_illegal_legendloc(debug=debug)
    showtc_hidden_axes(debug=debug)
    showtc_titles_and_labels(debug=debug)
    showtc_separate_format_with_title(debug=debug)
    showtc_fontscalefac(debug=debug)
    showtc_figure_geometry(debug=debug)
    showtc_debug_output(debug=debug)
    showtc_displays_when_no_output_file(debug=debug)
    showtc_saves_at_requested_resolution(debug=debug)


if __name__ == "__main__":
    test_showtc(debug=True)
