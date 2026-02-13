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
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rapidtide.workflows.calctexticc import (
    _get_parser,
    calctexticc,
    makdcommandlinelist,
    parsetextmeasurementlist,
)


def _capture_writenpvecs():
    """Return a side_effect function and its captured data list.

    calctexticc reuses a single numpy array for all writenpvecs calls,
    so the mock records a reference to the same mutating array.  This
    helper copies data at call time so each captured entry is independent.
    """
    captured = []

    def _side_effect(data, filename, **kwargs):
        captured.append((data.copy(), filename))

    return _side_effect, captured


# ---- _get_parser tests ----


def test_get_parser_returns_parser(debug=False):
    """Test that _get_parser returns an ArgumentParser."""
    parser = _get_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    if debug:
        print("Parser created successfully")


def test_get_parser_required_args(debug=False):
    """Test that parser requires the three positional arguments."""
    parser = _get_parser()
    # Should fail with no arguments
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_get_parser_with_valid_args(debug=False):
    """Test parser with valid positional arguments using a real temp file for measurementlist."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("0 1\n2 3\n")
        measfile = f.name

    parser = _get_parser()
    args = parser.parse_args(["data1.txt,data2.txt", measfile, "output_root"])
    assert args.datafile == "data1.txt,data2.txt"
    assert args.outputroot == "output_root"
    assert args.demedian is False
    assert args.demean is False
    assert args.nocache is False
    assert args.debug is False
    assert args.deepdebug is False

    import os

    os.unlink(measfile)

    if debug:
        print(f"Parsed args: {args}")


def test_get_parser_optional_flags(debug=False):
    """Test parser with all optional flags enabled."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("0 1\n")
        measfile = f.name

    parser = _get_parser()
    args = parser.parse_args(
        [
            "data.txt",
            measfile,
            "out",
            "--demedian",
            "--demean",
            "--nocache",
            "--debug",
            "--deepdebug",
        ]
    )
    assert args.demedian is True
    assert args.demean is True
    assert args.nocache is True
    assert args.debug is True
    assert args.deepdebug is True

    import os

    os.unlink(measfile)


# ---- parsetextmeasurementlist tests ----


def test_parsetextmeasurementlist_single_values(debug=False):
    """Test parsing measurement list with single values (volume only, file=0)."""
    measlist = np.array([["0", "1"], ["2", "3"]])
    numfiles = 1
    filesel, volumesel = parsetextmeasurementlist(measlist, numfiles)

    assert filesel.shape == (2, 2)
    assert volumesel.shape == (2, 2)
    # Single values => file index defaults to 0
    np.testing.assert_array_equal(filesel, np.zeros((2, 2), dtype=int))
    np.testing.assert_array_equal(volumesel, np.array([[0, 1], [2, 3]]))

    if debug:
        print(f"filesel:\n{filesel}")
        print(f"volumesel:\n{volumesel}")


def test_parsetextmeasurementlist_comma_separated(debug=False):
    """Test parsing measurement list with file,volume entries."""
    measlist = np.array([["0,1", "1,2"], ["0,3", "2,4"]])
    numfiles = 3
    filesel, volumesel = parsetextmeasurementlist(measlist, numfiles)

    expected_filesel = np.array([[0, 1], [0, 2]])
    expected_volumesel = np.array([[1, 2], [3, 4]])
    np.testing.assert_array_equal(filesel, expected_filesel)
    np.testing.assert_array_equal(volumesel, expected_volumesel)

    if debug:
        print(f"filesel:\n{filesel}")
        print(f"volumesel:\n{volumesel}")


def test_parsetextmeasurementlist_mixed_formats(debug=False):
    """Test parsing measurement list with mixed single and comma-separated entries."""
    measlist = np.array([["0,5", "3"], ["1,2", "0,7"]])
    numfiles = 2
    filesel, volumesel = parsetextmeasurementlist(measlist, numfiles)

    expected_filesel = np.array([[0, 0], [1, 0]])
    expected_volumesel = np.array([[5, 3], [2, 7]])
    np.testing.assert_array_equal(filesel, expected_filesel)
    np.testing.assert_array_equal(volumesel, expected_volumesel)


def test_parsetextmeasurementlist_debug_output(debug=False):
    """Test that debug mode prints without errors."""
    measlist = np.array([["0,1", "2"]])
    numfiles = 1
    filesel, volumesel = parsetextmeasurementlist(measlist, numfiles, debug=True)
    assert filesel.shape == (1, 2)


def test_parsetextmeasurementlist_too_many_commas(debug=False):
    """Test that entries with more than one comma cause sys.exit."""
    measlist = np.array([["0,1,2"]])
    numfiles = 3
    with pytest.raises(SystemExit):
        parsetextmeasurementlist(measlist, numfiles)


def test_parsetextmeasurementlist_file_out_of_range(debug=False):
    """Test that file index exceeding numfiles causes sys.exit."""
    measlist = np.array([["5,0"]])
    numfiles = 3  # valid file indices: 0, 1, 2
    with pytest.raises(SystemExit):
        parsetextmeasurementlist(measlist, numfiles)


def test_parsetextmeasurementlist_single_element(debug=False):
    """Test with a single element measurement list."""
    measlist = np.array([["0,0"]])
    numfiles = 1
    filesel, volumesel = parsetextmeasurementlist(measlist, numfiles)
    assert filesel.shape == (1, 1)
    assert volumesel.shape == (1, 1)
    assert filesel[0, 0] == 0
    assert volumesel[0, 0] == 0


# ---- makdcommandlinelist tests ----


def test_makdcommandlinelist_without_extra(debug=False):
    """Test command line list generation without extra info."""
    arglist = ["python", "calctexticc", "--debug", "data.txt"]
    starttime = time.time() - 5.0
    endtime = time.time()

    with patch("rapidtide.workflows.calctexticc.tide_util.version") as mock_version:
        mock_version.return_value = ("v2.9.0", "abc123", "2024-01-01", False)
        result = makdcommandlinelist(arglist, starttime, endtime)

    assert len(result) == 4
    assert result[0].startswith("# Processed on")
    assert "Processing took" in result[1]
    assert "v2.9.0" in result[2]
    assert result[3] == "python calctexticc --debug data.txt"

    if debug:
        for line in result:
            print(line)


def test_makdcommandlinelist_with_extra(debug=False):
    """Test command line list generation with extra info."""
    arglist = ["python", "calctexticc"]
    starttime = time.time() - 2.0
    endtime = time.time()

    with patch("rapidtide.workflows.calctexticc.tide_util.version") as mock_version:
        mock_version.return_value = ("v2.9.0", "abc123", "2024-01-01", False)
        result = makdcommandlinelist(arglist, starttime, endtime, extra="ICC time: 1.5 ms")

    assert len(result) == 5
    assert result[0].startswith("# Processed on")
    assert "Processing took" in result[1]
    assert "v2.9.0" in result[2]
    assert result[3] == "# ICC time: 1.5 ms"
    assert result[4] == "python calctexticc"

    if debug:
        for line in result:
            print(line)


def test_makdcommandlinelist_timing(debug=False):
    """Test that timing information is computed correctly."""
    arglist = ["test"]
    starttime = 1000.0
    endtime = 1010.5

    with patch("rapidtide.workflows.calctexticc.tide_util.version") as mock_version:
        mock_version.return_value = ("v1.0", "sha", "date", False)
        result = makdcommandlinelist(arglist, starttime, endtime)

    assert "10.500" in result[1]


def test_makdcommandlinelist_empty_arglist(debug=False):
    """Test with empty argument list."""
    arglist = []
    starttime = time.time()
    endtime = starttime + 1.0

    with patch("rapidtide.workflows.calctexticc.tide_util.version") as mock_version:
        mock_version.return_value = ("v1.0", "sha", "date", False)
        result = makdcommandlinelist(arglist, starttime, endtime)

    # Command line should be empty string
    assert result[-1] == ""


# ---- calctexticc tests ----


def _make_icc_test_data(numsubjs=4, nummeas=3, numvals=5, rng=None):
    """Helper to create synthetic ICC test data.

    Creates data where each subject has a consistent pattern across measurements
    plus some noise, so ICC should be moderately high.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    # Subject effects (consistent across measurements)
    subject_effects = rng.randn(numsubjs, numvals) * 5.0
    # Measurement noise
    noise = rng.randn(numsubjs, nummeas, numvals) * 1.0

    # data[val, subj*nummeas + meas] format
    data = np.zeros((numvals, numsubjs * nummeas))
    for subj in range(numsubjs):
        for meas in range(nummeas):
            data[:, subj * nummeas + meas] = subject_effects[subj, :] + noise[subj, meas, :]

    return data


def _make_measlist(numsubjs, nummeas):
    """Helper to create a simple measurement list for a single data file."""
    measlist = np.zeros((nummeas, numsubjs), dtype="U10")
    for subj in range(numsubjs):
        for meas in range(nummeas):
            measlist[meas, subj] = str(subj * nummeas + meas)
    return measlist


def test_calctexticc_basic(debug=False):
    """Test basic ICC calculation with synthetic data."""
    rng = np.random.RandomState(42)
    numsubjs = 4
    nummeas = 2
    numvals = 5

    data = _make_icc_test_data(numsubjs=numsubjs, nummeas=nummeas, numvals=numvals, rng=rng)
    measlist = _make_measlist(numsubjs, nummeas)

    side_effect, captured = _capture_writenpvecs()

    with tempfile.TemporaryDirectory() as tmpdir:
        outputroot = f"{tmpdir}/test_icc"

        args = argparse.Namespace(
            datafile="dummy.txt",
            measurementlist="dummy_meas.txt",
            outputroot=outputroot,
            debug=debug,
            deepdebug=False,
            demedian=False,
            demean=False,
            nocache=False,
        )

        with (
            patch("rapidtide.workflows.calctexticc.tide_io.readvecs") as mock_readvecs,
            patch("rapidtide.workflows.calctexticc.tide_io.writenpvecs", side_effect=side_effect),
            patch("rapidtide.workflows.calctexticc.tide_io.writevec"),
            patch("rapidtide.workflows.calctexticc.tide_util.version") as mock_version,
        ):
            mock_version.return_value = ("v2.9.0", "abc123", "2024-01-01", False)
            mock_readvecs.side_effect = [measlist, data, data]

            calctexticc(args)

            # Should have written 4 output files
            assert len(captured) == 4

            # Check ICC output shape
            icc_values, icc_filename = captured[0]
            assert icc_values.shape == (numvals,)

            # Check filenames
            assert icc_filename == f"{outputroot}_ICC.txt"
            assert captured[1][1] == f"{outputroot}_r_var.txt"
            assert captured[2][1] == f"{outputroot}_e_var.txt"
            assert captured[3][1] == f"{outputroot}_session_effect_F.txt"

        if debug:
            print(f"ICC values: {icc_values}")


def test_calctexticc_demedian(debug=False):
    """Test ICC calculation with demedian option."""
    rng = np.random.RandomState(99)
    numsubjs = 3
    nummeas = 2
    numvals = 4

    data = _make_icc_test_data(numsubjs=numsubjs, nummeas=nummeas, numvals=numvals, rng=rng)
    data += 100.0
    measlist = _make_measlist(numsubjs, nummeas)

    side_effect, captured = _capture_writenpvecs()

    with tempfile.TemporaryDirectory() as tmpdir:
        outputroot = f"{tmpdir}/test_demedian"

        args = argparse.Namespace(
            datafile="dummy.txt",
            measurementlist="dummy_meas.txt",
            outputroot=outputroot,
            debug=debug,
            deepdebug=False,
            demedian=True,
            demean=False,
            nocache=False,
        )

        with (
            patch("rapidtide.workflows.calctexticc.tide_io.readvecs") as mock_readvecs,
            patch("rapidtide.workflows.calctexticc.tide_io.writenpvecs", side_effect=side_effect),
            patch("rapidtide.workflows.calctexticc.tide_io.writevec"),
            patch("rapidtide.workflows.calctexticc.tide_util.version") as mock_version,
        ):
            mock_version.return_value = ("v2.9.0", "abc123", "2024-01-01", False)
            mock_readvecs.side_effect = [measlist, data, data]

            calctexticc(args)

            assert len(captured) == 4
            icc_values = captured[0][0]
            assert icc_values.shape == (numvals,)

            if debug:
                print(f"Demedian ICC values: {icc_values}")


def test_calctexticc_demean(debug=False):
    """Test ICC calculation with demean option."""
    rng = np.random.RandomState(77)
    numsubjs = 3
    nummeas = 2
    numvals = 4

    data = _make_icc_test_data(numsubjs=numsubjs, nummeas=nummeas, numvals=numvals, rng=rng)
    data += 50.0
    measlist = _make_measlist(numsubjs, nummeas)

    side_effect, captured = _capture_writenpvecs()

    with tempfile.TemporaryDirectory() as tmpdir:
        outputroot = f"{tmpdir}/test_demean"

        args = argparse.Namespace(
            datafile="dummy.txt",
            measurementlist="dummy_meas.txt",
            outputroot=outputroot,
            debug=debug,
            deepdebug=False,
            demedian=False,
            demean=True,
            nocache=False,
        )

        with (
            patch("rapidtide.workflows.calctexticc.tide_io.readvecs") as mock_readvecs,
            patch("rapidtide.workflows.calctexticc.tide_io.writenpvecs", side_effect=side_effect),
            patch("rapidtide.workflows.calctexticc.tide_io.writevec"),
            patch("rapidtide.workflows.calctexticc.tide_util.version") as mock_version,
        ):
            mock_version.return_value = ("v2.9.0", "abc123", "2024-01-01", False)
            mock_readvecs.side_effect = [measlist, data, data]

            calctexticc(args)

            assert len(captured) == 4
            icc_values = captured[0][0]
            assert icc_values.shape == (numvals,)

            if debug:
                print(f"Demean ICC values: {icc_values}")


def test_calctexticc_nocache(debug=False):
    """Test ICC calculation with nocache option."""
    rng = np.random.RandomState(55)
    numsubjs = 3
    nummeas = 2
    numvals = 3

    data = _make_icc_test_data(numsubjs=numsubjs, nummeas=nummeas, numvals=numvals, rng=rng)
    measlist = _make_measlist(numsubjs, nummeas)

    side_effect, captured = _capture_writenpvecs()

    with tempfile.TemporaryDirectory() as tmpdir:
        outputroot = f"{tmpdir}/test_nocache"

        args = argparse.Namespace(
            datafile="dummy.txt",
            measurementlist="dummy_meas.txt",
            outputroot=outputroot,
            debug=debug,
            deepdebug=False,
            demedian=False,
            demean=False,
            nocache=True,
        )

        with (
            patch("rapidtide.workflows.calctexticc.tide_io.readvecs") as mock_readvecs,
            patch("rapidtide.workflows.calctexticc.tide_io.writenpvecs", side_effect=side_effect),
            patch("rapidtide.workflows.calctexticc.tide_io.writevec"),
            patch("rapidtide.workflows.calctexticc.tide_util.version") as mock_version,
        ):
            mock_version.return_value = ("v2.9.0", "abc123", "2024-01-01", False)
            mock_readvecs.side_effect = [measlist, data, data]

            calctexticc(args)

            assert len(captured) == 4
            icc_values = captured[0][0]
            assert icc_values.shape == (numvals,)


def test_calctexticc_multiple_files(debug=False):
    """Test ICC calculation with data spread across multiple files."""
    rng = np.random.RandomState(33)
    numsubjs = 3
    nummeas = 2
    numvals = 4

    # Create two data files, each with a subset of volumes
    data_file0 = rng.randn(numvals, numsubjs)  # 3 volumes in file 0
    data_file1 = rng.randn(numvals, numsubjs)  # 3 volumes in file 1

    # Measurement list uses file,volume format
    measlist = np.array(
        [["0,0", "0,1", "0,2"], ["1,0", "1,1", "1,2"]],  # meas 0: all from file 0
        dtype="U10",  # meas 1: all from file 1
    )

    side_effect, captured = _capture_writenpvecs()

    with tempfile.TemporaryDirectory() as tmpdir:
        outputroot = f"{tmpdir}/test_multifile"

        args = argparse.Namespace(
            datafile="file0.txt,file1.txt",
            measurementlist="dummy_meas.txt",
            outputroot=outputroot,
            debug=debug,
            deepdebug=False,
            demedian=False,
            demean=False,
            nocache=False,
        )

        with (
            patch("rapidtide.workflows.calctexticc.tide_io.readvecs") as mock_readvecs,
            patch("rapidtide.workflows.calctexticc.tide_io.writenpvecs", side_effect=side_effect),
            patch("rapidtide.workflows.calctexticc.tide_io.writevec"),
            patch("rapidtide.workflows.calctexticc.tide_util.version") as mock_version,
        ):
            mock_version.return_value = ("v2.9.0", "abc123", "2024-01-01", False)
            # Calls: measlist, header0, header1, data0, data1
            mock_readvecs.side_effect = [
                measlist,
                data_file0,
                data_file1,
                data_file0,
                data_file1,
            ]

            calctexticc(args)

            assert len(captured) == 4


def test_calctexticc_deepdebug(debug=False):
    """Test ICC calculation with deepdebug output."""
    rng = np.random.RandomState(11)
    numsubjs = 2
    nummeas = 2
    numvals = 2

    data = _make_icc_test_data(numsubjs=numsubjs, nummeas=nummeas, numvals=numvals, rng=rng)
    measlist = _make_measlist(numsubjs, nummeas)

    with tempfile.TemporaryDirectory() as tmpdir:
        outputroot = f"{tmpdir}/test_deepdebug"

        args = argparse.Namespace(
            datafile="dummy.txt",
            measurementlist="dummy_meas.txt",
            outputroot=outputroot,
            debug=True,
            deepdebug=True,
            demedian=False,
            demean=False,
            nocache=False,
        )

        with (
            patch("rapidtide.workflows.calctexticc.tide_io.readvecs") as mock_readvecs,
            patch("rapidtide.workflows.calctexticc.tide_io.writenpvecs"),
            patch("rapidtide.workflows.calctexticc.tide_io.writevec"),
            patch("rapidtide.workflows.calctexticc.tide_util.version") as mock_version,
        ):
            mock_version.return_value = ("v2.9.0", "abc123", "2024-01-01", False)
            mock_readvecs.side_effect = [measlist, data, data]

            # Should not raise even with deepdebug enabled
            calctexticc(args)


def test_calctexticc_icc_values_correct(debug=False):
    """Test that ICC values are high for nearly-perfectly reproducible data."""
    rng = np.random.RandomState(42)
    numsubjs = 6
    nummeas = 4
    numvals = 3

    # Create nearly-perfect data: strong subject effects with tiny noise
    data = np.zeros((numvals, numsubjs * nummeas))
    subject_values = np.linspace(1.0, 10.0, numsubjs)
    for subj in range(numsubjs):
        for meas in range(nummeas):
            for v in range(numvals):
                data[v, subj * nummeas + meas] = (
                    subject_values[subj] * (v + 1) + rng.randn() * 0.01
                )

    measlist = _make_measlist(numsubjs, nummeas)
    side_effect, captured = _capture_writenpvecs()

    with tempfile.TemporaryDirectory() as tmpdir:
        outputroot = f"{tmpdir}/test_high_icc"

        args = argparse.Namespace(
            datafile="dummy.txt",
            measurementlist="dummy_meas.txt",
            outputroot=outputroot,
            debug=debug,
            deepdebug=False,
            demedian=False,
            demean=False,
            nocache=True,
        )

        with (
            patch("rapidtide.workflows.calctexticc.tide_io.readvecs") as mock_readvecs,
            patch("rapidtide.workflows.calctexticc.tide_io.writenpvecs", side_effect=side_effect),
            patch("rapidtide.workflows.calctexticc.tide_io.writevec"),
            patch("rapidtide.workflows.calctexticc.tide_util.version") as mock_version,
        ):
            mock_version.return_value = ("v2.9.0", "abc123", "2024-01-01", False)
            mock_readvecs.side_effect = [measlist, data, data]

            calctexticc(args)

            icc_values = captured[0][0]  # ICC
            e_var_values = captured[2][0]  # e_var

            # Near-perfect reproducibility should give ICC close to 1.0
            assert np.all(icc_values > 0.99), f"Expected ICC > 0.99, got {icc_values}"

            # Error variance should be very small
            assert np.all(e_var_values < 0.01), f"Expected small e_var, got {e_var_values}"

            if debug:
                print(f"High ICC values: {icc_values}")
                print(f"Error variance: {e_var_values}")


def test_calctexticc_zero_icc_data(debug=False):
    """Test ICC calculation with pure noise data (ICC should be near 0)."""
    rng = np.random.RandomState(123)
    numsubjs = 10
    nummeas = 3
    numvals = 3

    # Pure noise - no subject effects, ICC should be near 0
    data = rng.randn(numvals, numsubjs * nummeas)
    measlist = _make_measlist(numsubjs, nummeas)

    side_effect, captured = _capture_writenpvecs()

    with tempfile.TemporaryDirectory() as tmpdir:
        outputroot = f"{tmpdir}/test_noise"

        args = argparse.Namespace(
            datafile="dummy.txt",
            measurementlist="dummy_meas.txt",
            outputroot=outputroot,
            debug=debug,
            deepdebug=False,
            demedian=False,
            demean=False,
            nocache=True,
        )

        with (
            patch("rapidtide.workflows.calctexticc.tide_io.readvecs") as mock_readvecs,
            patch("rapidtide.workflows.calctexticc.tide_io.writenpvecs", side_effect=side_effect),
            patch("rapidtide.workflows.calctexticc.tide_io.writevec"),
            patch("rapidtide.workflows.calctexticc.tide_util.version") as mock_version,
        ):
            mock_version.return_value = ("v2.9.0", "abc123", "2024-01-01", False)
            mock_readvecs.side_effect = [measlist, data, data]

            calctexticc(args)

            icc_values = captured[0][0]
            # For pure noise, ICC should be close to 0 (but can be slightly negative)
            assert np.all(
                np.abs(icc_values) < 0.5
            ), f"Expected low ICC for noise, got {icc_values}"

            if debug:
                print(f"Noise ICC values: {icc_values}")


def test_calctexticc_output_variances(debug=False):
    """Test that e_var outputs are non-negative for well-behaved data."""
    rng = np.random.RandomState(88)
    numsubjs = 5
    nummeas = 3
    numvals = 4

    data = _make_icc_test_data(numsubjs=numsubjs, nummeas=nummeas, numvals=numvals, rng=rng)
    measlist = _make_measlist(numsubjs, nummeas)

    side_effect, captured = _capture_writenpvecs()

    with tempfile.TemporaryDirectory() as tmpdir:
        outputroot = f"{tmpdir}/test_var"

        args = argparse.Namespace(
            datafile="dummy.txt",
            measurementlist="dummy_meas.txt",
            outputroot=outputroot,
            debug=debug,
            deepdebug=False,
            demedian=False,
            demean=False,
            nocache=False,
        )

        with (
            patch("rapidtide.workflows.calctexticc.tide_io.readvecs") as mock_readvecs,
            patch("rapidtide.workflows.calctexticc.tide_io.writenpvecs", side_effect=side_effect),
            patch("rapidtide.workflows.calctexticc.tide_io.writevec"),
            patch("rapidtide.workflows.calctexticc.tide_util.version") as mock_version,
        ):
            mock_version.return_value = ("v2.9.0", "abc123", "2024-01-01", False)
            mock_readvecs.side_effect = [measlist, data, data]

            calctexticc(args)

            e_var_values = captured[2][0]
            session_f_values = captured[3][0]

            # Error variance should always be non-negative (it's MSE)
            assert np.all(e_var_values >= 0), f"e_var should be non-negative: {e_var_values}"
            # Session effect F should be non-negative
            assert np.all(session_f_values >= 0), f"F should be non-negative: {session_f_values}"

            if debug:
                print(f"e_var: {e_var_values}")
                print(f"session_F: {session_f_values}")


# ---- main test entry point ----


def test_calctexticc(debug=False):
    """Run all calctexticc sub-tests."""
    # _get_parser tests
    test_get_parser_returns_parser(debug=debug)
    test_get_parser_required_args(debug=debug)
    test_get_parser_with_valid_args(debug=debug)
    test_get_parser_optional_flags(debug=debug)

    # parsetextmeasurementlist tests
    test_parsetextmeasurementlist_single_values(debug=debug)
    test_parsetextmeasurementlist_comma_separated(debug=debug)
    test_parsetextmeasurementlist_mixed_formats(debug=debug)
    test_parsetextmeasurementlist_debug_output(debug=debug)
    test_parsetextmeasurementlist_too_many_commas(debug=debug)
    test_parsetextmeasurementlist_file_out_of_range(debug=debug)
    test_parsetextmeasurementlist_single_element(debug=debug)

    # makdcommandlinelist tests
    test_makdcommandlinelist_without_extra(debug=debug)
    test_makdcommandlinelist_with_extra(debug=debug)
    test_makdcommandlinelist_timing(debug=debug)
    test_makdcommandlinelist_empty_arglist(debug=debug)

    # calctexticc tests
    test_calctexticc_basic(debug=debug)
    test_calctexticc_demedian(debug=debug)
    test_calctexticc_demean(debug=debug)
    test_calctexticc_nocache(debug=debug)
    test_calctexticc_multiple_files(debug=debug)
    test_calctexticc_deepdebug(debug=debug)
    test_calctexticc_icc_values_correct(debug=debug)
    test_calctexticc_zero_icc_data(debug=debug)
    test_calctexticc_output_variances(debug=debug)


if __name__ == "__main__":
    test_calctexticc(debug=True)
