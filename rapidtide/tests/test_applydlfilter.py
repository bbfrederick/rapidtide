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
from unittest.mock import MagicMock, mock_open, patch

import numpy as np

from rapidtide.tests.utils import create_dir, get_test_temp_path
from rapidtide.workflows.applydlfilter import (
    DEFAULT_MODEL,
    _get_parser,
    applydlfilter,
    findfirst,
)

# ==================== Tests for findfirst ====================


def findfirst_found_first_element(debug=False):
    """Test findfirst when the first element of searchlist is in available."""
    available = ["alpha", "beta", "gamma", "delta"]
    searchlist = ["beta", "gamma"]
    index, name = findfirst(searchlist, available, debug=debug)
    assert index == 1
    assert name == "beta"


def findfirst_found_second_element(debug=False):
    """Test findfirst falls back to second element when first is not available."""
    available = ["alpha", "beta", "gamma", "delta"]
    searchlist = ["missing", "gamma"]
    index, name = findfirst(searchlist, available, debug=debug)
    assert index == 2
    assert name == "gamma"


def findfirst_not_found(debug=False):
    """Test findfirst when no element from searchlist is in available."""
    available = ["alpha", "beta", "gamma"]
    searchlist = ["missing1", "missing2"]
    index, name = findfirst(searchlist, available, debug=debug)
    assert index == -1
    assert name is None


def findfirst_single_element_found(debug=False):
    """Test findfirst with a single-element searchlist that is found."""
    available = ["x", "y", "z"]
    searchlist = ["z"]
    index, name = findfirst(searchlist, available, debug=debug)
    assert index == 2
    assert name == "z"


def findfirst_single_element_not_found(debug=False):
    """Test findfirst with a single-element searchlist that is not found."""
    available = ["x", "y", "z"]
    searchlist = ["w"]
    index, name = findfirst(searchlist, available, debug=debug)
    assert index == -1
    assert name is None


def findfirst_empty_available(debug=False):
    """Test findfirst with an empty available list."""
    available = []
    searchlist = ["something"]
    index, name = findfirst(searchlist, available, debug=debug)
    assert index == -1
    assert name is None


def findfirst_debug_output(debug=False):
    """Test findfirst with debug=True to exercise debug print paths."""
    available = ["a", "b", "c"]
    searchlist = ["b"]
    index, name = findfirst(searchlist, available, debug=True)
    assert index == 1
    assert name == "b"

    # Also exercise the debug path for a miss
    searchlist = ["x"]
    index, name = findfirst(searchlist, available, debug=True)
    assert index == -1
    assert name is None


def findfirst_duplicate_in_available(debug=False):
    """Test findfirst when the available list has duplicates - should find first occurrence."""
    available = ["a", "b", "a", "c"]
    searchlist = ["a"]
    index, name = findfirst(searchlist, available, debug=debug)
    assert index == 0
    assert name == "a"


# ==================== Tests for _get_parser ====================


def get_parser_returns_parser(debug=False):
    """Test that _get_parser returns an ArgumentParser."""
    parser = _get_parser()
    assert isinstance(parser, argparse.ArgumentParser)


def get_parser_default_model(debug=False):
    """Test that the default model is correctly set."""
    parser = _get_parser()
    # Parse with dummy files (need to bypass is_valid_file check)
    # We test the default by inspecting the parser's defaults
    defaults = {action.dest: action.default for action in parser._actions}
    assert defaults["model"] == DEFAULT_MODEL


def get_parser_default_flags(debug=False):
    """Test that default flag values are correct."""
    parser = _get_parser()
    defaults = {action.dest: action.default for action in parser._actions}
    assert defaults["filesarelists"] is False
    assert defaults["display"] is True
    assert defaults["verbose"] is False


def get_parser_prog_name(debug=False):
    """Test that the parser prog name is set correctly."""
    parser = _get_parser()
    assert parser.prog == "applydlfilter"


def get_parser_with_model_option(testtemproot, debug=False):
    """Test parser with --model option."""
    # Create a dummy input file
    infile = os.path.join(testtemproot, "parser_test_input.txt")
    with open(infile, "w") as f:
        f.write("1.0 2.0 3.0\n")

    parser = _get_parser()
    args = parser.parse_args([infile, "output.txt", "--model", "custom_model"])
    assert args.model == "custom_model"


def get_parser_with_filesarelists(testtemproot, debug=False):
    """Test parser with --filesarelists flag."""
    infile = os.path.join(testtemproot, "parser_test_input2.txt")
    with open(infile, "w") as f:
        f.write("file1.txt\nfile2.txt\n")

    parser = _get_parser()
    args = parser.parse_args([infile, "output.txt", "--filesarelists"])
    assert args.filesarelists is True


def get_parser_with_nodisplay(testtemproot, debug=False):
    """Test parser with --nodisplay flag."""
    infile = os.path.join(testtemproot, "parser_test_input3.txt")
    with open(infile, "w") as f:
        f.write("1.0 2.0 3.0\n")

    parser = _get_parser()
    args = parser.parse_args([infile, "output.txt", "--nodisplay"])
    assert args.display is False


def get_parser_with_verbose(testtemproot, debug=False):
    """Test parser with --verbose flag."""
    infile = os.path.join(testtemproot, "parser_test_input4.txt")
    with open(infile, "w") as f:
        f.write("1.0 2.0 3.0\n")

    parser = _get_parser()
    args = parser.parse_args([infile, "output.txt", "--verbose"])
    assert args.verbose is True


def get_parser_all_options(testtemproot, debug=False):
    """Test parser with all options combined."""
    infile = os.path.join(testtemproot, "parser_test_input5.txt")
    with open(infile, "w") as f:
        f.write("1.0\n")

    parser = _get_parser()
    args = parser.parse_args(
        [
            infile,
            "output.txt",
            "--model",
            "my_model",
            "--filesarelists",
            "--nodisplay",
            "--verbose",
        ]
    )
    assert args.model == "my_model"
    assert args.filesarelists is True
    assert args.display is False
    assert args.verbose is True


# ==================== Tests for applydlfilter ====================


def _make_mock_dlfilter(usebadpts=False):
    """Helper to create a mock DeepLearningFilter."""
    mock_filter = MagicMock()
    mock_filter.usebadpts = usebadpts
    mock_filter.apply.return_value = np.random.randn(100).astype(np.float32)
    return mock_filter


def applydlfilter_single_file(testtemproot, debug=False):
    """Test applydlfilter with a single input file (filesarelists=False)."""
    infile = os.path.join(testtemproot, "dlf_test_input.tsv")
    outfile = os.path.join(testtemproot, "dlf_test_output")

    signal_length = 100
    fmridata = np.random.randn(signal_length).astype(np.float32)
    filtered_data = np.random.randn(signal_length).astype(np.float32)

    args = argparse.Namespace(
        infilename=infile,
        outfilename=outfile,
        filesarelists=False,
        model=DEFAULT_MODEL,
        display=False,
        verbose=debug,
    )

    mock_filter = _make_mock_dlfilter(usebadpts=False)
    mock_filter.apply.return_value = filtered_data

    # readbidstsv returns: (samplerate, starttime, columns, data, compressed, columnsource, extrainfo)
    columns = ["cardiacfromfmri_25.0Hz"]
    data = fmridata.reshape(1, -1)

    with (
        patch("rapidtide.workflows.applydlfilter.tide_dlfilt.DeepLearningFilter") as mock_cls,
        patch("rapidtide.workflows.applydlfilter.tide_io.readbidstsv") as mock_read,
        patch("rapidtide.workflows.applydlfilter.tide_io.writebidstsv") as mock_write,
        patch("rapidtide.workflows.applydlfilter.happy_support.checkcardmatch") as mock_check,
    ):

        mock_cls.return_value = mock_filter
        mock_read.return_value = (25.0, 0.0, columns, data, False, "tsv", {})
        mock_check.return_value = (0.95, 0.0, "")

        applydlfilter(args)

        mock_filter.loadmodel.assert_called_once_with(DEFAULT_MODEL)
        mock_read.assert_called_once()
        mock_filter.apply.assert_called_once()
        mock_write.assert_called_once()

        # Verify the output filename passed to writebidstsv
        write_call_args = mock_write.call_args
        assert write_call_args[0][0] == outfile


def applydlfilter_single_file_with_pleth(testtemproot, debug=False):
    """Test applydlfilter with pleth data present."""
    infile = os.path.join(testtemproot, "dlf_test_pleth_input.tsv")
    outfile = os.path.join(testtemproot, "dlf_test_pleth_output")

    signal_length = 100
    fmridata = np.random.randn(signal_length).astype(np.float32)
    plethdata = np.random.randn(signal_length).astype(np.float32)
    filtered_data = np.random.randn(signal_length).astype(np.float32)

    args = argparse.Namespace(
        infilename=infile,
        outfilename=outfile,
        filesarelists=False,
        model=DEFAULT_MODEL,
        display=False,
        verbose=debug,
    )

    mock_filter = _make_mock_dlfilter(usebadpts=False)
    mock_filter.apply.return_value = filtered_data

    columns = ["cardiacfromfmri_25.0Hz", "pleth"]
    data = np.vstack([fmridata, plethdata])

    with (
        patch("rapidtide.workflows.applydlfilter.tide_dlfilt.DeepLearningFilter") as mock_cls,
        patch("rapidtide.workflows.applydlfilter.tide_io.readbidstsv") as mock_read,
        patch("rapidtide.workflows.applydlfilter.tide_io.writebidstsv") as mock_write,
        patch("rapidtide.workflows.applydlfilter.happy_support.checkcardmatch") as mock_check,
    ):

        mock_cls.return_value = mock_filter
        mock_read.return_value = (25.0, 0.0, columns, data, False, "tsv", {})
        mock_check.return_value = (0.90, 0.0, "")

        applydlfilter(args)

        # With pleth present, checkcardmatch should be called 3 times:
        # 1) raw vs dl filtered
        # 2) raw vs pleth
        # 3) dl filtered vs pleth
        assert mock_check.call_count == 3


def applydlfilter_single_file_with_badpts(testtemproot, debug=False):
    """Test applydlfilter with bad points data present."""
    infile = os.path.join(testtemproot, "dlf_test_badpts_input.tsv")
    outfile = os.path.join(testtemproot, "dlf_test_badpts_output")

    signal_length = 100
    fmridata = np.random.randn(signal_length).astype(np.float32)
    badptsdata = np.zeros(signal_length, dtype=np.float32)
    badptsdata[10:20] = 1.0
    filtered_data = np.random.randn(signal_length).astype(np.float32)

    args = argparse.Namespace(
        infilename=infile,
        outfilename=outfile,
        filesarelists=False,
        model=DEFAULT_MODEL,
        display=False,
        verbose=debug,
    )

    mock_filter = _make_mock_dlfilter(usebadpts=True)
    mock_filter.apply.return_value = filtered_data

    columns = ["cardiacfromfmri_25.0Hz", "badpts"]
    data = np.vstack([fmridata, badptsdata])

    with (
        patch("rapidtide.workflows.applydlfilter.tide_dlfilt.DeepLearningFilter") as mock_cls,
        patch("rapidtide.workflows.applydlfilter.tide_io.readbidstsv") as mock_read,
        patch("rapidtide.workflows.applydlfilter.tide_io.writebidstsv") as mock_write,
        patch("rapidtide.workflows.applydlfilter.happy_support.checkcardmatch") as mock_check,
    ):

        mock_cls.return_value = mock_filter
        mock_read.return_value = (25.0, 0.0, columns, data, False, "tsv", {})
        mock_check.return_value = (0.85, 0.0, "")

        applydlfilter(args)

        # Verify that apply was called with badpts
        call_args = mock_filter.apply.call_args
        assert call_args[1]["badpts"] is not None
        np.testing.assert_array_equal(call_args[1]["badpts"], badptsdata)


def applydlfilter_filesarelists(testtemproot, debug=False):
    """Test applydlfilter with filesarelists=True."""
    inlistfile = os.path.join(testtemproot, "dlf_inlist.txt")
    outlistfile = os.path.join(testtemproot, "dlf_outlist.txt")

    infile1 = os.path.join(testtemproot, "dlf_list_in1.tsv")
    infile2 = os.path.join(testtemproot, "dlf_list_in2.tsv")
    outfile1 = os.path.join(testtemproot, "dlf_list_out1")
    outfile2 = os.path.join(testtemproot, "dlf_list_out2")

    # Write the list files
    with open(inlistfile, "w") as f:
        f.write(f"{infile1}\n{infile2}\n")
    with open(outlistfile, "w") as f:
        f.write(f"{outfile1}\n{outfile2}\n")

    signal_length = 100
    fmridata = np.random.randn(signal_length).astype(np.float32)
    filtered_data = np.random.randn(signal_length).astype(np.float32)

    args = argparse.Namespace(
        infilename=inlistfile,
        outfilename=outlistfile,
        filesarelists=True,
        model=DEFAULT_MODEL,
        display=False,
        verbose=debug,
    )

    mock_filter = _make_mock_dlfilter(usebadpts=False)
    mock_filter.apply.return_value = filtered_data

    columns = ["cardiacfromfmri_25.0Hz"]
    data = fmridata.reshape(1, -1)

    with (
        patch("rapidtide.workflows.applydlfilter.tide_dlfilt.DeepLearningFilter") as mock_cls,
        patch("rapidtide.workflows.applydlfilter.tide_io.readbidstsv") as mock_read,
        patch("rapidtide.workflows.applydlfilter.tide_io.writebidstsv") as mock_write,
        patch("rapidtide.workflows.applydlfilter.happy_support.checkcardmatch") as mock_check,
    ):

        mock_cls.return_value = mock_filter
        mock_read.return_value = (25.0, 0.0, columns, data, False, "tsv", {})
        mock_check.return_value = (0.90, 0.0, "")

        applydlfilter(args)

        # Should process 2 files
        assert mock_read.call_count == 2
        assert mock_filter.apply.call_count == 2
        assert mock_write.call_count == 2


def applydlfilter_mismatched_lists(testtemproot, debug=False):
    """Test applydlfilter exits when list lengths don't match."""
    inlistfile = os.path.join(testtemproot, "dlf_inlist_mismatch.txt")
    outlistfile = os.path.join(testtemproot, "dlf_outlist_mismatch.txt")

    with open(inlistfile, "w") as f:
        f.write("file1.tsv\nfile2.tsv\nfile3.tsv\n")
    with open(outlistfile, "w") as f:
        f.write("out1\nout2\n")

    args = argparse.Namespace(
        infilename=inlistfile,
        outfilename=outlistfile,
        filesarelists=True,
        model=DEFAULT_MODEL,
        display=False,
        verbose=debug,
    )

    with (
        patch("rapidtide.workflows.applydlfilter.tide_dlfilt.DeepLearningFilter"),
        patch("rapidtide.workflows.applydlfilter.sys.exit") as mock_exit,
    ):
        mock_exit.side_effect = SystemExit(0)
        try:
            applydlfilter(args)
        except SystemExit:
            pass
        mock_exit.assert_called_once()


def applydlfilter_no_usable_data(testtemproot, debug=False):
    """Test applydlfilter exits when file has no usable data column."""
    infile = os.path.join(testtemproot, "dlf_nodata_input.tsv")
    outfile = os.path.join(testtemproot, "dlf_nodata_output")

    signal_length = 100

    args = argparse.Namespace(
        infilename=infile,
        outfilename=outfile,
        filesarelists=False,
        model=DEFAULT_MODEL,
        display=False,
        verbose=debug,
    )

    mock_filter = _make_mock_dlfilter(usebadpts=False)
    columns = ["unrecognized_column"]
    data = np.random.randn(1, signal_length).astype(np.float32)

    with (
        patch("rapidtide.workflows.applydlfilter.tide_dlfilt.DeepLearningFilter") as mock_cls,
        patch("rapidtide.workflows.applydlfilter.tide_io.readbidstsv") as mock_read,
        patch("rapidtide.workflows.applydlfilter.sys.exit") as mock_exit,
    ):

        mock_cls.return_value = mock_filter
        mock_read.return_value = (25.0, 0.0, columns, data, False, "tsv", {})
        mock_exit.side_effect = SystemExit(0)

        try:
            applydlfilter(args)
        except SystemExit:
            pass
        mock_exit.assert_called_once()


def applydlfilter_wrong_samplerate(testtemproot, debug=False):
    """Test applydlfilter exits when sample rate is not 25.0 Hz."""
    infile = os.path.join(testtemproot, "dlf_wrongsr_input.tsv")
    outfile = os.path.join(testtemproot, "dlf_wrongsr_output")

    signal_length = 100
    fmridata = np.random.randn(signal_length).astype(np.float32)

    args = argparse.Namespace(
        infilename=infile,
        outfilename=outfile,
        filesarelists=False,
        model=DEFAULT_MODEL,
        display=False,
        verbose=debug,
    )

    mock_filter = _make_mock_dlfilter(usebadpts=False)
    columns = ["cardiacfromfmri_25.0Hz"]
    data = fmridata.reshape(1, -1)

    with (
        patch("rapidtide.workflows.applydlfilter.tide_dlfilt.DeepLearningFilter") as mock_cls,
        patch("rapidtide.workflows.applydlfilter.tide_io.readbidstsv") as mock_read,
        patch("rapidtide.workflows.applydlfilter.sys.exit") as mock_exit,
    ):

        mock_cls.return_value = mock_filter
        mock_read.return_value = (50.0, 0.0, columns, data, False, "tsv", {})
        mock_exit.side_effect = SystemExit(0)

        try:
            applydlfilter(args)
        except SystemExit:
            pass
        mock_exit.assert_called_once()


def applydlfilter_usebadpts_missing_badpts(testtemproot, debug=False):
    """Test applydlfilter exits when model requires badpts but file has none."""
    infile = os.path.join(testtemproot, "dlf_nobadpts_input.tsv")
    outfile = os.path.join(testtemproot, "dlf_nobadpts_output")

    signal_length = 100
    fmridata = np.random.randn(signal_length).astype(np.float32)

    args = argparse.Namespace(
        infilename=infile,
        outfilename=outfile,
        filesarelists=False,
        model=DEFAULT_MODEL,
        display=False,
        verbose=debug,
    )

    mock_filter = _make_mock_dlfilter(usebadpts=True)
    columns = ["cardiacfromfmri_25.0Hz"]
    data = fmridata.reshape(1, -1)

    with (
        patch("rapidtide.workflows.applydlfilter.tide_dlfilt.DeepLearningFilter") as mock_cls,
        patch("rapidtide.workflows.applydlfilter.tide_io.readbidstsv") as mock_read,
        patch("rapidtide.workflows.applydlfilter.sys.exit") as mock_exit,
    ):

        mock_cls.return_value = mock_filter
        mock_read.return_value = (25.0, 0.0, columns, data, False, "tsv", {})
        mock_exit.side_effect = SystemExit(0)

        try:
            applydlfilter(args)
        except SystemExit:
            pass
        mock_exit.assert_called_once()


def applydlfilter_verbose(testtemproot, debug=False):
    """Test applydlfilter with verbose=True exercises verbose print paths."""
    infile = os.path.join(testtemproot, "dlf_verbose_input.tsv")
    outfile = os.path.join(testtemproot, "dlf_verbose_output")

    signal_length = 100
    fmridata = np.random.randn(signal_length).astype(np.float32)
    filtered_data = np.random.randn(signal_length).astype(np.float32)

    args = argparse.Namespace(
        infilename=infile,
        outfilename=outfile,
        filesarelists=False,
        model=DEFAULT_MODEL,
        display=False,
        verbose=True,
    )

    mock_filter = _make_mock_dlfilter(usebadpts=False)
    mock_filter.apply.return_value = filtered_data

    columns = ["cardiacfromfmri_25.0Hz"]
    data = fmridata.reshape(1, -1)

    with (
        patch("rapidtide.workflows.applydlfilter.tide_dlfilt.DeepLearningFilter") as mock_cls,
        patch("rapidtide.workflows.applydlfilter.tide_io.readbidstsv") as mock_read,
        patch("rapidtide.workflows.applydlfilter.tide_io.writebidstsv") as mock_write,
        patch("rapidtide.workflows.applydlfilter.happy_support.checkcardmatch") as mock_check,
    ):

        mock_cls.return_value = mock_filter
        mock_read.return_value = (25.0, 0.0, columns, data, False, "tsv", {})
        mock_check.return_value = (0.90, 0.0, "")

        applydlfilter(args)

        mock_write.assert_called_once()


def applydlfilter_alternative_datanames(testtemproot, debug=False):
    """Test applydlfilter with alternative recognized data column names."""
    infile = os.path.join(testtemproot, "dlf_altnames_input.tsv")
    outfile = os.path.join(testtemproot, "dlf_altnames_output")

    signal_length = 100
    fmridata = np.random.randn(signal_length).astype(np.float32)
    filtered_data = np.random.randn(signal_length).astype(np.float32)

    args = argparse.Namespace(
        infilename=infile,
        outfilename=outfile,
        filesarelists=False,
        model=DEFAULT_MODEL,
        display=False,
        verbose=debug,
    )

    mock_filter = _make_mock_dlfilter(usebadpts=False)
    mock_filter.apply.return_value = filtered_data

    # Use an alternative recognized column name (second in datanames list)
    columns = ["normcardiac_25.0Hz"]
    data = fmridata.reshape(1, -1)

    with (
        patch("rapidtide.workflows.applydlfilter.tide_dlfilt.DeepLearningFilter") as mock_cls,
        patch("rapidtide.workflows.applydlfilter.tide_io.readbidstsv") as mock_read,
        patch("rapidtide.workflows.applydlfilter.tide_io.writebidstsv") as mock_write,
        patch("rapidtide.workflows.applydlfilter.happy_support.checkcardmatch") as mock_check,
    ):

        mock_cls.return_value = mock_filter
        mock_read.return_value = (25.0, 0.0, columns, data, False, "tsv", {})
        mock_check.return_value = (0.85, 0.0, "")

        applydlfilter(args)

        mock_write.assert_called_once()


def applydlfilter_all_columns_present(testtemproot, debug=False):
    """Test applydlfilter with data, badpts, and pleth columns all present."""
    infile = os.path.join(testtemproot, "dlf_allcols_input.tsv")
    outfile = os.path.join(testtemproot, "dlf_allcols_output")

    signal_length = 100
    fmridata = np.random.randn(signal_length).astype(np.float32)
    badptsdata = np.zeros(signal_length, dtype=np.float32)
    plethdata = np.random.randn(signal_length).astype(np.float32)
    filtered_data = np.random.randn(signal_length).astype(np.float32)

    args = argparse.Namespace(
        infilename=infile,
        outfilename=outfile,
        filesarelists=False,
        model=DEFAULT_MODEL,
        display=False,
        verbose=debug,
    )

    mock_filter = _make_mock_dlfilter(usebadpts=False)
    mock_filter.apply.return_value = filtered_data

    columns = ["cardiacfromfmri_25.0Hz", "badpts", "pleth"]
    data = np.vstack([fmridata, badptsdata, plethdata])

    with (
        patch("rapidtide.workflows.applydlfilter.tide_dlfilt.DeepLearningFilter") as mock_cls,
        patch("rapidtide.workflows.applydlfilter.tide_io.readbidstsv") as mock_read,
        patch("rapidtide.workflows.applydlfilter.tide_io.writebidstsv") as mock_write,
        patch("rapidtide.workflows.applydlfilter.happy_support.checkcardmatch") as mock_check,
    ):

        mock_cls.return_value = mock_filter
        mock_read.return_value = (25.0, 0.0, columns, data, False, "tsv", {})
        mock_check.return_value = (0.88, 0.0, "")

        applydlfilter(args)

        # checkcardmatch called 3 times when pleth is present
        assert mock_check.call_count == 3
        mock_write.assert_called_once()

        # Verify extra header info includes pleth correlations
        write_call = mock_write.call_args
        extraheaderinfo = (
            write_call[1].get("extraheaderinfo") or write_call[0][3]
            if len(write_call[0]) > 3
            else write_call[1].get("extraheaderinfo")
        )
        # The function uses keyword args for writebidstsv
        if "extraheaderinfo" in write_call[1]:
            extraheaderinfo = write_call[1]["extraheaderinfo"]
            assert "corr_rawtodlfiltered" in extraheaderinfo
            assert "corr_rawtopleth" in extraheaderinfo
            assert "corr_dlfilteredtopleth" in extraheaderinfo


def applydlfilter_writes_correct_output(testtemproot, debug=False):
    """Test that applydlfilter passes the filtered data to writebidstsv."""
    infile = os.path.join(testtemproot, "dlf_output_check_input.tsv")
    outfile = os.path.join(testtemproot, "dlf_output_check_output")

    signal_length = 100
    fmridata = np.random.randn(signal_length).astype(np.float32)
    filtered_data = np.random.randn(signal_length).astype(np.float32)

    args = argparse.Namespace(
        infilename=infile,
        outfilename=outfile,
        filesarelists=False,
        model=DEFAULT_MODEL,
        display=False,
        verbose=debug,
    )

    mock_filter = _make_mock_dlfilter(usebadpts=False)
    mock_filter.apply.return_value = filtered_data

    columns = ["cardiacfromfmri_25.0Hz"]
    data = fmridata.reshape(1, -1)

    with (
        patch("rapidtide.workflows.applydlfilter.tide_dlfilt.DeepLearningFilter") as mock_cls,
        patch("rapidtide.workflows.applydlfilter.tide_io.readbidstsv") as mock_read,
        patch("rapidtide.workflows.applydlfilter.tide_io.writebidstsv") as mock_write,
        patch("rapidtide.workflows.applydlfilter.happy_support.checkcardmatch") as mock_check,
    ):

        mock_cls.return_value = mock_filter
        mock_read.return_value = (25.0, 0.0, columns, data, False, "tsv", {})
        mock_check.return_value = (0.90, 0.0, "")

        applydlfilter(args)

        # Check that writebidstsv received the predicted data
        write_call = mock_write.call_args
        written_data = write_call[0][1]
        np.testing.assert_array_equal(written_data, filtered_data)

        # Check that the sample rate is 25.0
        written_sr = write_call[0][2]
        assert written_sr == 25.0


# ==================== Tests for DEFAULT_MODEL constant ====================


def default_model_value(debug=False):
    """Test that DEFAULT_MODEL has the expected value."""
    assert DEFAULT_MODEL == "model_cnn_pytorch"


# ==================== Main test function ====================


def test_applydlfilter(debug=False, local=False):
    # set up temp directory
    testtemproot = get_test_temp_path(local)
    create_dir(testtemproot)

    # findfirst tests
    if debug:
        print("findfirst_found_first_element()")
    findfirst_found_first_element(debug=debug)

    if debug:
        print("findfirst_found_second_element()")
    findfirst_found_second_element(debug=debug)

    if debug:
        print("findfirst_not_found()")
    findfirst_not_found(debug=debug)

    if debug:
        print("findfirst_single_element_found()")
    findfirst_single_element_found(debug=debug)

    if debug:
        print("findfirst_single_element_not_found()")
    findfirst_single_element_not_found(debug=debug)

    if debug:
        print("findfirst_empty_available()")
    findfirst_empty_available(debug=debug)

    if debug:
        print("findfirst_debug_output()")
    findfirst_debug_output(debug=debug)

    if debug:
        print("findfirst_duplicate_in_available()")
    findfirst_duplicate_in_available(debug=debug)

    # _get_parser tests
    if debug:
        print("get_parser_returns_parser()")
    get_parser_returns_parser(debug=debug)

    if debug:
        print("get_parser_default_model()")
    get_parser_default_model(debug=debug)

    if debug:
        print("get_parser_default_flags()")
    get_parser_default_flags(debug=debug)

    if debug:
        print("get_parser_prog_name()")
    get_parser_prog_name(debug=debug)

    if debug:
        print("get_parser_with_model_option(testtemproot)")
    get_parser_with_model_option(testtemproot, debug=debug)

    if debug:
        print("get_parser_with_filesarelists(testtemproot)")
    get_parser_with_filesarelists(testtemproot, debug=debug)

    if debug:
        print("get_parser_with_nodisplay(testtemproot)")
    get_parser_with_nodisplay(testtemproot, debug=debug)

    if debug:
        print("get_parser_with_verbose(testtemproot)")
    get_parser_with_verbose(testtemproot, debug=debug)

    if debug:
        print("get_parser_all_options(testtemproot)")
    get_parser_all_options(testtemproot, debug=debug)

    # applydlfilter tests
    if debug:
        print("applydlfilter_single_file(testtemproot)")
    applydlfilter_single_file(testtemproot, debug=debug)

    if debug:
        print("applydlfilter_single_file_with_pleth(testtemproot)")
    applydlfilter_single_file_with_pleth(testtemproot, debug=debug)

    if debug:
        print("applydlfilter_single_file_with_badpts(testtemproot)")
    applydlfilter_single_file_with_badpts(testtemproot, debug=debug)

    if debug:
        print("applydlfilter_filesarelists(testtemproot)")
    applydlfilter_filesarelists(testtemproot, debug=debug)

    if debug:
        print("applydlfilter_mismatched_lists(testtemproot)")
    applydlfilter_mismatched_lists(testtemproot, debug=debug)

    if debug:
        print("applydlfilter_no_usable_data(testtemproot)")
    applydlfilter_no_usable_data(testtemproot, debug=debug)

    if debug:
        print("applydlfilter_wrong_samplerate(testtemproot)")
    applydlfilter_wrong_samplerate(testtemproot, debug=debug)

    if debug:
        print("applydlfilter_usebadpts_missing_badpts(testtemproot)")
    applydlfilter_usebadpts_missing_badpts(testtemproot, debug=debug)

    if debug:
        print("applydlfilter_verbose(testtemproot)")
    applydlfilter_verbose(testtemproot, debug=debug)

    if debug:
        print("applydlfilter_alternative_datanames(testtemproot)")
    applydlfilter_alternative_datanames(testtemproot, debug=debug)

    if debug:
        print("applydlfilter_all_columns_present(testtemproot)")
    applydlfilter_all_columns_present(testtemproot, debug=debug)

    if debug:
        print("applydlfilter_writes_correct_output(testtemproot)")
    applydlfilter_writes_correct_output(testtemproot, debug=debug)

    # DEFAULT_MODEL tests
    if debug:
        print("default_model_value()")
    default_model_value(debug=debug)


if __name__ == "__main__":
    test_applydlfilter(debug=True, local=True)
