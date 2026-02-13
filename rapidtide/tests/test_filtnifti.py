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
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rapidtide.workflows.filtnifti import _get_parser, filtnifti

# ==================== Helpers ====================


def _make_default_args(
    inputfilename="dummy_input.nii.gz",
    outputfilename="dummy_output.nii.gz",
    lowestfreq=0.01,
    highestfreq=0.1,
):
    return argparse.Namespace(
        inputfilename=inputfilename,
        outputfilename=outputfilename,
        lowestfreq=lowestfreq,
        highestfreq=highestfreq,
    )


def _make_dims(xsize, ysize, numslices, timepoints):
    return np.array([4, xsize, ysize, numslices, timepoints, 1, 1, 1])


def _make_sizes(tr):
    return np.array([1.0, 2.0, 2.0, 2.0, tr, 1.0, 1.0, 1.0])


def _make_hdr(units="sec"):
    hdr = MagicMock()
    hdr.get_xyzt_units = MagicMock(return_value=("mm", units))
    return hdr


def _make_4d_data(xsize=3, ysize=3, numslices=2, timepoints=10):
    rng = np.random.RandomState(0)
    return rng.randn(xsize, ysize, numslices, timepoints).astype(np.float64)


# ==================== _get_parser tests ====================


def parser_basic(debug=False):
    if debug:
        print("parser_basic")
    parser = _get_parser()
    assert parser is not None
    assert parser.prog == "filtnifti"


def parser_required_args(debug=False):
    if debug:
        print("parser_required_args")
    parser = _get_parser()
    actions = {a.dest: a for a in parser._actions}
    assert "inputfilename" in actions
    assert "outputfilename" in actions
    assert "lowestfreq" in actions
    assert "highestfreq" in actions


def parser_parsing(debug=False):
    if debug:
        print("parser_parsing")
    parser = _get_parser()
    args = parser.parse_args(["in.nii", "out.nii", "0.01", "0.1"])
    assert args.inputfilename == "in.nii"
    assert args.outputfilename == "out.nii"
    assert args.lowestfreq == 0.01
    assert args.highestfreq == 0.1


# ==================== filtnifti tests ====================


def filtnifti_basic_sec_units(debug=False):
    if debug:
        print("filtnifti_basic_sec_units")
    data = _make_4d_data()
    xsize, ysize, numslices, timepoints = data.shape
    dims = _make_dims(xsize, ysize, numslices, timepoints)
    sizes = _make_sizes(tr=2.0)
    hdr = _make_hdr(units="sec")

    saved = {}

    def mock_readfromnifti(fname):
        return (MagicMock(), data, hdr, dims, sizes)

    def mock_savetonifti(arr, hdr_arg, fname):
        saved["fname"] = fname
        saved["data"] = arr.copy()

    mock_filter = MagicMock()
    mock_filter.apply = MagicMock(side_effect=lambda Fs, ts: ts)

    with (
        patch("rapidtide.workflows.filtnifti.tide_io.fmritimeinfo", return_value=(2.0, timepoints)),
        patch("rapidtide.workflows.filtnifti.tide_io.readfromnifti", side_effect=mock_readfromnifti),
        patch("rapidtide.workflows.filtnifti.tide_io.savetonifti", side_effect=mock_savetonifti),
        patch("rapidtide.workflows.filtnifti.tide_filt.NoncausalFilter", return_value=mock_filter),
    ):
        args = _make_default_args()
        filtnifti(args)

    assert saved["fname"] == "dummy_output.nii.gz"
    assert saved["data"].shape == data.shape
    assert mock_filter.settype.called
    assert mock_filter.setfreqs.called
    assert mock_filter.apply.call_count == xsize * ysize * numslices


def filtnifti_msec_units(debug=False):
    if debug:
        print("filtnifti_msec_units")
    data = _make_4d_data()
    xsize, ysize, numslices, timepoints = data.shape
    dims = _make_dims(xsize, ysize, numslices, timepoints)
    sizes = _make_sizes(tr=2000.0)  # msec
    hdr = _make_hdr(units="msec")

    def mock_readfromnifti(fname):
        return (MagicMock(), data, hdr, dims, sizes)

    mock_filter = MagicMock()
    mock_filter.apply = MagicMock(side_effect=lambda Fs, ts: ts)

    with (
        patch("rapidtide.workflows.filtnifti.tide_io.fmritimeinfo", return_value=(2.0, timepoints)),
        patch("rapidtide.workflows.filtnifti.tide_io.readfromnifti", side_effect=mock_readfromnifti),
        patch("rapidtide.workflows.filtnifti.tide_io.savetonifti"),
        patch("rapidtide.workflows.filtnifti.tide_filt.NoncausalFilter", return_value=mock_filter),
    ):
        args = _make_default_args()
        filtnifti(args)

    # Ensure filter applied with Fs = 1 / (2000 msec -> 2 sec) = 0.5
    first_call = mock_filter.apply.call_args_list[0]
    Fs_passed = first_call[0][0]
    assert np.isclose(Fs_passed, 0.5)


def filtnifti_negative_freqs(debug=False):
    if debug:
        print("filtnifti_negative_freqs")
    data = _make_4d_data()
    xsize, ysize, numslices, timepoints = data.shape
    dims = _make_dims(xsize, ysize, numslices, timepoints)
    sizes = _make_sizes(tr=1.0)
    hdr = _make_hdr(units="sec")

    def mock_readfromnifti(fname):
        return (MagicMock(), data, hdr, dims, sizes)

    mock_filter = MagicMock()
    mock_filter.apply = MagicMock(side_effect=lambda Fs, ts: ts)

    with (
        patch("rapidtide.workflows.filtnifti.tide_io.fmritimeinfo", return_value=(1.0, timepoints)),
        patch("rapidtide.workflows.filtnifti.tide_io.readfromnifti", side_effect=mock_readfromnifti),
        patch("rapidtide.workflows.filtnifti.tide_io.savetonifti"),
        patch("rapidtide.workflows.filtnifti.tide_filt.NoncausalFilter", return_value=mock_filter),
    ):
        args = _make_default_args(lowestfreq=-1.0, highestfreq=-1.0)
        filtnifti(args)

    # lowestfreq should be clamped to 0.0
    assert args.lowestfreq == 0.0
    # highestfreq should remain negative (per current behavior)
    assert args.highestfreq == -1.0
    # setfreqs should be called with (0.0, 0.0, -1.0, -1.0)
    mock_filter.setfreqs.assert_called_with(0.0, 0.0, -1.0, -1.0)


# ==================== Main test function ====================


def test_filtnifti(debug=False):
    if debug:
        print("Running parser tests")
    parser_basic(debug=debug)
    parser_required_args(debug=debug)
    parser_parsing(debug=debug)

    if debug:
        print("Running filtnifti tests")
    filtnifti_basic_sec_units(debug=debug)
    filtnifti_msec_units(debug=debug)
    filtnifti_negative_freqs(debug=debug)


if __name__ == "__main__":
    test_filtnifti(debug=True)
