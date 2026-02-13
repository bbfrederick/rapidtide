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
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rapidtide.workflows.pairwisemergenifti import _get_parser, pairwisemergenifti

# ==================== Helpers ====================


def _make_args(
    inputfile="in.nii.gz",
    inputmask="mask.nii.gz",
    outputfile="out.nii.gz",
    maskmerge=False,
    debug=False,
):
    return argparse.Namespace(
        inputfile=inputfile,
        inputmask=inputmask,
        outputfile=outputfile,
        maskmerge=maskmerge,
        debug=debug,
    )


def _make_hdr(tr=1.5):
    # dict is enough: code uses copy() and output_hdr["pixdim"][4] assignment
    return {"pixdim": [1.0, 2.0, 2.0, 2.0, float(tr), 1.0, 1.0, 1.0]}


def _make_dims(x, y, z, t):
    return np.array([4, x, y, z, t, 1, 1, 1], dtype=int)


# ==================== _get_parser tests ====================


def parser_basic(debug=False):
    if debug:
        print("parser_basic")
    parser = _get_parser()
    assert parser is not None
    assert parser.prog == "pairwisemergenifti"


def parser_required_args(debug=False):
    if debug:
        print("parser_required_args")
    parser = _get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def parser_defaults_and_flags(debug=False):
    if debug:
        print("parser_defaults_and_flags")
    parser = _get_parser()
    args = parser.parse_args(["in.nii.gz", "mask.nii.gz", "out.nii.gz"])
    assert args.maskmerge is False
    assert args.debug is False

    args2 = parser.parse_args(["in.nii.gz", "mask.nii.gz", "out.nii.gz", "--maskmerge", "--debug"])
    assert args2.maskmerge is True
    assert args2.debug is True


# ==================== pairwisemergenifti tests ====================


def exits_on_space_dim_mismatch(debug=False):
    if debug:
        print("exits_on_space_dim_mismatch")

    x, y, z, t = 2, 1, 1, 4
    data = np.zeros((x, y, z, t), dtype=float)
    mask = np.ones((x, y, z, t), dtype=float)
    dims = _make_dims(x, y, z, t)
    hdr = _make_hdr()

    def mock_readfromnifti(fname):
        if "mask" in fname:
            return (MagicMock(), mask, hdr, dims, None)
        return (MagicMock(), data, hdr, dims, None)

    args = _make_args()

    with (
        patch("rapidtide.workflows.pairwisemergenifti.tide_io.readfromnifti", side_effect=mock_readfromnifti),
        patch("rapidtide.workflows.pairwisemergenifti.tide_io.parseniftidims", return_value=(x, y, z, t)),
        patch("rapidtide.workflows.pairwisemergenifti.tide_io.checkspacedimmatch", return_value=False),
        patch("rapidtide.workflows.pairwisemergenifti.exit", side_effect=SystemExit),
    ):
        with pytest.raises(SystemExit):
            pairwisemergenifti(args)


def exits_on_time_dim_mismatch(debug=False):
    if debug:
        print("exits_on_time_dim_mismatch")

    x, y, z, t = 2, 1, 1, 4
    data = np.zeros((x, y, z, t), dtype=float)
    mask = np.ones((x, y, z, t), dtype=float)
    dims = _make_dims(x, y, z, t)
    hdr = _make_hdr()

    def mock_readfromnifti(fname):
        if "mask" in fname:
            return (MagicMock(), mask, hdr, dims, None)
        return (MagicMock(), data, hdr, dims, None)

    args = _make_args()

    with (
        patch("rapidtide.workflows.pairwisemergenifti.tide_io.readfromnifti", side_effect=mock_readfromnifti),
        patch("rapidtide.workflows.pairwisemergenifti.tide_io.parseniftidims", return_value=(x, y, z, t)),
        patch("rapidtide.workflows.pairwisemergenifti.tide_io.checkspacedimmatch", return_value=True),
        patch("rapidtide.workflows.pairwisemergenifti.tide_io.checktimematch", return_value=False),
        patch("rapidtide.workflows.pairwisemergenifti.exit", side_effect=SystemExit),
    ):
        with pytest.raises(SystemExit):
            pairwisemergenifti(args)


def exits_on_odd_timepoints(debug=False):
    if debug:
        print("exits_on_odd_timepoints")

    x, y, z, t = 2, 1, 1, 3  # odd
    data = np.zeros((x, y, z, t), dtype=float)
    mask = np.ones((x, y, z, t), dtype=float)
    dims = _make_dims(x, y, z, t)
    hdr = _make_hdr()

    def mock_readfromnifti(fname):
        if "mask" in fname:
            return (MagicMock(), mask, hdr, dims, None)
        return (MagicMock(), data, hdr, dims, None)

    args = _make_args()

    with (
        patch("rapidtide.workflows.pairwisemergenifti.tide_io.readfromnifti", side_effect=mock_readfromnifti),
        patch("rapidtide.workflows.pairwisemergenifti.tide_io.parseniftidims", return_value=(x, y, z, t)),
        patch("rapidtide.workflows.pairwisemergenifti.tide_io.checkspacedimmatch", return_value=True),
        patch("rapidtide.workflows.pairwisemergenifti.tide_io.checktimematch", return_value=True),
        patch("rapidtide.workflows.pairwisemergenifti.exit", side_effect=SystemExit),
    ):
        with pytest.raises(SystemExit):
            pairwisemergenifti(args)


def merges_mask_when_maskmerge_true(debug=False):
    if debug:
        print("merges_mask_when_maskmerge_true")

    x, y, z, t = 2, 1, 1, 4
    # input data irrelevant for maskmerge=True, but must exist
    data = np.random.RandomState(0).randn(x, y, z, t).astype(float)

    # mask timepoints:
    # voxel0: [0,1,0,0] ; voxel1: [0,0,2,0]
    mask = np.zeros((x, y, z, t), dtype=float)
    mask[0, 0, 0, 1] = 1.0
    mask[1, 0, 0, 2] = 2.0

    dims = _make_dims(x, y, z, t)
    hdr = _make_hdr(tr=2.0)

    saved = {}

    def mock_readfromnifti(fname):
        if "mask" in fname:
            return (MagicMock(), mask, hdr, dims, None)
        return (MagicMock(), data, hdr, dims, None)

    def mock_savetonifti(arr, hdr_arg, outroot):
        saved["arr"] = np.array(arr, copy=True)
        saved["hdr"] = hdr_arg
        saved["outroot"] = outroot

    args = _make_args(maskmerge=True, outputfile="out.nii.gz")

    with (
        patch("rapidtide.workflows.pairwisemergenifti.tide_io.readfromnifti", side_effect=mock_readfromnifti),
        patch("rapidtide.workflows.pairwisemergenifti.tide_io.parseniftidims", return_value=(x, y, z, t)),
        patch("rapidtide.workflows.pairwisemergenifti.tide_io.checkspacedimmatch", return_value=True),
        patch("rapidtide.workflows.pairwisemergenifti.tide_io.checktimematch", return_value=True),
        patch("rapidtide.workflows.pairwisemergenifti.tide_io.niftisplitext", return_value=("out", ".nii.gz")),
        patch("rapidtide.workflows.pairwisemergenifti.tide_io.savetonifti", side_effect=mock_savetonifti),
    ):
        pairwisemergenifti(args)

    # Pair 0 merges timepoints (0,1), binarized sum:
    # voxel0: 0+1 -> 1 ; voxel1: 0+0 -> 0
    # Pair 1 merges timepoints (2,3), binarized sum:
    # voxel0: 0+0 -> 0 ; voxel1: 2+0 -> 1
    expected = np.zeros((x, y, z, t // 2), dtype=float)
    expected[0, 0, 0, 0] = 1.0
    expected[1, 0, 0, 1] = 1.0

    np.testing.assert_allclose(saved["arr"], expected)
    assert saved["outroot"] == "out"
    assert saved["hdr"]["pixdim"][4] == t // 2  # time pixdim set to n_timepoints_out


def merges_weighted_average_when_maskmerge_false(debug=False):
    if debug:
        print("merges_weighted_average_when_maskmerge_false")

    x, y, z, t = 2, 1, 1, 4
    # input data:
    # voxel0: [10, 30, 100, 200]
    # voxel1: [ 5, 15,  50,  60]
    data = np.zeros((x, y, z, t), dtype=float)
    data[0, 0, 0, :] = np.array([10.0, 30.0, 100.0, 200.0])
    data[1, 0, 0, :] = np.array([5.0, 15.0, 50.0, 60.0])

    # mask weights:
    # voxel0: [1,1,0,0] -> pair0 masksum=2, pair1 masksum=0
    # voxel1: [0,2,1,1] -> pair0 masksum=2, pair1 masksum=2
    mask = np.zeros((x, y, z, t), dtype=float)
    mask[0, 0, 0, 0] = 1.0
    mask[0, 0, 0, 1] = 1.0

    mask[1, 0, 0, 1] = 2.0
    mask[1, 0, 0, 2] = 1.0
    mask[1, 0, 0, 3] = 1.0

    dims = _make_dims(x, y, z, t)
    hdr = _make_hdr(tr=1.0)

    saved = {}

    def mock_readfromnifti(fname):
        if "mask" in fname:
            return (MagicMock(), mask, hdr, dims, None)
        return (MagicMock(), data, hdr, dims, None)

    def mock_savetonifti(arr, hdr_arg, outroot):
        saved["arr"] = np.array(arr, copy=True)
        saved["hdr"] = hdr_arg
        saved["outroot"] = outroot

    args = _make_args(maskmerge=False, outputfile="out.nii.gz")

    with (
        patch("rapidtide.workflows.pairwisemergenifti.tide_io.readfromnifti", side_effect=mock_readfromnifti),
        patch("rapidtide.workflows.pairwisemergenifti.tide_io.parseniftidims", return_value=(x, y, z, t)),
        patch("rapidtide.workflows.pairwisemergenifti.tide_io.checkspacedimmatch", return_value=True),
        patch("rapidtide.workflows.pairwisemergenifti.tide_io.checktimematch", return_value=True),
        patch("rapidtide.workflows.pairwisemergenifti.tide_io.niftisplitext", return_value=("out", ".nii.gz")),
        patch("rapidtide.workflows.pairwisemergenifti.tide_io.savetonifti", side_effect=mock_savetonifti),
    ):
        pairwisemergenifti(args)

    # Pair 0:
    # voxel0: (10+30)/2 = 20
    # voxel1: (5+15)/2 = 10  (masksum=0+2 -> 2)
    # Pair 1:
    # voxel0: (100+200)/0 -> 0 by where=masksum>0
    # voxel1: (50+60)/2 = 55
    expected = np.zeros((x, y, z, t // 2), dtype=float)
    expected[0, 0, 0, 0] = 20.0
    expected[1, 0, 0, 0] = 10.0
    expected[0, 0, 0, 1] = 0.0
    expected[1, 0, 0, 1] = 55.0

    np.testing.assert_allclose(saved["arr"], expected)
    assert saved["outroot"] == "out"
    assert saved["hdr"]["pixdim"][4] == t // 2


# ==================== Main test function ====================


def test_pairwisemergenifti(debug=False):
    # parser tests
    parser_basic(debug=debug)
    parser_required_args(debug=debug)
    parser_defaults_and_flags(debug=debug)

    # workflow tests
    exits_on_space_dim_mismatch(debug=debug)
    exits_on_time_dim_mismatch(debug=debug)
    exits_on_odd_timepoints(debug=debug)
    merges_mask_when_maskmerge_true(debug=debug)
    merges_weighted_average_when_maskmerge_false(debug=debug)


if __name__ == "__main__":
    test_pairwisemergenifti(debug=True)