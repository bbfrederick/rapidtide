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

import rapidtide.workflows.rankimage as ri

# ==================== Helpers ====================


def _make_args(inputfilename="in.nii.gz", maskfilename="mask.nii.gz", outputroot="out", debug=False):
    return argparse.Namespace(
        inputfilename=inputfilename,
        maskfilename=maskfilename,
        outputroot=outputroot,
        debug=debug,
    )


def _make_dims(x, y, z, t=1):
    return np.array([4, x, y, z, t, 1, 1, 1], dtype=int)


# ==================== _get_parser tests ====================


def parser_basic(debug=False):
    if debug:
        print("parser_basic")
    parser = ri._get_parser()
    assert parser is not None
    assert parser.prog == "rankimage"


def parser_required_args(debug=False):
    if debug:
        print("parser_required_args")
    parser = ri._get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def parser_defaults(debug=False):
    if debug:
        print("parser_defaults")
    parser = ri._get_parser()
    args = parser.parse_args(["in.nii.gz", "mask.nii.gz", "outroot"])
    assert args.debug is False


# ==================== imtopercentile tests ====================


def imtopercentile_basic(debug=False):
    if debug:
        print("imtopercentile_basic")

    # image with 4 valid voxels: values [0, 1, 2, 3] should map to [0, 33.33, 66.66, 100]
    img = np.array(
        [
            [[0.0], [1.0]],
            [[2.0], [3.0]],
        ],
        dtype=float,
    )  # shape (2,2,1)
    mask = np.ones_like(img)
    out = ri.imtopercentile(img, mask, debug=False)

    assert out.shape == img.shape
    assert np.min(out) >= 0.0
    assert np.max(out) <= 100.0
    # exact expected using dense ranks for unique values
    expected = np.array(
        [
            [[0.0], [100.0 / 3.0]],
            [[200.0 / 3.0], [100.0]],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)


def imtopercentile_masked_voxels_are_zero(debug=False):
    if debug:
        print("imtopercentile_masked_voxels_are_zero")

    img = np.random.RandomState(0).randn(3, 3, 1).astype(float)
    mask = np.ones_like(img)
    mask[0, :, :] = 0.0  # mask out first row
    out = ri.imtopercentile(img, mask, debug=False)
    assert np.all(out[0, :, :] == 0.0)


def imtopercentile_ties_dense_ranking(debug=False):
    if debug:
        print("imtopercentile_ties_dense_ranking")

    # valid voxels values: [1, 1, 2, 2] -> dense ranks [1,1,2,2]
    # percentiles: (rank-1)/(N-1)*100 with N=4 => [0,0,33.333,33.333]
    img = np.array([[[1.0], [1.0]], [[2.0], [2.0]]], dtype=float)  # (2,2,1)
    mask = np.ones_like(img)
    out = ri.imtopercentile(img, mask, debug=False)

    expected = np.array([[[0.0], [0.0]], [[100.0 / 3.0], [100.0 / 3.0]]], dtype=float)
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)


# ==================== rankimage workflow tests ====================


def rankimage_exits_on_dim_mismatch(debug=False):
    if debug:
        print("rankimage_exits_on_dim_mismatch")

    img = np.zeros((2, 2, 1), dtype=float)
    mask = np.zeros((3, 3, 1), dtype=float)
    img_dims = _make_dims(2, 2, 1, 1)
    mask_dims = _make_dims(3, 3, 1, 1)
    hdr = MagicMock()

    def mock_readfromnifti(fname):
        if "mask" in fname:
            return (MagicMock(), mask, hdr, mask_dims, None)
        return (MagicMock(), img, hdr, img_dims, None)

    args = _make_args()

    with (
        patch("rapidtide.workflows.rankimage.tide_io.readfromnifti", side_effect=mock_readfromnifti),
        patch("rapidtide.workflows.rankimage.tide_io.checkspacedimmatch", return_value=False),
        patch("rapidtide.workflows.rankimage.exit", side_effect=SystemExit),
    ):
        with pytest.raises(SystemExit):
            ri.rankimage(args)


def rankimage_3d_writes_outputs(debug=False):
    if debug:
        print("rankimage_3d_writes_outputs")

    img = np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=float)
    mask = np.ones_like(img)
    img_dims = _make_dims(2, 2, 1, 1)
    hdr = MagicMock()

    saved = {}

    def mock_readfromnifti(fname):
        if "mask" in fname:
            return (MagicMock(), mask, hdr, img_dims, None)
        return (MagicMock(), img, hdr, img_dims, None)

    def mock_savetonifti(arr, hdr_arg, savename):
        saved["nifti_name"] = savename
        saved["arr"] = np.array(arr, copy=True)

    def mock_writedicttojson(d, fname):
        saved["json_name"] = fname
        saved["json"] = d

    args = _make_args(outputroot="outroot", debug=False)

    with (
        patch("rapidtide.workflows.rankimage.tide_io.readfromnifti", side_effect=mock_readfromnifti),
        patch("rapidtide.workflows.rankimage.tide_io.checkspacedimmatch", return_value=True),
        patch("rapidtide.workflows.rankimage.tide_io.parseniftisizes", return_value=(2, 2, 1, 1)),
        patch("rapidtide.workflows.rankimage.tide_io.savetonifti", side_effect=mock_savetonifti),
        patch("rapidtide.workflows.rankimage.tide_io.writedicttojson", side_effect=mock_writedicttojson),
    ):
        ri.rankimage(args)

    assert saved["nifti_name"] == "outroot"
    assert saved["json_name"] == "outroot.json"
    assert saved["json"]["Units"] == "percentile"
    assert saved["json"]["RawSources"] == ["in.nii.gz", "mask.nii.gz"]
    assert saved["arr"].shape == img.shape
    assert np.min(saved["arr"]) >= 0.0
    assert np.max(saved["arr"]) <= 100.0


def rankimage_4d_processes_each_timepoint(debug=False):
    if debug:
        print("rankimage_4d_processes_each_timepoint")

    img = np.zeros((2, 2, 1, 2), dtype=float)
    img[:, :, :, 0] = np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=float)
    img[:, :, :, 1] = np.array([[[10.0], [20.0]], [[30.0], [40.0]]], dtype=float)
    mask = np.ones_like(img)

    dims = _make_dims(2, 2, 1, 2)
    hdr = MagicMock()

    saved = {}

    def mock_readfromnifti(fname):
        if "mask" in fname:
            return (MagicMock(), mask, hdr, dims, None)
        return (MagicMock(), img, hdr, dims, None)

    def mock_savetonifti(arr, hdr_arg, savename):
        saved["arr"] = np.array(arr, copy=True)

    args = _make_args(outputroot="outroot")

    with (
        patch("rapidtide.workflows.rankimage.tide_io.readfromnifti", side_effect=mock_readfromnifti),
        patch("rapidtide.workflows.rankimage.tide_io.checkspacedimmatch", return_value=True),
        patch("rapidtide.workflows.rankimage.tide_io.parseniftisizes", return_value=(2, 2, 1, 2)),
        patch("rapidtide.workflows.rankimage.tide_io.savetonifti", side_effect=mock_savetonifti),
        patch("rapidtide.workflows.rankimage.tide_io.writedicttojson"),
    ):
        ri.rankimage(args)

    assert saved["arr"].shape == img.shape
    # Each volume should independently span 0..100 for this monotonic data
    assert np.isclose(np.min(saved["arr"][:, :, :, 0]), 0.0)
    assert np.isclose(np.max(saved["arr"][:, :, :, 0]), 100.0)
    assert np.isclose(np.min(saved["arr"][:, :, :, 1]), 0.0)
    assert np.isclose(np.max(saved["arr"][:, :, :, 1]), 100.0)


# ==================== Main test function ====================


def test_rankimage(debug=False):
    # parser tests
    parser_basic(debug=debug)
    parser_required_args(debug=debug)
    parser_defaults(debug=debug)

    # imtopercentile tests
    imtopercentile_basic(debug=debug)
    imtopercentile_masked_voxels_are_zero(debug=debug)
    imtopercentile_ties_dense_ranking(debug=debug)

    # rankimage workflow tests
    rankimage_exits_on_dim_mismatch(debug=debug)
    rankimage_3d_writes_outputs(debug=debug)
    rankimage_4d_processes_each_timepoint(debug=debug)


if __name__ == "__main__":
    test_rankimage(debug=True)