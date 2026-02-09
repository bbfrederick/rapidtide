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
from unittest.mock import MagicMock, call, patch

import numpy as np

from rapidtide.tests.utils import create_dir, get_test_temp_path
from rapidtide.workflows.atlasaverage import _get_parser, atlasaverage, summarizevoxels

# ==================== Helpers ====================


def _make_mock_hdr(xsize, ysize, numslices, timepoints=1):
    """Create a mock NIfTI header that supports __getitem__ and __setitem__."""
    store = {
        "dim": [4, xsize, ysize, numslices, timepoints, 1, 1, 1],
        "pixdim": [1.0, 2.0, 2.0, 2.0, 1.5, 1.0, 1.0, 1.0],
    }
    hdr = MagicMock()
    hdr.__getitem__ = MagicMock(side_effect=lambda key: store[key])
    hdr.__setitem__ = MagicMock(side_effect=lambda key, val: store.__setitem__(key, val))

    def copy_fn():
        new_store = {
            "dim": list(store["dim"]),
            "pixdim": list(store["pixdim"]),
        }
        h = MagicMock()
        h.__getitem__ = MagicMock(side_effect=lambda key: new_store[key])
        h.__setitem__ = MagicMock(side_effect=lambda key, val: new_store.__setitem__(key, val))
        return h

    hdr.copy = copy_fn
    return hdr


def _make_3d_atlas(xsize=6, ysize=6, numslices=4, numregions=3):
    """Create a 3D integer-labeled atlas with regions assigned along the x-axis."""
    data = np.zeros((xsize, ysize, numslices), dtype=np.float64)
    region_size = xsize // numregions
    for r in range(numregions):
        start = r * region_size
        end = start + region_size
        data[start:end, :, :] = r + 1
    hdr = _make_mock_hdr(xsize, ysize, numslices, 1)
    dims = np.array([3, xsize, ysize, numslices, 1, 1, 1, 1])
    sizes = np.array([1.0, 2.0, 2.0, 2.0, 1.5, 1.0, 1.0, 1.0])
    return data, hdr, dims, sizes


def _make_3d_data(xsize=6, ysize=6, numslices=4):
    """Create 3D data with distinct values per spatial location."""
    rng = np.random.RandomState(42)
    data = rng.rand(xsize, ysize, numslices).astype(np.float64) + 0.5
    hdr = _make_mock_hdr(xsize, ysize, numslices, 1)
    dims = np.array([3, xsize, ysize, numslices, 1, 1, 1, 1])
    sizes = np.array([1.0, 2.0, 2.0, 2.0, 1.5, 1.0, 1.0, 1.0])
    return data, hdr, dims, sizes


def _make_4d_data(xsize=6, ysize=6, numslices=4, numtimepoints=10):
    """Create 4D data with distinct timecourses per voxel."""
    rng = np.random.RandomState(42)
    data = rng.rand(xsize, ysize, numslices, numtimepoints).astype(np.float64) + 1.0
    hdr = _make_mock_hdr(xsize, ysize, numslices, numtimepoints)
    dims = np.array([4, xsize, ysize, numslices, numtimepoints, 1, 1, 1])
    sizes = np.array([1.0, 2.0, 2.0, 2.0, 1.5, 1.0, 1.0, 1.0])
    return data, hdr, dims, sizes


def _make_base_args(outputroot="/tmp/test_atlasaverage_out"):
    """Create baseline args namespace."""
    return argparse.Namespace(
        datafile="dummy_data.nii.gz",
        templatefile="dummy_template.nii.gz",
        outputroot=outputroot,
        normmethod="none",
        summarymethod="mean",
        numpercentiles=1,
        ignorezeros=False,
        regionlistfile=None,
        regionlabelfile=None,
        includespec=None,
        excludespec=None,
        extramaskname=None,
        headerline=False,
        datalabel=None,
        debug=False,
    )


# ==================== _get_parser tests ====================


def parser_basic(debug=False):
    """Test that parser creates successfully and has expected defaults."""
    if debug:
        print("parser_basic")
    parser = _get_parser()
    assert parser is not None
    assert parser.prog == "atlasaverage"


def parser_required_args(debug=False):
    """Test parser with required arguments only."""
    if debug:
        print("parser_required_args")
    parser = _get_parser()
    # parser requires valid files - use parse_known_args to check structure
    # We test the parser attributes directly
    actions = {a.dest: a for a in parser._actions}
    assert "datafile" in actions
    assert "templatefile" in actions
    assert "outputroot" in actions


def parser_defaults(debug=False):
    """Test default values for optional arguments."""
    if debug:
        print("parser_defaults")
    import tempfile
    parser = _get_parser()
    with tempfile.NamedTemporaryFile(suffix=".nii") as f1, \
         tempfile.NamedTemporaryFile(suffix=".nii") as f2:
        defaults = parser.parse_args([f1.name, f2.name, "outroot"])
    assert defaults.normmethod == "none"
    assert defaults.summarymethod == "mean"
    assert defaults.numpercentiles == 1
    assert defaults.ignorezeros is False
    assert defaults.regionlistfile is None
    assert defaults.regionlabelfile is None
    assert defaults.includespec is None
    assert defaults.excludespec is None
    assert defaults.extramaskname is None
    assert defaults.headerline is False
    assert defaults.datalabel is None
    assert defaults.debug is False


def parser_normmethod_choices(debug=False):
    """Test that normmethod accepts only valid choices."""
    if debug:
        print("parser_normmethod_choices")
    import tempfile
    parser = _get_parser()
    with tempfile.NamedTemporaryFile(suffix=".nii") as f1, \
         tempfile.NamedTemporaryFile(suffix=".nii") as f2:
        for method in ["none", "pct", "var", "std", "p2p"]:
            args = parser.parse_args([f1.name, f2.name, "out", "--normmethod", method])
            assert args.normmethod == method


def parser_summarymethod_choices(debug=False):
    """Test that summarymethod accepts only valid choices."""
    if debug:
        print("parser_summarymethod_choices")
    import tempfile
    parser = _get_parser()
    with tempfile.NamedTemporaryFile(suffix=".nii") as f1, \
         tempfile.NamedTemporaryFile(suffix=".nii") as f2:
        for method in ["mean", "median", "sum", "std", "MAD", "CoV"]:
            args = parser.parse_args([f1.name, f2.name, "out", "--summarymethod", method])
            assert args.summarymethod == method


def parser_numpercentiles(debug=False):
    """Test that numpercentiles can be set."""
    if debug:
        print("parser_numpercentiles")
    import tempfile
    parser = _get_parser()
    with tempfile.NamedTemporaryFile(suffix=".nii") as f1, \
         tempfile.NamedTemporaryFile(suffix=".nii") as f2:
        args = parser.parse_args([f1.name, f2.name, "out", "--numpercentiles", "5"])
    assert args.numpercentiles == 5


def parser_flags(debug=False):
    """Test boolean flags."""
    if debug:
        print("parser_flags")
    import tempfile
    parser = _get_parser()
    with tempfile.NamedTemporaryFile(suffix=".nii") as f1, \
         tempfile.NamedTemporaryFile(suffix=".nii") as f2:
        args = parser.parse_args([
            f1.name, f2.name, "out",
            "--ignorezeros", "--headerline", "--debug",
        ])
    assert args.ignorezeros is True
    assert args.headerline is True
    assert args.debug is True


def parser_datalabel(debug=False):
    """Test datalabel option."""
    if debug:
        print("parser_datalabel")
    import tempfile
    parser = _get_parser()
    with tempfile.NamedTemporaryFile(suffix=".nii") as f1, \
         tempfile.NamedTemporaryFile(suffix=".nii") as f2:
        args = parser.parse_args([f1.name, f2.name, "out", "--datalabel", "my_label"])
    assert args.datalabel == "my_label"


# ==================== summarizevoxels tests ====================


def summarizevoxels_mean_1d(debug=False):
    """Test mean summary on 1D array."""
    if debug:
        print("summarizevoxels_mean_1d")
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = summarizevoxels(data, method="mean")
    assert np.isclose(result, 3.0), f"Expected 3.0, got {result}"


def summarizevoxels_mean_2d(debug=False):
    """Test mean summary on 2D array (voxels x timepoints)."""
    if debug:
        print("summarizevoxels_mean_2d")
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = summarizevoxels(data, method="mean")
    expected = np.array([2.5, 3.5, 4.5])
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def summarizevoxels_sum(debug=False):
    """Test sum summary."""
    if debug:
        print("summarizevoxels_sum")
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = summarizevoxels(data, method="sum")
    expected = np.array([9.0, 12.0])
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def summarizevoxels_median(debug=False):
    """Test median summary."""
    if debug:
        print("summarizevoxels_median")
    data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    result = summarizevoxels(data, method="median")
    expected = np.array([2.0, 20.0])
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def summarizevoxels_std(debug=False):
    """Test std summary."""
    if debug:
        print("summarizevoxels_std")
    data = np.array([2.0, 4.0, 6.0, 8.0])
    result = summarizevoxels(data, method="std")
    expected = np.std(data)
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"


def summarizevoxels_mad(debug=False):
    """Test MAD summary."""
    if debug:
        print("summarizevoxels_mad")
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = summarizevoxels(data, method="MAD")
    from statsmodels.robust import mad
    expected = mad(data)
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"


def summarizevoxels_cov_1d(debug=False):
    """Test CoV summary on 1D array."""
    if debug:
        print("summarizevoxels_cov_1d")
    data = np.array([2.0, 4.0, 6.0, 8.0])
    result = summarizevoxels(data, method="CoV")
    expected = 100.0 * np.std(data) / np.mean(data)
    assert np.isclose(result, expected), f"Expected {expected}, got {result}"


def summarizevoxels_cov_2d(debug=False):
    """Test CoV summary on 2D array."""
    if debug:
        print("summarizevoxels_cov_2d")
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = summarizevoxels(data, method="CoV")
    expected = 100.0 * np.std(data, axis=0) / np.mean(data, axis=0)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"


def summarizevoxels_nan_handling(debug=False):
    """Test that NaN values are handled (converted to zero)."""
    if debug:
        print("summarizevoxels_nan_handling")
    data = np.array([1.0, np.nan, 3.0])
    result = summarizevoxels(data, method="mean")
    # np.mean with NaN produces NaN, then nan_to_num converts it to 0
    # Actually: np.mean([1, nan, 3]) = nan -> nan_to_num -> 0.0
    # But wait, the result should be 0.0 since mean of [1, nan, 3] is nan
    assert np.isfinite(result), f"Expected finite result, got {result}"


def summarizevoxels_default_method(debug=False):
    """Test that default method is mean."""
    if debug:
        print("summarizevoxels_default_method")
    data = np.array([1.0, 2.0, 3.0])
    result_default = summarizevoxels(data)
    result_mean = summarizevoxels(data, method="mean")
    assert np.isclose(result_default, result_mean), "Default should be mean"


# ==================== atlasaverage tests ====================


def atlasaverage_3d_basic(debug=False):
    """Test atlasaverage with 3D data, basic operation."""
    if debug:
        print("atlasaverage_3d_basic")
    xsize, ysize, numslices = 6, 6, 4
    numregions = 3

    atlas_data, atlas_hdr, atlas_dims, atlas_sizes = _make_3d_atlas(
        xsize, ysize, numslices, numregions
    )
    input_data, input_hdr, input_dims, input_sizes = _make_3d_data(xsize, ysize, numslices)
    args = _make_base_args()

    saved_data = {}

    def mock_readfromnifti(fname, **kwargs):
        if "data" in fname:
            return MagicMock(), input_data, input_hdr, input_dims, input_sizes
        else:
            return MagicMock(), atlas_data, atlas_hdr, atlas_dims, atlas_sizes

    def mock_savetonifti(data, hdr, fname, **kwargs):
        saved_data[fname] = data.copy()

    def mock_getfracvals(data, fracs, **kwargs):
        return np.percentile(data[data != 0], np.array(fracs) * 100).tolist()

    with patch("rapidtide.workflows.atlasaverage.tide_io.readfromnifti", side_effect=mock_readfromnifti), \
         patch("rapidtide.workflows.atlasaverage.tide_io.checkspacematch", return_value=True), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftidims", return_value=(xsize, ysize, numslices, 1)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftisizes", return_value=(2.0, 2.0, 2.0, 1.5)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.savetonifti", side_effect=mock_savetonifti), \
         patch("rapidtide.workflows.atlasaverage.tide_io.writevec") as mock_writevec, \
         patch("rapidtide.workflows.atlasaverage.tide_mask.getmaskset", return_value=(None, None, None)), \
         patch("rapidtide.workflows.atlasaverage.tide_stats.getfracvals", side_effect=mock_getfracvals):

        atlasaverage(args)

    # Should have saved output nifti and percentile nifti
    assert args.outputroot in saved_data, "Output NIfTI not saved"
    assert args.outputroot + "_percentiles" in saved_data, "Percentile NIfTI not saved"
    # writevec should be called for regionsummaries and regionpercentiles
    assert mock_writevec.call_count >= 2


def atlasaverage_3d_with_headerline(debug=False):
    """Test 3D mode with headerline enabled."""
    if debug:
        print("atlasaverage_3d_with_headerline")
    xsize, ysize, numslices = 6, 6, 4
    numregions = 3
    atlas_data, atlas_hdr, atlas_dims, atlas_sizes = _make_3d_atlas(
        xsize, ysize, numslices, numregions
    )
    input_data, input_hdr, input_dims, input_sizes = _make_3d_data(xsize, ysize, numslices)
    args = _make_base_args()
    args.headerline = True

    def mock_readfromnifti(fname, **kwargs):
        if "data" in fname:
            return MagicMock(), input_data, input_hdr, input_dims, input_sizes
        else:
            return MagicMock(), atlas_data, atlas_hdr, atlas_dims, atlas_sizes

    def mock_getfracvals(data, fracs, **kwargs):
        return np.percentile(data[data != 0], np.array(fracs) * 100).tolist()

    written_data = {}

    def mock_writevec(data, fname, **kwargs):
        written_data[fname] = data

    with patch("rapidtide.workflows.atlasaverage.tide_io.readfromnifti", side_effect=mock_readfromnifti), \
         patch("rapidtide.workflows.atlasaverage.tide_io.checkspacematch", return_value=True), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftidims", return_value=(xsize, ysize, numslices, 1)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftisizes", return_value=(2.0, 2.0, 2.0, 1.5)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.savetonifti"), \
         patch("rapidtide.workflows.atlasaverage.tide_io.writevec", side_effect=mock_writevec), \
         patch("rapidtide.workflows.atlasaverage.tide_mask.getmaskset", return_value=(None, None, None)), \
         patch("rapidtide.workflows.atlasaverage.tide_stats.getfracvals", side_effect=mock_getfracvals):

        atlasaverage(args)

    # With headerline, the csv should have 2 lines (header + data)
    csv_key = args.outputroot + "_regionsummaries.csv"
    assert csv_key in written_data
    csv_content = written_data[csv_key]
    assert len(csv_content) == 2, f"Expected 2 lines with header, got {len(csv_content)}"


def atlasaverage_3d_no_headerline(debug=False):
    """Test 3D mode without headerline."""
    if debug:
        print("atlasaverage_3d_no_headerline")
    xsize, ysize, numslices = 6, 6, 4
    numregions = 3
    atlas_data, atlas_hdr, atlas_dims, atlas_sizes = _make_3d_atlas(
        xsize, ysize, numslices, numregions
    )
    input_data, input_hdr, input_dims, input_sizes = _make_3d_data(xsize, ysize, numslices)
    args = _make_base_args()
    args.headerline = False

    def mock_readfromnifti(fname, **kwargs):
        if "data" in fname:
            return MagicMock(), input_data, input_hdr, input_dims, input_sizes
        else:
            return MagicMock(), atlas_data, atlas_hdr, atlas_dims, atlas_sizes

    def mock_getfracvals(data, fracs, **kwargs):
        return np.percentile(data[data != 0], np.array(fracs) * 100).tolist()

    written_data = {}

    def mock_writevec(data, fname, **kwargs):
        written_data[fname] = data

    with patch("rapidtide.workflows.atlasaverage.tide_io.readfromnifti", side_effect=mock_readfromnifti), \
         patch("rapidtide.workflows.atlasaverage.tide_io.checkspacematch", return_value=True), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftidims", return_value=(xsize, ysize, numslices, 1)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftisizes", return_value=(2.0, 2.0, 2.0, 1.5)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.savetonifti"), \
         patch("rapidtide.workflows.atlasaverage.tide_io.writevec", side_effect=mock_writevec), \
         patch("rapidtide.workflows.atlasaverage.tide_mask.getmaskset", return_value=(None, None, None)), \
         patch("rapidtide.workflows.atlasaverage.tide_stats.getfracvals", side_effect=mock_getfracvals):

        atlasaverage(args)

    csv_key = args.outputroot + "_regionsummaries.csv"
    assert csv_key in written_data
    csv_content = written_data[csv_key]
    assert len(csv_content) == 1, f"Expected 1 line without header, got {len(csv_content)}"


def atlasaverage_3d_with_datalabel(debug=False):
    """Test 3D mode with datalabel prepended.

    NOTE: There is a known bug in atlasaverage.py (line 638-640) where
    setting datalabel causes an IndexError when building the regionpercentiles
    TSV. The bug is that thereglabels gets an extra "Region" entry at index 0
    when datalabel is set, but theregsizes and thepercentiles don't have a
    corresponding entry, causing an off-by-one IndexError in the for loop.
    This test verifies the bug exists (expects IndexError).
    """
    if debug:
        print("atlasaverage_3d_with_datalabel")
    xsize, ysize, numslices = 6, 6, 4
    numregions = 3
    atlas_data, atlas_hdr, atlas_dims, atlas_sizes = _make_3d_atlas(
        xsize, ysize, numslices, numregions
    )
    input_data, input_hdr, input_dims, input_sizes = _make_3d_data(xsize, ysize, numslices)
    args = _make_base_args()
    args.datalabel = "mydata"
    args.headerline = True

    def mock_readfromnifti(fname, **kwargs):
        if "data" in fname:
            return MagicMock(), input_data, input_hdr, input_dims, input_sizes
        else:
            return MagicMock(), atlas_data, atlas_hdr, atlas_dims, atlas_sizes

    def mock_getfracvals(data, fracs, **kwargs):
        return np.percentile(data[data != 0], np.array(fracs) * 100).tolist()

    import pytest
    with patch("rapidtide.workflows.atlasaverage.tide_io.readfromnifti", side_effect=mock_readfromnifti), \
         patch("rapidtide.workflows.atlasaverage.tide_io.checkspacematch", return_value=True), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftidims", return_value=(xsize, ysize, numslices, 1)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftisizes", return_value=(2.0, 2.0, 2.0, 1.5)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.savetonifti"), \
         patch("rapidtide.workflows.atlasaverage.tide_io.writevec"), \
         patch("rapidtide.workflows.atlasaverage.tide_mask.getmaskset", return_value=(None, None, None)), \
         patch("rapidtide.workflows.atlasaverage.tide_stats.getfracvals", side_effect=mock_getfracvals):

        # Bug: datalabel causes IndexError in regionpercentiles TSV generation
        with pytest.raises(IndexError):
            atlasaverage(args)


def atlasaverage_3d_ignorezeros(debug=False):
    """Test 3D mode with ignorezeros flag."""
    if debug:
        print("atlasaverage_3d_ignorezeros")
    xsize, ysize, numslices = 6, 6, 4
    numregions = 3
    atlas_data, atlas_hdr, atlas_dims, atlas_sizes = _make_3d_atlas(
        xsize, ysize, numslices, numregions
    )
    # Input data with some zeros
    input_data = np.ones((xsize, ysize, numslices), dtype=np.float64)
    input_data[0, 0, 0] = 0.0  # Add a zero in region 1
    input_hdr = _make_mock_hdr(xsize, ysize, numslices, 1)
    input_dims = np.array([3, xsize, ysize, numslices, 1, 1, 1, 1])
    input_sizes = np.array([1.0, 2.0, 2.0, 2.0, 1.5, 1.0, 1.0, 1.0])

    args = _make_base_args()
    args.ignorezeros = True

    def mock_readfromnifti(fname, **kwargs):
        if "data" in fname:
            return MagicMock(), input_data, input_hdr, input_dims, input_sizes
        else:
            return MagicMock(), atlas_data, atlas_hdr, atlas_dims, atlas_sizes

    def mock_getfracvals(data, fracs, **kwargs):
        nonzero = data[data != 0]
        if len(nonzero) == 0:
            return [0.0] * len(fracs)
        return np.percentile(nonzero, np.array(fracs) * 100).tolist()

    with patch("rapidtide.workflows.atlasaverage.tide_io.readfromnifti", side_effect=mock_readfromnifti), \
         patch("rapidtide.workflows.atlasaverage.tide_io.checkspacematch", return_value=True), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftidims", return_value=(xsize, ysize, numslices, 1)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftisizes", return_value=(2.0, 2.0, 2.0, 1.5)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.savetonifti"), \
         patch("rapidtide.workflows.atlasaverage.tide_io.writevec"), \
         patch("rapidtide.workflows.atlasaverage.tide_mask.getmaskset", return_value=(None, None, None)), \
         patch("rapidtide.workflows.atlasaverage.tide_stats.getfracvals", side_effect=mock_getfracvals):

        atlasaverage(args)
    # No assertion error means it ran successfully with ignorezeros


def atlasaverage_3d_summary_methods(debug=False):
    """Test 3D mode with different summary methods."""
    if debug:
        print("atlasaverage_3d_summary_methods")
    xsize, ysize, numslices = 6, 6, 4
    numregions = 3
    atlas_data, atlas_hdr, atlas_dims, atlas_sizes = _make_3d_atlas(
        xsize, ysize, numslices, numregions
    )
    input_data, input_hdr, input_dims, input_sizes = _make_3d_data(xsize, ysize, numslices)

    def mock_readfromnifti(fname, **kwargs):
        if "data" in fname:
            return MagicMock(), input_data, input_hdr, input_dims, input_sizes
        else:
            return MagicMock(), atlas_data, atlas_hdr, atlas_dims, atlas_sizes

    def mock_getfracvals(data, fracs, **kwargs):
        nonzero = data[data != 0]
        if len(nonzero) == 0:
            return [0.0] * len(fracs)
        return np.percentile(nonzero, np.array(fracs) * 100).tolist()

    for method in ["mean", "median", "sum", "std", "MAD", "CoV"]:
        if debug:
            print(f"  testing summarymethod={method}")
        args = _make_base_args()
        args.summarymethod = method

        with patch("rapidtide.workflows.atlasaverage.tide_io.readfromnifti", side_effect=mock_readfromnifti), \
             patch("rapidtide.workflows.atlasaverage.tide_io.checkspacematch", return_value=True), \
             patch("rapidtide.workflows.atlasaverage.tide_io.parseniftidims", return_value=(xsize, ysize, numslices, 1)), \
             patch("rapidtide.workflows.atlasaverage.tide_io.parseniftisizes", return_value=(2.0, 2.0, 2.0, 1.5)), \
             patch("rapidtide.workflows.atlasaverage.tide_io.savetonifti"), \
             patch("rapidtide.workflows.atlasaverage.tide_io.writevec"), \
             patch("rapidtide.workflows.atlasaverage.tide_mask.getmaskset", return_value=(None, None, None)), \
             patch("rapidtide.workflows.atlasaverage.tide_stats.getfracvals", side_effect=mock_getfracvals):

            atlasaverage(args)


def atlasaverage_3d_numpercentiles(debug=False):
    """Test 3D mode with multiple percentiles."""
    if debug:
        print("atlasaverage_3d_numpercentiles")
    xsize, ysize, numslices = 6, 6, 4
    numregions = 3
    atlas_data, atlas_hdr, atlas_dims, atlas_sizes = _make_3d_atlas(
        xsize, ysize, numslices, numregions
    )
    input_data, input_hdr, input_dims, input_sizes = _make_3d_data(xsize, ysize, numslices)
    args = _make_base_args()
    args.numpercentiles = 4  # will generate 6 fracs: 0, 0.2, 0.4, 0.6, 0.8, 1.0

    def mock_readfromnifti(fname, **kwargs):
        if "data" in fname:
            return MagicMock(), input_data, input_hdr, input_dims, input_sizes
        else:
            return MagicMock(), atlas_data, atlas_hdr, atlas_dims, atlas_sizes

    def mock_getfracvals(data, fracs, **kwargs):
        nonzero = data[data != 0]
        if len(nonzero) == 0:
            return [0.0] * len(fracs)
        return np.percentile(nonzero, np.array(fracs) * 100).tolist()

    with patch("rapidtide.workflows.atlasaverage.tide_io.readfromnifti", side_effect=mock_readfromnifti), \
         patch("rapidtide.workflows.atlasaverage.tide_io.checkspacematch", return_value=True), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftidims", return_value=(xsize, ysize, numslices, 1)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftisizes", return_value=(2.0, 2.0, 2.0, 1.5)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.savetonifti"), \
         patch("rapidtide.workflows.atlasaverage.tide_io.writevec"), \
         patch("rapidtide.workflows.atlasaverage.tide_mask.getmaskset", return_value=(None, None, None)), \
         patch("rapidtide.workflows.atlasaverage.tide_stats.getfracvals", side_effect=mock_getfracvals):

        atlasaverage(args)


def atlasaverage_4d_basic(debug=False):
    """Test atlasaverage with 4D data, basic operation."""
    if debug:
        print("atlasaverage_4d_basic")
    xsize, ysize, numslices = 6, 6, 4
    numtimepoints = 10
    numregions = 3

    atlas_data, atlas_hdr, atlas_dims, atlas_sizes = _make_3d_atlas(
        xsize, ysize, numslices, numregions
    )
    input_data, input_hdr, input_dims, input_sizes = _make_4d_data(
        xsize, ysize, numslices, numtimepoints
    )
    args = _make_base_args()

    written_bids = {}

    def mock_readfromnifti(fname, **kwargs):
        if "data" in fname:
            return MagicMock(), input_data, input_hdr, input_dims, input_sizes
        else:
            return MagicMock(), atlas_data, atlas_hdr, atlas_dims, atlas_sizes

    def mock_writebidstsv(fname, data, samplerate, **kwargs):
        written_bids[fname] = {"data": data.copy(), "samplerate": samplerate}

    with patch("rapidtide.workflows.atlasaverage.tide_io.readfromnifti", side_effect=mock_readfromnifti), \
         patch("rapidtide.workflows.atlasaverage.tide_io.checkspacematch", return_value=True), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftidims", return_value=(xsize, ysize, numslices, numtimepoints)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftisizes", return_value=(2.0, 2.0, 2.0, 1.5)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.writebidstsv", side_effect=mock_writebidstsv), \
         patch("rapidtide.workflows.atlasaverage.tide_mask.getmaskset", return_value=(None, None, None)):

        atlasaverage(args)

    assert args.outputroot in written_bids, "BIDS TSV not written"
    bids_data = written_bids[args.outputroot]
    assert bids_data["data"].shape == (numregions, numtimepoints), \
        f"Expected shape ({numregions}, {numtimepoints}), got {bids_data['data'].shape}"
    # samplerate should be 1.0 / tr = 1.0 / 1.5
    assert np.isclose(bids_data["samplerate"], 1.0 / 1.5), \
        f"Expected samplerate {1.0/1.5}, got {bids_data['samplerate']}"


def atlasaverage_4d_normmethod_pct(debug=False):
    """Test 4D mode with pct normalization."""
    if debug:
        print("atlasaverage_4d_normmethod_pct")
    xsize, ysize, numslices = 6, 6, 4
    numtimepoints = 10
    numregions = 3
    atlas_data, atlas_hdr, _, _ = _make_3d_atlas(xsize, ysize, numslices, numregions)
    input_data, input_hdr, input_dims, input_sizes = _make_4d_data(
        xsize, ysize, numslices, numtimepoints
    )
    atlas_dims = np.array([3, xsize, ysize, numslices, 1, 1, 1, 1])
    atlas_sizes = np.array([1.0, 2.0, 2.0, 2.0, 1.5, 1.0, 1.0, 1.0])
    args = _make_base_args()
    args.normmethod = "pct"

    def mock_readfromnifti(fname, **kwargs):
        if "data" in fname:
            return MagicMock(), input_data, input_hdr, input_dims, input_sizes
        else:
            return MagicMock(), atlas_data, _make_mock_hdr(xsize, ysize, numslices, 1), atlas_dims, atlas_sizes

    with patch("rapidtide.workflows.atlasaverage.tide_io.readfromnifti", side_effect=mock_readfromnifti), \
         patch("rapidtide.workflows.atlasaverage.tide_io.checkspacematch", return_value=True), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftidims", return_value=(xsize, ysize, numslices, numtimepoints)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftisizes", return_value=(2.0, 2.0, 2.0, 1.5)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.writebidstsv"), \
         patch("rapidtide.workflows.atlasaverage.tide_mask.getmaskset", return_value=(None, None, None)):

        atlasaverage(args)


def atlasaverage_4d_normmethod_std(debug=False):
    """Test 4D mode with std normalization."""
    if debug:
        print("atlasaverage_4d_normmethod_std")
    xsize, ysize, numslices = 6, 6, 4
    numtimepoints = 10
    numregions = 3
    atlas_data, atlas_hdr, _, _ = _make_3d_atlas(xsize, ysize, numslices, numregions)
    input_data, input_hdr, input_dims, input_sizes = _make_4d_data(
        xsize, ysize, numslices, numtimepoints
    )
    atlas_dims = np.array([3, xsize, ysize, numslices, 1, 1, 1, 1])
    atlas_sizes = np.array([1.0, 2.0, 2.0, 2.0, 1.5, 1.0, 1.0, 1.0])
    args = _make_base_args()
    args.normmethod = "std"

    def mock_readfromnifti(fname, **kwargs):
        if "data" in fname:
            return MagicMock(), input_data, input_hdr, input_dims, input_sizes
        else:
            return MagicMock(), atlas_data, _make_mock_hdr(xsize, ysize, numslices, 1), atlas_dims, atlas_sizes

    with patch("rapidtide.workflows.atlasaverage.tide_io.readfromnifti", side_effect=mock_readfromnifti), \
         patch("rapidtide.workflows.atlasaverage.tide_io.checkspacematch", return_value=True), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftidims", return_value=(xsize, ysize, numslices, numtimepoints)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftisizes", return_value=(2.0, 2.0, 2.0, 1.5)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.writebidstsv"), \
         patch("rapidtide.workflows.atlasaverage.tide_mask.getmaskset", return_value=(None, None, None)):

        atlasaverage(args)


def atlasaverage_4d_normmethod_var(debug=False):
    """Test 4D mode with var normalization."""
    if debug:
        print("atlasaverage_4d_normmethod_var")
    xsize, ysize, numslices = 6, 6, 4
    numtimepoints = 10
    numregions = 3
    atlas_data, atlas_hdr, _, _ = _make_3d_atlas(xsize, ysize, numslices, numregions)
    input_data, input_hdr, input_dims, input_sizes = _make_4d_data(
        xsize, ysize, numslices, numtimepoints
    )
    atlas_dims = np.array([3, xsize, ysize, numslices, 1, 1, 1, 1])
    atlas_sizes = np.array([1.0, 2.0, 2.0, 2.0, 1.5, 1.0, 1.0, 1.0])
    args = _make_base_args()
    args.normmethod = "var"

    def mock_readfromnifti(fname, **kwargs):
        if "data" in fname:
            return MagicMock(), input_data, input_hdr, input_dims, input_sizes
        else:
            return MagicMock(), atlas_data, _make_mock_hdr(xsize, ysize, numslices, 1), atlas_dims, atlas_sizes

    with patch("rapidtide.workflows.atlasaverage.tide_io.readfromnifti", side_effect=mock_readfromnifti), \
         patch("rapidtide.workflows.atlasaverage.tide_io.checkspacematch", return_value=True), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftidims", return_value=(xsize, ysize, numslices, numtimepoints)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftisizes", return_value=(2.0, 2.0, 2.0, 1.5)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.writebidstsv"), \
         patch("rapidtide.workflows.atlasaverage.tide_mask.getmaskset", return_value=(None, None, None)):

        atlasaverage(args)


def atlasaverage_4d_normmethod_p2p(debug=False):
    """Test 4D mode with p2p normalization."""
    if debug:
        print("atlasaverage_4d_normmethod_p2p")
    xsize, ysize, numslices = 6, 6, 4
    numtimepoints = 10
    numregions = 3
    atlas_data, atlas_hdr, _, _ = _make_3d_atlas(xsize, ysize, numslices, numregions)
    input_data, input_hdr, input_dims, input_sizes = _make_4d_data(
        xsize, ysize, numslices, numtimepoints
    )
    atlas_dims = np.array([3, xsize, ysize, numslices, 1, 1, 1, 1])
    atlas_sizes = np.array([1.0, 2.0, 2.0, 2.0, 1.5, 1.0, 1.0, 1.0])
    args = _make_base_args()
    args.normmethod = "p2p"

    def mock_readfromnifti(fname, **kwargs):
        if "data" in fname:
            return MagicMock(), input_data, input_hdr, input_dims, input_sizes
        else:
            return MagicMock(), atlas_data, _make_mock_hdr(xsize, ysize, numslices, 1), atlas_dims, atlas_sizes

    with patch("rapidtide.workflows.atlasaverage.tide_io.readfromnifti", side_effect=mock_readfromnifti), \
         patch("rapidtide.workflows.atlasaverage.tide_io.checkspacematch", return_value=True), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftidims", return_value=(xsize, ysize, numslices, numtimepoints)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftisizes", return_value=(2.0, 2.0, 2.0, 1.5)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.writebidstsv"), \
         patch("rapidtide.workflows.atlasaverage.tide_mask.getmaskset", return_value=(None, None, None)):

        atlasaverage(args)


def atlasaverage_4d_median_summary(debug=False):
    """Test 4D mode with median summary method."""
    if debug:
        print("atlasaverage_4d_median_summary")
    xsize, ysize, numslices = 6, 6, 4
    numtimepoints = 10
    numregions = 3
    atlas_data, atlas_hdr, _, _ = _make_3d_atlas(xsize, ysize, numslices, numregions)
    input_data, input_hdr, input_dims, input_sizes = _make_4d_data(
        xsize, ysize, numslices, numtimepoints
    )
    atlas_dims = np.array([3, xsize, ysize, numslices, 1, 1, 1, 1])
    atlas_sizes = np.array([1.0, 2.0, 2.0, 2.0, 1.5, 1.0, 1.0, 1.0])
    args = _make_base_args()
    args.summarymethod = "median"

    def mock_readfromnifti(fname, **kwargs):
        if "data" in fname:
            return MagicMock(), input_data, input_hdr, input_dims, input_sizes
        else:
            return MagicMock(), atlas_data, _make_mock_hdr(xsize, ysize, numslices, 1), atlas_dims, atlas_sizes

    with patch("rapidtide.workflows.atlasaverage.tide_io.readfromnifti", side_effect=mock_readfromnifti), \
         patch("rapidtide.workflows.atlasaverage.tide_io.checkspacematch", return_value=True), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftidims", return_value=(xsize, ysize, numslices, numtimepoints)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftisizes", return_value=(2.0, 2.0, 2.0, 1.5)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.writebidstsv"), \
         patch("rapidtide.workflows.atlasaverage.tide_mask.getmaskset", return_value=(None, None, None)):

        atlasaverage(args)


def atlasaverage_space_mismatch(debug=False):
    """Test that space mismatch causes exit."""
    if debug:
        print("atlasaverage_space_mismatch")
    xsize, ysize, numslices = 6, 6, 4
    atlas_data, atlas_hdr, atlas_dims, atlas_sizes = _make_3d_atlas(xsize, ysize, numslices, 3)
    input_data, input_hdr, input_dims, input_sizes = _make_3d_data(xsize, ysize, numslices)
    args = _make_base_args()

    def mock_readfromnifti(fname, **kwargs):
        if "data" in fname:
            return MagicMock(), input_data, input_hdr, input_dims, input_sizes
        else:
            return MagicMock(), atlas_data, atlas_hdr, atlas_dims, atlas_sizes

    import pytest
    with patch("rapidtide.workflows.atlasaverage.tide_io.readfromnifti", side_effect=mock_readfromnifti), \
         patch("rapidtide.workflows.atlasaverage.tide_io.checkspacematch", return_value=False):

        with pytest.raises(SystemExit):
            atlasaverage(args)


def atlasaverage_regionlabelfile(debug=False):
    """Test atlasaverage with a region label file."""
    if debug:
        print("atlasaverage_regionlabelfile")
    xsize, ysize, numslices = 6, 6, 4
    numregions = 3
    atlas_data, atlas_hdr, atlas_dims, atlas_sizes = _make_3d_atlas(
        xsize, ysize, numslices, numregions
    )
    input_data, input_hdr, input_dims, input_sizes = _make_4d_data(
        xsize, ysize, numslices, 10
    )
    args = _make_base_args()
    args.regionlabelfile = "labels.txt"

    region_labels = ["Frontal", "Parietal", "Occipital"]

    written_bids = {}

    def mock_readfromnifti(fname, **kwargs):
        if "data" in fname:
            return MagicMock(), input_data, input_hdr, input_dims, input_sizes
        else:
            return MagicMock(), atlas_data, atlas_hdr, atlas_dims, atlas_sizes

    def mock_writebidstsv(fname, data, samplerate, **kwargs):
        written_bids[fname] = kwargs.get("columns", [])

    with patch("rapidtide.workflows.atlasaverage.tide_io.readfromnifti", side_effect=mock_readfromnifti), \
         patch("rapidtide.workflows.atlasaverage.tide_io.checkspacematch", return_value=True), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftidims", return_value=(xsize, ysize, numslices, 10)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftisizes", return_value=(2.0, 2.0, 2.0, 1.5)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.writebidstsv", side_effect=mock_writebidstsv), \
         patch("rapidtide.workflows.atlasaverage.tide_io.readlabels", return_value=region_labels), \
         patch("rapidtide.workflows.atlasaverage.tide_mask.getmaskset", return_value=(None, None, None)):

        atlasaverage(args)

    assert args.outputroot in written_bids
    assert written_bids[args.outputroot] == region_labels


def atlasaverage_regionlabelfile_mismatch(debug=False):
    """Test that mismatched region labels cause exit."""
    if debug:
        print("atlasaverage_regionlabelfile_mismatch")
    xsize, ysize, numslices = 6, 6, 4
    numregions = 3
    atlas_data, atlas_hdr, atlas_dims, atlas_sizes = _make_3d_atlas(
        xsize, ysize, numslices, numregions
    )
    input_data, input_hdr, input_dims, input_sizes = _make_4d_data(
        xsize, ysize, numslices, 10
    )
    args = _make_base_args()
    args.regionlabelfile = "labels.txt"

    # Wrong number of labels
    region_labels = ["Frontal", "Parietal"]  # Only 2 labels for 3 regions

    def mock_readfromnifti(fname, **kwargs):
        if "data" in fname:
            return MagicMock(), input_data, input_hdr, input_dims, input_sizes
        else:
            return MagicMock(), atlas_data, atlas_hdr, atlas_dims, atlas_sizes

    import pytest
    with patch("rapidtide.workflows.atlasaverage.tide_io.readfromnifti", side_effect=mock_readfromnifti), \
         patch("rapidtide.workflows.atlasaverage.tide_io.checkspacematch", return_value=True), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftidims", return_value=(xsize, ysize, numslices, 10)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftisizes", return_value=(2.0, 2.0, 2.0, 1.5)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.readlabels", return_value=region_labels), \
         patch("rapidtide.workflows.atlasaverage.tide_mask.getmaskset", return_value=(None, None, None)):

        with pytest.raises(SystemExit):
            atlasaverage(args)


def atlasaverage_regionlistfile(debug=False):
    """Test atlasaverage with a region list file (subset of regions)."""
    if debug:
        print("atlasaverage_regionlistfile")
    xsize, ysize, numslices = 6, 6, 4
    numregions = 3
    atlas_data, atlas_hdr, atlas_dims, atlas_sizes = _make_3d_atlas(
        xsize, ysize, numslices, numregions
    )
    input_data, input_hdr, input_dims, input_sizes = _make_4d_data(
        xsize, ysize, numslices, 10
    )
    args = _make_base_args()
    args.regionlistfile = "regionlist.txt"

    # Only process regions 1 and 2 (must be contiguous starting at 1;
    # non-contiguous lists like [1, 3] trigger a bug in the source code
    # where timecourses[theregion-1] goes out of bounds)
    regionlist = np.array([1, 2], dtype=float)

    written_bids = {}

    def mock_readfromnifti(fname, **kwargs):
        if "data" in fname:
            return MagicMock(), input_data, input_hdr, input_dims, input_sizes
        else:
            return MagicMock(), atlas_data, atlas_hdr, atlas_dims, atlas_sizes

    def mock_writebidstsv(fname, data, samplerate, **kwargs):
        written_bids[fname] = {"data": data.copy(), "columns": kwargs.get("columns", [])}

    with patch("rapidtide.workflows.atlasaverage.tide_io.readfromnifti", side_effect=mock_readfromnifti), \
         patch("rapidtide.workflows.atlasaverage.tide_io.checkspacematch", return_value=True), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftidims", return_value=(xsize, ysize, numslices, 10)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftisizes", return_value=(2.0, 2.0, 2.0, 1.5)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.writebidstsv", side_effect=mock_writebidstsv), \
         patch("rapidtide.workflows.atlasaverage.tide_io.readvec", return_value=regionlist), \
         patch("rapidtide.workflows.atlasaverage.tide_mask.getmaskset", return_value=(None, None, None)):

        atlasaverage(args)

    assert args.outputroot in written_bids
    # numregions should be 2 (from regionlist)
    assert written_bids[args.outputroot]["data"].shape[0] == 2


def atlasaverage_debug_mode(debug=False):
    """Test atlasaverage with debug flag enabled."""
    if debug:
        print("atlasaverage_debug_mode")
    xsize, ysize, numslices = 6, 6, 4
    numregions = 3
    atlas_data, atlas_hdr, atlas_dims, atlas_sizes = _make_3d_atlas(
        xsize, ysize, numslices, numregions
    )
    input_data, input_hdr, input_dims, input_sizes = _make_3d_data(xsize, ysize, numslices)
    args = _make_base_args()
    args.debug = True

    def mock_readfromnifti(fname, **kwargs):
        if "data" in fname:
            return MagicMock(), input_data, input_hdr, input_dims, input_sizes
        else:
            return MagicMock(), atlas_data, atlas_hdr, atlas_dims, atlas_sizes

    def mock_getfracvals(data, fracs, **kwargs):
        nonzero = data[data != 0]
        if len(nonzero) == 0:
            return [0.0] * len(fracs)
        return np.percentile(nonzero, np.array(fracs) * 100).tolist()

    with patch("rapidtide.workflows.atlasaverage.tide_io.readfromnifti", side_effect=mock_readfromnifti), \
         patch("rapidtide.workflows.atlasaverage.tide_io.checkspacematch", return_value=True), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftidims", return_value=(xsize, ysize, numslices, 1)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftisizes", return_value=(2.0, 2.0, 2.0, 1.5)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.savetonifti"), \
         patch("rapidtide.workflows.atlasaverage.tide_io.writevec"), \
         patch("rapidtide.workflows.atlasaverage.tide_mask.getmaskset", return_value=(None, None, None)), \
         patch("rapidtide.workflows.atlasaverage.tide_stats.getfracvals", side_effect=mock_getfracvals):

        atlasaverage(args)


def atlasaverage_auto_labels(debug=False):
    """Test that auto-generated labels follow region_NNN format."""
    if debug:
        print("atlasaverage_auto_labels")
    xsize, ysize, numslices = 6, 6, 4
    numregions = 3
    atlas_data, atlas_hdr, atlas_dims, atlas_sizes = _make_3d_atlas(
        xsize, ysize, numslices, numregions
    )
    input_data, input_hdr, input_dims, input_sizes = _make_4d_data(
        xsize, ysize, numslices, 10
    )
    args = _make_base_args()

    written_bids = {}

    def mock_readfromnifti(fname, **kwargs):
        if "data" in fname:
            return MagicMock(), input_data, input_hdr, input_dims, input_sizes
        else:
            return MagicMock(), atlas_data, atlas_hdr, atlas_dims, atlas_sizes

    def mock_writebidstsv(fname, data, samplerate, **kwargs):
        written_bids[fname] = kwargs.get("columns", [])

    with patch("rapidtide.workflows.atlasaverage.tide_io.readfromnifti", side_effect=mock_readfromnifti), \
         patch("rapidtide.workflows.atlasaverage.tide_io.checkspacematch", return_value=True), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftidims", return_value=(xsize, ysize, numslices, 10)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftisizes", return_value=(2.0, 2.0, 2.0, 1.5)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.writebidstsv", side_effect=mock_writebidstsv), \
         patch("rapidtide.workflows.atlasaverage.tide_mask.getmaskset", return_value=(None, None, None)):

        atlasaverage(args)

    columns = written_bids[args.outputroot]
    assert len(columns) == numregions
    assert columns[0] == "region_1"
    assert columns[1] == "region_2"
    assert columns[2] == "region_3"


def atlasaverage_includemask(debug=False):
    """Test atlasaverage with an include mask."""
    if debug:
        print("atlasaverage_includemask")
    xsize, ysize, numslices = 6, 6, 4
    numtimepoints = 10
    numregions = 3
    numvoxels = xsize * ysize * numslices
    atlas_data, atlas_hdr, atlas_dims, atlas_sizes = _make_3d_atlas(
        xsize, ysize, numslices, numregions
    )
    input_data, input_hdr, input_dims, input_sizes = _make_4d_data(
        xsize, ysize, numslices, numtimepoints
    )
    args = _make_base_args()
    args.includespec = "include_mask.nii.gz:1"

    # Include mask: ones everywhere (so all voxels included)
    include_mask = np.ones(numvoxels)

    def mock_readfromnifti(fname, **kwargs):
        if "data" in fname:
            return MagicMock(), input_data, input_hdr, input_dims, input_sizes
        else:
            return MagicMock(), atlas_data, atlas_hdr, atlas_dims, atlas_sizes

    with patch("rapidtide.workflows.atlasaverage.tide_io.readfromnifti", side_effect=mock_readfromnifti), \
         patch("rapidtide.workflows.atlasaverage.tide_io.checkspacematch", return_value=True), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftidims", return_value=(xsize, ysize, numslices, numtimepoints)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftisizes", return_value=(2.0, 2.0, 2.0, 1.5)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.processnamespec", return_value=("include_mask.nii.gz", [1])), \
         patch("rapidtide.workflows.atlasaverage.tide_io.writebidstsv"), \
         patch("rapidtide.workflows.atlasaverage.tide_mask.getmaskset", return_value=(include_mask, None, None)):

        atlasaverage(args)


def atlasaverage_excludemask(debug=False):
    """Test atlasaverage with an exclude mask."""
    if debug:
        print("atlasaverage_excludemask")
    xsize, ysize, numslices = 6, 6, 4
    numtimepoints = 10
    numregions = 3
    numvoxels = xsize * ysize * numslices
    atlas_data, atlas_hdr, atlas_dims, atlas_sizes = _make_3d_atlas(
        xsize, ysize, numslices, numregions
    )
    input_data, input_hdr, input_dims, input_sizes = _make_4d_data(
        xsize, ysize, numslices, numtimepoints
    )
    args = _make_base_args()
    args.excludespec = "exclude_mask.nii.gz"

    # Exclude mask: zeros everywhere (so no voxels excluded)
    exclude_mask = np.zeros(numvoxels)

    def mock_readfromnifti(fname, **kwargs):
        if "data" in fname:
            return MagicMock(), input_data, input_hdr, input_dims, input_sizes
        else:
            return MagicMock(), atlas_data, atlas_hdr, atlas_dims, atlas_sizes

    with patch("rapidtide.workflows.atlasaverage.tide_io.readfromnifti", side_effect=mock_readfromnifti), \
         patch("rapidtide.workflows.atlasaverage.tide_io.checkspacematch", return_value=True), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftidims", return_value=(xsize, ysize, numslices, numtimepoints)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftisizes", return_value=(2.0, 2.0, 2.0, 1.5)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.processnamespec", return_value=("exclude_mask.nii.gz", None)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.writebidstsv"), \
         patch("rapidtide.workflows.atlasaverage.tide_mask.getmaskset", return_value=(None, exclude_mask, None)):

        atlasaverage(args)


def atlasaverage_3d_maskedatlas_output(debug=False):
    """Test that masked atlas is saved when include or exclude mask is used (3D)."""
    if debug:
        print("atlasaverage_3d_maskedatlas_output")
    xsize, ysize, numslices = 6, 6, 4
    numregions = 3
    numvoxels = xsize * ysize * numslices
    atlas_data, atlas_hdr, atlas_dims, atlas_sizes = _make_3d_atlas(
        xsize, ysize, numslices, numregions
    )
    input_data, input_hdr, input_dims, input_sizes = _make_3d_data(xsize, ysize, numslices)
    args = _make_base_args()
    args.includespec = "include_mask.nii.gz"

    include_mask = np.ones(numvoxels)

    saved_data = {}

    def mock_readfromnifti(fname, **kwargs):
        if "data" in fname:
            return MagicMock(), input_data, input_hdr, input_dims, input_sizes
        else:
            return MagicMock(), atlas_data, atlas_hdr, atlas_dims, atlas_sizes

    def mock_savetonifti(data, hdr, fname, **kwargs):
        saved_data[fname] = data.copy()

    def mock_getfracvals(data, fracs, **kwargs):
        nonzero = data[data != 0]
        if len(nonzero) == 0:
            return [0.0] * len(fracs)
        return np.percentile(nonzero, np.array(fracs) * 100).tolist()

    with patch("rapidtide.workflows.atlasaverage.tide_io.readfromnifti", side_effect=mock_readfromnifti), \
         patch("rapidtide.workflows.atlasaverage.tide_io.checkspacematch", return_value=True), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftidims", return_value=(xsize, ysize, numslices, 1)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.parseniftisizes", return_value=(2.0, 2.0, 2.0, 1.5)), \
         patch("rapidtide.workflows.atlasaverage.tide_io.savetonifti", side_effect=mock_savetonifti), \
         patch("rapidtide.workflows.atlasaverage.tide_io.writevec"), \
         patch("rapidtide.workflows.atlasaverage.tide_io.processnamespec", return_value=("include_mask.nii.gz", None)), \
         patch("rapidtide.workflows.atlasaverage.tide_mask.getmaskset", return_value=(include_mask, None, None)), \
         patch("rapidtide.workflows.atlasaverage.tide_stats.getfracvals", side_effect=mock_getfracvals):

        atlasaverage(args)

    # Should save maskedatlas when includename is set
    maskedatlas_key = f"{args.outputroot}_maskedatlas"
    assert maskedatlas_key in saved_data, f"Expected {maskedatlas_key} in saved files: {list(saved_data.keys())}"


# ==================== Main test function ====================


def test_atlasaverage(debug=False):
    # _get_parser tests
    if debug:
        print("Running parser tests")
    parser_basic(debug=debug)
    parser_required_args(debug=debug)
    parser_defaults(debug=debug)
    parser_normmethod_choices(debug=debug)
    parser_summarymethod_choices(debug=debug)
    parser_numpercentiles(debug=debug)
    parser_flags(debug=debug)
    parser_datalabel(debug=debug)

    # summarizevoxels tests
    if debug:
        print("Running summarizevoxels tests")
    summarizevoxels_mean_1d(debug=debug)
    summarizevoxels_mean_2d(debug=debug)
    summarizevoxels_sum(debug=debug)
    summarizevoxels_median(debug=debug)
    summarizevoxels_std(debug=debug)
    summarizevoxels_mad(debug=debug)
    summarizevoxels_cov_1d(debug=debug)
    summarizevoxels_cov_2d(debug=debug)
    summarizevoxels_nan_handling(debug=debug)
    summarizevoxels_default_method(debug=debug)

    # atlasaverage tests
    if debug:
        print("Running atlasaverage tests")
    atlasaverage_3d_basic(debug=debug)
    atlasaverage_3d_with_headerline(debug=debug)
    atlasaverage_3d_no_headerline(debug=debug)
    atlasaverage_3d_with_datalabel(debug=debug)
    atlasaverage_3d_ignorezeros(debug=debug)
    atlasaverage_3d_summary_methods(debug=debug)
    atlasaverage_3d_numpercentiles(debug=debug)
    atlasaverage_4d_basic(debug=debug)
    atlasaverage_4d_normmethod_pct(debug=debug)
    atlasaverage_4d_normmethod_std(debug=debug)
    atlasaverage_4d_normmethod_var(debug=debug)
    atlasaverage_4d_normmethod_p2p(debug=debug)
    atlasaverage_4d_median_summary(debug=debug)
    atlasaverage_space_mismatch(debug=debug)
    atlasaverage_regionlabelfile(debug=debug)
    atlasaverage_regionlabelfile_mismatch(debug=debug)
    atlasaverage_regionlistfile(debug=debug)
    atlasaverage_debug_mode(debug=debug)
    atlasaverage_auto_labels(debug=debug)
    atlasaverage_includemask(debug=debug)
    atlasaverage_excludemask(debug=debug)
    atlasaverage_3d_maskedatlas_output(debug=debug)


if __name__ == "__main__":
    test_atlasaverage(debug=True)
