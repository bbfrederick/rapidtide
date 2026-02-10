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
import copy
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rapidtide.tests.utils import create_dir, get_test_temp_path
from rapidtide.workflows.histnifti import _get_parser, histnifti

# ==================== Helpers ====================


def _make_mock_hdr(xsize, ysize, numslices, timepoints=1):
    """Create a mock NIfTI header that supports __getitem__, __setitem__, and deepcopy."""
    store = {
        "dim": [4, xsize, ysize, numslices, timepoints, 1, 1, 1],
        "pixdim": [1.0, 2.0, 2.0, 2.0, 1.5, 1.0, 1.0, 1.0],
        "toffset": 0.0,
    }

    def _build_hdr(s):
        hdr = MagicMock()
        hdr.__getitem__ = MagicMock(side_effect=lambda key: s[key])
        hdr.__setitem__ = MagicMock(side_effect=lambda key, val: s.__setitem__(key, val))
        hdr.__deepcopy__ = MagicMock(side_effect=lambda memo: _build_hdr({
            "dim": list(s["dim"]),
            "pixdim": list(s["pixdim"]),
            "toffset": s["toffset"],
        }))
        return hdr

    return _build_hdr(store)


def _make_dims(xsize, ysize, numslices, timepoints):
    return np.array([4, xsize, ysize, numslices, timepoints, 1, 1, 1])


def _make_sizes():
    return np.array([1.0, 2.0, 2.0, 2.0, 1.5, 1.0, 1.0, 1.0])


def _make_4d_data(xsize=3, ysize=3, numslices=2, timepoints=20):
    """Create synthetic 4D data with known distribution for histogram testing."""
    rng = np.random.RandomState(42)
    data = rng.randn(xsize, ysize, numslices, timepoints).astype(np.float64)
    # Scale to have a reasonable range
    data = data * 10.0 + 50.0
    return data


def _make_3d_data(xsize=3, ysize=3, numslices=2):
    """Create synthetic 3D data with known distribution."""
    rng = np.random.RandomState(42)
    data = rng.randn(xsize, ysize, numslices).astype(np.float64)
    data = data * 10.0 + 50.0
    return data


def _make_default_args(inputfile="dummy_input.nii.gz", outputroot="/tmp/test_histnifti"):
    """Create a default Namespace with all required args."""
    return argparse.Namespace(
        inputfile=inputfile,
        outputroot=outputroot,
        maskfile=None,
        histlen=None,
        minval=None,
        maxval=None,
        robustrange=False,
        transform=False,
        nozero=False,
        nozerothresh=0.01,
        normhist=False,
        display=False,
    )


def _run_histnifti(data, args, mask_data=None, mask_mismatch=False):
    """Helper to run histnifti with mocked IO. Returns saved files dict."""
    if data.ndim == 4:
        xsize, ysize, numslices, timepoints = data.shape
    else:
        xsize, ysize, numslices = data.shape
        timepoints = 1

    dims = _make_dims(xsize, ysize, numslices, timepoints)
    sizes = _make_sizes()
    hdr = _make_mock_hdr(xsize, ysize, numslices, timepoints)

    saved_nifti = {}

    def mock_readfromnifti(fname, **kwargs):
        if "mask" in fname.lower():
            m_data = mask_data if mask_data is not None else np.ones((xsize, ysize, numslices))
            m_dims = _make_dims(xsize, ysize, numslices, 1)
            m_hdr = _make_mock_hdr(xsize, ysize, numslices, 1)
            return (MagicMock(), m_data, m_hdr, m_dims, sizes)
        return (MagicMock(), data, hdr, dims, sizes)

    def mock_savetonifti(arr, hdr_arg, fname, **kwargs):
        saved_nifti[fname] = arr.copy()

    def mock_checkspacematch(hdr1, hdr2):
        return not mask_mismatch

    with patch("rapidtide.workflows.histnifti.tide_io.readfromnifti",
               side_effect=mock_readfromnifti), \
         patch("rapidtide.workflows.histnifti.tide_io.checkspacematch",
               side_effect=mock_checkspacematch), \
         patch("rapidtide.workflows.histnifti.tide_io.savetonifti",
               side_effect=mock_savetonifti), \
         patch("rapidtide.workflows.histnifti.tide_stats.makeandsavehistogram"):

        histnifti(args)

    return saved_nifti


# ==================== _get_parser tests ====================


def parser_basic(debug=False):
    """Test that parser creates successfully."""
    if debug:
        print("parser_basic")
    parser = _get_parser()
    assert parser is not None
    assert parser.prog == "histnifti"


def parser_required_args(debug=False):
    """Test parser has required positional arguments."""
    if debug:
        print("parser_required_args")
    parser = _get_parser()
    actions = {a.dest: a for a in parser._actions}
    assert "inputfile" in actions
    assert "outputroot" in actions


def parser_defaults(debug=False):
    """Test default values for optional arguments."""
    if debug:
        print("parser_defaults")
    parser = _get_parser()
    args = parser.parse_args(["input.nii", "outroot"])
    assert args.histlen is None
    assert args.minval is None
    assert args.maxval is None
    assert args.robustrange is False
    assert args.transform is False
    assert args.nozero is False
    assert args.nozerothresh == 0.01
    assert args.normhist is False
    assert args.maskfile is None
    assert args.display is True


def parser_histlen(debug=False):
    """Test --histlen option."""
    if debug:
        print("parser_histlen")
    parser = _get_parser()
    args = parser.parse_args(["input.nii", "outroot", "--histlen", "50"])
    assert args.histlen == 50


def parser_minval_maxval(debug=False):
    """Test --minval and --maxval options."""
    if debug:
        print("parser_minval_maxval")
    parser = _get_parser()
    args = parser.parse_args(["input.nii", "outroot", "--minval", "0.5", "--maxval", "100.0"])
    assert args.minval == 0.5
    assert args.maxval == 100.0


def parser_boolean_flags(debug=False):
    """Test boolean flag options."""
    if debug:
        print("parser_boolean_flags")
    parser = _get_parser()
    args = parser.parse_args([
        "input.nii", "outroot",
        "--robustrange", "--transform", "--nozero", "--normhist", "--nodisplay",
    ])
    assert args.robustrange is True
    assert args.transform is True
    assert args.nozero is True
    assert args.normhist is True
    assert args.display is False


def parser_nozerothresh(debug=False):
    """Test --nozerothresh option."""
    if debug:
        print("parser_nozerothresh")
    parser = _get_parser()
    args = parser.parse_args(["input.nii", "outroot", "--nozerothresh", "0.05"])
    assert args.nozerothresh == 0.05


def parser_maskfile(debug=False):
    """Test --maskfile option with valid file."""
    if debug:
        print("parser_maskfile")
    parser = _get_parser()
    with tempfile.NamedTemporaryFile(suffix=".nii") as f:
        args = parser.parse_args(["input.nii", "outroot", "--maskfile", f.name])
    assert args.maskfile is not None


# ==================== histnifti 4D tests ====================


def histnifti_4d_basic(debug=False):
    """Test basic 4D workflow produces sorted, pcts, and hists outputs."""
    if debug:
        print("histnifti_4d_basic")
    xsize, ysize, numslices, timepoints = 3, 3, 2, 20
    data = _make_4d_data(xsize, ysize, numslices, timepoints)
    args = _make_default_args()

    saved = _run_histnifti(data, args)

    assert "/tmp/test_histnifti_sorted" in saved
    assert "/tmp/test_histnifti_pcts" in saved
    assert "/tmp/test_histnifti_hists" in saved


def histnifti_4d_sorted_shape(debug=False):
    """Test that sorted output has same shape as input data."""
    if debug:
        print("histnifti_4d_sorted_shape")
    xsize, ysize, numslices, timepoints = 3, 3, 2, 20
    data = _make_4d_data(xsize, ysize, numslices, timepoints)
    args = _make_default_args()

    saved = _run_histnifti(data, args)

    sorted_data = saved["/tmp/test_histnifti_sorted"]
    assert sorted_data.shape == (xsize, ysize, numslices, timepoints)


def histnifti_4d_sorted_is_sorted(debug=False):
    """Test that sorted output is actually sorted along time axis."""
    if debug:
        print("histnifti_4d_sorted_is_sorted")
    xsize, ysize, numslices, timepoints = 3, 3, 2, 20
    data = _make_4d_data(xsize, ysize, numslices, timepoints)
    args = _make_default_args()

    saved = _run_histnifti(data, args)

    sorted_data = saved["/tmp/test_histnifti_sorted"]
    # For each voxel, values should be sorted along the time axis
    for x in range(xsize):
        for y in range(ysize):
            for z in range(numslices):
                voxel_ts = sorted_data[x, y, z, :]
                assert np.all(voxel_ts[:-1] <= voxel_ts[1:]), \
                    f"Voxel ({x},{y},{z}) not sorted"


def histnifti_4d_pcts_shape(debug=False):
    """Test that percentiles output has correct shape."""
    if debug:
        print("histnifti_4d_pcts_shape")
    xsize, ysize, numslices, timepoints = 3, 3, 2, 20
    data = _make_4d_data(xsize, ysize, numslices, timepoints)
    args = _make_default_args()

    saved = _run_histnifti(data, args)

    pcts = saved["/tmp/test_histnifti_pcts"]
    # 5 percentiles: [0.2, 0.25, 0.5, 0.75, 0.98]
    assert pcts.shape == (xsize, ysize, numslices, 5)


def histnifti_4d_pcts_ordered(debug=False):
    """Test that percentile values are monotonically increasing."""
    if debug:
        print("histnifti_4d_pcts_ordered")
    xsize, ysize, numslices, timepoints = 3, 3, 2, 20
    data = _make_4d_data(xsize, ysize, numslices, timepoints)
    args = _make_default_args()

    saved = _run_histnifti(data, args)

    pcts = saved["/tmp/test_histnifti_pcts"]
    for x in range(xsize):
        for y in range(ysize):
            for z in range(numslices):
                vals = pcts[x, y, z, :]
                assert np.all(vals[:-1] <= vals[1:]), \
                    f"Percentiles at ({x},{y},{z}) not ordered: {vals}"


def histnifti_4d_hists_shape(debug=False):
    """Test that histogram output has correct shape."""
    if debug:
        print("histnifti_4d_hists_shape")
    xsize, ysize, numslices, timepoints = 3, 3, 2, 20
    data = _make_4d_data(xsize, ysize, numslices, timepoints)
    args = _make_default_args()

    saved = _run_histnifti(data, args)

    hists = saved["/tmp/test_histnifti_hists"]
    # Auto histlen = 2 * floor(sqrt(timepoints)) + 1 = 2 * 4 + 1 = 9
    expected_histlen = 2 * int(np.floor(np.sqrt(timepoints))) + 1
    assert hists.shape == (xsize, ysize, numslices, expected_histlen)


def histnifti_4d_hists_nonnegative(debug=False):
    """Test that histogram counts are non-negative."""
    if debug:
        print("histnifti_4d_hists_nonnegative")
    xsize, ysize, numslices, timepoints = 3, 3, 2, 20
    data = _make_4d_data(xsize, ysize, numslices, timepoints)
    args = _make_default_args()

    saved = _run_histnifti(data, args)

    hists = saved["/tmp/test_histnifti_hists"]
    assert np.all(hists >= 0), "Histogram has negative counts"


def histnifti_4d_custom_histlen(debug=False):
    """Test 4D workflow with custom histogram length."""
    if debug:
        print("histnifti_4d_custom_histlen")
    xsize, ysize, numslices, timepoints = 3, 3, 2, 20
    data = _make_4d_data(xsize, ysize, numslices, timepoints)
    args = _make_default_args()
    args.histlen = 15

    saved = _run_histnifti(data, args)

    hists = saved["/tmp/test_histnifti_hists"]
    assert hists.shape == (xsize, ysize, numslices, 15)


def histnifti_4d_minval_maxval(debug=False):
    """Test 4D workflow with explicit min/max values."""
    if debug:
        print("histnifti_4d_minval_maxval")
    xsize, ysize, numslices, timepoints = 3, 3, 2, 20
    data = _make_4d_data(xsize, ysize, numslices, timepoints)
    args = _make_default_args()
    args.minval = 30.0
    args.maxval = 70.0

    saved = _run_histnifti(data, args)

    assert "/tmp/test_histnifti_hists" in saved


def histnifti_4d_robustrange(debug=False):
    """Test 4D workflow with robustrange enabled."""
    if debug:
        print("histnifti_4d_robustrange")
    xsize, ysize, numslices, timepoints = 3, 3, 2, 20
    data = _make_4d_data(xsize, ysize, numslices, timepoints)
    args = _make_default_args()
    args.robustrange = True

    saved = _run_histnifti(data, args)

    assert "/tmp/test_histnifti_sorted" in saved
    assert "/tmp/test_histnifti_pcts" in saved
    assert "/tmp/test_histnifti_hists" in saved


def histnifti_4d_nozero(debug=False):
    """Test 4D workflow with nozero enabled (uses getfracvals for percentiles)."""
    if debug:
        print("histnifti_4d_nozero")
    xsize, ysize, numslices, timepoints = 3, 3, 2, 20
    data = _make_4d_data(xsize, ysize, numslices, timepoints)
    # Set some values to zero
    data[0, 0, 0, :5] = 0.0
    args = _make_default_args()
    args.nozero = True

    saved = _run_histnifti(data, args)

    assert "/tmp/test_histnifti_sorted" in saved
    assert "/tmp/test_histnifti_pcts" in saved
    assert "/tmp/test_histnifti_hists" in saved


def histnifti_4d_with_mask(debug=False):
    """Test 4D workflow with a mask file."""
    if debug:
        print("histnifti_4d_with_mask")
    xsize, ysize, numslices, timepoints = 3, 3, 2, 20
    data = _make_4d_data(xsize, ysize, numslices, timepoints)
    mask = np.ones((xsize, ysize, numslices), dtype=np.float64)
    # Mask out half the voxels
    mask[0, :, :] = 0.0
    args = _make_default_args()
    args.maskfile = "dummy_mask.nii.gz"

    saved = _run_histnifti(data, args, mask_data=mask)

    sorted_out = saved["/tmp/test_histnifti_sorted"]
    # Masked-out voxels should be zero
    assert np.all(sorted_out[0, :, :, :] == 0.0), \
        "Masked-out voxels should have zero values in sorted output"


def histnifti_4d_mask_mismatch(debug=False):
    """Test that mask dimension mismatch causes exit."""
    if debug:
        print("histnifti_4d_mask_mismatch")
    xsize, ysize, numslices, timepoints = 3, 3, 2, 20
    data = _make_4d_data(xsize, ysize, numslices, timepoints)
    mask = np.ones((xsize, ysize, numslices), dtype=np.float64)
    args = _make_default_args()
    args.maskfile = "dummy_mask.nii.gz"

    with pytest.raises(SystemExit):
        _run_histnifti(data, args, mask_data=mask, mask_mismatch=True)


# ==================== histnifti 3D tests ====================


def histnifti_3d_basic(debug=False):
    """Test basic 3D workflow calls makeandsavehistogram."""
    if debug:
        print("histnifti_3d_basic")
    xsize, ysize, numslices = 3, 3, 2
    data = _make_3d_data(xsize, ysize, numslices)
    args = _make_default_args()

    dims = _make_dims(xsize, ysize, numslices, 1)
    sizes = _make_sizes()
    hdr = _make_mock_hdr(xsize, ysize, numslices, 1)

    makeandsavehistogram_called = {"called": False, "kwargs": {}}

    def mock_readfromnifti(fname, **kwargs):
        return (MagicMock(), data, hdr, dims, sizes)

    def mock_makeandsavehistogram(indata, histlen, endtrim, outname, **kwargs):
        makeandsavehistogram_called["called"] = True
        makeandsavehistogram_called["kwargs"] = kwargs
        makeandsavehistogram_called["outname"] = outname
        makeandsavehistogram_called["histlen"] = histlen

    with patch("rapidtide.workflows.histnifti.tide_io.readfromnifti",
               side_effect=mock_readfromnifti), \
         patch("rapidtide.workflows.histnifti.tide_io.savetonifti"), \
         patch("rapidtide.workflows.histnifti.tide_stats.makeandsavehistogram",
               side_effect=mock_makeandsavehistogram):

        histnifti(args)

    assert makeandsavehistogram_called["called"], "makeandsavehistogram should be called for 3D data"
    assert makeandsavehistogram_called["outname"] == "/tmp/test_histnifti_hist"


def histnifti_3d_nozero(debug=False):
    """Test 3D workflow with nozero enabled filters out zero-like values."""
    if debug:
        print("histnifti_3d_nozero")
    xsize, ysize, numslices = 3, 3, 2
    data = _make_3d_data(xsize, ysize, numslices)
    # Set some values to zero
    data[0, 0, 0] = 0.0
    data[1, 1, 0] = 0.005  # Below default nozerothresh of 0.01
    args = _make_default_args()
    args.nozero = True

    dims = _make_dims(xsize, ysize, numslices, 1)
    sizes = _make_sizes()
    hdr = _make_mock_hdr(xsize, ysize, numslices, 1)

    captured_data = {}

    def mock_readfromnifti(fname, **kwargs):
        return (MagicMock(), data, hdr, dims, sizes)

    def mock_makeandsavehistogram(indata, histlen, endtrim, outname, **kwargs):
        captured_data["indata"] = np.array(indata).copy()

    with patch("rapidtide.workflows.histnifti.tide_io.readfromnifti",
               side_effect=mock_readfromnifti), \
         patch("rapidtide.workflows.histnifti.tide_io.savetonifti"), \
         patch("rapidtide.workflows.histnifti.tide_stats.makeandsavehistogram",
               side_effect=mock_makeandsavehistogram):

        histnifti(args)

    # The total number of valid voxels is numspatiallocs (all ones mask)
    # nozero should filter out voxels with abs value < nozerothresh
    total_voxels = xsize * ysize * numslices
    assert len(captured_data["indata"]) < total_voxels, \
        "nozero should exclude some values"


def histnifti_3d_transform(debug=False):
    """Test 3D workflow with transform enabled saves transformed output."""
    if debug:
        print("histnifti_3d_transform")
    xsize, ysize, numslices = 3, 3, 2
    data = _make_3d_data(xsize, ysize, numslices)
    args = _make_default_args()
    args.transform = True

    dims = _make_dims(xsize, ysize, numslices, 1)
    sizes = _make_sizes()
    hdr = _make_mock_hdr(xsize, ysize, numslices, 1)

    saved_nifti = {}

    def mock_readfromnifti(fname, **kwargs):
        return (MagicMock(), data, hdr, dims, sizes)

    def mock_savetonifti(arr, hdr_arg, fname, **kwargs):
        saved_nifti[fname] = arr.copy()

    with patch("rapidtide.workflows.histnifti.tide_io.readfromnifti",
               side_effect=mock_readfromnifti), \
         patch("rapidtide.workflows.histnifti.tide_io.savetonifti",
               side_effect=mock_savetonifti), \
         patch("rapidtide.workflows.histnifti.tide_stats.makeandsavehistogram"):

        histnifti(args)

    assert "/tmp/test_histnifti_transformed" in saved_nifti
    transformed = saved_nifti["/tmp/test_histnifti_transformed"]
    assert transformed.shape == (xsize, ysize, numslices)
    # Transformed values should be percentile scores (0-100 range)
    assert np.min(transformed) >= 0.0
    assert np.max(transformed) <= 100.0


def histnifti_3d_custom_histlen(debug=False):
    """Test 3D workflow with custom histogram length."""
    if debug:
        print("histnifti_3d_custom_histlen")
    xsize, ysize, numslices = 3, 3, 2
    data = _make_3d_data(xsize, ysize, numslices)
    args = _make_default_args()
    args.histlen = 25

    dims = _make_dims(xsize, ysize, numslices, 1)
    sizes = _make_sizes()
    hdr = _make_mock_hdr(xsize, ysize, numslices, 1)

    captured = {}

    def mock_readfromnifti(fname, **kwargs):
        return (MagicMock(), data, hdr, dims, sizes)

    def mock_makeandsavehistogram(indata, histlen, endtrim, outname, **kwargs):
        captured["histlen"] = histlen

    with patch("rapidtide.workflows.histnifti.tide_io.readfromnifti",
               side_effect=mock_readfromnifti), \
         patch("rapidtide.workflows.histnifti.tide_io.savetonifti"), \
         patch("rapidtide.workflows.histnifti.tide_stats.makeandsavehistogram",
               side_effect=mock_makeandsavehistogram):

        histnifti(args)

    assert captured["histlen"] == 25


def histnifti_3d_normhist(debug=False):
    """Test 3D workflow passes normalize flag to makeandsavehistogram."""
    if debug:
        print("histnifti_3d_normhist")
    xsize, ysize, numslices = 3, 3, 2
    data = _make_3d_data(xsize, ysize, numslices)
    args = _make_default_args()
    args.normhist = True

    dims = _make_dims(xsize, ysize, numslices, 1)
    sizes = _make_sizes()
    hdr = _make_mock_hdr(xsize, ysize, numslices, 1)

    captured = {}

    def mock_readfromnifti(fname, **kwargs):
        return (MagicMock(), data, hdr, dims, sizes)

    def mock_makeandsavehistogram(indata, histlen, endtrim, outname, **kwargs):
        captured["normalize"] = kwargs.get("normalize", False)

    with patch("rapidtide.workflows.histnifti.tide_io.readfromnifti",
               side_effect=mock_readfromnifti), \
         patch("rapidtide.workflows.histnifti.tide_io.savetonifti"), \
         patch("rapidtide.workflows.histnifti.tide_stats.makeandsavehistogram",
               side_effect=mock_makeandsavehistogram):

        histnifti(args)

    assert captured["normalize"] is True


def histnifti_3d_display_flag(debug=False):
    """Test 3D workflow passes display flag to makeandsavehistogram."""
    if debug:
        print("histnifti_3d_display_flag")
    xsize, ysize, numslices = 3, 3, 2
    data = _make_3d_data(xsize, ysize, numslices)
    args = _make_default_args()
    args.display = False

    dims = _make_dims(xsize, ysize, numslices, 1)
    sizes = _make_sizes()
    hdr = _make_mock_hdr(xsize, ysize, numslices, 1)

    captured = {}

    def mock_readfromnifti(fname, **kwargs):
        return (MagicMock(), data, hdr, dims, sizes)

    def mock_makeandsavehistogram(indata, histlen, endtrim, outname, **kwargs):
        captured["displayplots"] = kwargs.get("displayplots", True)

    with patch("rapidtide.workflows.histnifti.tide_io.readfromnifti",
               side_effect=mock_readfromnifti), \
         patch("rapidtide.workflows.histnifti.tide_io.savetonifti"), \
         patch("rapidtide.workflows.histnifti.tide_stats.makeandsavehistogram",
               side_effect=mock_makeandsavehistogram):

        histnifti(args)

    assert captured["displayplots"] is False


# ==================== Main test function ====================


def test_histnifti(debug=False):
    # _get_parser tests
    if debug:
        print("Running parser tests")
    parser_basic(debug=debug)
    parser_required_args(debug=debug)
    parser_defaults(debug=debug)
    parser_histlen(debug=debug)
    parser_minval_maxval(debug=debug)
    parser_boolean_flags(debug=debug)
    parser_nozerothresh(debug=debug)
    parser_maskfile(debug=debug)

    # histnifti 4D tests
    if debug:
        print("Running histnifti 4D tests")
    histnifti_4d_basic(debug=debug)
    histnifti_4d_sorted_shape(debug=debug)
    histnifti_4d_sorted_is_sorted(debug=debug)
    histnifti_4d_pcts_shape(debug=debug)
    histnifti_4d_pcts_ordered(debug=debug)
    histnifti_4d_hists_shape(debug=debug)
    histnifti_4d_hists_nonnegative(debug=debug)
    histnifti_4d_custom_histlen(debug=debug)
    histnifti_4d_minval_maxval(debug=debug)
    histnifti_4d_robustrange(debug=debug)
    histnifti_4d_nozero(debug=debug)
    histnifti_4d_with_mask(debug=debug)
    histnifti_4d_mask_mismatch(debug=debug)

    # histnifti 3D tests
    if debug:
        print("Running histnifti 3D tests")
    histnifti_3d_basic(debug=debug)
    histnifti_3d_nozero(debug=debug)
    histnifti_3d_transform(debug=debug)
    histnifti_3d_custom_histlen(debug=debug)
    histnifti_3d_normhist(debug=debug)
    histnifti_3d_display_flag(debug=debug)


if __name__ == "__main__":
    test_histnifti(debug=True)
