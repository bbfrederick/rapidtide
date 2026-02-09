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

from rapidtide.tests.utils import create_dir, get_test_temp_path
from rapidtide.workflows.spatialmi import _get_parser, getMI, getneighborhood, spatialmi

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


def _make_dims(xsize, ysize, numslices, timepoints):
    return np.array([4, xsize, ysize, numslices, timepoints, 1, 1, 1])


def _make_sizes():
    return np.array([1.0, 2.0, 2.0, 2.0, 1.5, 1.0, 1.0, 1.0])


def _reset_getneighborhood_globals():
    """Reset the global variables used by getneighborhood for caching."""
    import rapidtide.workflows.spatialmi as sm
    for gname in ("kernel", "usedwidth", "indexlist", "usedradius"):
        if gname in dir(sm):
            try:
                delattr(sm, gname)
            except AttributeError:
                pass
    # Also reset via globals dict in the module
    for gname in ("kernel", "usedwidth", "indexlist", "usedradius"):
        if gname in sm.__dict__:
            del sm.__dict__[gname]


def _reset_getMI_globals():
    """Reset the global thebins variable used by getMI."""
    import rapidtide.workflows.spatialmi as sm
    sm.thebins = None


def _make_default_args(outputroot="/tmp/test_spatialmi"):
    """Create a default Namespace with all required args."""
    return argparse.Namespace(
        inputfilename1="dummy_input1.nii.gz",
        maskfilename1="dummy_mask1.nii.gz",
        inputfilename2="dummy_input2.nii.gz",
        maskfilename2="dummy_mask2.nii.gz",
        outputroot=outputroot,
        radius=2.0,
        sigma=None,
        spherical=False,
        kernelwidth=None,
        norm=True,
        prebin=True,
        index1=0,
        index2=0,
        debug=False,
    )


# ==================== _get_parser tests ====================


def parser_basic(debug=False):
    """Test that parser creates successfully."""
    if debug:
        print("parser_basic")
    parser = _get_parser()
    assert parser is not None
    assert parser.prog == "spatialmi"


def parser_required_args(debug=False):
    """Test parser has required positional arguments."""
    if debug:
        print("parser_required_args")
    parser = _get_parser()
    actions = {a.dest: a for a in parser._actions}
    assert "inputfilename1" in actions
    assert "maskfilename1" in actions
    assert "inputfilename2" in actions
    assert "maskfilename2" in actions
    assert "outputroot" in actions


def parser_defaults(debug=False):
    """Test default values for optional arguments."""
    if debug:
        print("parser_defaults")
    parser = _get_parser()
    args = parser.parse_args(["in1.nii", "mask1.nii", "in2.nii", "mask2.nii", "outroot"])
    assert args.prebin is True
    assert args.norm is True
    assert args.radius == 2.0
    assert args.sigma is None
    assert args.kernelwidth is None
    assert args.spherical is False
    assert args.index1 == 0
    assert args.index2 == 0
    assert args.debug is False


def parser_optional_args(debug=False):
    """Test optional arguments parsing."""
    if debug:
        print("parser_optional_args")
    parser = _get_parser()
    args = parser.parse_args([
        "in1.nii", "mask1.nii", "in2.nii", "mask2.nii", "outroot",
        "--noprebin", "--nonorm", "--radius", "3.0", "--sigma", "1.5",
        "--kernelwidth", "2.0", "--spherical", "--index1", "2", "--index2", "3",
        "--debug",
    ])
    assert args.prebin is False
    assert args.norm is False
    assert args.radius == 3.0
    assert args.sigma == 1.5
    assert args.kernelwidth == 2.0
    assert args.spherical is True
    assert args.index1 == 2
    assert args.index2 == 3
    assert args.debug is True


# ==================== getneighborhood tests ====================


def getneighborhood_cubic_center(debug=False):
    """Test cubic neighborhood extraction at center of volume."""
    if debug:
        print("getneighborhood_cubic_center")
    _reset_getneighborhood_globals()

    xsize, ysize, zsize = 10, 10, 10
    data = np.ones((xsize, ysize, zsize), dtype=np.float64)
    radius = 1.0

    result = getneighborhood(
        data, 5, 5, 5, xsize, ysize, zsize, radius,
        spherical=False, kernelwidth=None,
    )
    # With radius=1, cubic neighborhood is (2*1+1)^3 = 27 voxels
    assert len(result) == 27
    # All values are 1.0 (kernel is all 1.0 when kernelwidth=None)
    assert np.allclose(result, 1.0)


def getneighborhood_cubic_with_kernel(debug=False):
    """Test cubic neighborhood with gaussian kernel weighting."""
    if debug:
        print("getneighborhood_cubic_with_kernel")
    _reset_getneighborhood_globals()

    xsize, ysize, zsize = 10, 10, 10
    data = np.ones((xsize, ysize, zsize), dtype=np.float64)
    radius = 1.0

    result = getneighborhood(
        data, 5, 5, 5, xsize, ysize, zsize, radius,
        spherical=False, kernelwidth=1.5,
    )
    # Should have 27 elements
    assert len(result) == 27
    # Since data is all 1.0, the result is the kernel values
    # The max value should be 1.0 (Gaussian peak at distance 0)
    assert np.max(result) == pytest.approx(1.0, abs=0.01)
    # Corner values (distance=sqrt(3)) should be less than center
    assert np.min(result) < np.max(result)


def getneighborhood_cubic_edge(debug=False):
    """Test cubic neighborhood at edge of volume (clipped)."""
    if debug:
        print("getneighborhood_cubic_edge")
    _reset_getneighborhood_globals()

    xsize, ysize, zsize = 10, 10, 10
    data = np.ones((xsize, ysize, zsize), dtype=np.float64)
    radius = 2.0

    # At corner (0,0,0) with radius=2, neighborhood is clipped
    result = getneighborhood(
        data, 0, 0, 0, xsize, ysize, zsize, radius,
        spherical=False, kernelwidth=None,
    )
    # At corner, we get 0:3 in each dimension = 3^3 = 27 voxels (clipped)
    expected_size = 3 * 3 * 3  # 0 to radius+1 in each dim
    assert len(result) == expected_size


def getneighborhood_cubic_values(debug=False):
    """Test that cubic neighborhood extracts correct values."""
    if debug:
        print("getneighborhood_cubic_values")
    _reset_getneighborhood_globals()

    xsize, ysize, zsize = 10, 10, 10
    data = np.zeros((xsize, ysize, zsize), dtype=np.float64)
    # Set a known pattern
    data[4:7, 4:7, 4:7] = 1.0
    radius = 1.0

    result = getneighborhood(
        data, 5, 5, 5, xsize, ysize, zsize, radius,
        spherical=False, kernelwidth=None,
    )
    # The entire (2*1+1)^3 = 3^3 block should be 1.0
    assert np.allclose(result, 1.0)


def getneighborhood_spherical_center(debug=False):
    """Test spherical neighborhood extraction at center of volume."""
    if debug:
        print("getneighborhood_spherical_center")
    _reset_getneighborhood_globals()

    xsize, ysize, zsize = 10, 10, 10
    data = np.ones((xsize, ysize, zsize), dtype=np.float64)
    radius = 2.0

    result = getneighborhood(
        data, 5, 5, 5, xsize, ysize, zsize, radius,
        spherical=True,
    )
    # Spherical neighborhood with radius=2 should have fewer voxels
    # than cubic (5^3=125), but more than just the center
    assert len(result) > 1
    assert len(result) < 125
    # All values should be 1.0
    assert np.allclose(result, 1.0)


def getneighborhood_spherical_edge(debug=False):
    """Test spherical neighborhood at edge is clipped to valid indices."""
    if debug:
        print("getneighborhood_spherical_edge")
    _reset_getneighborhood_globals()

    xsize, ysize, zsize = 10, 10, 10
    data = np.ones((xsize, ysize, zsize), dtype=np.float64)
    radius = 2.0

    # At edge, should get fewer voxels than at center
    result_center = getneighborhood(
        data, 5, 5, 5, xsize, ysize, zsize, radius,
        spherical=True,
    )
    _reset_getneighborhood_globals()
    result_edge = getneighborhood(
        data, 0, 0, 0, xsize, ysize, zsize, radius,
        spherical=True,
    )
    assert len(result_edge) < len(result_center)


def getneighborhood_spherical_values(debug=False):
    """Test spherical neighborhood returns correct data values."""
    if debug:
        print("getneighborhood_spherical_values")
    _reset_getneighborhood_globals()

    xsize, ysize, zsize = 10, 10, 10
    rng = np.random.RandomState(42)
    data = rng.randn(xsize, ysize, zsize).astype(np.float64)
    radius = 1.0

    result = getneighborhood(
        data, 5, 5, 5, xsize, ysize, zsize, radius,
        spherical=True,
    )
    # With radius=1, spherical neighborhood includes center + 6 face neighbors = 7
    assert len(result) == 7
    # Center value should be included
    assert data[5, 5, 5] in result


def getneighborhood_radius_size_increases(debug=False):
    """Test that larger radius produces larger neighborhood."""
    if debug:
        print("getneighborhood_radius_size_increases")
    xsize, ysize, zsize = 15, 15, 15
    data = np.ones((xsize, ysize, zsize), dtype=np.float64)

    _reset_getneighborhood_globals()
    result_r1 = getneighborhood(
        data, 7, 7, 7, xsize, ysize, zsize, 1.0,
        spherical=True,
    )
    _reset_getneighborhood_globals()
    result_r2 = getneighborhood(
        data, 7, 7, 7, xsize, ysize, zsize, 2.0,
        spherical=True,
    )
    _reset_getneighborhood_globals()
    result_r3 = getneighborhood(
        data, 7, 7, 7, xsize, ysize, zsize, 3.0,
        spherical=True,
    )
    assert len(result_r1) < len(result_r2) < len(result_r3)


# ==================== getMI tests ====================


def getMI_identical_signals(debug=False):
    """Test getMI with identical signals gives high MI."""
    if debug:
        print("getMI_identical_signals")
    _reset_getMI_globals()

    rng = np.random.RandomState(42)
    x = rng.randn(100).astype(np.float64)
    y = x.copy()

    mi = getMI(x, y, norm=True, bins=10, init=True, prebin=True)
    assert mi is not None
    assert mi > 0.0


def getMI_independent_signals(debug=False):
    """Test getMI with independent signals gives lower MI than identical."""
    if debug:
        print("getMI_independent_signals")
    _reset_getMI_globals()

    rng = np.random.RandomState(42)
    x = rng.randn(200).astype(np.float64)
    y = rng.randn(200).astype(np.float64)

    mi_indep = getMI(x, y, norm=True, bins=10, init=True, prebin=True)

    _reset_getMI_globals()
    mi_same = getMI(x, x.copy(), norm=True, bins=10, init=True, prebin=True)

    assert mi_same > mi_indep, \
        f"MI of identical ({mi_same}) should exceed MI of independent ({mi_indep})"


def getMI_correlated_signals(debug=False):
    """Test getMI with correlated signals gives higher MI than independent."""
    if debug:
        print("getMI_correlated_signals")
    _reset_getMI_globals()

    rng = np.random.RandomState(42)
    x = rng.randn(200).astype(np.float64)
    y = 0.8 * x + 0.2 * rng.randn(200).astype(np.float64)

    mi_corr = getMI(x, y, norm=True, bins=10, init=True, prebin=True)

    _reset_getMI_globals()
    y_indep = rng.randn(200).astype(np.float64)
    mi_indep = getMI(x, y_indep, norm=True, bins=10, init=True, prebin=True)

    assert mi_corr > mi_indep, \
        f"MI of correlated ({mi_corr}) should exceed MI of independent ({mi_indep})"


def getMI_no_norm(debug=False):
    """Test getMI without normalization."""
    if debug:
        print("getMI_no_norm")
    _reset_getMI_globals()

    rng = np.random.RandomState(42)
    x = rng.randn(100).astype(np.float64)
    y = x.copy()

    mi = getMI(x, y, norm=False, bins=10, init=True, prebin=True)
    assert mi is not None
    assert mi > 0.0


def getMI_no_prebin(debug=False):
    """Test getMI with prebin=False uses slow path."""
    if debug:
        print("getMI_no_prebin")
    _reset_getMI_globals()

    rng = np.random.RandomState(42)
    x = rng.randn(100).astype(np.float64)
    y = x.copy()

    mi = getMI(x, y, norm=True, bins=10, init=True, prebin=False)
    assert mi is not None
    assert mi > 0.0


def getMI_auto_bins(debug=False):
    """Test getMI with bins=-1 auto-selects bin count."""
    if debug:
        print("getMI_auto_bins")
    _reset_getMI_globals()

    rng = np.random.RandomState(42)
    x = rng.randn(200).astype(np.float64)
    y = x.copy()

    mi = getMI(x, y, norm=True, bins=-1, init=True, prebin=True)
    assert mi is not None
    assert mi > 0.0


def getMI_nonnegative(debug=False):
    """Test getMI always returns non-negative values."""
    if debug:
        print("getMI_nonnegative")
    _reset_getMI_globals()

    rng = np.random.RandomState(42)
    for _ in range(5):
        x = rng.randn(100).astype(np.float64)
        y = rng.randn(100).astype(np.float64)
        mi = getMI(x, y, norm=True, bins=10, init=True, prebin=True)
        assert mi >= 0.0, f"MI should be non-negative, got {mi}"
        _reset_getMI_globals()


# ==================== spatialmi tests ====================


def _run_spatialmi(image1, image2, mask1, mask2, args, image1_4d=False, image2_4d=False):
    """Helper to run spatialmi with mocked IO and CLI parsing. Returns saved files dict."""
    xsize, ysize, numslices = image1.shape[:3]

    if image1_4d:
        timepoints1 = image1.shape[3]
        data1 = image1
    else:
        timepoints1 = 1
        data1 = image1

    if image2_4d:
        timepoints2 = image2.shape[3]
        data2 = image2
    else:
        timepoints2 = 1
        data2 = image2

    dims1 = _make_dims(xsize, ysize, numslices, timepoints1)
    dims2 = _make_dims(xsize, ysize, numslices, timepoints2)
    mask_dims = _make_dims(xsize, ysize, numslices, 1)
    sizes = _make_sizes()

    hdr1 = _make_mock_hdr(xsize, ysize, numslices, timepoints1)
    hdr2 = _make_mock_hdr(xsize, ysize, numslices, timepoints2)
    mask_hdr1 = _make_mock_hdr(xsize, ysize, numslices, 1)
    mask_hdr2 = _make_mock_hdr(xsize, ysize, numslices, 1)

    saved_nifti = {}

    read_count = {"n": 0}

    def mock_readfromnifti(fname, **kwargs):
        read_count["n"] += 1
        if read_count["n"] == 1:
            return (MagicMock(), data1, hdr1, dims1, sizes)
        elif read_count["n"] == 2:
            return (MagicMock(), mask1, mask_hdr1, mask_dims, sizes)
        elif read_count["n"] == 3:
            return (MagicMock(), data2, hdr2, dims2, sizes)
        elif read_count["n"] == 4:
            return (MagicMock(), mask2, mask_hdr2, mask_dims, sizes)

    def mock_savetonifti(arr, hdr_arg, fname, **kwargs):
        saved_nifti[fname] = arr.copy()

    _reset_getneighborhood_globals()
    _reset_getMI_globals()

    with patch("rapidtide.workflows.spatialmi.tide_io.readfromnifti",
               side_effect=mock_readfromnifti), \
         patch("rapidtide.workflows.spatialmi.tide_io.checkspacedimmatch",
               return_value=True), \
         patch("rapidtide.workflows.spatialmi.tide_io.savetonifti",
               side_effect=mock_savetonifti):

        spatialmi(args)

    return saved_nifti


def spatialmi_basic_3d(debug=False):
    """Test basic spatialmi workflow with 3D inputs produces output."""
    if debug:
        print("spatialmi_basic_3d")
    xsize, ysize, numslices = 7, 7, 5
    rng = np.random.RandomState(42)
    image1 = rng.randn(xsize, ysize, numslices).astype(np.float64)
    image2 = rng.randn(xsize, ysize, numslices).astype(np.float64)
    mask1 = np.ones((xsize, ysize, numslices), dtype=np.float64)
    mask2 = np.ones((xsize, ysize, numslices), dtype=np.float64)

    args = _make_default_args()
    args.radius = 1.0

    saved = _run_spatialmi(image1, image2, mask1, mask2, args)

    assert "/tmp/test_spatialmi_result" in saved
    assert saved["/tmp/test_spatialmi_result"].shape == (xsize, ysize, numslices)


def spatialmi_identical_images(debug=False):
    """Test that identical images produce higher MI than different images."""
    if debug:
        print("spatialmi_identical_images")
    xsize, ysize, numslices = 7, 7, 5
    rng = np.random.RandomState(42)
    image1 = rng.randn(xsize, ysize, numslices).astype(np.float64) * 10.0
    mask1 = np.ones((xsize, ysize, numslices), dtype=np.float64)
    mask2 = np.ones((xsize, ysize, numslices), dtype=np.float64)

    # Identical images
    args = _make_default_args(outputroot="/tmp/test_spatialmi_same")
    args.radius = 1.0
    saved_same = _run_spatialmi(image1, image1.copy(), mask1, mask2, args)

    # Different images
    image2 = rng.randn(xsize, ysize, numslices).astype(np.float64) * 10.0
    args2 = _make_default_args(outputroot="/tmp/test_spatialmi_diff")
    args2.radius = 1.0
    saved_diff = _run_spatialmi(image1, image2, mask1, mask2, args2)

    mi_same = saved_same["/tmp/test_spatialmi_same_result"]
    mi_diff = saved_diff["/tmp/test_spatialmi_diff_result"]

    # Mean MI should be higher for identical images
    assert np.mean(mi_same) > np.mean(mi_diff), \
        f"Mean MI for identical ({np.mean(mi_same)}) should exceed different ({np.mean(mi_diff)})"


def spatialmi_output_nonnegative(debug=False):
    """Test that output MI values are non-negative after nan_to_num."""
    if debug:
        print("spatialmi_output_nonnegative")
    xsize, ysize, numslices = 7, 7, 5
    rng = np.random.RandomState(42)
    image1 = rng.randn(xsize, ysize, numslices).astype(np.float64)
    image2 = rng.randn(xsize, ysize, numslices).astype(np.float64)
    mask1 = np.ones((xsize, ysize, numslices), dtype=np.float64)
    mask2 = np.ones((xsize, ysize, numslices), dtype=np.float64)

    args = _make_default_args()
    args.radius = 1.0

    saved = _run_spatialmi(image1, image2, mask1, mask2, args)

    result = saved["/tmp/test_spatialmi_result"]
    assert np.all(np.isfinite(result)), "Output should have no NaN/Inf values"
    # MI values should be non-negative (allow tiny floating point noise)
    assert np.all(result >= -1e-10), \
        f"MI values should be non-negative, min={np.min(result)}"


def spatialmi_with_mask(debug=False):
    """Test that masked-out voxels have zero MI in output."""
    if debug:
        print("spatialmi_with_mask")
    xsize, ysize, numslices = 7, 7, 5
    rng = np.random.RandomState(42)
    image1 = rng.randn(xsize, ysize, numslices).astype(np.float64) * 10.0
    image2 = rng.randn(xsize, ysize, numslices).astype(np.float64) * 10.0
    mask1 = np.ones((xsize, ysize, numslices), dtype=np.float64)
    mask2 = np.ones((xsize, ysize, numslices), dtype=np.float64)
    # Mask out a region
    mask1[0, :, :] = 0.0

    args = _make_default_args()
    args.radius = 1.0

    saved = _run_spatialmi(image1, image2, mask1, mask2, args)

    result = saved["/tmp/test_spatialmi_result"]
    # Where totalmask (mask1 * mask2) is 0, output should be 0
    assert np.all(result[0, :, :] == 0.0), \
        "Masked-out voxels should have zero MI"


def spatialmi_radius_check(debug=False):
    """Test that radius < 1.0 causes exit."""
    if debug:
        print("spatialmi_radius_check")
    args = _make_default_args()
    args.radius = 0.5

    with pytest.raises(SystemExit):
        spatialmi(args)


def spatialmi_dim_mismatch_input_mask(debug=False):
    """Test that dimension mismatch between input and mask causes exit."""
    if debug:
        print("spatialmi_dim_mismatch_input_mask")
    xsize, ysize, numslices = 7, 7, 5
    image1 = np.ones((xsize, ysize, numslices), dtype=np.float64)

    dims = _make_dims(xsize, ysize, numslices, 1)
    sizes = _make_sizes()
    hdr = _make_mock_hdr(xsize, ysize, numslices, 1)

    def mock_readfromnifti(fname, **kwargs):
        return (MagicMock(), image1, hdr, dims, sizes)

    call_count = {"n": 0}

    def mock_checkspacedimmatch(d1, d2, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return False  # input1 vs mask1 mismatch
        return True

    args = _make_default_args()

    with patch("rapidtide.workflows.spatialmi.tide_io.readfromnifti",
               side_effect=mock_readfromnifti), \
         patch("rapidtide.workflows.spatialmi.tide_io.checkspacedimmatch",
               side_effect=mock_checkspacedimmatch):
        with pytest.raises(SystemExit):
            spatialmi(args)


def spatialmi_4d_input(debug=False):
    """Test spatialmi with 4D input selects correct timepoint."""
    if debug:
        print("spatialmi_4d_input")
    xsize, ysize, numslices = 7, 7, 5
    rng = np.random.RandomState(42)
    # 4D data with 3 timepoints
    data_4d = rng.randn(xsize, ysize, numslices, 3).astype(np.float64) * 10.0
    image2 = rng.randn(xsize, ysize, numslices).astype(np.float64) * 10.0
    mask1 = np.ones((xsize, ysize, numslices), dtype=np.float64)
    mask2 = np.ones((xsize, ysize, numslices), dtype=np.float64)

    args = _make_default_args()
    args.radius = 1.0
    args.index1 = 1  # Select second timepoint

    saved = _run_spatialmi(data_4d, image2, mask1, mask2, args, image1_4d=True)

    assert "/tmp/test_spatialmi_result" in saved
    assert saved["/tmp/test_spatialmi_result"].shape == (xsize, ysize, numslices)


def spatialmi_spherical_neighborhood(debug=False):
    """Test spatialmi with spherical neighborhood option."""
    if debug:
        print("spatialmi_spherical_neighborhood")
    xsize, ysize, numslices = 7, 7, 5
    rng = np.random.RandomState(42)
    image1 = rng.randn(xsize, ysize, numslices).astype(np.float64) * 10.0
    image2 = rng.randn(xsize, ysize, numslices).astype(np.float64) * 10.0
    mask1 = np.ones((xsize, ysize, numslices), dtype=np.float64)
    mask2 = np.ones((xsize, ysize, numslices), dtype=np.float64)

    args = _make_default_args()
    args.radius = 1.0
    args.spherical = True

    saved = _run_spatialmi(image1, image2, mask1, mask2, args)

    assert "/tmp/test_spatialmi_result" in saved
    assert saved["/tmp/test_spatialmi_result"].shape == (xsize, ysize, numslices)


# ==================== Main test function ====================


def test_spatialmi(debug=False):
    # _get_parser tests
    if debug:
        print("Running parser tests")
    parser_basic(debug=debug)
    parser_required_args(debug=debug)
    parser_defaults(debug=debug)
    parser_optional_args(debug=debug)

    # getneighborhood tests
    if debug:
        print("Running getneighborhood tests")
    getneighborhood_cubic_center(debug=debug)
    getneighborhood_cubic_with_kernel(debug=debug)
    getneighborhood_cubic_edge(debug=debug)
    getneighborhood_cubic_values(debug=debug)
    getneighborhood_spherical_center(debug=debug)
    getneighborhood_spherical_edge(debug=debug)
    getneighborhood_spherical_values(debug=debug)
    getneighborhood_radius_size_increases(debug=debug)

    # getMI tests
    if debug:
        print("Running getMI tests")
    getMI_identical_signals(debug=debug)
    getMI_independent_signals(debug=debug)
    getMI_correlated_signals(debug=debug)
    getMI_no_norm(debug=debug)
    getMI_no_prebin(debug=debug)
    getMI_auto_bins(debug=debug)
    getMI_nonnegative(debug=debug)

    # spatialmi workflow tests
    if debug:
        print("Running spatialmi workflow tests")
    spatialmi_basic_3d(debug=debug)
    spatialmi_identical_images(debug=debug)
    spatialmi_output_nonnegative(debug=debug)
    spatialmi_with_mask(debug=debug)
    spatialmi_radius_check(debug=debug)
    spatialmi_dim_mismatch_input_mask(debug=debug)
    spatialmi_4d_input(debug=debug)
    spatialmi_spherical_neighborhood(debug=debug)


if __name__ == "__main__":
    test_spatialmi(debug=True)
