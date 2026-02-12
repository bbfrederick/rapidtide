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
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rapidtide.tests.utils import create_dir, get_test_temp_path
from rapidtide.workflows.niftidecomp import (
    _get_parser,
    _get_parser_spatial,
    _get_parser_temporal,
    main,
    main_spatial,
    main_temporal,
    niftidecomp_workflow,
    transposeifspatial,
)

# ==================== Helpers ====================


def _make_mock_hdr(xsize, ysize, numslices, timepoints=1):
    """Create a mock NIfTI header that supports __getitem__, __setitem__, copy."""
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


def _make_4d_data(xsize=4, ysize=4, numslices=3, numtimepoints=20):
    """Create 4D data with varying timecourses suitable for decomposition."""
    rng = np.random.RandomState(42)
    # Create data with actual signal structure for PCA to work on
    numvoxels = xsize * ysize * numslices
    # Generate a few source signals
    t = np.linspace(0, 4 * np.pi, numtimepoints)
    sources = np.array(
        [
            np.sin(t),
            np.cos(t),
            np.sin(2 * t),
        ]
    )
    # Mix sources into voxels
    mixing = rng.rand(numvoxels, 3)
    data_2d = mixing @ sources + 0.1 * rng.randn(numvoxels, numtimepoints)
    # Ensure non-zero variance everywhere and add a baseline
    data_2d += 100.0
    data = data_2d.reshape((xsize, ysize, numslices, numtimepoints))
    hdr = _make_mock_hdr(xsize, ysize, numslices, numtimepoints)
    dims = np.array([4, xsize, ysize, numslices, numtimepoints, 1, 1, 1])
    sizes = np.array([1.0, 2.0, 2.0, 2.0, 1.5, 1.0, 1.0, 1.0])
    return data, hdr, dims, sizes


# ==================== _get_parser tests ====================


def parser_temporal_basic(debug=False):
    """Test parser creation for temporal mode."""
    if debug:
        print("parser_temporal_basic")
    parser = _get_parser("temporal")
    assert parser is not None
    assert parser.prog == "temporaldecomp"


def parser_spatial_basic(debug=False):
    """Test parser creation for spatial mode."""
    if debug:
        print("parser_spatial_basic")
    parser = _get_parser("spatial")
    assert parser is not None
    assert parser.prog == "spatialdecomp"


def parser_invalid_axis(debug=False):
    """Test that invalid axis raises ValueError."""
    if debug:
        print("parser_invalid_axis")
    with pytest.raises(ValueError):
        _get_parser("invalid")


def parser_defaults(debug=False):
    """Test parser default values."""
    if debug:
        print("parser_defaults")
    parser = _get_parser("temporal")
    with tempfile.NamedTemporaryFile(suffix=".nii") as f:
        args = parser.parse_args([f.name, "outroot"])
    assert args.datamaskname is None
    assert args.ncomp == -1.0
    assert args.sigma == 0.0
    assert args.decomptype == "pca"
    assert args.demean is True
    assert args.varnorm is True


def parser_decomptype_choices(debug=False):
    """Test that decomptype accepts pca, ica, sparse."""
    if debug:
        print("parser_decomptype_choices")
    parser = _get_parser("temporal")
    with tempfile.NamedTemporaryFile(suffix=".nii") as f:
        for dtype in ["pca", "ica", "sparse"]:
            args = parser.parse_args([f.name, "out", "--type", dtype])
            assert args.decomptype == dtype


def parser_nodemean(debug=False):
    """Test --nodemean flag."""
    if debug:
        print("parser_nodemean")
    parser = _get_parser("temporal")
    with tempfile.NamedTemporaryFile(suffix=".nii") as f:
        args = parser.parse_args([f.name, "out", "--nodemean"])
    assert args.demean is False


def parser_novarnorm(debug=False):
    """Test --novarnorm flag."""
    if debug:
        print("parser_novarnorm")
    parser = _get_parser("temporal")
    with tempfile.NamedTemporaryFile(suffix=".nii") as f:
        args = parser.parse_args([f.name, "out", "--novarnorm"])
    assert args.varnorm is False


def parser_ncomp(debug=False):
    """Test --ncomp option."""
    if debug:
        print("parser_ncomp")
    parser = _get_parser("temporal")
    with tempfile.NamedTemporaryFile(suffix=".nii") as f:
        args = parser.parse_args([f.name, "out", "--ncomp", "5"])
    assert args.ncomp == 5.0


def parser_smooth(debug=False):
    """Test --smooth option."""
    if debug:
        print("parser_smooth")
    parser = _get_parser("spatial")
    with tempfile.NamedTemporaryFile(suffix=".nii") as f:
        args = parser.parse_args([f.name, "out", "--smooth", "3.5"])
    assert args.sigma == 3.5


# ==================== _get_parser_temporal / _get_parser_spatial tests ====================


def parser_temporal_wrapper(debug=False):
    """Test _get_parser_temporal convenience wrapper."""
    if debug:
        print("parser_temporal_wrapper")
    parser = _get_parser_temporal()
    assert parser.prog == "temporaldecomp"


def parser_spatial_wrapper(debug=False):
    """Test _get_parser_spatial convenience wrapper."""
    if debug:
        print("parser_spatial_wrapper")
    parser = _get_parser_spatial()
    assert parser.prog == "spatialdecomp"


# ==================== transposeifspatial tests ====================


def transpose_spatial(debug=False):
    """Test that spatial axis transposes data."""
    if debug:
        print("transpose_spatial")
    data = np.array([[1, 2, 3], [4, 5, 6]])
    result = transposeifspatial(data, decompaxis="spatial")
    expected = np.array([[1, 4], [2, 5], [3, 6]])
    assert np.array_equal(result, expected)


def transpose_temporal(debug=False):
    """Test that temporal axis does not transpose data."""
    if debug:
        print("transpose_temporal")
    data = np.array([[1, 2, 3], [4, 5, 6]])
    result = transposeifspatial(data, decompaxis="temporal")
    assert np.array_equal(result, data)


def transpose_default(debug=False):
    """Test that default axis is temporal (no transpose)."""
    if debug:
        print("transpose_default")
    data = np.array([[1, 2], [3, 4], [5, 6]])
    result = transposeifspatial(data)
    assert np.array_equal(result, data)


def transpose_1d(debug=False):
    """Test transposeifspatial with 1D array."""
    if debug:
        print("transpose_1d")
    data = np.array([1, 2, 3])
    result = transposeifspatial(data, decompaxis="spatial")
    # np.transpose on 1D is a no-op
    assert np.array_equal(result, data)


def transpose_3d(debug=False):
    """Test transposeifspatial with 3D array."""
    if debug:
        print("transpose_3d")
    data = np.ones((2, 3, 4))
    result = transposeifspatial(data, decompaxis="spatial")
    assert result.shape == (4, 3, 2)


# ==================== niftidecomp_workflow tests ====================


def _run_workflow(
    decompaxis="temporal",
    decomptype="pca",
    pcacomponents=0.5,
    icacomponents=None,
    varnorm=True,
    demean=True,
    sigma=0.0,
    datamaskname=None,
    mask_data=None,
    mask_dims=None,
    debug=False,
):
    """Helper to run niftidecomp_workflow with mocked IO."""
    xsize, ysize, numslices, numtimepoints = 4, 4, 3, 20
    input_data, input_hdr, input_dims, input_sizes = _make_4d_data(
        xsize, ysize, numslices, numtimepoints
    )

    call_count = {"n": 0}

    def mock_readfromnifti(fname, **kwargs):
        call_count["n"] += 1
        if datamaskname is not None and "mask" in fname:
            mask_hdr = _make_mock_hdr(xsize, ysize, numslices, 1)
            return MagicMock(), mask_data, mask_hdr, mask_dims, input_sizes
        return MagicMock(), input_data, input_hdr, input_dims, input_sizes

    with (
        patch(
            "rapidtide.workflows.niftidecomp.tide_io.readfromnifti", side_effect=mock_readfromnifti
        ),
        patch(
            "rapidtide.workflows.niftidecomp.tide_io.parseniftidims",
            return_value=(xsize, ysize, numslices, numtimepoints),
        ),
        patch(
            "rapidtide.workflows.niftidecomp.tide_io.parseniftisizes",
            return_value=(2.0, 2.0, 2.0, 1.5),
        ),
        patch("rapidtide.workflows.niftidecomp.tide_io.checkspacedimmatch", return_value=True),
        patch("rapidtide.workflows.niftidecomp.tide_io.checktimematch", return_value=True),
    ):

        result = niftidecomp_workflow(
            decompaxis,
            ["dummy_data.nii.gz"],
            datamaskname=datamaskname,
            decomptype=decomptype,
            pcacomponents=pcacomponents,
            icacomponents=icacomponents,
            varnorm=varnorm,
            demean=demean,
            sigma=sigma,
        )

    return result, xsize, ysize, numslices, numtimepoints


def workflow_pca_temporal(debug=False):
    """Test PCA temporal decomposition."""
    if debug:
        print("workflow_pca_temporal")
    result, xsize, ysize, numslices, numtimepoints = _run_workflow(
        decompaxis="temporal", decomptype="pca", pcacomponents=0.5
    )
    (outputcomponents, outputcoefficients, outinvtrans, exp_var, exp_var_pct, hdr, dims, sizes) = (
        result
    )

    # Components should be (ncomponents, numtimepoints)
    assert outputcomponents.shape[1] == numtimepoints
    # Coefficients should be (xsize, ysize, numslices, ncomponents)
    assert outputcoefficients.shape[:3] == (xsize, ysize, numslices)
    # Inverse transform should be (xsize, ysize, numslices, numtimepoints)
    assert outinvtrans.shape == (xsize, ysize, numslices, numtimepoints)
    # Explained variance arrays should exist
    assert len(exp_var) > 0
    assert len(exp_var_pct) > 0
    # Variance percentages should sum to ~100 (for all components)
    assert np.sum(exp_var_pct) <= 100.1


def workflow_pca_spatial(debug=False):
    """Test PCA spatial decomposition."""
    if debug:
        print("workflow_pca_spatial")
    result, xsize, ysize, numslices, numtimepoints = _run_workflow(
        decompaxis="spatial", decomptype="pca", pcacomponents=0.5
    )
    (outputcomponents, outputcoefficients, outinvtrans, exp_var, exp_var_pct, hdr, dims, sizes) = (
        result
    )

    # Components should be (xsize, ysize, numslices, ncomponents)
    assert outputcomponents.shape[:3] == (xsize, ysize, numslices)
    # Coefficients should be (ncomponents, numtimepoints)
    assert outputcoefficients.ndim == 2
    # Inverse transform should be (xsize, ysize, numslices, numtimepoints)
    assert outinvtrans.shape == (xsize, ysize, numslices, numtimepoints)
    assert len(exp_var) > 0


def workflow_pca_ncomp_fixed(debug=False):
    """Test PCA with fixed number of components."""
    if debug:
        print("workflow_pca_ncomp_fixed")
    ncomp = 3
    result, xsize, ysize, numslices, numtimepoints = _run_workflow(
        decompaxis="temporal", decomptype="pca", pcacomponents=ncomp
    )
    (outputcomponents, outputcoefficients, outinvtrans, exp_var, exp_var_pct, hdr, dims, sizes) = (
        result
    )

    assert outputcomponents.shape[0] == ncomp
    assert outputcomponents.shape[1] == numtimepoints
    assert len(exp_var) == ncomp


def workflow_nodemean(debug=False):
    """Test workflow with demeaning disabled."""
    if debug:
        print("workflow_nodemean")
    result, xsize, ysize, numslices, numtimepoints = _run_workflow(
        decompaxis="temporal",
        decomptype="pca",
        pcacomponents=0.5,
        demean=False,
    )
    (outputcomponents, outputcoefficients, outinvtrans, exp_var, exp_var_pct, hdr, dims, sizes) = (
        result
    )
    assert outinvtrans.shape == (xsize, ysize, numslices, numtimepoints)


def workflow_novarnorm(debug=False):
    """Test workflow with variance normalization disabled."""
    if debug:
        print("workflow_novarnorm")
    result, xsize, ysize, numslices, numtimepoints = _run_workflow(
        decompaxis="temporal",
        decomptype="pca",
        pcacomponents=0.5,
        varnorm=False,
    )
    (outputcomponents, outputcoefficients, outinvtrans, exp_var, exp_var_pct, hdr, dims, sizes) = (
        result
    )
    assert outinvtrans.shape == (xsize, ysize, numslices, numtimepoints)


def workflow_nodemean_novarnorm(debug=False):
    """Test workflow with both demeaning and varnorm disabled."""
    if debug:
        print("workflow_nodemean_novarnorm")
    result, xsize, ysize, numslices, numtimepoints = _run_workflow(
        decompaxis="temporal",
        decomptype="pca",
        pcacomponents=0.5,
        demean=False,
        varnorm=False,
    )
    (outputcomponents, outputcoefficients, outinvtrans, exp_var, exp_var_pct, hdr, dims, sizes) = (
        result
    )
    assert outinvtrans.shape == (xsize, ysize, numslices, numtimepoints)


def workflow_with_3d_mask(debug=False):
    """Test workflow with a 3D mask."""
    if debug:
        print("workflow_with_3d_mask")
    xsize, ysize, numslices = 4, 4, 3
    # Create a mask that includes all voxels
    mask_data = np.ones((xsize, ysize, numslices), dtype=np.float64)
    mask_dims = np.array([3, xsize, ysize, numslices, 1, 1, 1, 1])

    result, _, _, _, numtimepoints = _run_workflow(
        decompaxis="temporal",
        decomptype="pca",
        pcacomponents=0.5,
        datamaskname="dummy_mask.nii.gz",
        mask_data=mask_data,
        mask_dims=mask_dims,
    )
    (outputcomponents, outputcoefficients, outinvtrans, exp_var, exp_var_pct, hdr, dims, sizes) = (
        result
    )
    assert outinvtrans.shape == (xsize, ysize, numslices, numtimepoints)


def workflow_sparse_pca(debug=False):
    """Test SparsePCA decomposition returns zero explained variance."""
    if debug:
        print("workflow_sparse_pca")
    result, xsize, ysize, numslices, numtimepoints = _run_workflow(
        decompaxis="temporal",
        decomptype="sparse",
        pcacomponents=3,
    )
    (outputcomponents, outputcoefficients, outinvtrans, exp_var, exp_var_pct, hdr, dims, sizes) = (
        result
    )
    assert outputcomponents.shape[0] == 3
    # SparsePCA doesn't provide explained variance, so zeros are returned
    assert np.all(exp_var == 0.0)
    assert np.all(exp_var_pct == 0.0)


def workflow_ica_temporal(debug=False):
    """Test ICA temporal decomposition."""
    if debug:
        print("workflow_ica_temporal")
    result, xsize, ysize, numslices, numtimepoints = _run_workflow(
        decompaxis="temporal",
        decomptype="ica",
        icacomponents=3,
    )
    (outputcomponents, outputcoefficients, outinvtrans, exp_var, exp_var_pct, hdr, dims, sizes) = (
        result
    )

    # Components should be (ncomponents, numtimepoints)
    assert outputcomponents.shape[1] == numtimepoints
    # Coefficients should be (xsize, ysize, numslices, ncomponents)
    assert outputcoefficients.shape[:3] == (xsize, ysize, numslices)
    # Inverse transform should be (xsize, ysize, numslices, numtimepoints)
    assert outinvtrans.shape == (xsize, ysize, numslices, numtimepoints)
    # ICA does not provide explained variance, so zeros are returned
    assert np.all(exp_var == 0.0)
    assert np.all(exp_var_pct == 0.0)


def workflow_multiple_files(debug=False):
    """Test workflow with multiple input files (dimension matching)."""
    if debug:
        print("workflow_multiple_files")
    xsize, ysize, numslices, numtimepoints = 4, 4, 3, 20
    input_data, input_hdr, input_dims, input_sizes = _make_4d_data(
        xsize, ysize, numslices, numtimepoints
    )

    def mock_readfromnifti(fname, **kwargs):
        return MagicMock(), input_data, input_hdr, input_dims, input_sizes

    with (
        patch(
            "rapidtide.workflows.niftidecomp.tide_io.readfromnifti", side_effect=mock_readfromnifti
        ),
        patch(
            "rapidtide.workflows.niftidecomp.tide_io.parseniftidims",
            return_value=(xsize, ysize, numslices, numtimepoints),
        ),
        patch(
            "rapidtide.workflows.niftidecomp.tide_io.parseniftisizes",
            return_value=(2.0, 2.0, 2.0, 1.5),
        ),
        patch("rapidtide.workflows.niftidecomp.tide_io.checkspacedimmatch", return_value=True),
        patch("rapidtide.workflows.niftidecomp.tide_io.checktimematch", return_value=True),
    ):

        result = niftidecomp_workflow(
            "temporal",
            ["file1.nii.gz", "file2.nii.gz"],
            decomptype="pca",
            pcacomponents=0.5,
        )

    (outputcomponents, outputcoefficients, outinvtrans, exp_var, exp_var_pct, hdr, dims, sizes) = (
        result
    )
    # With 2 files, totaltimepoints = 2 * numtimepoints = 40
    assert outinvtrans.shape == (xsize, ysize, numslices, numtimepoints * 2)


def workflow_spatial_pca_fixed(debug=False):
    """Test spatial PCA with fixed number of components."""
    if debug:
        print("workflow_spatial_pca_fixed")
    ncomp = 3
    result, xsize, ysize, numslices, numtimepoints = _run_workflow(
        decompaxis="spatial",
        decomptype="pca",
        pcacomponents=ncomp,
    )
    (outputcomponents, outputcoefficients, outinvtrans, exp_var, exp_var_pct, hdr, dims, sizes) = (
        result
    )

    # Spatial components should be (xsize, ysize, numslices, ncomponents)
    assert outputcomponents.shape == (xsize, ysize, numslices, ncomp)
    assert len(exp_var) == ncomp


def workflow_smoothing(debug=False):
    """Test workflow with spatial smoothing enabled."""
    if debug:
        print("workflow_smoothing")
    xsize, ysize, numslices, numtimepoints = 4, 4, 3, 20
    input_data, input_hdr, input_dims, input_sizes = _make_4d_data(
        xsize, ysize, numslices, numtimepoints
    )

    smooth_called = {"count": 0}

    def mock_readfromnifti(fname, **kwargs):
        return MagicMock(), input_data.copy(), input_hdr, input_dims, input_sizes

    def mock_ssmooth(xd, yd, sd, sigma, data):
        smooth_called["count"] += 1
        return data  # Return data unchanged

    with (
        patch(
            "rapidtide.workflows.niftidecomp.tide_io.readfromnifti", side_effect=mock_readfromnifti
        ),
        patch(
            "rapidtide.workflows.niftidecomp.tide_io.parseniftidims",
            return_value=(xsize, ysize, numslices, numtimepoints),
        ),
        patch(
            "rapidtide.workflows.niftidecomp.tide_io.parseniftisizes",
            return_value=(2.0, 2.0, 2.0, 1.5),
        ),
        patch("rapidtide.workflows.niftidecomp.tide_io.checkspacedimmatch", return_value=True),
        patch("rapidtide.workflows.niftidecomp.tide_io.checktimematch", return_value=True),
        patch("rapidtide.workflows.niftidecomp.tide_filt.ssmooth", side_effect=mock_ssmooth),
    ):

        result = niftidecomp_workflow(
            "temporal",
            ["dummy.nii.gz"],
            decomptype="pca",
            pcacomponents=0.5,
            sigma=2.0,
        )

    # ssmooth should be called once per timepoint
    assert smooth_called["count"] == numtimepoints


# ==================== main tests ====================


def main_temporal_pca(debug=False):
    """Test main function with temporal PCA."""
    if debug:
        print("main_temporal_pca")
    xsize, ysize, numslices, numtimepoints = 4, 4, 3, 20
    input_data, input_hdr, input_dims, input_sizes = _make_4d_data(
        xsize, ysize, numslices, numtimepoints
    )

    def mock_readfromnifti(fname, **kwargs):
        return MagicMock(), input_data, input_hdr, input_dims, input_sizes

    args = {
        "ncomp": -1.0,
        "datafile": "dummy_data.nii.gz",
        "datamaskname": None,
        "decomptype": "pca",
        "varnorm": True,
        "demean": True,
        "sigma": 0.0,
        "outputroot": "/tmp/test_niftidecomp_temporal",
    }

    with (
        patch(
            "rapidtide.workflows.niftidecomp.tide_io.readfromnifti", side_effect=mock_readfromnifti
        ),
        patch(
            "rapidtide.workflows.niftidecomp.tide_io.parseniftidims",
            return_value=(xsize, ysize, numslices, numtimepoints),
        ),
        patch(
            "rapidtide.workflows.niftidecomp.tide_io.parseniftisizes",
            return_value=(2.0, 2.0, 2.0, 1.5),
        ),
        patch("rapidtide.workflows.niftidecomp.tide_io.checkspacedimmatch", return_value=True),
        patch("rapidtide.workflows.niftidecomp.tide_io.checktimematch", return_value=True),
        patch("rapidtide.workflows.niftidecomp.tide_io.writevec"),
        patch("rapidtide.workflows.niftidecomp.tide_io.writenpvecs") as mock_writenpvecs,
        patch("rapidtide.workflows.niftidecomp.tide_io.savetonifti") as mock_savetonifti,
    ):

        main("temporal", args)

    # ncomp < 0 => pcacomponents = 0.5, icacomponents = None
    assert args["pcacomponents"] == 0.5
    assert args["icacomponents"] is None

    # Should write explained variance, components, coefficients, and fit
    assert mock_writenpvecs.call_count == 3  # exp_var, exp_var_pct, components
    assert mock_savetonifti.call_count == 2  # coefficients, fit


def main_spatial_pca(debug=False):
    """Test main function with spatial PCA."""
    if debug:
        print("main_spatial_pca")
    xsize, ysize, numslices, numtimepoints = 4, 4, 3, 20
    input_data, input_hdr, input_dims, input_sizes = _make_4d_data(
        xsize, ysize, numslices, numtimepoints
    )

    def mock_readfromnifti(fname, **kwargs):
        return MagicMock(), input_data, input_hdr, input_dims, input_sizes

    args = {
        "ncomp": 3.0,
        "datafile": "dummy_data.nii.gz",
        "datamaskname": None,
        "decomptype": "pca",
        "varnorm": True,
        "demean": True,
        "sigma": 0.0,
        "outputroot": "/tmp/test_niftidecomp_spatial",
    }

    with (
        patch(
            "rapidtide.workflows.niftidecomp.tide_io.readfromnifti", side_effect=mock_readfromnifti
        ),
        patch(
            "rapidtide.workflows.niftidecomp.tide_io.parseniftidims",
            return_value=(xsize, ysize, numslices, numtimepoints),
        ),
        patch(
            "rapidtide.workflows.niftidecomp.tide_io.parseniftisizes",
            return_value=(2.0, 2.0, 2.0, 1.5),
        ),
        patch("rapidtide.workflows.niftidecomp.tide_io.checkspacedimmatch", return_value=True),
        patch("rapidtide.workflows.niftidecomp.tide_io.checktimematch", return_value=True),
        patch("rapidtide.workflows.niftidecomp.tide_io.writevec"),
        patch("rapidtide.workflows.niftidecomp.tide_io.writenpvecs") as mock_writenpvecs,
        patch("rapidtide.workflows.niftidecomp.tide_io.savetonifti") as mock_savetonifti,
    ):

        main("spatial", args)

    # ncomp >= 1.0 => pcacomponents = int(ncomp), icacomponents = int(ncomp)
    assert args["pcacomponents"] == 3
    assert args["icacomponents"] == 3

    # Should write: exp_var, exp_var_pct, coefficients (3 writenpvecs)
    # and: components, fit (2 savetonifti)
    assert mock_writenpvecs.call_count == 3
    assert mock_savetonifti.call_count == 2


def main_ncomp_fractional(debug=False):
    """Test main with fractional ncomp (between 0 and 1)."""
    if debug:
        print("main_ncomp_fractional")
    xsize, ysize, numslices, numtimepoints = 4, 4, 3, 20
    input_data, input_hdr, input_dims, input_sizes = _make_4d_data(
        xsize, ysize, numslices, numtimepoints
    )

    def mock_readfromnifti(fname, **kwargs):
        return MagicMock(), input_data, input_hdr, input_dims, input_sizes

    args = {
        "ncomp": 0.8,
        "datafile": "dummy_data.nii.gz",
        "datamaskname": None,
        "decomptype": "pca",
        "varnorm": True,
        "demean": True,
        "sigma": 0.0,
        "outputroot": "/tmp/test_niftidecomp_frac",
    }

    with (
        patch(
            "rapidtide.workflows.niftidecomp.tide_io.readfromnifti", side_effect=mock_readfromnifti
        ),
        patch(
            "rapidtide.workflows.niftidecomp.tide_io.parseniftidims",
            return_value=(xsize, ysize, numslices, numtimepoints),
        ),
        patch(
            "rapidtide.workflows.niftidecomp.tide_io.parseniftisizes",
            return_value=(2.0, 2.0, 2.0, 1.5),
        ),
        patch("rapidtide.workflows.niftidecomp.tide_io.checkspacedimmatch", return_value=True),
        patch("rapidtide.workflows.niftidecomp.tide_io.checktimematch", return_value=True),
        patch("rapidtide.workflows.niftidecomp.tide_io.writevec"),
        patch("rapidtide.workflows.niftidecomp.tide_io.writenpvecs"),
        patch("rapidtide.workflows.niftidecomp.tide_io.savetonifti"),
    ):

        main("temporal", args)

    assert args["pcacomponents"] == 0.8
    assert args["icacomponents"] is None


def main_temporal_wrapper_test(debug=False):
    """Test main_temporal convenience wrapper."""
    if debug:
        print("main_temporal_wrapper_test")
    xsize, ysize, numslices, numtimepoints = 4, 4, 3, 20
    input_data, input_hdr, input_dims, input_sizes = _make_4d_data(
        xsize, ysize, numslices, numtimepoints
    )

    def mock_readfromnifti(fname, **kwargs):
        return MagicMock(), input_data, input_hdr, input_dims, input_sizes

    ns = argparse.Namespace(
        ncomp=-1.0,
        datafile="dummy_data.nii.gz",
        datamaskname=None,
        decomptype="pca",
        varnorm=True,
        demean=True,
        sigma=0.0,
        outputroot="/tmp/test_niftidecomp_temwrap",
    )

    with (
        patch(
            "rapidtide.workflows.niftidecomp.tide_io.readfromnifti", side_effect=mock_readfromnifti
        ),
        patch(
            "rapidtide.workflows.niftidecomp.tide_io.parseniftidims",
            return_value=(xsize, ysize, numslices, numtimepoints),
        ),
        patch(
            "rapidtide.workflows.niftidecomp.tide_io.parseniftisizes",
            return_value=(2.0, 2.0, 2.0, 1.5),
        ),
        patch("rapidtide.workflows.niftidecomp.tide_io.checkspacedimmatch", return_value=True),
        patch("rapidtide.workflows.niftidecomp.tide_io.checktimematch", return_value=True),
        patch("rapidtide.workflows.niftidecomp.tide_io.writevec"),
        patch("rapidtide.workflows.niftidecomp.tide_io.writenpvecs"),
        patch("rapidtide.workflows.niftidecomp.tide_io.savetonifti"),
    ):

        main_temporal(ns)


def main_spatial_wrapper_test(debug=False):
    """Test main_spatial convenience wrapper."""
    if debug:
        print("main_spatial_wrapper_test")
    xsize, ysize, numslices, numtimepoints = 4, 4, 3, 20
    input_data, input_hdr, input_dims, input_sizes = _make_4d_data(
        xsize, ysize, numslices, numtimepoints
    )

    def mock_readfromnifti(fname, **kwargs):
        return MagicMock(), input_data, input_hdr, input_dims, input_sizes

    ns = argparse.Namespace(
        ncomp=3.0,
        datafile="dummy_data.nii.gz",
        datamaskname=None,
        decomptype="pca",
        varnorm=True,
        demean=True,
        sigma=0.0,
        outputroot="/tmp/test_niftidecomp_spawrap",
    )

    with (
        patch(
            "rapidtide.workflows.niftidecomp.tide_io.readfromnifti", side_effect=mock_readfromnifti
        ),
        patch(
            "rapidtide.workflows.niftidecomp.tide_io.parseniftidims",
            return_value=(xsize, ysize, numslices, numtimepoints),
        ),
        patch(
            "rapidtide.workflows.niftidecomp.tide_io.parseniftisizes",
            return_value=(2.0, 2.0, 2.0, 1.5),
        ),
        patch("rapidtide.workflows.niftidecomp.tide_io.checkspacedimmatch", return_value=True),
        patch("rapidtide.workflows.niftidecomp.tide_io.checktimematch", return_value=True),
        patch("rapidtide.workflows.niftidecomp.tide_io.writevec"),
        patch("rapidtide.workflows.niftidecomp.tide_io.writenpvecs"),
        patch("rapidtide.workflows.niftidecomp.tide_io.savetonifti"),
    ):

        main_spatial(ns)


# ==================== Main test function ====================


def test_niftidecomp(debug=False):
    # _get_parser tests
    if debug:
        print("Running parser tests")
    parser_temporal_basic(debug=debug)
    parser_spatial_basic(debug=debug)
    parser_invalid_axis(debug=debug)
    parser_defaults(debug=debug)
    parser_decomptype_choices(debug=debug)
    parser_nodemean(debug=debug)
    parser_novarnorm(debug=debug)
    parser_ncomp(debug=debug)
    parser_smooth(debug=debug)

    # _get_parser_temporal / _get_parser_spatial tests
    if debug:
        print("Running parser wrapper tests")
    parser_temporal_wrapper(debug=debug)
    parser_spatial_wrapper(debug=debug)

    # transposeifspatial tests
    if debug:
        print("Running transposeifspatial tests")
    transpose_spatial(debug=debug)
    transpose_temporal(debug=debug)
    transpose_default(debug=debug)
    transpose_1d(debug=debug)
    transpose_3d(debug=debug)

    # niftidecomp_workflow tests
    if debug:
        print("Running niftidecomp_workflow tests")
    workflow_pca_temporal(debug=debug)
    workflow_pca_spatial(debug=debug)
    workflow_pca_ncomp_fixed(debug=debug)
    workflow_ica_temporal(debug=debug)
    workflow_nodemean(debug=debug)
    workflow_novarnorm(debug=debug)
    workflow_nodemean_novarnorm(debug=debug)
    workflow_with_3d_mask(debug=debug)
    workflow_sparse_pca(debug=debug)
    workflow_multiple_files(debug=debug)
    workflow_spatial_pca_fixed(debug=debug)
    workflow_smoothing(debug=debug)

    # main tests
    if debug:
        print("Running main tests")
    main_temporal_pca(debug=debug)
    main_spatial_pca(debug=debug)
    main_ncomp_fractional(debug=debug)
    main_temporal_wrapper_test(debug=debug)
    main_spatial_wrapper_test(debug=debug)


if __name__ == "__main__":
    test_niftidecomp(debug=True)
