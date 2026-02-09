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
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rapidtide.tests.utils import create_dir, get_test_temp_path
from rapidtide.workflows.polyfitim import _get_parser, main, polyfitim

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


def _make_test_data(xsize=4, ysize=4, numslices=2, timepoints=5, order=1):
    """Create synthetic data with a known polynomial relationship to a template.

    Returns data, template, masks, and expected relationship parameters.
    data[voxel, t] = intercept + slope * template[voxel] + noise
    For higher orders, adds template^2, template^3, etc.
    """
    rng = np.random.RandomState(42)
    numvoxels = xsize * ysize * numslices

    # Template: smooth spatial gradient
    template = np.zeros((xsize, ysize, numslices), dtype=np.float64)
    for x in range(xsize):
        for y in range(ysize):
            for z in range(numslices):
                template[x, y, z] = (x + 1.0) / xsize + (y + 1.0) / ysize

    # Data: polynomial function of template + noise, varying across timepoints
    data = np.zeros((xsize, ysize, numslices, timepoints), dtype=np.float64)
    for t in range(timepoints):
        intercept = 10.0 + t * 0.5
        for o in range(1, order + 1):
            coeff = 5.0 / (o + 1) + t * 0.1
            data[:, :, :, t] += coeff * (template ** o)
        data[:, :, :, t] += intercept
        data[:, :, :, t] += rng.randn(xsize, ysize, numslices) * 0.01

    # Masks: all ones
    datamask = np.ones((xsize, ysize, numslices), dtype=np.float64)
    templatemask = np.ones((xsize, ysize, numslices), dtype=np.float64)

    return data, template, datamask, templatemask


def _make_dims(xsize, ysize, numslices, timepoints):
    return np.array([4, xsize, ysize, numslices, timepoints, 1, 1, 1])


def _make_sizes():
    return np.array([1.0, 2.0, 2.0, 2.0, 1.5, 1.0, 1.0, 1.0])


# ==================== _get_parser tests ====================


def parser_basic(debug=False):
    """Test that parser creates successfully."""
    if debug:
        print("parser_basic")
    parser = _get_parser()
    assert parser is not None
    assert parser.prog == "polyfitim"


def parser_required_args(debug=False):
    """Test parser has required arguments."""
    if debug:
        print("parser_required_args")
    parser = _get_parser()
    actions = {a.dest: a for a in parser._actions}
    assert "datafile" in actions
    assert "datamask" in actions
    assert "templatefile" in actions
    assert "templatemask" in actions
    assert "outputroot" in actions


def parser_defaults(debug=False):
    """Test default values for optional arguments."""
    if debug:
        print("parser_defaults")
    parser = _get_parser()
    with tempfile.NamedTemporaryFile(suffix=".nii") as f1, \
         tempfile.NamedTemporaryFile(suffix=".nii") as f2, \
         tempfile.NamedTemporaryFile(suffix=".nii") as f3, \
         tempfile.NamedTemporaryFile(suffix=".nii") as f4:
        args = parser.parse_args([f1.name, f2.name, f3.name, f4.name, "outroot"])
    assert args.order == 1
    assert args.regionatlas is None


def parser_order(debug=False):
    """Test --order option."""
    if debug:
        print("parser_order")
    parser = _get_parser()
    with tempfile.NamedTemporaryFile(suffix=".nii") as f1, \
         tempfile.NamedTemporaryFile(suffix=".nii") as f2, \
         tempfile.NamedTemporaryFile(suffix=".nii") as f3, \
         tempfile.NamedTemporaryFile(suffix=".nii") as f4:
        args = parser.parse_args([f1.name, f2.name, f3.name, f4.name, "out", "--order", "3"])
    assert args.order == 3


def parser_regionatlas(debug=False):
    """Test --regionatlas option."""
    if debug:
        print("parser_regionatlas")
    parser = _get_parser()
    with tempfile.NamedTemporaryFile(suffix=".nii") as f1, \
         tempfile.NamedTemporaryFile(suffix=".nii") as f2, \
         tempfile.NamedTemporaryFile(suffix=".nii") as f3, \
         tempfile.NamedTemporaryFile(suffix=".nii") as f4, \
         tempfile.NamedTemporaryFile(suffix=".nii") as f5:
        args = parser.parse_args([
            f1.name, f2.name, f3.name, f4.name, "out",
            "--regionatlas", f5.name,
        ])
    assert args.regionatlas is not None


# ==================== polyfitim tests ====================


def _run_polyfitim(xsize=4, ysize=4, numslices=2, timepoints=5,
                   order=1, regionatlas_data=None, datamask_4d=False):
    """Helper to run polyfitim with mocked IO. Returns saved files dict."""
    data, template, datamask, templatemask = _make_test_data(
        xsize, ysize, numslices, timepoints, order=order,
    )
    sizes = _make_sizes()

    data_dims = _make_dims(xsize, ysize, numslices, timepoints)
    data_hdr = _make_mock_hdr(xsize, ysize, numslices, timepoints)

    template_dims = _make_dims(xsize, ysize, numslices, 1)
    template_hdr = _make_mock_hdr(xsize, ysize, numslices, 1)

    tmask_dims = _make_dims(xsize, ysize, numslices, 1)

    if datamask_4d:
        datamask_full = np.ones((xsize, ysize, numslices, timepoints), dtype=np.float64)
        dmask_dims = _make_dims(xsize, ysize, numslices, timepoints)
    else:
        datamask_full = datamask
        dmask_dims = _make_dims(xsize, ysize, numslices, 1)

    if regionatlas_data is not None:
        atlas_dims = _make_dims(xsize, ysize, numslices, 1)

    saved_nifti = {}
    saved_text = {}

    def mock_readfromnifti(fname, **kwargs):
        if "atlas" in fname:
            return (MagicMock(), regionatlas_data,
                    _make_mock_hdr(xsize, ysize, numslices, 1), atlas_dims, sizes)
        elif "template_mask" in fname or "templatemask" in fname:
            return (MagicMock(), templatemask,
                    _make_mock_hdr(xsize, ysize, numslices, 1), tmask_dims, sizes)
        elif "template" in fname:
            return (MagicMock(), template, template_hdr, template_dims, sizes)
        elif "datamask" in fname or "data_mask" in fname:
            return (MagicMock(), datamask_full,
                    _make_mock_hdr(xsize, ysize, numslices, dmask_dims[4]), dmask_dims, sizes)
        else:
            return MagicMock(), data, data_hdr, data_dims, sizes

    def mock_savetonifti(arr, hdr, fname, **kwargs):
        saved_nifti[fname] = arr.copy()

    def mock_writenpvecs(arr, fname, **kwargs):
        saved_text[fname] = np.array(arr).copy()

    with patch("rapidtide.workflows.polyfitim.tide_io.readfromnifti", side_effect=mock_readfromnifti), \
         patch("rapidtide.workflows.polyfitim.tide_io.checkspacedimmatch", return_value=True), \
         patch("rapidtide.workflows.polyfitim.tide_io.checktimematch", return_value=True), \
         patch("rapidtide.workflows.polyfitim.tide_io.savetonifti", side_effect=mock_savetonifti), \
         patch("rapidtide.workflows.polyfitim.tide_io.writenpvecs", side_effect=mock_writenpvecs):

        polyfitim(
            datafile="dummy_data.nii.gz",
            datamask="dummy_datamask.nii.gz",
            templatefile="dummy_template.nii.gz",
            templatemask="dummy_templatemask.nii.gz",
            outputroot="/tmp/test_polyfitim",
            regionatlas="dummy_atlas.nii.gz" if regionatlas_data is not None else None,
            order=order,
        )

    return saved_nifti, saved_text


def polyfitim_linear_basic(debug=False):
    """Test basic linear (order=1) polyfitim without region atlas."""
    if debug:
        print("polyfitim_linear_basic")
    xsize, ysize, numslices, timepoints = 4, 4, 2, 5

    saved_nifti, saved_text = _run_polyfitim(
        xsize, ysize, numslices, timepoints, order=1,
    )

    # Should save fit and residuals
    assert "/tmp/test_polyfitim_fit" in saved_nifti
    assert "/tmp/test_polyfitim_residuals" in saved_nifti
    assert saved_nifti["/tmp/test_polyfitim_fit"].shape == (xsize, ysize, numslices, timepoints)
    assert saved_nifti["/tmp/test_polyfitim_residuals"].shape == (xsize, ysize, numslices, timepoints)

    # Should save r2vals and coefficient files
    assert "/tmp/test_polyfitim_r2vals.txt" in saved_text


def polyfitim_quadratic(debug=False):
    """Test quadratic (order=2) polyfitim."""
    if debug:
        print("polyfitim_quadratic")
    xsize, ysize, numslices, timepoints = 4, 4, 2, 5

    saved_nifti, saved_text = _run_polyfitim(
        xsize, ysize, numslices, timepoints, order=2,
    )

    assert "/tmp/test_polyfitim_fit" in saved_nifti
    assert "/tmp/test_polyfitim_residuals" in saved_nifti
    assert "/tmp/test_polyfitim_r2vals.txt" in saved_text


def polyfitim_cubic(debug=False):
    """Test cubic (order=3) polyfitim."""
    if debug:
        print("polyfitim_cubic")
    xsize, ysize, numslices, timepoints = 4, 4, 2, 5

    saved_nifti, saved_text = _run_polyfitim(
        xsize, ysize, numslices, timepoints, order=3,
    )

    assert "/tmp/test_polyfitim_fit" in saved_nifti
    assert "/tmp/test_polyfitim_residuals" in saved_nifti


def polyfitim_r2_quality(debug=False):
    """Test that R2 values are high for data with known polynomial relationship."""
    if debug:
        print("polyfitim_r2_quality")
    xsize, ysize, numslices, timepoints = 4, 4, 2, 5

    saved_nifti, saved_text = _run_polyfitim(
        xsize, ysize, numslices, timepoints, order=1,
    )

    r2 = saved_text["/tmp/test_polyfitim_r2vals.txt"]
    # Data is linear in template with very low noise, R2 should be high
    for t in range(timepoints):
        assert r2[t] > 0.95, f"R2 at timepoint {t} = {r2[t]}, expected > 0.95"


def polyfitim_residuals_small(debug=False):
    """Test that residuals are small for data matching the polynomial model."""
    if debug:
        print("polyfitim_residuals_small")
    xsize, ysize, numslices, timepoints = 4, 4, 2, 5

    saved_nifti, saved_text = _run_polyfitim(
        xsize, ysize, numslices, timepoints, order=1,
    )

    residuals = saved_nifti["/tmp/test_polyfitim_residuals"]
    fit = saved_nifti["/tmp/test_polyfitim_fit"]
    # Residuals should be much smaller than the fit values
    assert np.std(residuals) < 0.1 * np.std(fit), \
        f"Residuals std ({np.std(residuals)}) too large relative to fit std ({np.std(fit)})"


def polyfitim_with_regionatlas(debug=False):
    """Test polyfitim with a region atlas for per-region fitting."""
    if debug:
        print("polyfitim_with_regionatlas")
    xsize, ysize, numslices, timepoints = 4, 4, 2, 5

    # Create atlas with 2 regions split along x-axis
    atlas = np.zeros((xsize, ysize, numslices), dtype=np.float64)
    atlas[:xsize // 2, :, :] = 1
    atlas[xsize // 2:, :, :] = 2

    saved_nifti, saved_text = _run_polyfitim(
        xsize, ysize, numslices, timepoints, order=1,
        regionatlas_data=atlas,
    )

    assert "/tmp/test_polyfitim_fit" in saved_nifti
    assert "/tmp/test_polyfitim_residuals" in saved_nifti
    assert "/tmp/test_polyfitim_r2vals.txt" in saved_text

    # R2 should have shape (numregions, timepoints)
    r2 = saved_text["/tmp/test_polyfitim_r2vals.txt"]
    assert r2.shape == (2, timepoints), f"Expected R2 shape (2, {timepoints}), got {r2.shape}"


def polyfitim_4d_datamask(debug=False):
    """Test polyfitim with a 4D data mask."""
    if debug:
        print("polyfitim_4d_datamask")
    xsize, ysize, numslices, timepoints = 4, 4, 2, 5

    saved_nifti, saved_text = _run_polyfitim(
        xsize, ysize, numslices, timepoints, order=1,
        datamask_4d=True,
    )

    assert "/tmp/test_polyfitim_fit" in saved_nifti
    assert "/tmp/test_polyfitim_residuals" in saved_nifti


def polyfitim_order_zero_exit(debug=False):
    """Test that order < 1 causes exit."""
    if debug:
        print("polyfitim_order_zero_exit")
    with pytest.raises(SystemExit):
        polyfitim(
            datafile="dummy.nii.gz",
            datamask="dummy_mask.nii.gz",
            templatefile="dummy_template.nii.gz",
            templatemask="dummy_tmask.nii.gz",
            outputroot="/tmp/test_polyfitim_bad",
            order=0,
        )


def polyfitim_space_mismatch(debug=False):
    """Test that spatial dimension mismatch causes exit."""
    if debug:
        print("polyfitim_space_mismatch")
    xsize, ysize, numslices, timepoints = 4, 4, 2, 5
    data, template, datamask, templatemask = _make_test_data(
        xsize, ysize, numslices, timepoints,
    )
    sizes = _make_sizes()
    data_dims = _make_dims(xsize, ysize, numslices, timepoints)
    data_hdr = _make_mock_hdr(xsize, ysize, numslices, timepoints)
    dmask_dims = _make_dims(xsize, ysize, numslices, 1)

    call_count = {"n": 0}

    def mock_readfromnifti(fname, **kwargs):
        return (MagicMock(), data, data_hdr, data_dims, sizes)

    def mock_checkspacedimmatch(dims1, dims2, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return False  # First call: data vs datamask fails
        return True

    with patch("rapidtide.workflows.polyfitim.tide_io.readfromnifti", side_effect=mock_readfromnifti), \
         patch("rapidtide.workflows.polyfitim.tide_io.checkspacedimmatch", side_effect=mock_checkspacedimmatch):

        with pytest.raises(SystemExit):
            polyfitim(
                datafile="dummy_data.nii.gz",
                datamask="dummy_datamask.nii.gz",
                templatefile="dummy_template.nii.gz",
                templatemask="dummy_templatemask.nii.gz",
                outputroot="/tmp/test_polyfitim_mismatch",
                order=1,
            )


def polyfitim_fit_plus_residuals_equals_data(debug=False):
    """Test that fit + residuals = original data."""
    if debug:
        print("polyfitim_fit_plus_residuals_equals_data")
    xsize, ysize, numslices, timepoints = 4, 4, 2, 5

    data, template, datamask, templatemask = _make_test_data(
        xsize, ysize, numslices, timepoints,
    )
    sizes = _make_sizes()
    data_dims = _make_dims(xsize, ysize, numslices, timepoints)
    data_hdr = _make_mock_hdr(xsize, ysize, numslices, timepoints)
    tmask_dims = _make_dims(xsize, ysize, numslices, 1)
    dmask_dims = _make_dims(xsize, ysize, numslices, 1)
    template_dims = _make_dims(xsize, ysize, numslices, 1)

    saved_nifti = {}

    def mock_readfromnifti(fname, **kwargs):
        if "templatemask" in fname:
            return (MagicMock(), templatemask,
                    _make_mock_hdr(xsize, ysize, numslices, 1), tmask_dims, sizes)
        elif "template" in fname:
            return (MagicMock(), template,
                    _make_mock_hdr(xsize, ysize, numslices, 1), template_dims, sizes)
        elif "datamask" in fname:
            return (MagicMock(), datamask,
                    _make_mock_hdr(xsize, ysize, numslices, 1), dmask_dims, sizes)
        else:
            return MagicMock(), data, data_hdr, data_dims, sizes

    def mock_savetonifti(arr, hdr, fname, **kwargs):
        saved_nifti[fname] = arr.copy()

    with patch("rapidtide.workflows.polyfitim.tide_io.readfromnifti", side_effect=mock_readfromnifti), \
         patch("rapidtide.workflows.polyfitim.tide_io.checkspacedimmatch", return_value=True), \
         patch("rapidtide.workflows.polyfitim.tide_io.checktimematch", return_value=True), \
         patch("rapidtide.workflows.polyfitim.tide_io.savetonifti", side_effect=mock_savetonifti), \
         patch("rapidtide.workflows.polyfitim.tide_io.writenpvecs"):

        polyfitim(
            datafile="dummy_data.nii.gz",
            datamask="dummy_datamask.nii.gz",
            templatefile="dummy_template.nii.gz",
            templatemask="dummy_templatemask.nii.gz",
            outputroot="/tmp/test_polyfitim_sum",
            order=1,
        )

    fit = saved_nifti["/tmp/test_polyfitim_sum_fit"]
    residuals = saved_nifti["/tmp/test_polyfitim_sum_residuals"]
    # fit + residuals should equal original data (within floating point precision)
    reconstructed = fit + residuals
    assert np.allclose(reconstructed, data, atol=1e-10), \
        f"Max difference: {np.max(np.abs(reconstructed - data))}"


# ==================== main tests ====================


def main_function(debug=False):
    """Test main function dispatches correctly."""
    if debug:
        print("main_function")
    xsize, ysize, numslices, timepoints = 4, 4, 2, 5
    data, template, datamask, templatemask = _make_test_data(
        xsize, ysize, numslices, timepoints,
    )
    sizes = _make_sizes()
    data_dims = _make_dims(xsize, ysize, numslices, timepoints)
    data_hdr = _make_mock_hdr(xsize, ysize, numslices, timepoints)
    tmask_dims = _make_dims(xsize, ysize, numslices, 1)
    dmask_dims = _make_dims(xsize, ysize, numslices, 1)
    template_dims = _make_dims(xsize, ysize, numslices, 1)

    def mock_readfromnifti(fname, **kwargs):
        if "templatemask" in fname:
            return (MagicMock(), templatemask,
                    _make_mock_hdr(xsize, ysize, numslices, 1), tmask_dims, sizes)
        elif "template" in fname:
            return (MagicMock(), template,
                    _make_mock_hdr(xsize, ysize, numslices, 1), template_dims, sizes)
        elif "datamask" in fname:
            return (MagicMock(), datamask,
                    _make_mock_hdr(xsize, ysize, numslices, 1), dmask_dims, sizes)
        else:
            return MagicMock(), data, data_hdr, data_dims, sizes

    args = argparse.Namespace(
        datafile="dummy_data.nii.gz",
        datamask="dummy_datamask.nii.gz",
        templatefile="dummy_template.nii.gz",
        templatemask="dummy_templatemask.nii.gz",
        outputroot="/tmp/test_polyfitim_main",
        regionatlas=None,
        order=1,
    )

    with patch("rapidtide.workflows.polyfitim.tide_io.readfromnifti", side_effect=mock_readfromnifti), \
         patch("rapidtide.workflows.polyfitim.tide_io.checkspacedimmatch", return_value=True), \
         patch("rapidtide.workflows.polyfitim.tide_io.checktimematch", return_value=True), \
         patch("rapidtide.workflows.polyfitim.tide_io.savetonifti"), \
         patch("rapidtide.workflows.polyfitim.tide_io.writenpvecs"):

        main(args)


# ==================== Main test function ====================


def test_polyfitim(debug=False):
    # _get_parser tests
    if debug:
        print("Running parser tests")
    parser_basic(debug=debug)
    parser_required_args(debug=debug)
    parser_defaults(debug=debug)
    parser_order(debug=debug)
    parser_regionatlas(debug=debug)

    # polyfitim tests
    if debug:
        print("Running polyfitim tests")
    polyfitim_linear_basic(debug=debug)
    polyfitim_quadratic(debug=debug)
    polyfitim_cubic(debug=debug)
    polyfitim_r2_quality(debug=debug)
    polyfitim_residuals_small(debug=debug)
    polyfitim_with_regionatlas(debug=debug)
    polyfitim_4d_datamask(debug=debug)
    polyfitim_order_zero_exit(debug=debug)
    polyfitim_space_mismatch(debug=debug)
    polyfitim_fit_plus_residuals_equals_data(debug=debug)

    # main tests
    if debug:
        print("Running main tests")
    main_function(debug=debug)


if __name__ == "__main__":
    test_polyfitim(debug=True)
