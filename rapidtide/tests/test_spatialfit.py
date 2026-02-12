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

from rapidtide.workflows.spatialfit import _get_parser, spatialfit

# ---- helpers ----


def _capture_writenpvecs():
    """Capture writenpvecs calls, copying data at call time."""
    captured = []

    def _side_effect(data, filename, **kwargs):
        captured.append((np.array(data, copy=True), filename))

    return _side_effect, captured


def _capture_savetonifti():
    """Capture savetonifti calls, copying data at call time."""
    captured = []

    def _side_effect(data, hdr, filename, **kwargs):
        captured.append((np.array(data, copy=True), filename))

    return _side_effect, captured


def _make_test_data(xsize=2, ysize=2, numslices=1, timepoints=3):
    """Create consistent test data for spatialfit tests.

    Template = [1, 2, 3, 4] (reshaped to xsize x ysize x numslices).
    Data at each timepoint = slope * template (no intercept for clean math).
    Slopes = [2.0, 3.0, 1.0] for 3 timepoints.

    Returns dict with all mock nifti returns and expected results.
    """
    numvoxels = xsize * ysize * numslices

    # Template: [1, 2, 3, 4]
    template_flat = np.arange(1, numvoxels + 1, dtype=np.float64)
    template_3d = template_flat.reshape((xsize, ysize, numslices))

    # Data: slope_t * template for each timepoint
    rng = np.random.RandomState(42)
    slopes = 1.0 + rng.rand(timepoints) * 3.0
    intercepts = np.zeros(timepoints)
    r2_vals = 0.85 + rng.rand(timepoints) * 0.15

    data_4d = np.zeros((xsize, ysize, numslices, timepoints), dtype=np.float64)
    for t in range(timepoints):
        data_4d[:, :, :, t] = slopes[t] * template_3d + intercepts[t]

    # Dims arrays (NIfTI format: [ndim, x, y, z, t, ...])
    data_dims = np.array([4, xsize, ysize, numslices, timepoints, 1, 1, 1])
    template_dims = np.array([3, xsize, ysize, numslices, 1, 1, 1, 1])
    data_sizes = np.array([4, 2.0, 2.0, 3.0, 1.5, 1, 1, 1])
    template_sizes = np.array([3, 2.0, 2.0, 3.0, 1, 1, 1, 1])

    # Masks (all ones)
    data_mask = np.ones((xsize, ysize, numslices, timepoints), dtype=np.float64)
    template_mask = np.ones((xsize, ysize, numslices), dtype=np.float64)
    dmask_dims = data_dims.copy()
    tmask_dims = template_dims.copy()

    # mlregress returns for each timepoint
    mlregress_returns = []
    for t in range(timepoints):
        thefit = np.array([[intercepts[t], slopes[t]]])
        mlregress_returns.append((thefit, r2_vals[t]))

    return {
        "data_4d": data_4d,
        "template_3d": template_3d,
        "data_dims": data_dims,
        "template_dims": template_dims,
        "data_sizes": data_sizes,
        "template_sizes": template_sizes,
        "data_mask": data_mask,
        "template_mask": template_mask,
        "dmask_dims": dmask_dims,
        "tmask_dims": tmask_dims,
        "slopes": slopes,
        "intercepts": intercepts,
        "r2_vals": r2_vals,
        "mlregress_returns": mlregress_returns,
        "template_flat": template_flat,
        "xsize": xsize,
        "ysize": ysize,
        "numslices": numslices,
        "timepoints": timepoints,
    }


def _run_spatialfit(argv_list, readfromnifti_returns, mlregress_returns):
    """Run spatialfit with mocked IO and return captured outputs.

    Parameters
    ----------
    argv_list : list
        sys.argv replacement
    readfromnifti_returns : list
        Ordered return values for each readfromnifti call
    mlregress_returns : list
        Return values for each mlregress call (one per timepoint)

    Returns
    -------
    (writenpvecs_captured, savetonifti_captured)
    """
    wnp_effect, wnp_captured = _capture_writenpvecs()
    stn_effect, stn_captured = _capture_savetonifti()

    with (
        patch("sys.argv", argv_list),
        patch(
            "rapidtide.workflows.parser_funcs.is_valid_file",
            side_effect=lambda parser, x: x,
        ),
        patch(
            "rapidtide.workflows.spatialfit.tide_io.readfromnifti",
            side_effect=readfromnifti_returns,
        ),
        patch(
            "rapidtide.workflows.spatialfit.tide_fit.mlregress",
            side_effect=mlregress_returns,
        ),
        patch(
            "rapidtide.workflows.spatialfit.tide_io.writenpvecs",
            side_effect=wnp_effect,
        ),
        patch(
            "rapidtide.workflows.spatialfit.tide_io.savetonifti",
            side_effect=stn_effect,
        ),
    ):
        spatialfit(None)

    return wnp_captured, stn_captured


# ---- _get_parser tests ----


def test_get_parser_returns_parser(debug=False):
    """Test that _get_parser returns an ArgumentParser."""
    parser = _get_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    if debug:
        print("Parser created successfully")


def test_get_parser_required_args(debug=False):
    """Test that parser requires three positional arguments."""
    parser = _get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_get_parser_with_valid_args(debug=False):
    """Test parser with valid positional arguments and default values."""
    parser = _get_parser()
    with patch(
        "rapidtide.workflows.parser_funcs.is_valid_file",
        side_effect=lambda p, x: x,
    ):
        args = parser.parse_args(["data.nii.gz", "template.nii.gz", "output_root"])

    assert args.datafile == "data.nii.gz"
    assert args.templatefile == "template.nii.gz"
    assert args.outputroot == "output_root"
    assert args.dmask is None
    assert args.tmask is None
    assert args.order == 1
    assert args.debug is False

    if debug:
        print(f"Parsed args: {args}")


def test_get_parser_optional_flags(debug=False):
    """Test parser with all optional flags."""
    parser = _get_parser()
    with patch(
        "rapidtide.workflows.parser_funcs.is_valid_file",
        side_effect=lambda p, x: x,
    ):
        args = parser.parse_args([
            "data.nii.gz", "template.nii.gz", "out",
            "--datamask", "dmask.nii.gz",
            "--templatemask", "tmask.nii.gz",
            "--order", "3",
            "--debug",
        ])

    assert args.dmask == "dmask.nii.gz"
    assert args.tmask == "tmask.nii.gz"
    assert args.order == 3
    assert args.debug is True


def test_get_parser_order_is_int(debug=False):
    """Test that --order requires an integer value."""
    parser = _get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["data.nii", "tmpl.nii", "out", "--order", "abc"])


def test_get_parser_default_order(debug=False):
    """Test that default order is 1."""
    parser = _get_parser()
    with patch(
        "rapidtide.workflows.parser_funcs.is_valid_file",
        side_effect=lambda p, x: x,
    ):
        args = parser.parse_args(["data.nii", "tmpl.nii", "out"])
    assert args.order == 1


# ---- spatialfit basic tests ----


def test_spatialfit_basic_outputs(debug=False):
    """Test basic spatialfit run produces correct number and names of output files."""
    td = _make_test_data()

    # readfromnifti called 3 times: data, template, tmask
    read_returns = [
        (MagicMock(), td["data_4d"], MagicMock(), td["data_dims"], td["data_sizes"]),
        (MagicMock(), td["template_3d"], MagicMock(), td["template_dims"], td["template_sizes"]),
        (MagicMock(), td["template_mask"], MagicMock(), td["tmask_dims"], td["template_sizes"]),
    ]

    argv = [
        "spatialfit", "data.nii.gz", "template.nii.gz", "outroot",
        "--templatemask", "tmask.nii.gz",
    ]

    wnp, stn = _run_spatialfit(argv, read_returns, td["mlregress_returns"])

    # 3 text files: lincoffs, offsets, r2vals
    assert len(wnp) == 3
    wnp_filenames = [c[1] for c in wnp]
    assert wnp_filenames[0] == "outroot_lincoffs.txt"
    assert wnp_filenames[1] == "outroot_offsets.txt"
    assert wnp_filenames[2] == "outroot_r2vals.txt"

    # 4 nifti files: fit, residuals, normalized, newtemplate
    assert len(stn) == 4
    stn_filenames = [c[1] for c in stn]
    assert stn_filenames[0] == "outroot_fit"
    assert stn_filenames[1] == "outroot_residuals"
    assert stn_filenames[2] == "outroot_normalized"
    assert stn_filenames[3] == "outroot_newtemplate"

    if debug:
        print(f"Text files: {wnp_filenames}")
        print(f"NIfTI files: {stn_filenames}")


def test_spatialfit_lincoffs_offsets_r2(debug=False):
    """Test that lincoffs, offsets, and r2vals match mlregress returns."""
    td = _make_test_data()

    read_returns = [
        (MagicMock(), td["data_4d"], MagicMock(), td["data_dims"], td["data_sizes"]),
        (MagicMock(), td["template_3d"], MagicMock(), td["template_dims"], td["template_sizes"]),
        (MagicMock(), td["template_mask"], MagicMock(), td["tmask_dims"], td["template_sizes"]),
    ]
    argv = [
        "spatialfit", "data.nii.gz", "template.nii.gz", "outroot",
        "--templatemask", "tmask.nii.gz",
    ]

    wnp, stn = _run_spatialfit(argv, read_returns, td["mlregress_returns"])

    lincoffs = wnp[0][0]
    offsets = wnp[1][0]
    r2vals = wnp[2][0]

    np.testing.assert_allclose(lincoffs, td["slopes"], atol=1e-12)
    np.testing.assert_allclose(offsets, td["intercepts"], atol=1e-12)
    np.testing.assert_allclose(r2vals, td["r2_vals"], atol=1e-12)

    if debug:
        print(f"lincoffs: {lincoffs}")
        print(f"offsets: {offsets}")
        print(f"r2vals: {r2vals}")


def test_spatialfit_fitdata(debug=False):
    """Test that fit data = slope * template (with mask applied)."""
    td = _make_test_data()

    read_returns = [
        (MagicMock(), td["data_4d"], MagicMock(), td["data_dims"], td["data_sizes"]),
        (MagicMock(), td["template_3d"], MagicMock(), td["template_dims"], td["template_sizes"]),
        (MagicMock(), td["template_mask"], MagicMock(), td["tmask_dims"], td["template_sizes"]),
    ]
    argv = [
        "spatialfit", "data.nii.gz", "template.nii.gz", "outroot",
        "--templatemask", "tmask.nii.gz",
    ]

    wnp, stn = _run_spatialfit(argv, read_returns, td["mlregress_returns"])

    fitdata = stn[0][0]
    xsize, ysize, numslices, timepoints = td["xsize"], td["ysize"], td["numslices"], td["timepoints"]
    assert fitdata.shape == (xsize, ysize, numslices, timepoints)

    # Expected: fitdata[:, :, :, t] = slope_t * template_3d
    for t in range(timepoints):
        expected = td["slopes"][t] * td["template_3d"]
        np.testing.assert_allclose(fitdata[:, :, :, t], expected, atol=1e-12)

    if debug:
        print(f"Fitdata verified for {timepoints} timepoints")


def test_spatialfit_residuals(debug=False):
    """Test that residuals = data - fitdata."""
    td = _make_test_data()

    read_returns = [
        (MagicMock(), td["data_4d"], MagicMock(), td["data_dims"], td["data_sizes"]),
        (MagicMock(), td["template_3d"], MagicMock(), td["template_dims"], td["template_sizes"]),
        (MagicMock(), td["template_mask"], MagicMock(), td["tmask_dims"], td["template_sizes"]),
    ]
    argv = [
        "spatialfit", "data.nii.gz", "template.nii.gz", "outroot",
        "--templatemask", "tmask.nii.gz",
    ]

    wnp, stn = _run_spatialfit(argv, read_returns, td["mlregress_returns"])

    fitdata = stn[0][0]
    residuals = stn[1][0]

    # Since data = slope * template + 0, and fitdata = slope * template,
    # residuals should be all zeros.
    expected_residuals = td["data_4d"] - fitdata
    np.testing.assert_allclose(residuals, expected_residuals, atol=1e-12)

    # For our zero-intercept test data, residuals are zero
    np.testing.assert_allclose(residuals, 0.0, atol=1e-12)

    if debug:
        print(f"Residuals max abs: {np.max(np.abs(residuals)):.2e}")


def test_spatialfit_normalized(debug=False):
    """Test that normalized = (data - offset) / slope equals the template."""
    td = _make_test_data()

    read_returns = [
        (MagicMock(), td["data_4d"], MagicMock(), td["data_dims"], td["data_sizes"]),
        (MagicMock(), td["template_3d"], MagicMock(), td["template_dims"], td["template_sizes"]),
        (MagicMock(), td["template_mask"], MagicMock(), td["tmask_dims"], td["template_sizes"]),
    ]
    argv = [
        "spatialfit", "data.nii.gz", "template.nii.gz", "outroot",
        "--templatemask", "tmask.nii.gz",
    ]

    wnp, stn = _run_spatialfit(argv, read_returns, td["mlregress_returns"])

    normalized = stn[2][0]
    timepoints = td["timepoints"]

    # For data = slope * template, normalized = (data - 0) / slope = template
    for t in range(timepoints):
        np.testing.assert_allclose(normalized[:, :, :, t], td["template_3d"], atol=1e-12)

    if debug:
        print("Normalized data matches template at all timepoints")


def test_spatialfit_newtemplate(debug=False):
    """Test that newtemplate is correctly averaged across timepoints."""
    td = _make_test_data()

    read_returns = [
        (MagicMock(), td["data_4d"], MagicMock(), td["data_dims"], td["data_sizes"]),
        (MagicMock(), td["template_3d"], MagicMock(), td["template_dims"], td["template_sizes"]),
        (MagicMock(), td["template_mask"], MagicMock(), td["tmask_dims"], td["template_sizes"]),
    ]
    argv = [
        "spatialfit", "data.nii.gz", "template.nii.gz", "outroot",
        "--templatemask", "tmask.nii.gz",
    ]

    wnp, stn = _run_spatialfit(argv, read_returns, td["mlregress_returns"])

    newtemplate = stn[3][0]

    # newtemplate = mean over timepoints of (data[:, t] / slope_t)
    # Since data[:, t] = slope_t * template, data[:, t] / slope_t = template
    # So newtemplate = template
    np.testing.assert_allclose(newtemplate, td["template_3d"], atol=1e-12)

    if debug:
        print(f"Newtemplate matches original template")


def test_spatialfit_with_nonzero_intercepts(debug=False):
    """Test spatialfit with non-zero intercepts to verify offset handling."""
    xsize, ysize, numslices, timepoints = 2, 2, 1, 2
    numvoxels = xsize * ysize * numslices

    template_flat = np.array([1.0, 2.0, 3.0, 4.0])
    template_3d = template_flat.reshape((xsize, ysize, numslices))

    slopes = np.array([2.0, 0.5])
    intercepts = np.array([1.0, -0.5])
    r2_vals = np.array([0.98, 0.92])

    data_4d = np.zeros((xsize, ysize, numslices, timepoints))
    for t in range(timepoints):
        data_4d[:, :, :, t] = slopes[t] * template_3d + intercepts[t]

    data_dims = np.array([4, xsize, ysize, numslices, timepoints, 1, 1, 1])
    template_dims = np.array([3, xsize, ysize, numslices, 1, 1, 1, 1])
    sizes = np.array([4, 2.0, 2.0, 3.0, 1.5, 1, 1, 1])
    tmask = np.ones((xsize, ysize, numslices), dtype=np.float64)
    tmask_dims = template_dims.copy()

    mlregress_returns = [
        (np.array([[intercepts[t], slopes[t]]]), r2_vals[t]) for t in range(timepoints)
    ]

    read_returns = [
        (MagicMock(), data_4d, MagicMock(), data_dims, sizes),
        (MagicMock(), template_3d, MagicMock(), template_dims, sizes),
        (MagicMock(), tmask, MagicMock(), tmask_dims, sizes),
    ]
    argv = [
        "spatialfit", "data.nii.gz", "template.nii.gz", "outroot",
        "--templatemask", "tmask.nii.gz",
    ]

    wnp, stn = _run_spatialfit(argv, read_returns, mlregress_returns)

    lincoffs = wnp[0][0]
    offsets = wnp[1][0]
    np.testing.assert_allclose(lincoffs, slopes, atol=1e-12)
    np.testing.assert_allclose(offsets, intercepts, atol=1e-12)

    # Residuals should be the intercept offset (fitdata doesn't include intercept)
    residuals = stn[1][0]
    for t in range(timepoints):
        expected_residual = intercepts[t] * np.ones_like(template_3d)
        np.testing.assert_allclose(residuals[:, :, :, t], expected_residual, atol=1e-12)

    # Normalized: (data - offset) / slope = template
    normalized = stn[2][0]
    for t in range(timepoints):
        np.testing.assert_allclose(normalized[:, :, :, t], template_3d, atol=1e-12)

    if debug:
        print(f"Non-zero intercept test: offsets = {offsets}")


def test_spatialfit_with_both_masks(debug=False):
    """Test spatialfit with both data mask and template mask."""
    td = _make_test_data()

    # Create a data mask that zeros out one voxel at one timepoint
    dmask = td["data_mask"].copy()
    dmask[0, 0, 0, 1] = 0.0  # mask out voxel (0,0,0) at timepoint 1

    read_returns = [
        (MagicMock(), td["data_4d"], MagicMock(), td["data_dims"], td["data_sizes"]),
        (MagicMock(), dmask, MagicMock(), td["dmask_dims"], td["data_sizes"]),
        (MagicMock(), td["template_3d"], MagicMock(), td["template_dims"], td["template_sizes"]),
        (MagicMock(), td["template_mask"], MagicMock(), td["tmask_dims"], td["template_sizes"]),
    ]
    argv = [
        "spatialfit", "data.nii.gz", "template.nii.gz", "outroot",
        "--datamask", "dmask.nii.gz",
        "--templatemask", "tmask.nii.gz",
    ]

    wnp, stn = _run_spatialfit(argv, read_returns, td["mlregress_returns"])

    # Should still produce all outputs
    assert len(wnp) == 3
    assert len(stn) == 4

    lincoffs = wnp[0][0]
    np.testing.assert_allclose(lincoffs, td["slopes"], atol=1e-12)

    if debug:
        print("Both masks test passed")


def test_spatialfit_output_shapes(debug=False):
    """Test that output arrays have correct shapes."""
    td = _make_test_data(xsize=3, ysize=2, numslices=2, timepoints=4)

    # Need 4 mlregress returns for 4 timepoints
    slopes = np.array([2.0, 3.0, 1.0, 0.5])
    mlregress_returns = [
        (np.array([[0.0, slopes[t]]]), 0.9) for t in range(4)
    ]
    tmask = np.ones((3, 2, 2), dtype=np.float64)
    tmask_dims = np.array([3, 3, 2, 2, 1, 1, 1, 1])

    read_returns = [
        (MagicMock(), td["data_4d"], MagicMock(), td["data_dims"], td["data_sizes"]),
        (MagicMock(), td["template_3d"], MagicMock(), td["template_dims"], td["template_sizes"]),
        (MagicMock(), tmask, MagicMock(), tmask_dims, td["template_sizes"]),
    ]
    argv = [
        "spatialfit", "data.nii.gz", "template.nii.gz", "outroot",
        "--templatemask", "tmask.nii.gz",
    ]

    wnp, stn = _run_spatialfit(argv, read_returns, mlregress_returns)

    # Text outputs: each should have length = timepoints
    for data, _ in wnp:
        assert len(data) == 4

    # NIfTI outputs: fit, residuals, normalized are 4D; newtemplate is 3D
    assert stn[0][0].shape == (3, 2, 2, 4)  # fit
    assert stn[1][0].shape == (3, 2, 2, 4)  # residuals
    assert stn[2][0].shape == (3, 2, 2, 4)  # normalized
    assert stn[3][0].shape == (3, 2, 2)      # newtemplate

    if debug:
        print("All output shapes verified")


# ---- spatialfit error path tests ----


def test_spatialfit_spatial_dim_mismatch_data_template(debug=False):
    """Test that mismatched spatial dims between data and template cause exit."""
    td = _make_test_data()

    # Make template with different spatial dims
    bad_template_dims = np.array([3, 3, 3, 1, 1, 1, 1, 1])  # 3x3x1 vs 2x2x1

    read_returns = [
        (MagicMock(), td["data_4d"], MagicMock(), td["data_dims"], td["data_sizes"]),
        (MagicMock(), td["template_3d"], MagicMock(), bad_template_dims, td["template_sizes"]),
    ]
    argv = ["spatialfit", "data.nii.gz", "template.nii.gz", "outroot"]

    with (
        patch("sys.argv", argv),
        patch(
            "rapidtide.workflows.parser_funcs.is_valid_file",
            side_effect=lambda parser, x: x,
        ),
        patch(
            "rapidtide.workflows.spatialfit.tide_io.readfromnifti",
            side_effect=read_returns,
        ),
        pytest.raises(SystemExit),
    ):
        spatialfit(None)

    if debug:
        print("Spatial dim mismatch correctly exits")


def test_spatialfit_template_timedim_not_one(debug=False):
    """Test that template with time dimension != 1 causes exit."""
    td = _make_test_data()

    # Template with timedim = 3 instead of 1
    bad_template_dims = np.array([4, 2, 2, 1, 3, 1, 1, 1])

    read_returns = [
        (MagicMock(), td["data_4d"], MagicMock(), td["data_dims"], td["data_sizes"]),
        (MagicMock(), td["template_3d"], MagicMock(), bad_template_dims, td["template_sizes"]),
    ]
    argv = ["spatialfit", "data.nii.gz", "template.nii.gz", "outroot"]

    with (
        patch("sys.argv", argv),
        patch(
            "rapidtide.workflows.parser_funcs.is_valid_file",
            side_effect=lambda parser, x: x,
        ),
        patch(
            "rapidtide.workflows.spatialfit.tide_io.readfromnifti",
            side_effect=read_returns,
        ),
        pytest.raises(SystemExit),
    ):
        spatialfit(None)

    if debug:
        print("Template timedim != 1 correctly exits")


def test_spatialfit_datamask_spatial_mismatch(debug=False):
    """Test that data mask with mismatched spatial dims causes exit."""
    td = _make_test_data()

    bad_dmask_dims = np.array([4, 3, 3, 1, 3, 1, 1, 1])  # different spatial dims

    read_returns = [
        (MagicMock(), td["data_4d"], MagicMock(), td["data_dims"], td["data_sizes"]),
        (MagicMock(), td["data_mask"], MagicMock(), bad_dmask_dims, td["data_sizes"]),
        (MagicMock(), td["template_3d"], MagicMock(), td["template_dims"], td["template_sizes"]),
    ]
    argv = [
        "spatialfit", "data.nii.gz", "template.nii.gz", "outroot",
        "--datamask", "dmask.nii.gz",
    ]

    with (
        patch("sys.argv", argv),
        patch(
            "rapidtide.workflows.parser_funcs.is_valid_file",
            side_effect=lambda parser, x: x,
        ),
        patch(
            "rapidtide.workflows.spatialfit.tide_io.readfromnifti",
            side_effect=read_returns,
        ),
        pytest.raises(SystemExit),
    ):
        spatialfit(None)

    if debug:
        print("Data mask spatial mismatch correctly exits")


def test_spatialfit_datamask_time_mismatch(debug=False):
    """Test that data mask with mismatched time dim causes exit."""
    td = _make_test_data()

    bad_dmask_dims = np.array([4, 2, 2, 1, 5, 1, 1, 1])  # 5 timepoints vs 3

    read_returns = [
        (MagicMock(), td["data_4d"], MagicMock(), td["data_dims"], td["data_sizes"]),
        (MagicMock(), td["data_mask"], MagicMock(), bad_dmask_dims, td["data_sizes"]),
        (MagicMock(), td["template_3d"], MagicMock(), td["template_dims"], td["template_sizes"]),
    ]
    argv = [
        "spatialfit", "data.nii.gz", "template.nii.gz", "outroot",
        "--datamask", "dmask.nii.gz",
    ]

    with (
        patch("sys.argv", argv),
        patch(
            "rapidtide.workflows.parser_funcs.is_valid_file",
            side_effect=lambda parser, x: x,
        ),
        patch(
            "rapidtide.workflows.spatialfit.tide_io.readfromnifti",
            side_effect=read_returns,
        ),
        pytest.raises(SystemExit),
    ):
        spatialfit(None)

    if debug:
        print("Data mask time mismatch correctly exits")


def test_spatialfit_templatemask_spatial_mismatch(debug=False):
    """Test that template mask with mismatched spatial dims causes exit."""
    td = _make_test_data()

    bad_tmask_dims = np.array([3, 3, 3, 1, 1, 1, 1, 1])  # 3x3 vs 2x2

    read_returns = [
        (MagicMock(), td["data_4d"], MagicMock(), td["data_dims"], td["data_sizes"]),
        (MagicMock(), td["template_3d"], MagicMock(), td["template_dims"], td["template_sizes"]),
        (MagicMock(), td["template_mask"], MagicMock(), bad_tmask_dims, td["template_sizes"]),
    ]
    argv = [
        "spatialfit", "data.nii.gz", "template.nii.gz", "outroot",
        "--templatemask", "tmask.nii.gz",
    ]

    with (
        patch("sys.argv", argv),
        patch(
            "rapidtide.workflows.parser_funcs.is_valid_file",
            side_effect=lambda parser, x: x,
        ),
        patch(
            "rapidtide.workflows.spatialfit.tide_io.readfromnifti",
            side_effect=read_returns,
        ),
        pytest.raises(SystemExit),
    ):
        spatialfit(None)

    if debug:
        print("Template mask spatial mismatch correctly exits")


def test_spatialfit_templatemask_timedim_not_one(debug=False):
    """Test that template mask with time dim != 1 causes exit."""
    td = _make_test_data()

    bad_tmask_dims = np.array([4, 2, 2, 1, 3, 1, 1, 1])  # timedim = 3

    read_returns = [
        (MagicMock(), td["data_4d"], MagicMock(), td["data_dims"], td["data_sizes"]),
        (MagicMock(), td["template_3d"], MagicMock(), td["template_dims"], td["template_sizes"]),
        (MagicMock(), td["template_mask"], MagicMock(), bad_tmask_dims, td["template_sizes"]),
    ]
    argv = [
        "spatialfit", "data.nii.gz", "template.nii.gz", "outroot",
        "--templatemask", "tmask.nii.gz",
    ]

    with (
        patch("sys.argv", argv),
        patch(
            "rapidtide.workflows.parser_funcs.is_valid_file",
            side_effect=lambda parser, x: x,
        ),
        patch(
            "rapidtide.workflows.spatialfit.tide_io.readfromnifti",
            side_effect=read_returns,
        ),
        pytest.raises(SystemExit),
    ):
        spatialfit(None)

    if debug:
        print("Template mask timedim != 1 correctly exits")


# ---- spatialfit special cases ----


def test_spatialfit_debug_flag(debug=False):
    """Test that debug flag doesn't cause errors."""
    td = _make_test_data()

    read_returns = [
        (MagicMock(), td["data_4d"], MagicMock(), td["data_dims"], td["data_sizes"]),
        (MagicMock(), td["template_3d"], MagicMock(), td["template_dims"], td["template_sizes"]),
        (MagicMock(), td["template_mask"], MagicMock(), td["tmask_dims"], td["template_sizes"]),
    ]
    argv = [
        "spatialfit", "data.nii.gz", "template.nii.gz", "outroot",
        "--templatemask", "tmask.nii.gz",
        "--debug",
    ]

    # Should not raise
    wnp, stn = _run_spatialfit(argv, read_returns, td["mlregress_returns"])
    assert len(wnp) == 3
    assert len(stn) == 4

    if debug:
        print("Debug flag test passed")


def test_spatialfit_mlregress_called_per_timepoint(debug=False):
    """Test that mlregress is called once per timepoint."""
    td = _make_test_data(timepoints=3)

    read_returns = [
        (MagicMock(), td["data_4d"], MagicMock(), td["data_dims"], td["data_sizes"]),
        (MagicMock(), td["template_3d"], MagicMock(), td["template_dims"], td["template_sizes"]),
        (MagicMock(), td["template_mask"], MagicMock(), td["tmask_dims"], td["template_sizes"]),
    ]
    argv = [
        "spatialfit", "data.nii.gz", "template.nii.gz", "outroot",
        "--templatemask", "tmask.nii.gz",
    ]

    wnp_effect, _ = _capture_writenpvecs()
    stn_effect, _ = _capture_savetonifti()

    with (
        patch("sys.argv", argv),
        patch(
            "rapidtide.workflows.parser_funcs.is_valid_file",
            side_effect=lambda parser, x: x,
        ),
        patch(
            "rapidtide.workflows.spatialfit.tide_io.readfromnifti",
            side_effect=read_returns,
        ),
        patch(
            "rapidtide.workflows.spatialfit.tide_fit.mlregress",
            side_effect=td["mlregress_returns"],
        ) as mock_mlregress,
        patch(
            "rapidtide.workflows.spatialfit.tide_io.writenpvecs",
            side_effect=wnp_effect,
        ),
        patch(
            "rapidtide.workflows.spatialfit.tide_io.savetonifti",
            side_effect=stn_effect,
        ),
    ):
        spatialfit(None)
        assert mock_mlregress.call_count == 3

    if debug:
        print("mlregress called 3 times (once per timepoint)")


def test_spatialfit_single_timepoint(debug=False):
    """Test spatialfit with a single timepoint (4D with t=1)."""
    xsize, ysize, numslices, timepoints = 2, 2, 1, 1
    numvoxels = xsize * ysize * numslices

    template_3d = np.array([1.0, 2.0, 3.0, 4.0]).reshape((xsize, ysize, numslices))
    data_4d = (2.0 * template_3d[:, :, :, np.newaxis]).copy()  # shape (2,2,1,1)

    data_dims = np.array([4, xsize, ysize, numslices, timepoints, 1, 1, 1])
    template_dims = np.array([3, xsize, ysize, numslices, 1, 1, 1, 1])
    sizes = np.array([4, 2.0, 2.0, 3.0, 1.5, 1, 1, 1])
    tmask = np.ones((xsize, ysize, numslices), dtype=np.float64)
    tmask_dims = template_dims.copy()

    mlregress_returns = [(np.array([[0.0, 2.0]]), 1.0)]

    read_returns = [
        (MagicMock(), data_4d, MagicMock(), data_dims, sizes),
        (MagicMock(), template_3d, MagicMock(), template_dims, sizes),
        (MagicMock(), tmask, MagicMock(), tmask_dims, sizes),
    ]
    argv = [
        "spatialfit", "data.nii.gz", "template.nii.gz", "outroot",
        "--templatemask", "tmask.nii.gz",
    ]

    wnp, stn = _run_spatialfit(argv, read_returns, mlregress_returns)

    lincoffs = wnp[0][0]
    assert len(lincoffs) == 1
    assert lincoffs[0] == 2.0

    newtemplate = stn[3][0]
    np.testing.assert_allclose(newtemplate, template_3d, atol=1e-12)

    if debug:
        print("Single timepoint test passed")


def test_spatialfit_partial_mask(debug=False):
    """Test spatialfit with a template mask that zeros out some voxels."""
    xsize, ysize, numslices, timepoints = 2, 2, 1, 2
    numvoxels = xsize * ysize * numslices

    template_3d = np.array([1.0, 2.0, 3.0, 4.0]).reshape((xsize, ysize, numslices))
    data_4d = np.zeros((xsize, ysize, numslices, timepoints))
    slopes = [2.0, 3.0]
    for t in range(timepoints):
        data_4d[:, :, :, t] = slopes[t] * template_3d

    # Template mask: zero out the last voxel
    tmask = np.array([1.0, 1.0, 1.0, 0.0]).reshape((xsize, ysize, numslices))

    data_dims = np.array([4, xsize, ysize, numslices, timepoints, 1, 1, 1])
    template_dims = np.array([3, xsize, ysize, numslices, 1, 1, 1, 1])
    sizes = np.array([4, 2.0, 2.0, 3.0, 1.5, 1, 1, 1])
    tmask_dims = template_dims.copy()

    mlregress_returns = [
        (np.array([[0.0, slopes[t]]]), 0.95) for t in range(timepoints)
    ]

    read_returns = [
        (MagicMock(), data_4d, MagicMock(), data_dims, sizes),
        (MagicMock(), template_3d, MagicMock(), template_dims, sizes),
        (MagicMock(), tmask, MagicMock(), tmask_dims, sizes),
    ]
    argv = [
        "spatialfit", "data.nii.gz", "template.nii.gz", "outroot",
        "--templatemask", "tmask.nii.gz",
    ]

    wnp, stn = _run_spatialfit(argv, read_returns, mlregress_returns)

    fitdata = stn[0][0]
    # The masked-out voxel (index 3, which is (1,1,0)) should have zero fitdata
    # because maskedtemplate for that voxel is 0
    assert fitdata[1, 1, 0, 0] == 0.0
    assert fitdata[1, 1, 0, 1] == 0.0

    # Unmasked voxels should have fitdata = slope * template
    np.testing.assert_allclose(fitdata[0, 0, 0, 0], 2.0 * 1.0, atol=1e-12)
    np.testing.assert_allclose(fitdata[0, 1, 0, 0], 2.0 * 2.0, atol=1e-12)

    if debug:
        print("Partial mask test passed")


def test_spatialfit_no_templatemask(debug=False):
    """Test spatialfit without a template mask (previously crashed for timepoints > 1)."""
    td = _make_test_data()

    # Only 2 readfromnifti calls: data, template (no tmask)
    read_returns = [
        (MagicMock(), td["data_4d"], MagicMock(), td["data_dims"], td["data_sizes"]),
        (MagicMock(), td["template_3d"], MagicMock(), td["template_dims"], td["template_sizes"]),
    ]
    argv = ["spatialfit", "data.nii.gz", "template.nii.gz", "outroot"]

    wnp, stn = _run_spatialfit(argv, read_returns, td["mlregress_returns"])

    # Should produce all outputs
    assert len(wnp) == 3
    assert len(stn) == 4

    lincoffs = wnp[0][0]
    np.testing.assert_allclose(lincoffs, td["slopes"], atol=1e-12)

    # newtemplate should match original template (data = slope * template, no intercept)
    newtemplate = stn[3][0]
    np.testing.assert_allclose(newtemplate, td["template_3d"], atol=1e-12)

    if debug:
        print("No template mask test passed")


def test_spatialfit_output_root_path(debug=False):
    """Test that output root path is correctly prepended to all output filenames."""
    td = _make_test_data()

    read_returns = [
        (MagicMock(), td["data_4d"], MagicMock(), td["data_dims"], td["data_sizes"]),
        (MagicMock(), td["template_3d"], MagicMock(), td["template_dims"], td["template_sizes"]),
        (MagicMock(), td["template_mask"], MagicMock(), td["tmask_dims"], td["template_sizes"]),
    ]
    argv = [
        "spatialfit", "data.nii.gz", "template.nii.gz", "/tmp/results/myfit",
        "--templatemask", "tmask.nii.gz",
    ]

    wnp, stn = _run_spatialfit(argv, read_returns, td["mlregress_returns"])

    assert wnp[0][1] == "/tmp/results/myfit_lincoffs.txt"
    assert wnp[1][1] == "/tmp/results/myfit_offsets.txt"
    assert wnp[2][1] == "/tmp/results/myfit_r2vals.txt"
    assert stn[0][1] == "/tmp/results/myfit_fit"
    assert stn[1][1] == "/tmp/results/myfit_residuals"
    assert stn[2][1] == "/tmp/results/myfit_normalized"
    assert stn[3][1] == "/tmp/results/myfit_newtemplate"

    if debug:
        print("Output root path test passed")


# ---- main test entry point ----


def test_spatialfit_all(debug=False):
    """Run all spatialfit sub-tests."""
    # _get_parser tests
    test_get_parser_returns_parser(debug=debug)
    test_get_parser_required_args(debug=debug)
    test_get_parser_with_valid_args(debug=debug)
    test_get_parser_optional_flags(debug=debug)
    test_get_parser_order_is_int(debug=debug)
    test_get_parser_default_order(debug=debug)

    # spatialfit happy path tests
    test_spatialfit_basic_outputs(debug=debug)
    test_spatialfit_lincoffs_offsets_r2(debug=debug)
    test_spatialfit_fitdata(debug=debug)
    test_spatialfit_residuals(debug=debug)
    test_spatialfit_normalized(debug=debug)
    test_spatialfit_newtemplate(debug=debug)
    test_spatialfit_with_nonzero_intercepts(debug=debug)
    test_spatialfit_with_both_masks(debug=debug)
    test_spatialfit_output_shapes(debug=debug)

    # error path tests
    test_spatialfit_spatial_dim_mismatch_data_template(debug=debug)
    test_spatialfit_template_timedim_not_one(debug=debug)
    test_spatialfit_datamask_spatial_mismatch(debug=debug)
    test_spatialfit_datamask_time_mismatch(debug=debug)
    test_spatialfit_templatemask_spatial_mismatch(debug=debug)
    test_spatialfit_templatemask_timedim_not_one(debug=debug)

    # special cases
    test_spatialfit_debug_flag(debug=debug)
    test_spatialfit_mlregress_called_per_timepoint(debug=debug)
    test_spatialfit_single_timepoint(debug=debug)
    test_spatialfit_partial_mask(debug=debug)
    test_spatialfit_no_templatemask(debug=debug)
    test_spatialfit_output_root_path(debug=debug)


if __name__ == "__main__":
    test_spatialfit_all(debug=True)
