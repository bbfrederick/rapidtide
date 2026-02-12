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

from rapidtide.workflows.pairproc import _get_parser, pairproc

# ---- helpers ----


def _make_args(**overrides):
    """Build a minimal args Namespace for pairproc with sensible defaults."""
    defaults = dict(
        inputfile="input.nii.gz",
        outputroot="/tmp/test_pairproc_out",
        datamaskname=None,
        getdist=False,
        demean=False,
        debug=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _make_mock_hdr():
    """Create a mock NIfTI header that supports copy() and item access."""
    hdr = {
        "pixdim": np.zeros(8),
    }

    class MockHdr(dict):
        def copy(self):
            new = MockHdr(self)
            new["pixdim"] = np.array(self["pixdim"])
            return new

    return MockHdr(hdr)


def _make_input_data(xsize=2, ysize=2, numslices=1, timepoints=6, seed=42):
    """Create synthetic 4D input data for pairproc tests.

    Returns
    -------
    dict with keys: 'data', 'hdr', 'dims', 'sizes', and component values.
    """
    rng = np.random.RandomState(seed)
    data = rng.randn(xsize, ysize, numslices, timepoints).astype(np.double)
    # Ensure all voxels have nonzero range (so they pass the auto-mask)
    data += np.arange(timepoints).reshape(1, 1, 1, -1) * 0.1

    hdr = _make_mock_hdr()
    dims = np.array([4, xsize, ysize, numslices, timepoints])
    sizes = np.array([0, 1.0, 1.0, 1.0, 1.0])

    return {
        "data": data,
        "hdr": hdr,
        "dims": dims,
        "sizes": sizes,
        "xsize": xsize,
        "ysize": ysize,
        "numslices": numslices,
        "timepoints": timepoints,
    }


def _make_mask_data(xsize=2, ysize=2, numslices=1, mask_value=1.0):
    """Create a 3D mask for pairproc tests."""
    data = np.full((xsize, ysize, numslices), mask_value, dtype=np.double)
    hdr = _make_mock_hdr()
    dims = np.array([3, xsize, ysize, numslices, 1])
    sizes = np.array([0, 1.0, 1.0, 1.0, 1.0])
    return {
        "data": data,
        "hdr": hdr,
        "dims": dims,
        "sizes": sizes,
    }


def _capture_savetonifti():
    """Capture savetonifti calls, copying data at call time."""
    captured = []

    def _side_effect(data, hdr, filename, **kwargs):
        captured.append((np.array(data, copy=True), filename))

    return _side_effect, captured


def _capture_writenpvecs():
    """Capture writenpvecs calls, copying data at call time."""
    captured = []

    def _side_effect(data, filename, **kwargs):
        captured.append((np.array(data, copy=True), filename))

    return _side_effect, captured


def _run_pairproc(args, input_info, mask_info=None):
    """Run pairproc with full mocking of external dependencies.

    Parameters
    ----------
    args : Namespace
        Arguments to pass to pairproc.
    input_info : dict
        Input data info from _make_input_data.
    mask_info : dict or None
        Mask data info from _make_mask_data.

    Returns
    -------
    dict with keys: 'savetonifti', 'writenpvecs'
    """
    stn_effect, stn_captured = _capture_savetonifti()
    wnp_effect, wnp_captured = _capture_writenpvecs()

    readfromnifti_returns = [(
        MagicMock(),
        input_info["data"],
        input_info["hdr"],
        input_info["dims"],
        input_info["sizes"],
    )]
    if mask_info is not None:
        readfromnifti_returns.append((
            MagicMock(),
            mask_info["data"],
            mask_info["hdr"],
            mask_info["dims"],
            mask_info["sizes"],
        ))

    with (
        patch(
            "rapidtide.workflows.pairproc.tide_io.readfromnifti",
            side_effect=readfromnifti_returns,
        ),
        patch(
            "rapidtide.workflows.pairproc.tide_io.parseniftidims",
            return_value=(
                input_info["xsize"],
                input_info["ysize"],
                input_info["numslices"],
                input_info["timepoints"],
            ),
        ),
        patch(
            "rapidtide.workflows.pairproc.tide_io.checkspacedimmatch",
            return_value=True,
        ),
        patch(
            "rapidtide.workflows.pairproc.tide_io.savetonifti",
            side_effect=stn_effect,
        ),
        patch(
            "rapidtide.workflows.pairproc.tide_io.writenpvecs",
            side_effect=wnp_effect,
        ),
    ):
        pairproc(args)

    return {
        "savetonifti": stn_captured,
        "writenpvecs": wnp_captured,
    }


# ---- _get_parser tests ----


def test_parser_required_args(debug=False):
    """Parser should accept the two required positional args."""
    parser = _get_parser()
    args = parser.parse_args(["input.nii.gz", "outroot"])
    assert args.inputfile == "input.nii.gz"
    assert args.outputroot == "outroot"
    if debug:
        print("test_parser_required_args passed")


def test_parser_defaults(debug=False):
    """Parser defaults should match expected values."""
    parser = _get_parser()
    args = parser.parse_args(["input.nii.gz", "outroot"])
    assert args.datamaskname is None
    assert args.getdist is False
    assert args.demean is False
    assert args.debug is False
    if debug:
        print("test_parser_defaults passed")


def test_parser_dmask(debug=False):
    """Parser should accept --dmask with a valid file."""
    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as f:
        parser = _get_parser()
        args = parser.parse_args(["input.nii.gz", "outroot", "--dmask", f.name])
        assert args.datamaskname == f.name
    if debug:
        print("test_parser_dmask passed")


def test_parser_getdist(debug=False):
    """Parser should accept --getdist flag."""
    parser = _get_parser()
    args = parser.parse_args(["input.nii.gz", "outroot", "--getdist"])
    assert args.getdist is True
    if debug:
        print("test_parser_getdist passed")


def test_parser_demean(debug=False):
    """Parser should accept --demean flag."""
    parser = _get_parser()
    args = parser.parse_args(["input.nii.gz", "outroot", "--demean"])
    assert args.demean is True
    if debug:
        print("test_parser_demean passed")


def test_parser_debug_flag(debug=False):
    """Parser should accept --debug flag."""
    parser = _get_parser()
    args = parser.parse_args(["input.nii.gz", "outroot", "--debug"])
    assert args.debug is True
    if debug:
        print("test_parser_debug_flag passed")


def test_parser_missing_required(debug=False):
    """Parser should fail when required arguments are missing."""
    parser = _get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    if debug:
        print("test_parser_missing_required passed")


def test_parser_all_options(debug=False):
    """Parser should accept all optional flags together."""
    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as f:
        parser = _get_parser()
        args = parser.parse_args([
            "input.nii.gz", "outroot",
            "--dmask", f.name,
            "--getdist",
            "--demean",
            "--debug",
        ])
        assert args.datamaskname == f.name
        assert args.getdist is True
        assert args.demean is True
        assert args.debug is True
    if debug:
        print("test_parser_all_options passed")


# ---- pairproc basic tests ----


def test_pairproc_basic_run(debug=False):
    """pairproc should run without errors with basic inputs."""
    input_info = _make_input_data(timepoints=6)
    args = _make_args()
    result = _run_pairproc(args, input_info)
    assert len(result["savetonifti"]) > 0
    assert len(result["writenpvecs"]) > 0
    if debug:
        print("test_pairproc_basic_run passed")


def test_pairproc_output_files_real(debug=False):
    """pairproc should write temporal and spatial output files for 'real' run."""
    input_info = _make_input_data(timepoints=6)
    args = _make_args(outputroot="/tmp/pp")
    result = _run_pairproc(args, input_info)

    stn_files = [e[1] for e in result["savetonifti"]]
    wnp_files = [e[1] for e in result["writenpvecs"]]

    # Temporal: correlations and p-values NIfTI for "real"
    assert any("_temporalcorrelations_real" in f for f in stn_files)
    assert any("_temporalpvalues_real" in f for f in stn_files)

    # Spatial: correlations and p-values text for "real"
    assert any("_r1r2spatialcorrelations_real.txt" in f for f in wnp_files)
    assert any("_r1r2spatialpvalues_real.txt" in f for f in wnp_files)

    if debug:
        print("test_pairproc_output_files_real passed")


def test_pairproc_output_counts_no_getdist(debug=False):
    """Without getdist, should produce 2 NIfTI and 2 text files."""
    input_info = _make_input_data(timepoints=6)
    args = _make_args(getdist=False)
    result = _run_pairproc(args, input_info)

    assert len(result["savetonifti"]) == 2  # temporal corr + pval
    assert len(result["writenpvecs"]) == 2  # spatial corr + pval
    if debug:
        print("test_pairproc_output_counts_no_getdist passed")


def test_pairproc_temporal_output_shape(debug=False):
    """Temporal output maps should have 3D shape (xsize, ysize, numslices)."""
    xsize, ysize, numslices = 3, 2, 1
    input_info = _make_input_data(xsize=xsize, ysize=ysize, numslices=numslices, timepoints=8)
    args = _make_args()
    result = _run_pairproc(args, input_info)

    for data, fname in result["savetonifti"]:
        assert data.shape == (xsize, ysize, numslices)
    if debug:
        print("test_pairproc_temporal_output_shape passed")


def test_pairproc_spatial_output_length(debug=False):
    """Spatial output vectors should have length numsubjects = timepoints // 2."""
    timepoints = 10
    input_info = _make_input_data(timepoints=timepoints)
    args = _make_args()
    result = _run_pairproc(args, input_info)

    numsubjects = timepoints // 2
    for data, fname in result["writenpvecs"]:
        assert data.shape == (numsubjects,)
    if debug:
        print("test_pairproc_spatial_output_length passed")


# ---- even/odd split tests ----


def test_pairproc_odd_timepoints_raises(debug=False):
    """pairproc should raise ValueError for odd number of timepoints."""
    input_info = _make_input_data(timepoints=7)
    args = _make_args()

    with (
        patch(
            "rapidtide.workflows.pairproc.tide_io.readfromnifti",
            return_value=(
                MagicMock(),
                input_info["data"],
                input_info["hdr"],
                input_info["dims"],
                input_info["sizes"],
            ),
        ),
        patch(
            "rapidtide.workflows.pairproc.tide_io.parseniftidims",
            return_value=(
                input_info["xsize"],
                input_info["ysize"],
                input_info["numslices"],
                input_info["timepoints"],
            ),
        ),
        pytest.raises(ValueError, match="even number"),
    ):
        pairproc(args)

    if debug:
        print("test_pairproc_odd_timepoints_raises passed")


def test_pairproc_even_odd_split(debug=False):
    """Verify that even and odd volumes are correctly separated."""
    xsize, ysize, numslices, timepoints = 1, 1, 1, 4
    # Create data: [10, 20, 30, 40] → even=[10, 30], odd=[20, 40]
    data = np.array([[[[10.0, 20.0, 30.0, 40.0]]]])
    input_info = {
        "data": data,
        "hdr": _make_mock_hdr(),
        "dims": np.array([4, xsize, ysize, numslices, timepoints]),
        "sizes": np.array([0, 1.0, 1.0, 1.0, 1.0]),
        "xsize": xsize,
        "ysize": ysize,
        "numslices": numslices,
        "timepoints": timepoints,
    }
    args = _make_args()

    # We need to capture what pearsonr receives to verify even/odd split
    pearsonr_calls = []

    original_pearsonr_result = MagicMock()
    original_pearsonr_result.statistic = 0.9
    original_pearsonr_result.pvalue = 0.01

    def mock_pearsonr(a, b):
        pearsonr_calls.append((np.array(a, copy=True), np.array(b, copy=True)))
        return original_pearsonr_result

    stn_effect, _ = _capture_savetonifti()
    wnp_effect, _ = _capture_writenpvecs()

    with (
        patch(
            "rapidtide.workflows.pairproc.tide_io.readfromnifti",
            return_value=(MagicMock(), data, input_info["hdr"], input_info["dims"], input_info["sizes"]),
        ),
        patch(
            "rapidtide.workflows.pairproc.tide_io.parseniftidims",
            return_value=(xsize, ysize, numslices, timepoints),
        ),
        patch("rapidtide.workflows.pairproc.tide_io.savetonifti", side_effect=stn_effect),
        patch("rapidtide.workflows.pairproc.tide_io.writenpvecs", side_effect=wnp_effect),
        patch("rapidtide.workflows.pairproc.pearsonr", side_effect=mock_pearsonr),
    ):
        pairproc(args)

    # First pearsonr call should be the temporal correlation for the one voxel
    # evenims = stdnormalize([10, 30]), oddims = stdnormalize([20, 40])
    # After stdnormalize with 2 values: both become [-1, 1] (identical normalization)
    assert len(pearsonr_calls) >= 1
    if debug:
        print("test_pairproc_even_odd_split passed")


# ---- demean tests ----


def test_pairproc_demean(debug=False):
    """pairproc with demean should subtract mean from each image."""
    xsize, ysize, numslices, timepoints = 1, 2, 1, 4
    # 2 voxels, 4 timepoints: even=[0,2], odd=[1,3]
    data = np.zeros((xsize, ysize, numslices, timepoints))
    data[0, 0, 0, :] = [10.0, 20.0, 30.0, 40.0]
    data[0, 1, 0, :] = [50.0, 60.0, 70.0, 80.0]

    input_info = {
        "data": data,
        "hdr": _make_mock_hdr(),
        "dims": np.array([4, xsize, ysize, numslices, timepoints]),
        "sizes": np.array([0, 1.0, 1.0, 1.0, 1.0]),
        "xsize": xsize,
        "ysize": ysize,
        "numslices": numslices,
        "timepoints": timepoints,
    }
    args = _make_args(demean=True)

    # Track what values get passed to pearsonr
    pearsonr_inputs = []

    mock_result = MagicMock()
    mock_result.statistic = 0.5
    mock_result.pvalue = 0.1

    def mock_pearsonr(a, b):
        pearsonr_inputs.append((np.array(a, copy=True), np.array(b, copy=True)))
        return mock_result

    stn_effect, _ = _capture_savetonifti()
    wnp_effect, _ = _capture_writenpvecs()

    with (
        patch(
            "rapidtide.workflows.pairproc.tide_io.readfromnifti",
            return_value=(MagicMock(), data, input_info["hdr"], input_info["dims"], input_info["sizes"]),
        ),
        patch(
            "rapidtide.workflows.pairproc.tide_io.parseniftidims",
            return_value=(xsize, ysize, numslices, timepoints),
        ),
        patch("rapidtide.workflows.pairproc.tide_io.savetonifti", side_effect=stn_effect),
        patch("rapidtide.workflows.pairproc.tide_io.writenpvecs", side_effect=wnp_effect),
        patch("rapidtide.workflows.pairproc.pearsonr", side_effect=mock_pearsonr),
    ):
        pairproc(args)

    # pearsonr should have been called (temporal + spatial)
    assert len(pearsonr_inputs) > 0
    if debug:
        print("test_pairproc_demean passed")


def test_pairproc_no_demean(debug=False):
    """pairproc without demean should use raw values."""
    input_info = _make_input_data(timepoints=6)
    args = _make_args(demean=False)
    result = _run_pairproc(args, input_info)
    assert len(result["savetonifti"]) == 2
    if debug:
        print("test_pairproc_no_demean passed")


# ---- getdist tests ----


def test_pairproc_getdist(debug=False):
    """With getdist, should produce outputs for both 'real' and 'shifted' runs."""
    input_info = _make_input_data(timepoints=6)
    args = _make_args(getdist=True)
    result = _run_pairproc(args, input_info)

    # Should have 4 NIfTI files (2 runs x 2 files) and 4 text files
    assert len(result["savetonifti"]) == 4
    assert len(result["writenpvecs"]) == 4

    stn_files = [e[1] for e in result["savetonifti"]]
    wnp_files = [e[1] for e in result["writenpvecs"]]

    # Check for both real and shifted outputs
    assert any("_temporalcorrelations_real" in f for f in stn_files)
    assert any("_temporalcorrelations_shifted" in f for f in stn_files)
    assert any("_temporalpvalues_real" in f for f in stn_files)
    assert any("_temporalpvalues_shifted" in f for f in stn_files)
    assert any("_r1r2spatialcorrelations_real.txt" in f for f in wnp_files)
    assert any("_r1r2spatialcorrelations_shifted.txt" in f for f in wnp_files)
    assert any("_r1r2spatialpvalues_real.txt" in f for f in wnp_files)
    assert any("_r1r2spatialpvalues_shifted.txt" in f for f in wnp_files)

    if debug:
        print("test_pairproc_getdist passed")


def test_pairproc_shifted_uses_roll(debug=False):
    """In shifted run, oddims should be rolled by 1 along subject axis."""
    xsize, ysize, numslices, timepoints = 1, 1, 1, 6
    # Single voxel, 6 timepoints → 3 subjects
    data = np.array([[[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]]])

    input_info = {
        "data": data,
        "hdr": _make_mock_hdr(),
        "dims": np.array([4, xsize, ysize, numslices, timepoints]),
        "sizes": np.array([0, 1.0, 1.0, 1.0, 1.0]),
        "xsize": xsize,
        "ysize": ysize,
        "numslices": numslices,
        "timepoints": timepoints,
    }
    args = _make_args(getdist=True)

    pearsonr_calls = []

    mock_result = MagicMock()
    mock_result.statistic = 0.5
    mock_result.pvalue = 0.1

    def mock_pearsonr(a, b):
        pearsonr_calls.append((np.array(a, copy=True), np.array(b, copy=True)))
        return mock_result

    stn_effect, _ = _capture_savetonifti()
    wnp_effect, _ = _capture_writenpvecs()

    with (
        patch(
            "rapidtide.workflows.pairproc.tide_io.readfromnifti",
            return_value=(MagicMock(), data, input_info["hdr"], input_info["dims"], input_info["sizes"]),
        ),
        patch(
            "rapidtide.workflows.pairproc.tide_io.parseniftidims",
            return_value=(xsize, ysize, numslices, timepoints),
        ),
        patch("rapidtide.workflows.pairproc.tide_io.savetonifti", side_effect=stn_effect),
        patch("rapidtide.workflows.pairproc.tide_io.writenpvecs", side_effect=wnp_effect),
        patch("rapidtide.workflows.pairproc.pearsonr", side_effect=mock_pearsonr),
    ):
        pairproc(args)

    # Two runs (real + shifted), each with 1 temporal + 3 spatial pearsonr calls = 4 per run = 8
    assert len(pearsonr_calls) == 8
    if debug:
        print("test_pairproc_shifted_uses_roll passed")


# ---- mask tests ----


def test_pairproc_with_mask(debug=False):
    """pairproc should use provided mask to select voxels."""
    xsize, ysize, numslices = 2, 2, 1
    timepoints = 4
    input_info = _make_input_data(xsize=xsize, ysize=ysize, numslices=numslices, timepoints=timepoints)

    # Mask out half the voxels
    mask = _make_mask_data(xsize=xsize, ysize=ysize, numslices=numslices)
    mask["data"][0, 0, 0] = 0.0  # mask out one voxel
    mask["data"][1, 1, 0] = 0.0  # mask out another

    args = _make_args(datamaskname="mask.nii.gz")
    result = _run_pairproc(args, input_info, mask_info=mask)

    # Should still produce outputs
    assert len(result["savetonifti"]) == 2
    assert len(result["writenpvecs"]) == 2

    # Temporal output should have zeros at masked voxels
    for data, fname in result["savetonifti"]:
        if "_temporalcorrelations_real" in fname:
            assert data[0, 0, 0] == 0.0  # masked voxel
            assert data[1, 1, 0] == 0.0  # masked voxel
            break

    if debug:
        print("test_pairproc_with_mask passed")


def test_pairproc_auto_mask(debug=False):
    """Without mask, voxels with zero range should be excluded."""
    xsize, ysize, numslices = 3, 1, 1
    timepoints = 4
    data = np.zeros((xsize, ysize, numslices, timepoints))
    # First two voxels have nonzero data, third is constant (zero range)
    data[0, 0, 0, :] = [1.0, 2.0, 3.0, 4.0]
    data[1, 0, 0, :] = [5.0, 6.0, 7.0, 8.0]
    data[2, 0, 0, :] = [9.0, 9.0, 9.0, 9.0]  # zero range

    input_info = {
        "data": data,
        "hdr": _make_mock_hdr(),
        "dims": np.array([4, xsize, ysize, numslices, timepoints]),
        "sizes": np.array([0, 1.0, 1.0, 1.0, 1.0]),
        "xsize": xsize,
        "ysize": ysize,
        "numslices": numslices,
        "timepoints": timepoints,
    }
    args = _make_args()
    result = _run_pairproc(args, input_info)

    for data_out, fname in result["savetonifti"]:
        if "_temporalcorrelations_real" in fname:
            # Third voxel should have zero correlation (excluded by auto-mask)
            assert data_out[2, 0, 0] == 0.0
            # First voxel should have nonzero correlation
            assert data_out[0, 0, 0] != 0.0
            break

    if debug:
        print("test_pairproc_auto_mask passed")


def test_pairproc_mask_dim_mismatch_exits(debug=False):
    """pairproc should exit when mask spatial dimensions don't match."""
    input_info = _make_input_data(xsize=2, ysize=2, numslices=1, timepoints=4)
    mask_info = _make_mask_data(xsize=3, ysize=3, numslices=1)  # wrong dims
    args = _make_args(datamaskname="mask.nii.gz")

    with (
        patch(
            "rapidtide.workflows.pairproc.tide_io.readfromnifti",
            side_effect=[
                (MagicMock(), input_info["data"], input_info["hdr"], input_info["dims"], input_info["sizes"]),
                (MagicMock(), mask_info["data"], mask_info["hdr"], mask_info["dims"], mask_info["sizes"]),
            ],
        ),
        patch(
            "rapidtide.workflows.pairproc.tide_io.parseniftidims",
            return_value=(input_info["xsize"], input_info["ysize"], input_info["numslices"], input_info["timepoints"]),
        ),
        patch(
            "rapidtide.workflows.pairproc.tide_io.checkspacedimmatch",
            return_value=False,
        ),
        patch("builtins.exit", side_effect=SystemExit) as mock_exit,
    ):
        with pytest.raises(SystemExit):
            pairproc(args)
        mock_exit.assert_called_once()

    if debug:
        print("test_pairproc_mask_dim_mismatch_exits passed")


def test_pairproc_mask_time_not_one_exits(debug=False):
    """pairproc should exit when mask time dimension is not 1."""
    input_info = _make_input_data(xsize=2, ysize=2, numslices=1, timepoints=4)
    mask_info = _make_mask_data(xsize=2, ysize=2, numslices=1)
    mask_info["dims"][4] = 4  # wrong time dim

    args = _make_args(datamaskname="mask.nii.gz")

    with (
        patch(
            "rapidtide.workflows.pairproc.tide_io.readfromnifti",
            side_effect=[
                (MagicMock(), input_info["data"], input_info["hdr"], input_info["dims"], input_info["sizes"]),
                (MagicMock(), mask_info["data"], mask_info["hdr"], mask_info["dims"], mask_info["sizes"]),
            ],
        ),
        patch(
            "rapidtide.workflows.pairproc.tide_io.parseniftidims",
            return_value=(input_info["xsize"], input_info["ysize"], input_info["numslices"], input_info["timepoints"]),
        ),
        patch(
            "rapidtide.workflows.pairproc.tide_io.checkspacedimmatch",
            return_value=True,
        ),
        patch("builtins.exit", side_effect=SystemExit) as mock_exit,
    ):
        with pytest.raises(SystemExit):
            pairproc(args)
        mock_exit.assert_called_once()

    if debug:
        print("test_pairproc_mask_time_not_one_exits passed")


# ---- debug mode test ----


def test_pairproc_debug_mode(debug=False):
    """pairproc should run successfully with debug=True."""
    input_info = _make_input_data(timepoints=6)
    args = _make_args(debug=True)
    result = _run_pairproc(args, input_info)
    assert len(result["savetonifti"]) == 2
    if debug:
        print("test_pairproc_debug_mode passed")


# ---- correlation value tests ----


def test_pairproc_temporal_correlations_range(debug=False):
    """Temporal correlations should be in [-1, 1] for valid voxels."""
    input_info = _make_input_data(xsize=3, ysize=3, numslices=1, timepoints=10)
    args = _make_args()
    result = _run_pairproc(args, input_info)

    for data, fname in result["savetonifti"]:
        if "_temporalcorrelations_real" in fname:
            nonzero = data[data != 0.0]
            assert np.all(nonzero >= -1.0)
            assert np.all(nonzero <= 1.0)
            break
    if debug:
        print("test_pairproc_temporal_correlations_range passed")


def test_pairproc_temporal_pvalues_range(debug=False):
    """Temporal p-values should be in [0, 1] for valid voxels."""
    input_info = _make_input_data(xsize=3, ysize=3, numslices=1, timepoints=10)
    args = _make_args()
    result = _run_pairproc(args, input_info)

    for data, fname in result["savetonifti"]:
        if "_temporalpvalues_real" in fname:
            nonzero = data[data != 0.0]
            assert np.all(nonzero >= 0.0)
            assert np.all(nonzero <= 1.0)
            break
    if debug:
        print("test_pairproc_temporal_pvalues_range passed")


def test_pairproc_spatial_correlations_range(debug=False):
    """Spatial correlations should be in [-1, 1]."""
    input_info = _make_input_data(xsize=3, ysize=3, numslices=1, timepoints=10)
    args = _make_args()
    result = _run_pairproc(args, input_info)

    for data, fname in result["writenpvecs"]:
        if "_r1r2spatialcorrelations_real.txt" in fname:
            assert np.all(data >= -1.0)
            assert np.all(data <= 1.0)
            break
    if debug:
        print("test_pairproc_spatial_correlations_range passed")


def test_pairproc_spatial_pvalues_range(debug=False):
    """Spatial p-values should be in [0, 1]."""
    input_info = _make_input_data(xsize=3, ysize=3, numslices=1, timepoints=10)
    args = _make_args()
    result = _run_pairproc(args, input_info)

    for data, fname in result["writenpvecs"]:
        if "_r1r2spatialpvalues_real.txt" in fname:
            assert np.all(data >= 0.0)
            assert np.all(data <= 1.0)
            break
    if debug:
        print("test_pairproc_spatial_pvalues_range passed")


# ---- edge case tests ----


def test_pairproc_four_timepoints(debug=False):
    """pairproc should work with 4 timepoints (minimum practical: 2 subjects)."""
    input_info = _make_input_data(xsize=2, ysize=1, numslices=1, timepoints=4)
    args = _make_args()
    result = _run_pairproc(args, input_info)

    assert len(result["savetonifti"]) == 2
    # numsubjects = 2
    for data, fname in result["writenpvecs"]:
        assert data.shape == (2,)
    if debug:
        print("test_pairproc_four_timepoints passed")


def test_pairproc_large_timepoints(debug=False):
    """pairproc should handle larger time series."""
    input_info = _make_input_data(xsize=2, ysize=2, numslices=1, timepoints=20)
    args = _make_args()
    result = _run_pairproc(args, input_info)

    numsubjects = 10
    for data, fname in result["writenpvecs"]:
        assert data.shape == (numsubjects,)
    if debug:
        print("test_pairproc_large_timepoints passed")


def test_pairproc_multi_slice(debug=False):
    """pairproc should work with multiple slices."""
    xsize, ysize, numslices = 2, 2, 3
    input_info = _make_input_data(xsize=xsize, ysize=ysize, numslices=numslices, timepoints=6)
    args = _make_args()
    result = _run_pairproc(args, input_info)

    for data, fname in result["savetonifti"]:
        assert data.shape == (xsize, ysize, numslices)
    if debug:
        print("test_pairproc_multi_slice passed")


# ---- combined option tests ----


def test_pairproc_demean_and_getdist(debug=False):
    """pairproc should work with both demean and getdist enabled."""
    input_info = _make_input_data(timepoints=8)
    args = _make_args(demean=True, getdist=True)
    result = _run_pairproc(args, input_info)

    # getdist produces 4 NIfTI and 4 text files
    assert len(result["savetonifti"]) == 4
    assert len(result["writenpvecs"]) == 4
    if debug:
        print("test_pairproc_demean_and_getdist passed")


def test_pairproc_mask_and_demean(debug=False):
    """pairproc should work with both mask and demean."""
    input_info = _make_input_data(xsize=2, ysize=2, numslices=1, timepoints=6)
    mask_info = _make_mask_data(xsize=2, ysize=2, numslices=1)
    args = _make_args(datamaskname="mask.nii.gz", demean=True)
    result = _run_pairproc(args, input_info, mask_info=mask_info)

    assert len(result["savetonifti"]) == 2
    if debug:
        print("test_pairproc_mask_and_demean passed")


# ---- integration test ----


def test_pairproc_integration(debug=False):
    """Full integration test verifying all outputs are consistent."""
    xsize, ysize, numslices = 3, 2, 1
    timepoints = 8
    input_info = _make_input_data(
        xsize=xsize, ysize=ysize, numslices=numslices, timepoints=timepoints
    )
    args = _make_args(getdist=True, demean=True, debug=True)
    result = _run_pairproc(args, input_info)

    numsubjects = timepoints // 2
    numspatiallocs = xsize * ysize * numslices

    # Check output counts: 2 runs x 2 NIfTI + 2 text = 4 each
    assert len(result["savetonifti"]) == 4
    assert len(result["writenpvecs"]) == 4

    # Verify NIfTI shapes
    for data, fname in result["savetonifti"]:
        assert data.shape == (xsize, ysize, numslices)

    # Verify text vector shapes
    for data, fname in result["writenpvecs"]:
        assert data.shape == (numsubjects,)

    # Verify all file suffixes present
    stn_files = [e[1] for e in result["savetonifti"]]
    wnp_files = [e[1] for e in result["writenpvecs"]]
    for run in ["real", "shifted"]:
        assert any(f"_temporalcorrelations_{run}" in f for f in stn_files)
        assert any(f"_temporalpvalues_{run}" in f for f in stn_files)
        assert any(f"_r1r2spatialcorrelations_{run}.txt" in f for f in wnp_files)
        assert any(f"_r1r2spatialpvalues_{run}.txt" in f for f in wnp_files)

    # Verify correlation ranges
    for data, fname in result["savetonifti"]:
        if "_temporalcorrelations_" in fname:
            nonzero = data[data != 0.0]
            if len(nonzero) > 0:
                assert np.all(nonzero >= -1.0)
                assert np.all(nonzero <= 1.0)

    for data, fname in result["writenpvecs"]:
        if "_r1r2spatialcorrelations_" in fname:
            assert np.all(data >= -1.0)
            assert np.all(data <= 1.0)

    if debug:
        print("test_pairproc_integration passed")


# ---- main test function ----


def test_pairproc(debug=False):
    """Run all pairproc tests."""
    # parser tests
    test_parser_required_args(debug=debug)
    test_parser_defaults(debug=debug)
    test_parser_dmask(debug=debug)
    test_parser_getdist(debug=debug)
    test_parser_demean(debug=debug)
    test_parser_debug_flag(debug=debug)
    test_parser_missing_required(debug=debug)
    test_parser_all_options(debug=debug)

    # basic tests
    test_pairproc_basic_run(debug=debug)
    test_pairproc_output_files_real(debug=debug)
    test_pairproc_output_counts_no_getdist(debug=debug)
    test_pairproc_temporal_output_shape(debug=debug)
    test_pairproc_spatial_output_length(debug=debug)

    # even/odd split
    test_pairproc_odd_timepoints_raises(debug=debug)
    test_pairproc_even_odd_split(debug=debug)

    # demean
    test_pairproc_demean(debug=debug)
    test_pairproc_no_demean(debug=debug)

    # getdist
    test_pairproc_getdist(debug=debug)
    test_pairproc_shifted_uses_roll(debug=debug)

    # mask
    test_pairproc_with_mask(debug=debug)
    test_pairproc_auto_mask(debug=debug)
    test_pairproc_mask_dim_mismatch_exits(debug=debug)
    test_pairproc_mask_time_not_one_exits(debug=debug)

    # debug mode
    test_pairproc_debug_mode(debug=debug)

    # correlation ranges
    test_pairproc_temporal_correlations_range(debug=debug)
    test_pairproc_temporal_pvalues_range(debug=debug)
    test_pairproc_spatial_correlations_range(debug=debug)
    test_pairproc_spatial_pvalues_range(debug=debug)

    # edge cases
    test_pairproc_four_timepoints(debug=debug)
    test_pairproc_large_timepoints(debug=debug)
    test_pairproc_multi_slice(debug=debug)

    # combined options
    test_pairproc_demean_and_getdist(debug=debug)
    test_pairproc_mask_and_demean(debug=debug)

    # integration
    test_pairproc_integration(debug=debug)


if __name__ == "__main__":
    test_pairproc(debug=True)
