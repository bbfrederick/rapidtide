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
import io
import os
import tempfile
from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from rapidtide.workflows.pixelcomp import (
    _get_parser,
    bland_altman_plot,
    pairdata,
    pixelcomp,
)

# ---- helpers ----


def _make_default_args(**overrides):
    """Create default args Namespace for pixelcomp."""
    defaults = dict(
        inputfilename1="input1.nii.gz",
        maskfilename1="mask1.nii.gz",
        inputfilename2="input2.nii.gz",
        maskfilename2="mask2.nii.gz",
        outputroot="/tmp/test_pixelcomp_out",
        scatter=False,
        fitonly=False,
        display=False,
        fitorder=1,
        usex=False,
        histbins=51,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _make_3d_test_data(shape=(4, 5, 3), rng=None):
    """Create synthetic 3D image data, masks, and dims arrays.

    Returns (input1, mask1, input2, mask2, dims, sizes).
    Input2 = 2*Input1 + 1 + noise for a known linear relationship.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    input1 = rng.normal(10.0, 3.0, shape)
    input2 = 2.0 * input1 + 1.0 + rng.normal(0, 0.1, shape)

    # masks: all ones (all voxels valid)
    mask1 = np.ones(shape, dtype=np.float64)
    mask2 = np.ones(shape, dtype=np.float64)

    dims = np.array([3, shape[0], shape[1], shape[2], 1, 1, 1, 1])
    sizes = np.array([3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    return input1, mask1, input2, mask2, dims, sizes


def _run_pixelcomp_with_mocks(args, input1, mask1, input2, mask2, dims, sizes,
                               dim_match_results=None):
    """Run pixelcomp with fully mocked I/O.

    Returns dict with captured output.
    """
    captured = {
        "savefig_calls": [],
        "written_files": {},
    }

    file_map = {
        args.inputfilename1: (MagicMock(), input1.copy(), MagicMock(), dims.copy(), sizes.copy()),
        args.maskfilename1: (MagicMock(), mask1.copy(), MagicMock(), dims.copy(), sizes.copy()),
        args.inputfilename2: (MagicMock(), input2.copy(), MagicMock(), dims.copy(), sizes.copy()),
        args.maskfilename2: (MagicMock(), mask2.copy(), MagicMock(), dims.copy(), sizes.copy()),
    }

    def mock_readfromnifti(fname, headeronly=False):
        if fname in file_map:
            return file_map[fname]
        raise ValueError(f"Unexpected file: {fname}")

    if dim_match_results is None:
        def mock_checkspacedimmatch(d1, d2):
            return True
    else:
        call_idx = [0]

        def mock_checkspacedimmatch(d1, d2):
            result = dim_match_results[call_idx[0]]
            call_idx[0] += 1
            return result

    def mock_savefig(fname, **kwargs):
        captured["savefig_calls"].append(fname)

    original_open = open

    def mock_open(fname, mode="r", *a, **kw):
        if "w" in mode and args.outputroot in str(fname):
            sio = io.StringIO()
            sio.close_original = sio.close
            sio.name = fname

            def mock_close():
                captured["written_files"][fname] = sio.getvalue()
                sio.close_original()

            sio.close = mock_close
            return sio
        return original_open(fname, mode, *a, **kw)

    with (
        patch("rapidtide.workflows.pixelcomp.tide_io.readfromnifti",
              side_effect=mock_readfromnifti),
        patch("rapidtide.workflows.pixelcomp.tide_io.checkspacedimmatch",
              side_effect=mock_checkspacedimmatch),
        patch("rapidtide.workflows.pixelcomp.plt.savefig",
              side_effect=mock_savefig),
        patch("rapidtide.workflows.pixelcomp.plt.show"),
        patch("builtins.open", side_effect=mock_open),
    ):
        pixelcomp(args)

    plt.close("all")
    return captured


# ---- _get_parser tests ----


def parser_basic(debug=False):
    """Test that _get_parser returns a valid parser."""
    parser = _get_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    assert parser.prog == "pixelcomp"

    if debug:
        print("parser_basic passed")


def parser_required_args(debug=False):
    """Test that parser requires all five positional arguments."""
    parser = _get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])

    with pytest.raises(SystemExit):
        parser.parse_args(["in1.nii"])

    with pytest.raises(SystemExit):
        parser.parse_args(["in1.nii", "mask1.nii", "in2.nii", "mask2.nii"])

    if debug:
        print("parser_required_args passed")


def parser_all_positional(debug=False):
    """Test parser with all five positional arguments."""
    parser = _get_parser()
    args = parser.parse_args(["in1.nii", "mask1.nii", "in2.nii", "mask2.nii", "output"])
    assert args.inputfilename1 == "in1.nii"
    assert args.maskfilename1 == "mask1.nii"
    assert args.inputfilename2 == "in2.nii"
    assert args.maskfilename2 == "mask2.nii"
    assert args.outputroot == "output"

    if debug:
        print("parser_all_positional passed")


def parser_defaults(debug=False):
    """Test default values from the parser."""
    parser = _get_parser()
    args = parser.parse_args(["in1", "m1", "in2", "m2", "out"])

    assert args.scatter is False
    assert args.fitonly is False
    assert args.display is True
    assert args.fitorder == 1
    assert args.usex is False
    assert args.histbins == 51

    if debug:
        print("parser_defaults passed")


def parser_scatter(debug=False):
    """Test --scatter flag."""
    parser = _get_parser()
    args = parser.parse_args(["in1", "m1", "in2", "m2", "out", "--scatter"])
    assert args.scatter is True

    if debug:
        print("parser_scatter passed")


def parser_fitonly(debug=False):
    """Test --fitonly flag."""
    parser = _get_parser()
    args = parser.parse_args(["in1", "m1", "in2", "m2", "out", "--fitonly"])
    assert args.fitonly is True

    if debug:
        print("parser_fitonly passed")


def parser_nodisplay(debug=False):
    """Test --nodisplay flag."""
    parser = _get_parser()
    args = parser.parse_args(["in1", "m1", "in2", "m2", "out", "--nodisplay"])
    assert args.display is False

    if debug:
        print("parser_nodisplay passed")


def parser_fitorder(debug=False):
    """Test --fitorder option."""
    parser = _get_parser()
    args = parser.parse_args(["in1", "m1", "in2", "m2", "out", "--fitorder", "3"])
    assert args.fitorder == 3

    if debug:
        print("parser_fitorder passed")


def parser_usex(debug=False):
    """Test --usex flag."""
    parser = _get_parser()
    args = parser.parse_args(["in1", "m1", "in2", "m2", "out", "--usex"])
    assert args.usex is True

    if debug:
        print("parser_usex passed")


def parser_histbins(debug=False):
    """Test --histbins option."""
    parser = _get_parser()
    args = parser.parse_args(["in1", "m1", "in2", "m2", "out", "--histbins", "100"])
    assert args.histbins == 100

    if debug:
        print("parser_histbins passed")


# ---- bland_altman_plot tests ----


def bland_altman_basic(debug=False):
    """Test bland_altman_plot runs with basic inputs."""
    data1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    data2 = np.array([1.1, 2.2, 2.8, 4.1, 4.9])

    fig, ax = plt.subplots()
    bland_altman_plot(data1, data2)
    plt.close(fig)

    if debug:
        print("bland_altman_basic passed")


def bland_altman_usex_false(debug=False):
    """Test bland_altman_plot with usex=False uses mean of data1 and data2."""
    data1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    data2 = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

    with patch("rapidtide.workflows.pixelcomp.plt.scatter") as mock_scatter:
        bland_altman_plot(data1, data2, usex=False)

        scatter_args = mock_scatter.call_args[0]
        mean_arg = scatter_args[0]
        # usex=False: mean = np.mean([data1, data2], axis=0) = element-wise mean
        expected_mean = np.mean([data1, data2], axis=0)
        np.testing.assert_allclose(mean_arg, expected_mean)

    if debug:
        print("bland_altman_usex_false passed")


def bland_altman_usex_true(debug=False):
    """Test bland_altman_plot with usex=True uses mean of data1 (scalar)."""
    data1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    data2 = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

    with patch("rapidtide.workflows.pixelcomp.plt.scatter") as mock_scatter:
        bland_altman_plot(data1, data2, usex=True)

        scatter_args = mock_scatter.call_args[0]
        mean_arg = scatter_args[0]
        # usex=True: mean = np.mean(data1) = scalar
        assert np.isclose(mean_arg, np.mean(data1))

    if debug:
        print("bland_altman_usex_true passed")


def bland_altman_diff(debug=False):
    """Test bland_altman_plot passes correct diff to scatter."""
    data1 = np.array([1.0, 2.0, 3.0])
    data2 = np.array([1.5, 2.0, 3.5])

    with patch("rapidtide.workflows.pixelcomp.plt.scatter") as mock_scatter:
        bland_altman_plot(data1, data2)

        scatter_args = mock_scatter.call_args[0]
        diff_arg = scatter_args[1]
        expected_diff = data2 - data1
        np.testing.assert_allclose(diff_arg, expected_diff)

    if debug:
        print("bland_altman_diff passed")


def bland_altman_axhlines(debug=False):
    """Test bland_altman_plot draws three horizontal lines."""
    data1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    data2 = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

    with patch("rapidtide.workflows.pixelcomp.plt.axhline") as mock_axhline:
        bland_altman_plot(data1, data2)
        assert mock_axhline.call_count == 3

    if debug:
        print("bland_altman_axhlines passed")


def bland_altman_axhline_values(debug=False):
    """Test bland_altman_plot draws lines at md, md+2sd, md-2sd."""
    data1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    data2 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

    diff = data2 - data1
    md = np.mean(diff)
    sd = np.std(diff)

    with patch("rapidtide.workflows.pixelcomp.plt.axhline") as mock_axhline:
        bland_altman_plot(data1, data2)

        called_values = [c[0][0] for c in mock_axhline.call_args_list]
        assert np.isclose(called_values[0], md)
        assert np.isclose(called_values[1], md + 2 * sd)
        assert np.isclose(called_values[2], md - 2 * sd)

    if debug:
        print("bland_altman_axhline_values passed")


def bland_altman_identical_data(debug=False):
    """Test bland_altman_plot with identical data (diff=0)."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    with patch("rapidtide.workflows.pixelcomp.plt.axhline") as mock_axhline:
        bland_altman_plot(data, data)

        called_values = [c[0][0] for c in mock_axhline.call_args_list]
        # all three lines should be at 0
        for val in called_values:
            assert np.isclose(val, 0.0)

    if debug:
        print("bland_altman_identical_data passed")


# ---- pairdata tests ----


def pairdata_basic(debug=False):
    """Test pairdata with basic 3D arrays."""
    input1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64)
    input2 = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.float64)
    mask = np.array([[[1, 0], [0, 1]], [[1, 1], [0, 0]]], dtype=np.float64)

    result = pairdata(input1, input2, mask)

    assert result.ndim == 2
    assert result.shape[1] == 2

    # number of pairs = number of nonzero mask elements
    assert result.shape[0] == np.count_nonzero(mask)

    if debug:
        print(f"pairdata_basic: result shape = {result.shape}")
        print("pairdata_basic passed")


def pairdata_values(debug=False):
    """Test that pairdata extracts correct value pairs."""
    input1 = np.zeros((2, 2, 2), dtype=np.float64)
    input2 = np.zeros((2, 2, 2), dtype=np.float64)
    mask = np.zeros((2, 2, 2), dtype=np.float64)

    # set specific values
    input1[0, 0, 0] = 1.0
    input2[0, 0, 0] = 10.0
    mask[0, 0, 0] = 1.0

    input1[1, 1, 1] = 2.0
    input2[1, 1, 1] = 20.0
    mask[1, 1, 1] = 1.0

    result = pairdata(input1, input2, mask)

    assert result.shape == (2, 2)
    # check that the pairs are correct (order from np.where)
    assert 1.0 in result[:, 0]
    assert 10.0 in result[:, 1]
    assert 2.0 in result[:, 0]
    assert 20.0 in result[:, 1]

    if debug:
        print("pairdata_values passed")


def pairdata_all_masked(debug=False):
    """Test pairdata with all-ones mask (all voxels paired)."""
    shape = (3, 4, 2)
    rng = np.random.RandomState(42)
    input1 = rng.normal(0, 1, shape)
    input2 = rng.normal(0, 1, shape)
    mask = np.ones(shape, dtype=np.float64)

    result = pairdata(input1, input2, mask)

    total_voxels = np.prod(shape)
    assert result.shape == (total_voxels, 2)

    if debug:
        print("pairdata_all_masked passed")


def pairdata_empty_mask(debug=False):
    """Test pairdata with all-zeros mask (no voxels paired)."""
    shape = (3, 4, 2)
    input1 = np.ones(shape, dtype=np.float64)
    input2 = np.ones(shape, dtype=np.float64)
    mask = np.zeros(shape, dtype=np.float64)

    result = pairdata(input1, input2, mask)

    assert result.shape[0] == 0

    if debug:
        print("pairdata_empty_mask passed")


def pairdata_partial_mask(debug=False):
    """Test pairdata with partial mask."""
    shape = (4, 4, 4)
    rng = np.random.RandomState(42)
    input1 = rng.normal(0, 1, shape)
    input2 = rng.normal(0, 1, shape)

    mask = np.zeros(shape, dtype=np.float64)
    mask[0:2, :, :] = 1.0  # half the voxels

    result = pairdata(input1, input2, mask)

    expected_pairs = int(np.count_nonzero(mask))
    assert result.shape == (expected_pairs, 2)

    if debug:
        print("pairdata_partial_mask passed")


def pairdata_first_col_from_input1(debug=False):
    """Test that first column of result comes from input1."""
    shape = (2, 2, 2)
    input1 = np.arange(8, dtype=np.float64).reshape(shape)
    input2 = (np.arange(8, dtype=np.float64) * 10).reshape(shape)
    mask = np.ones(shape, dtype=np.float64)

    result = pairdata(input1, input2, mask)

    # col 0 should contain input1 values, col 1 input2 values
    np.testing.assert_array_equal(sorted(result[:, 0]), sorted(input1.ravel()))
    np.testing.assert_array_equal(sorted(result[:, 1]), sorted(input2.ravel()))

    if debug:
        print("pairdata_first_col_from_input1 passed")


# ---- pixelcomp workflow tests ----


def pixelcomp_scatter(debug=False):
    """Test pixelcomp with scatter plot mode."""
    input1, mask1, input2, mask2, dims, sizes = _make_3d_test_data()
    args = _make_default_args(scatter=True, fitonly=False)

    captured = _run_pixelcomp_with_mocks(args, input1, mask1, input2, mask2, dims, sizes)

    # should save scatter plot and bland-altman plot
    scatter_saves = [f for f in captured["savefig_calls"] if "scatter" in f]
    ba_saves = [f for f in captured["savefig_calls"] if "blandaltman" in f]
    assert len(scatter_saves) == 1
    assert len(ba_saves) == 1

    if debug:
        print("pixelcomp_scatter passed")


def pixelcomp_contour(debug=False):
    """Test pixelcomp with contour plot mode (default)."""
    input1, mask1, input2, mask2, dims, sizes = _make_3d_test_data()
    args = _make_default_args(scatter=False, fitonly=False)

    captured = _run_pixelcomp_with_mocks(args, input1, mask1, input2, mask2, dims, sizes)

    contour_saves = [f for f in captured["savefig_calls"] if "contour" in f]
    assert len(contour_saves) == 1

    if debug:
        print("pixelcomp_contour passed")


def pixelcomp_fitonly(debug=False):
    """Test pixelcomp with fitonly=True skips pairdata file and bland-altman."""
    input1, mask1, input2, mask2, dims, sizes = _make_3d_test_data()
    args = _make_default_args(fitonly=True, scatter=True)

    captured = _run_pixelcomp_with_mocks(args, input1, mask1, input2, mask2, dims, sizes)

    # should NOT save bland-altman plot
    ba_saves = [f for f in captured["savefig_calls"] if "blandaltman" in f]
    assert len(ba_saves) == 0

    # should still save the fit coefficients file
    fit_files = [f for f in captured["written_files"] if "fit" in f]
    assert len(fit_files) >= 1

    if debug:
        print("pixelcomp_fitonly passed")


def pixelcomp_fit_coefficients(debug=False):
    """Test that pixelcomp writes polynomial fit coefficients."""
    input1, mask1, input2, mask2, dims, sizes = _make_3d_test_data()
    args = _make_default_args(fitorder=1, scatter=True)

    captured = _run_pixelcomp_with_mocks(args, input1, mask1, input2, mask2, dims, sizes)

    fit_files = {k: v for k, v in captured["written_files"].items() if "order_1_fit" in k}
    assert len(fit_files) == 1

    if debug:
        print("pixelcomp_fit_coefficients passed")


def pixelcomp_higher_fitorder(debug=False):
    """Test pixelcomp with higher fit order."""
    input1, mask1, input2, mask2, dims, sizes = _make_3d_test_data()
    args = _make_default_args(fitorder=2, scatter=True)

    captured = _run_pixelcomp_with_mocks(args, input1, mask1, input2, mask2, dims, sizes)

    fit_files = {k: v for k, v in captured["written_files"].items() if "order_2_fit" in k}
    assert len(fit_files) == 1

    if debug:
        print("pixelcomp_higher_fitorder passed")


def pixelcomp_pairdata_file(debug=False):
    """Test that pixelcomp writes pairdata file when not fitonly."""
    input1, mask1, input2, mask2, dims, sizes = _make_3d_test_data()
    args = _make_default_args(fitonly=False, scatter=True, outputroot="/tmp/test_pixelcomp_out")

    captured = _run_pixelcomp_with_mocks(args, input1, mask1, input2, mask2, dims, sizes)

    # the pairdata file is written to args.outputroot directly
    assert "/tmp/test_pixelcomp_out" in captured["written_files"]

    if debug:
        print("pixelcomp_pairdata_file passed")


def pixelcomp_dim_mismatch_input1_mask1(debug=False):
    """Test that pixelcomp exits when input1 dims don't match mask1."""
    input1, mask1, input2, mask2, dims, sizes = _make_3d_test_data()
    args = _make_default_args()

    # first checkspacedimmatch call fails (input1 vs mask1)
    with pytest.raises(SystemExit):
        _run_pixelcomp_with_mocks(
            args, input1, mask1, input2, mask2, dims, sizes,
            dim_match_results=[False, True, True],
        )

    if debug:
        print("pixelcomp_dim_mismatch_input1_mask1 passed")


def pixelcomp_dim_mismatch_input2_mask2(debug=False):
    """Test that pixelcomp exits when input2 dims don't match mask2."""
    input1, mask1, input2, mask2, dims, sizes = _make_3d_test_data()
    args = _make_default_args()

    # second checkspacedimmatch call fails (input2 vs mask2)
    with pytest.raises(SystemExit):
        _run_pixelcomp_with_mocks(
            args, input1, mask1, input2, mask2, dims, sizes,
            dim_match_results=[True, False, True],
        )

    if debug:
        print("pixelcomp_dim_mismatch_input2_mask2 passed")


def pixelcomp_dim_mismatch_input1_input2(debug=False):
    """Test that pixelcomp exits when input1 and input2 dims don't match."""
    input1, mask1, input2, mask2, dims, sizes = _make_3d_test_data()
    args = _make_default_args()

    # third checkspacedimmatch call fails (input1 vs input2)
    with pytest.raises(SystemExit):
        _run_pixelcomp_with_mocks(
            args, input1, mask1, input2, mask2, dims, sizes,
            dim_match_results=[True, True, False],
        )

    if debug:
        print("pixelcomp_dim_mismatch_input1_input2 passed")


def pixelcomp_mask_intersection(debug=False):
    """Test that pixelcomp uses intersection of both masks."""
    shape = (4, 5, 3)
    rng = np.random.RandomState(42)
    input1 = rng.normal(10.0, 3.0, shape)
    input2 = 2.0 * input1 + 1.0

    # mask1 has top half, mask2 has bottom half -> small overlap
    mask1 = np.zeros(shape, dtype=np.float64)
    mask1[0:3, :, :] = 1.0
    mask2 = np.zeros(shape, dtype=np.float64)
    mask2[2:4, :, :] = 1.0

    # intersection: only row 2
    expected_pairs = int(np.count_nonzero(mask1 * mask2))

    dims = np.array([3, shape[0], shape[1], shape[2], 1, 1, 1, 1])
    sizes = np.array([3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    args = _make_default_args(scatter=True, fitonly=False)

    captured = _run_pixelcomp_with_mocks(args, input1, mask1, input2, mask2, dims, sizes)

    # check that the pairdata file has the right number of lines
    pairdata_content = captured["written_files"].get(args.outputroot, "")
    lines = [l for l in pairdata_content.strip().split("\n") if l.strip()]
    assert len(lines) == expected_pairs, (
        f"Expected {expected_pairs} pairs, got {len(lines)}"
    )

    if debug:
        print("pixelcomp_mask_intersection passed")


def pixelcomp_output_root(debug=False):
    """Test that output files use the correct outputroot."""
    input1, mask1, input2, mask2, dims, sizes = _make_3d_test_data()
    args = _make_default_args(outputroot="/tmp/myanalysis", scatter=True)

    captured = _run_pixelcomp_with_mocks(args, input1, mask1, input2, mask2, dims, sizes)

    for fname in captured["savefig_calls"]:
        assert fname.startswith("/tmp/myanalysis")

    if debug:
        print("pixelcomp_output_root passed")


def pixelcomp_prints_coefficients(debug=False):
    """Test that pixelcomp prints fit coefficients."""
    input1, mask1, input2, mask2, dims, sizes = _make_3d_test_data()
    args = _make_default_args(scatter=True)

    captured_stdout = io.StringIO()
    with patch("sys.stdout", captured_stdout):
        _run_pixelcomp_with_mocks(args, input1, mask1, input2, mask2, dims, sizes)

    output = captured_stdout.getvalue()
    assert "thecoffs=" in output

    if debug:
        print("pixelcomp_prints_coefficients passed")


# ---- main test function ----


def test_pixelcomp(debug=False):
    # parser tests
    parser_basic(debug=debug)
    parser_required_args(debug=debug)
    parser_all_positional(debug=debug)
    parser_defaults(debug=debug)
    parser_scatter(debug=debug)
    parser_fitonly(debug=debug)
    parser_nodisplay(debug=debug)
    parser_fitorder(debug=debug)
    parser_usex(debug=debug)
    parser_histbins(debug=debug)

    # bland_altman_plot tests
    bland_altman_basic(debug=debug)
    bland_altman_usex_false(debug=debug)
    bland_altman_usex_true(debug=debug)
    bland_altman_diff(debug=debug)
    bland_altman_axhlines(debug=debug)
    bland_altman_axhline_values(debug=debug)
    bland_altman_identical_data(debug=debug)

    # pairdata tests
    pairdata_basic(debug=debug)
    pairdata_values(debug=debug)
    pairdata_all_masked(debug=debug)
    pairdata_empty_mask(debug=debug)
    pairdata_partial_mask(debug=debug)
    pairdata_first_col_from_input1(debug=debug)

    # pixelcomp workflow tests
    pixelcomp_scatter(debug=debug)
    pixelcomp_contour(debug=debug)
    pixelcomp_fitonly(debug=debug)
    pixelcomp_fit_coefficients(debug=debug)
    pixelcomp_higher_fitorder(debug=debug)
    pixelcomp_pairdata_file(debug=debug)
    pixelcomp_dim_mismatch_input1_mask1(debug=debug)
    pixelcomp_dim_mismatch_input2_mask2(debug=debug)
    pixelcomp_dim_mismatch_input1_input2(debug=debug)
    pixelcomp_mask_intersection(debug=debug)
    pixelcomp_output_root(debug=debug)
    pixelcomp_prints_coefficients(debug=debug)


if __name__ == "__main__":
    test_pixelcomp(debug=True)
