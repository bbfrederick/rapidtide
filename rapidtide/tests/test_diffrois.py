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
from io import StringIO
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest

from rapidtide.workflows.diffrois import _get_parser, diffrois

# ---- helpers ----


def _make_args(**overrides):
    """Build a minimal args Namespace for diffrois with sensible defaults."""
    defaults = dict(
        datafile="input.csv",
        outputroot="/tmp/test_diffrois_out",
        keyfile=None,
        maxlines=None,
        debug=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _make_csv_df(numregions=3, numlabels=5, nan_positions=None, seed=42):
    """Create a synthetic DataFrame mimicking ROI CSV data.

    Parameters
    ----------
    numregions : int
        Number of ROI columns.
    numlabels : int
        Number of rows (time points / subjects).
    nan_positions : list of (row, col) tuples, optional
        Positions to insert NaN values (col is 0-indexed into region columns).
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame with a "Region" column and numregions data columns.
    """
    rng = np.random.RandomState(seed)
    region_names = [f"ROI_{i}" for i in range(numregions)]
    labels = [f"label_{z}" for z in range(numlabels)]
    data = rng.randn(numlabels, numregions)

    if nan_positions is not None:
        for row, col in nan_positions:
            data[row, col] = np.nan

    df = pd.DataFrame(data, columns=region_names)
    df.insert(0, "Region", labels)
    return df, region_names


def _capture_savetonifti():
    """Capture savetonifti calls, copying data at call time."""
    captured = []

    def _side_effect(data, hdr, filename, **kwargs):
        captured.append((np.array(data, copy=True), filename))

    return _side_effect, captured


def _run_diffrois(args, df, keyfile_content=None):
    """Run diffrois with full mocking of external dependencies.

    Parameters
    ----------
    args : Namespace
        Arguments to pass to diffrois.
    df : pd.DataFrame
        DataFrame to return from pd.read_csv.
    keyfile_content : str or None
        Content to return when keyfile is opened.  If None, no keyfile mock is set up.

    Returns
    -------
    dict with keys: 'savetonifti'
    """
    stn_effect, stn_captured = _capture_savetonifti()

    mock_nifti_hdr = {
        "dim": np.zeros(8, dtype=int),
        "pixdim": np.zeros(8, dtype=float),
    }
    mock_nifti_img = MagicMock()
    mock_nifti_img.header = mock_nifti_hdr

    patches = [
        patch("rapidtide.workflows.diffrois.pd.read_csv", return_value=df),
        patch("rapidtide.workflows.diffrois.nib.Nifti1Image", return_value=mock_nifti_img),
        patch("rapidtide.workflows.diffrois.tide_io.savetonifti", side_effect=stn_effect),
    ]

    if keyfile_content is not None:
        patches.append(
            patch("builtins.open", mock_open(read_data=keyfile_content))
        )

    # Enter all context managers
    for p in patches:
        p.start()
    try:
        diffrois(args)
    finally:
        for p in patches:
            p.stop()

    return {
        "savetonifti": stn_captured,
    }


# ---- _get_parser tests ----


def test_parser_required_args(debug=False):
    """Parser should accept the two required positional args."""
    with tempfile.NamedTemporaryFile(suffix=".csv") as f:
        parser = _get_parser()
        args = parser.parse_args([f.name, "outroot"])
        assert args.datafile == f.name
        assert args.outputroot == "outroot"
    if debug:
        print("test_parser_required_args passed")


def test_parser_defaults(debug=False):
    """Parser defaults should match expected values."""
    with tempfile.NamedTemporaryFile(suffix=".csv") as f:
        parser = _get_parser()
        args = parser.parse_args([f.name, "outroot"])
        assert args.keyfile is None
        assert args.maxlines is None
        assert args.debug is False
    if debug:
        print("test_parser_defaults passed")


def test_parser_keyfile(debug=False):
    """Parser should accept --keyfile with a valid file."""
    with (
        tempfile.NamedTemporaryFile(suffix=".csv") as data_f,
        tempfile.NamedTemporaryFile(suffix=".txt") as key_f,
    ):
        parser = _get_parser()
        args = parser.parse_args([data_f.name, "outroot", "--keyfile", key_f.name])
        assert args.keyfile == key_f.name
    if debug:
        print("test_parser_keyfile passed")


def test_parser_maxlines(debug=False):
    """Parser should accept --maxlines with an integer."""
    with tempfile.NamedTemporaryFile(suffix=".csv") as f:
        parser = _get_parser()
        args = parser.parse_args([f.name, "outroot", "--maxlines", "10"])
        assert args.maxlines == 10
    if debug:
        print("test_parser_maxlines passed")


def test_parser_debug_flag(debug=False):
    """Parser should accept --debug flag."""
    with tempfile.NamedTemporaryFile(suffix=".csv") as f:
        parser = _get_parser()
        args = parser.parse_args([f.name, "outroot", "--debug"])
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


# ---- diffrois basic tests ----


def test_diffrois_basic_run(debug=False):
    """diffrois should run without errors with basic inputs."""
    df, _ = _make_csv_df(numregions=3, numlabels=5)
    args = _make_args()
    result = _run_diffrois(args, df)
    assert len(result["savetonifti"]) == 5
    if debug:
        print("test_diffrois_basic_run passed")


def test_diffrois_output_files(debug=False):
    """diffrois should write all 5 expected output files."""
    df, _ = _make_csv_df(numregions=3, numlabels=5)
    args = _make_args(outputroot="/tmp/dr")
    result = _run_diffrois(args, df)

    filenames = [entry[1] for entry in result["savetonifti"]]
    expected_suffixes = ["_diffs", "_mask", "_meandiffs", "_stddiffs", "_demeaneddiffs"]
    for suffix in expected_suffixes:
        assert any(suffix in f for f in filenames), f"Missing output file with suffix {suffix}"
    if debug:
        print("test_diffrois_output_files passed")


def test_diffrois_diffs_shape(debug=False):
    """Diffs output should have shape (numregions, numregions, 1, numlabels)."""
    numregions = 3
    numlabels = 5
    df, _ = _make_csv_df(numregions=numregions, numlabels=numlabels)
    args = _make_args()
    result = _run_diffrois(args, df)

    for data, fname in result["savetonifti"]:
        if "_diffs" in fname and "_meandiffs" not in fname and "_stddiffs" not in fname and "_demeaneddiffs" not in fname:
            assert data.shape == (numregions, numregions, 1, numlabels)
            break
    else:
        pytest.fail("_diffs output not found")
    if debug:
        print("test_diffrois_diffs_shape passed")


def test_diffrois_mask_shape(debug=False):
    """Mask output should have shape (numregions, numregions, 1, numlabels)."""
    numregions = 4
    numlabels = 6
    df, _ = _make_csv_df(numregions=numregions, numlabels=numlabels)
    args = _make_args()
    result = _run_diffrois(args, df)

    for data, fname in result["savetonifti"]:
        if fname.endswith("_mask"):
            assert data.shape == (numregions, numregions, 1, numlabels)
            break
    else:
        pytest.fail("_mask output not found")
    if debug:
        print("test_diffrois_mask_shape passed")


def test_diffrois_meandiffs_shape(debug=False):
    """Mean diffs output should have shape (numregions, numregions, 1, 1)."""
    numregions = 3
    numlabels = 5
    df, _ = _make_csv_df(numregions=numregions, numlabels=numlabels)
    args = _make_args()
    result = _run_diffrois(args, df)

    for data, fname in result["savetonifti"]:
        if "_meandiffs" in fname:
            assert data.shape == (numregions, numregions, 1, 1)
            break
    else:
        pytest.fail("_meandiffs output not found")
    if debug:
        print("test_diffrois_meandiffs_shape passed")


def test_diffrois_stddiffs_shape(debug=False):
    """Std diffs output should have shape (numregions, numregions, 1, 1)."""
    numregions = 3
    numlabels = 5
    df, _ = _make_csv_df(numregions=numregions, numlabels=numlabels)
    args = _make_args()
    result = _run_diffrois(args, df)

    for data, fname in result["savetonifti"]:
        if "_stddiffs" in fname:
            assert data.shape == (numregions, numregions, 1, 1)
            break
    else:
        pytest.fail("_stddiffs output not found")
    if debug:
        print("test_diffrois_stddiffs_shape passed")


def test_diffrois_demeaneddiffs_shape(debug=False):
    """Demeaned diffs output should have shape (numregions, numregions, 1, numlabels)."""
    numregions = 3
    numlabels = 5
    df, _ = _make_csv_df(numregions=numregions, numlabels=numlabels)
    args = _make_args()
    result = _run_diffrois(args, df)

    for data, fname in result["savetonifti"]:
        if "_demeaneddiffs" in fname:
            assert data.shape == (numregions, numregions, 1, numlabels)
            break
    else:
        pytest.fail("_demeaneddiffs output not found")
    if debug:
        print("test_diffrois_demeaneddiffs_shape passed")


# ---- diffrois correctness tests ----


def test_diffrois_diagonal_zero(debug=False):
    """Diagonal elements (self-differences) should be zero."""
    numregions = 3
    numlabels = 4
    df, _ = _make_csv_df(numregions=numregions, numlabels=numlabels)
    args = _make_args()
    result = _run_diffrois(args, df)

    for data, fname in result["savetonifti"]:
        if "_diffs" in fname and "_meandiffs" not in fname and "_stddiffs" not in fname and "_demeaneddiffs" not in fname:
            for i in range(numregions):
                np.testing.assert_array_equal(data[i, i, 0, :], 0.0)
            break
    if debug:
        print("test_diffrois_diagonal_zero passed")


def test_diffrois_symmetric_values(debug=False):
    """Matrix [i,j] and [j,i] should have the same value (symmetric, not antisymmetric)."""
    numregions = 3
    numlabels = 4
    df, _ = _make_csv_df(numregions=numregions, numlabels=numlabels)
    args = _make_args()
    result = _run_diffrois(args, df)

    for data, fname in result["savetonifti"]:
        if "_diffs" in fname and "_meandiffs" not in fname and "_stddiffs" not in fname and "_demeaneddiffs" not in fname:
            for i in range(numregions):
                for j in range(i + 1, numregions):
                    np.testing.assert_array_equal(data[i, j, 0, :], data[j, i, 0, :])
            break
    if debug:
        print("test_diffrois_symmetric_values passed")


def test_diffrois_difference_values(debug=False):
    """Verify computed differences match manual calculation."""
    numregions = 2
    numlabels = 3
    # Create deterministic data
    data_vals = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    region_names = ["ROI_0", "ROI_1"]
    labels = ["label_0", "label_1", "label_2"]
    df = pd.DataFrame(data_vals, columns=region_names)
    df.insert(0, "Region", labels)

    args = _make_args()
    result = _run_diffrois(args, df)

    for data, fname in result["savetonifti"]:
        if "_diffs" in fname and "_meandiffs" not in fname and "_stddiffs" not in fname and "_demeaneddiffs" not in fname:
            # ROI_0[z] - ROI_1[z] for each z: [1-4, 2-5, 3-6] = [-3, -3, -3]
            expected_01 = np.array([-3.0, -3.0, -3.0])
            np.testing.assert_array_almost_equal(data[0, 1, 0, :], expected_01)
            # Symmetric: [1,0] should be the same as [0,1]
            np.testing.assert_array_almost_equal(data[1, 0, 0, :], expected_01)
            # Diagonal should be zero
            np.testing.assert_array_equal(data[0, 0, 0, :], 0.0)
            np.testing.assert_array_equal(data[1, 1, 0, :], 0.0)
            break
    if debug:
        print("test_diffrois_difference_values passed")


def test_diffrois_mean_values(debug=False):
    """Verify mean differences are computed correctly."""
    numregions = 2
    numlabels = 4
    # Create data where ROI_0 - ROI_1 = [2, 4, 6, 8] -> mean = 5.0
    data_vals = np.array([[3.0, 1.0], [5.0, 1.0], [7.0, 1.0], [9.0, 1.0]])
    region_names = ["ROI_0", "ROI_1"]
    labels = [f"label_{z}" for z in range(numlabels)]
    df = pd.DataFrame(data_vals, columns=region_names)
    df.insert(0, "Region", labels)

    args = _make_args()
    result = _run_diffrois(args, df)

    for data, fname in result["savetonifti"]:
        if "_meandiffs" in fname:
            # Mean of [2, 4, 6, 8] = 5.0
            assert abs(data[0, 1, 0, 0] - 5.0) < 1e-10
            # Diagonal mean should be 0.0
            assert abs(data[0, 0, 0, 0]) < 1e-10
            break
    if debug:
        print("test_diffrois_mean_values passed")


def test_diffrois_std_values(debug=False):
    """Verify std differences are computed correctly."""
    numregions = 2
    numlabels = 4
    # ROI_0 - ROI_1 = [2, 4, 6, 8] -> std = np.std([2,4,6,8])
    data_vals = np.array([[3.0, 1.0], [5.0, 1.0], [7.0, 1.0], [9.0, 1.0]])
    region_names = ["ROI_0", "ROI_1"]
    labels = [f"label_{z}" for z in range(numlabels)]
    df = pd.DataFrame(data_vals, columns=region_names)
    df.insert(0, "Region", labels)

    args = _make_args()
    result = _run_diffrois(args, df)

    expected_std = np.std([2.0, 4.0, 6.0, 8.0])
    for data, fname in result["savetonifti"]:
        if "_stddiffs" in fname:
            assert abs(data[0, 1, 0, 0] - expected_std) < 1e-10
            break
    if debug:
        print("test_diffrois_std_values passed")


def test_diffrois_demeaned_values(debug=False):
    """Verify demeaned differences are computed correctly."""
    numregions = 2
    numlabels = 4
    # ROI_0 - ROI_1 = [2, 4, 6, 8], mean = 5.0
    # demeaned = [-3, -1, 1, 3]
    data_vals = np.array([[3.0, 1.0], [5.0, 1.0], [7.0, 1.0], [9.0, 1.0]])
    region_names = ["ROI_0", "ROI_1"]
    labels = [f"label_{z}" for z in range(numlabels)]
    df = pd.DataFrame(data_vals, columns=region_names)
    df.insert(0, "Region", labels)

    args = _make_args()
    result = _run_diffrois(args, df)

    expected_demeaned = np.array([-3.0, -1.0, 1.0, 3.0])
    for data, fname in result["savetonifti"]:
        if "_demeaneddiffs" in fname:
            np.testing.assert_array_almost_equal(data[0, 1, 0, :], expected_demeaned)
            break
    if debug:
        print("test_diffrois_demeaned_values passed")


# ---- NaN handling tests ----


def test_diffrois_nan_masking(debug=False):
    """NaN values should be masked out (mask=0 for affected pairs)."""
    numregions = 2
    numlabels = 3
    # Put NaN in ROI_0 at row 1
    df, _ = _make_csv_df(numregions=numregions, numlabels=numlabels, nan_positions=[(1, 0)])
    args = _make_args()
    result = _run_diffrois(args, df)

    for data, fname in result["savetonifti"]:
        if fname.endswith("_mask"):
            # Row 1, pair (0,1) and (1,0) should be masked
            assert data[0, 1, 0, 1] == 0
            assert data[1, 0, 0, 1] == 0
            # Row 0 should be valid for all pairs
            assert data[0, 1, 0, 0] == 1
            break
    if debug:
        print("test_diffrois_nan_masking passed")


def test_diffrois_nan_diffs_zero(debug=False):
    """Diffs at NaN positions should remain zero (default array value)."""
    numregions = 2
    numlabels = 3
    df, _ = _make_csv_df(numregions=numregions, numlabels=numlabels, nan_positions=[(1, 0)])
    args = _make_args()
    result = _run_diffrois(args, df)

    for data, fname in result["savetonifti"]:
        if "_diffs" in fname and "_meandiffs" not in fname and "_stddiffs" not in fname and "_demeaneddiffs" not in fname:
            # Row 1, pair (0,1) should be 0 (NaN in region 0)
            assert data[0, 1, 0, 1] == 0.0
            break
    if debug:
        print("test_diffrois_nan_diffs_zero passed")


def test_diffrois_nan_diagonal_mask(debug=False):
    """Diagonal mask should be 0 when the diagonal region has a NaN."""
    numregions = 2
    numlabels = 3
    # NaN at ROI_0, row 1 — diagonal [0,0] at z=1 should have mask=0
    df, _ = _make_csv_df(numregions=numregions, numlabels=numlabels, nan_positions=[(1, 0)])
    args = _make_args()
    result = _run_diffrois(args, df)

    for data, fname in result["savetonifti"]:
        if fname.endswith("_mask"):
            # Diagonal [0,0] at z=1: both ival and jval come from ROI_0 which is NaN
            assert data[0, 0, 0, 1] == 0
            break
    if debug:
        print("test_diffrois_nan_diagonal_mask passed")


def test_diffrois_all_nan_column(debug=False):
    """Column with all NaN should produce all-zero mask for affected pairs."""
    numregions = 2
    numlabels = 3
    # All rows of ROI_0 are NaN
    nan_positions = [(i, 0) for i in range(numlabels)]
    df, _ = _make_csv_df(numregions=numregions, numlabels=numlabels, nan_positions=nan_positions)
    args = _make_args()
    result = _run_diffrois(args, df)

    for data, fname in result["savetonifti"]:
        if fname.endswith("_mask"):
            # All pairs involving ROI_0 should be masked
            np.testing.assert_array_equal(data[0, :, 0, :], 0)
            np.testing.assert_array_equal(data[:, 0, 0, :], 0)
            # ROI_1 diagonal should still be valid
            np.testing.assert_array_equal(data[1, 1, 0, :], 1)
            break
    if debug:
        print("test_diffrois_all_nan_column passed")


def test_diffrois_demeaned_nan_zeroed(debug=False):
    """Demeaned diffs at NaN positions should be zeroed out."""
    numregions = 2
    numlabels = 3
    df, _ = _make_csv_df(numregions=numregions, numlabels=numlabels, nan_positions=[(1, 0)])
    args = _make_args()
    result = _run_diffrois(args, df)

    for data, fname in result["savetonifti"]:
        if "_demeaneddiffs" in fname:
            # z=1 pair (0,1) was NaN, so demeaned should be 0
            assert data[0, 1, 0, 1] == 0.0
            break
    if debug:
        print("test_diffrois_demeaned_nan_zeroed passed")


# ---- maxlines tests ----


def test_diffrois_maxlines(debug=False):
    """--maxlines should limit processing to the first N rows."""
    numregions = 2
    numlabels = 10
    maxlines = 4
    df, _ = _make_csv_df(numregions=numregions, numlabels=numlabels)
    args = _make_args(maxlines=maxlines)
    result = _run_diffrois(args, df)

    for data, fname in result["savetonifti"]:
        if "_diffs" in fname and "_meandiffs" not in fname and "_stddiffs" not in fname and "_demeaneddiffs" not in fname:
            # 4th dimension should be maxlines, not numlabels
            assert data.shape == (numregions, numregions, 1, maxlines)
            break
    if debug:
        print("test_diffrois_maxlines passed")


def test_diffrois_maxlines_larger_than_data(debug=False):
    """--maxlines larger than data should use all available rows."""
    numregions = 2
    numlabels = 5
    maxlines = 100
    df, _ = _make_csv_df(numregions=numregions, numlabels=numlabels)
    args = _make_args(maxlines=maxlines)
    result = _run_diffrois(args, df)

    for data, fname in result["savetonifti"]:
        if "_diffs" in fname and "_meandiffs" not in fname and "_stddiffs" not in fname and "_demeaneddiffs" not in fname:
            assert data.shape == (numregions, numregions, 1, numlabels)
            break
    if debug:
        print("test_diffrois_maxlines_larger_than_data passed")


# ---- keyfile tests ----


def test_diffrois_keyfile_reorder(debug=False):
    """Keyfile should reorder regions in the output."""
    numregions = 3
    numlabels = 4
    # Create data with known values
    data_vals = np.array([
        [10.0, 20.0, 30.0],
        [11.0, 21.0, 31.0],
        [12.0, 22.0, 32.0],
        [13.0, 23.0, 33.0],
    ])
    region_names = ["ROI_0", "ROI_1", "ROI_2"]
    labels = [f"label_{z}" for z in range(numlabels)]
    df = pd.DataFrame(data_vals, columns=region_names)
    df.insert(0, "Region", labels)

    # Keyfile reverses the order
    keyfile_content = "ROI_2\nROI_1\nROI_0\n"
    args = _make_args(keyfile="keys.txt")
    result = _run_diffrois(args, df, keyfile_content=keyfile_content)

    for data, fname in result["savetonifti"]:
        if "_diffs" in fname and "_meandiffs" not in fname and "_stddiffs" not in fname and "_demeaneddiffs" not in fname:
            # With reordered keys: colkeys = [ROI_2, ROI_1, ROI_0]
            # [0,1] = ROI_2 - ROI_1 = 30-20=10 at z=0
            assert abs(data[0, 1, 0, 0] - 10.0) < 1e-10
            # [0,2] = ROI_2 - ROI_0 = 30-10=20 at z=0
            assert abs(data[0, 2, 0, 0] - 20.0) < 1e-10
            # [1,2] = ROI_1 - ROI_0 = 20-10=10 at z=0
            assert abs(data[1, 2, 0, 0] - 10.0) < 1e-10
            break
    if debug:
        print("test_diffrois_keyfile_reorder passed")


def test_diffrois_keyfile_subset(debug=False):
    """Keyfile with subset of regions should only produce that subset."""
    numregions = 3
    numlabels = 3
    data_vals = np.array([
        [10.0, 20.0, 30.0],
        [11.0, 21.0, 31.0],
        [12.0, 22.0, 32.0],
    ])
    region_names = ["ROI_0", "ROI_1", "ROI_2"]
    labels = [f"label_{z}" for z in range(numlabels)]
    df = pd.DataFrame(data_vals, columns=region_names)
    df.insert(0, "Region", labels)

    # Keyfile with only 2 of 3 regions
    keyfile_content = "ROI_0\nROI_2\n"
    args = _make_args(keyfile="keys.txt")
    result = _run_diffrois(args, df, keyfile_content=keyfile_content)

    for data, fname in result["savetonifti"]:
        if "_diffs" in fname and "_meandiffs" not in fname and "_stddiffs" not in fname and "_demeaneddiffs" not in fname:
            # Output should be 2x2 (only 2 regions selected)
            assert data.shape == (2, 2, 1, numlabels)
            # [0,1] = ROI_0 - ROI_2 = 10-30=-20 at z=0
            assert abs(data[0, 1, 0, 0] - (-20.0)) < 1e-10
            break
    if debug:
        print("test_diffrois_keyfile_subset passed")


def test_diffrois_no_keyfile(debug=False):
    """Without keyfile, regions should be in CSV column order."""
    numregions = 3
    numlabels = 3
    data_vals = np.array([
        [10.0, 20.0, 30.0],
        [11.0, 21.0, 31.0],
        [12.0, 22.0, 32.0],
    ])
    region_names = ["ROI_0", "ROI_1", "ROI_2"]
    labels = [f"label_{z}" for z in range(numlabels)]
    df = pd.DataFrame(data_vals, columns=region_names)
    df.insert(0, "Region", labels)

    args = _make_args()
    result = _run_diffrois(args, df)

    for data, fname in result["savetonifti"]:
        if "_diffs" in fname and "_meandiffs" not in fname and "_stddiffs" not in fname and "_demeaneddiffs" not in fname:
            # [0,1] = ROI_0 - ROI_1 = 10-20=-10 at z=0
            assert abs(data[0, 1, 0, 0] - (-10.0)) < 1e-10
            break
    if debug:
        print("test_diffrois_no_keyfile passed")


# ---- debug mode test ----


def test_diffrois_debug_mode(debug=False):
    """diffrois should run successfully with debug=True."""
    df, _ = _make_csv_df(numregions=2, numlabels=3)
    args = _make_args(debug=True)
    result = _run_diffrois(args, df)
    assert len(result["savetonifti"]) == 5
    if debug:
        print("test_diffrois_debug_mode passed")


# ---- edge case tests ----


def test_diffrois_single_region(debug=False):
    """diffrois should handle a single region (1x1 output)."""
    data_vals = np.array([[5.0], [6.0], [7.0]])
    region_names = ["ROI_0"]
    labels = ["label_0", "label_1", "label_2"]
    df = pd.DataFrame(data_vals, columns=region_names)
    df.insert(0, "Region", labels)

    args = _make_args()
    result = _run_diffrois(args, df)

    for data, fname in result["savetonifti"]:
        if "_diffs" in fname and "_meandiffs" not in fname and "_stddiffs" not in fname and "_demeaneddiffs" not in fname:
            assert data.shape == (1, 1, 1, 3)
            # Self-difference should be 0
            np.testing.assert_array_equal(data[0, 0, 0, :], 0.0)
            break
    if debug:
        print("test_diffrois_single_region passed")


def test_diffrois_single_label(debug=False):
    """diffrois should handle a single time point."""
    data_vals = np.array([[1.0, 2.0, 3.0]])
    region_names = ["ROI_0", "ROI_1", "ROI_2"]
    labels = ["label_0"]
    df = pd.DataFrame(data_vals, columns=region_names)
    df.insert(0, "Region", labels)

    args = _make_args()
    result = _run_diffrois(args, df)

    for data, fname in result["savetonifti"]:
        if "_diffs" in fname and "_meandiffs" not in fname and "_stddiffs" not in fname and "_demeaneddiffs" not in fname:
            assert data.shape == (3, 3, 1, 1)
            # [0,1] = ROI_0 - ROI_1 = 1-2 = -1
            assert abs(data[0, 1, 0, 0] - (-1.0)) < 1e-10
            # [0,2] = ROI_0 - ROI_2 = 1-3 = -2
            assert abs(data[0, 2, 0, 0] - (-2.0)) < 1e-10
            break
    if debug:
        print("test_diffrois_single_label passed")


def test_diffrois_mask_all_ones_no_nan(debug=False):
    """Mask should be all ones when there are no NaN values."""
    numregions = 3
    numlabels = 4
    df, _ = _make_csv_df(numregions=numregions, numlabels=numlabels)
    args = _make_args()
    result = _run_diffrois(args, df)

    for data, fname in result["savetonifti"]:
        if fname.endswith("_mask"):
            np.testing.assert_array_equal(data, 1)
            break
    if debug:
        print("test_diffrois_mask_all_ones_no_nan passed")


def test_diffrois_many_regions(debug=False):
    """diffrois should handle larger numbers of regions."""
    numregions = 10
    numlabels = 8
    df, _ = _make_csv_df(numregions=numregions, numlabels=numlabels)
    args = _make_args()
    result = _run_diffrois(args, df)

    for data, fname in result["savetonifti"]:
        if "_diffs" in fname and "_meandiffs" not in fname and "_stddiffs" not in fname and "_demeaneddiffs" not in fname:
            assert data.shape == (numregions, numregions, 1, numlabels)
            break
    if debug:
        print("test_diffrois_many_regions passed")


# ---- integration test ----


def test_diffrois_integration(debug=False):
    """Full integration test verifying all outputs are consistent."""
    numregions = 3
    numlabels = 6
    data_vals = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0],
        [16.0, 17.0, 18.0],
    ])
    region_names = ["ROI_0", "ROI_1", "ROI_2"]
    labels = [f"label_{z}" for z in range(numlabels)]
    df = pd.DataFrame(data_vals, columns=region_names)
    df.insert(0, "Region", labels)

    args = _make_args()
    result = _run_diffrois(args, df)

    # Collect outputs by name
    outputs = {}
    for data, fname in result["savetonifti"]:
        for suffix in ["_diffs", "_mask", "_meandiffs", "_stddiffs", "_demeaneddiffs"]:
            if fname.endswith(suffix):
                outputs[suffix] = data
                break

    assert len(outputs) == 5

    # All diffs between adjacent regions should be -1 (constant difference)
    diffs = outputs["_diffs"]
    for z in range(numlabels):
        # ROI_0 - ROI_1 = -1 for all z
        assert abs(diffs[0, 1, 0, z] - (-1.0)) < 1e-10
        # ROI_0 - ROI_2 = -2 for all z
        assert abs(diffs[0, 2, 0, z] - (-2.0)) < 1e-10
        # ROI_1 - ROI_2 = -1 for all z
        assert abs(diffs[1, 2, 0, z] - (-1.0)) < 1e-10

    # Mean of constant differences should be the same constant
    meandiffs = outputs["_meandiffs"]
    assert abs(meandiffs[0, 1, 0, 0] - (-1.0)) < 1e-10
    assert abs(meandiffs[0, 2, 0, 0] - (-2.0)) < 1e-10

    # Std of constant differences should be 0
    stddiffs = outputs["_stddiffs"]
    assert abs(stddiffs[0, 1, 0, 0]) < 1e-10
    assert abs(stddiffs[0, 2, 0, 0]) < 1e-10

    # Demeaned values of constant differences should be 0
    demeaneddiffs = outputs["_demeaneddiffs"]
    np.testing.assert_array_almost_equal(demeaneddiffs[0, 1, 0, :], 0.0)
    np.testing.assert_array_almost_equal(demeaneddiffs[0, 2, 0, :], 0.0)

    # Mask should be all ones (no NaN)
    mask = outputs["_mask"]
    np.testing.assert_array_equal(mask, 1)

    if debug:
        print("test_diffrois_integration passed")


# ---- main test function ----


def test_diffrois(debug=False):
    """Run all diffrois tests."""
    # parser tests
    test_parser_required_args(debug=debug)
    test_parser_defaults(debug=debug)
    test_parser_keyfile(debug=debug)
    test_parser_maxlines(debug=debug)
    test_parser_debug_flag(debug=debug)
    test_parser_missing_required(debug=debug)

    # basic tests
    test_diffrois_basic_run(debug=debug)
    test_diffrois_output_files(debug=debug)
    test_diffrois_diffs_shape(debug=debug)
    test_diffrois_mask_shape(debug=debug)
    test_diffrois_meandiffs_shape(debug=debug)
    test_diffrois_stddiffs_shape(debug=debug)
    test_diffrois_demeaneddiffs_shape(debug=debug)

    # correctness tests
    test_diffrois_diagonal_zero(debug=debug)
    test_diffrois_symmetric_values(debug=debug)
    test_diffrois_difference_values(debug=debug)
    test_diffrois_mean_values(debug=debug)
    test_diffrois_std_values(debug=debug)
    test_diffrois_demeaned_values(debug=debug)

    # NaN handling
    test_diffrois_nan_masking(debug=debug)
    test_diffrois_nan_diffs_zero(debug=debug)
    test_diffrois_nan_diagonal_mask(debug=debug)
    test_diffrois_all_nan_column(debug=debug)
    test_diffrois_demeaned_nan_zeroed(debug=debug)

    # maxlines
    test_diffrois_maxlines(debug=debug)
    test_diffrois_maxlines_larger_than_data(debug=debug)

    # keyfile
    test_diffrois_keyfile_reorder(debug=debug)
    test_diffrois_keyfile_subset(debug=debug)
    test_diffrois_no_keyfile(debug=debug)

    # debug mode
    test_diffrois_debug_mode(debug=debug)

    # edge cases
    test_diffrois_single_region(debug=debug)
    test_diffrois_single_label(debug=debug)
    test_diffrois_mask_all_ones_no_nan(debug=debug)
    test_diffrois_many_regions(debug=debug)

    # integration
    test_diffrois_integration(debug=debug)


if __name__ == "__main__":
    test_diffrois(debug=True)
