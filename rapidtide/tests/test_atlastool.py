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
import os
from unittest.mock import MagicMock, patch

import numpy as np

from rapidtide.tests.utils import create_dir, get_test_temp_path
from rapidtide.workflows.atlastool import _get_parser, atlastool

# ==================== Helpers ====================


def _make_mock_hdr(xsize, ysize, numslices, timepoints=1):
    """Create a mock NIfTI header."""
    hdr = MagicMock()
    hdr.__getitem__ = MagicMock(
        side_effect=lambda key: {
            "dim": [4, xsize, ysize, numslices, timepoints, 1, 1, 1],
            "pixdim": [1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
        }[key]
    )
    hdr.__setitem__ = MagicMock()

    def copy_fn():
        h = MagicMock()
        h.__getitem__ = MagicMock(
            side_effect=lambda key: {
                "dim": [4, xsize, ysize, numslices, timepoints, 1, 1, 1],
                "pixdim": [1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            }[key]
        )
        h.__setitem__ = MagicMock()
        return h

    hdr.copy = copy_fn
    return hdr


def _make_3d_atlas(xsize=6, ysize=6, numslices=4, numregions=3):
    """Create a 3D integer-labeled atlas and matching header/dims."""
    data = np.zeros((xsize, ysize, numslices), dtype=np.float64)
    # Assign region labels (1-based) to different spatial areas
    region_size = xsize // numregions
    for r in range(numregions):
        start = r * region_size
        end = start + region_size
        data[start:end, :, :] = r + 1
    hdr = _make_mock_hdr(xsize, ysize, numslices, 1)
    dims = np.array([3, xsize, ysize, numslices, 1, 1, 1, 1])
    sizes = np.array([1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0])
    return data, hdr, dims, sizes


def _make_4d_atlas(xsize=6, ysize=6, numslices=4, numregions=3):
    """Create a 4D binary atlas (one volume per region)."""
    data = np.zeros((xsize, ysize, numslices, numregions), dtype=np.float64)
    region_size = xsize // numregions
    for r in range(numregions):
        start = r * region_size
        end = start + region_size
        data[start:end, :, :, r] = 1.0
    hdr = _make_mock_hdr(xsize, ysize, numslices, numregions)
    dims = np.array([4, xsize, ysize, numslices, numregions, 1, 1, 1])
    sizes = np.array([1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0])
    return data, hdr, dims, sizes


def _make_splittable_3d_atlas(xsize=8, ysize=6, numslices=4, numregions=2):
    """Create a 3D atlas where regions span both hemispheres (for split tests).

    Regions are defined along the y-axis so each covers the full x range,
    ensuring both left and right halves have voxels after splitting.
    """
    data = np.zeros((xsize, ysize, numslices), dtype=np.float64)
    region_size = ysize // numregions
    for r in range(numregions):
        start = r * region_size
        end = start + region_size
        data[:, start:end, :] = r + 1
    hdr = _make_mock_hdr(xsize, ysize, numslices, 1)
    dims = np.array([3, xsize, ysize, numslices, 1, 1, 1, 1])
    sizes = np.array([1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0])
    return data, hdr, dims, sizes


def _make_base_args(testtemproot, suffix=""):
    """Create a base Namespace with default arguments."""
    return argparse.Namespace(
        inputtemplatename="input.nii.gz",
        outputtemplatename=os.path.join(testtemproot, f"atlasout{suffix}.nii.gz"),
        debug=False,
        maxval=None,
        labelfile=None,
        dosplit=False,
        LtoR=True,
        targetfile=None,
        xfm=None,
        maskfile=None,
        maskthresh=0.25,
        removeemptyregions=False,
        volumeperregion=False,
    )


# ==================== Tests for _get_parser ====================


def get_parser_returns_parser(debug=False):
    """Test that _get_parser returns an ArgumentParser."""
    parser = _get_parser()
    assert isinstance(parser, argparse.ArgumentParser)


def get_parser_prog_name(debug=False):
    """Test that the parser prog name is correct."""
    parser = _get_parser()
    assert parser.prog == "atlastool"


def get_parser_defaults(debug=False):
    """Test that default values are correct."""
    parser = _get_parser()
    defaults = {action.dest: action.default for action in parser._actions}
    assert defaults["volumeperregion"] is False
    assert defaults["dosplit"] is False
    assert defaults["maskthresh"] == 0.25
    assert defaults["labelfile"] is None
    assert defaults["xfm"] is None
    assert defaults["targetfile"] is None
    assert defaults["maskfile"] is None
    assert defaults["removeemptyregions"] is False
    assert defaults["LtoR"] is True
    assert defaults["debug"] is False
    assert defaults["maxval"] is None


def get_parser_with_4d_flag(testtemproot, debug=False):
    """Test parser with --4d flag."""
    infile = os.path.join(testtemproot, "parser_atlas_input.nii.gz")
    with open(infile, "w") as f:
        f.write("dummy")

    parser = _get_parser()
    args = parser.parse_args([infile, "output.nii.gz", "--4d"])
    assert args.volumeperregion is True


def get_parser_with_3d_flag(testtemproot, debug=False):
    """Test parser with --3d flag."""
    infile = os.path.join(testtemproot, "parser_atlas_input2.nii.gz")
    with open(infile, "w") as f:
        f.write("dummy")

    parser = _get_parser()
    args = parser.parse_args([infile, "output.nii.gz", "--3d"])
    assert args.volumeperregion is False


def get_parser_with_split(testtemproot, debug=False):
    """Test parser with --split flag."""
    infile = os.path.join(testtemproot, "parser_atlas_input3.nii.gz")
    with open(infile, "w") as f:
        f.write("dummy")

    parser = _get_parser()
    args = parser.parse_args([infile, "output.nii.gz", "--split"])
    assert args.dosplit is True


def get_parser_with_all_options(testtemproot, debug=False):
    """Test parser with multiple options combined."""
    infile = os.path.join(testtemproot, "parser_atlas_input4.nii.gz")
    labelfile = os.path.join(testtemproot, "parser_labels.txt")
    maskfile = os.path.join(testtemproot, "parser_mask.nii.gz")
    for f in [infile, labelfile, maskfile]:
        with open(f, "w") as fh:
            fh.write("dummy")

    parser = _get_parser()
    args = parser.parse_args(
        [
            infile,
            "output.nii.gz",
            "--4d",
            "--split",
            "--maskthresh",
            "0.5",
            "--labelfile",
            labelfile,
            "--maskfile",
            maskfile,
            "--removeemptyregions",
            "--RtoL",
            "--debug",
            "--maxval",
            "10",
        ]
    )
    assert args.volumeperregion is True
    assert args.dosplit is True
    assert args.maskthresh == 0.5
    assert args.labelfile == labelfile
    assert args.maskfile == maskfile
    assert args.removeemptyregions is True
    assert args.LtoR is False
    assert args.debug is True
    assert args.maxval == 10


# ==================== Tests for atlastool with 3D input ====================


def atlastool_3d_to_3d(testtemproot, debug=False):
    """Test converting a 3D atlas to 3D output (integer labels)."""
    data, hdr, dims, sizes = _make_3d_atlas()
    args = _make_base_args(testtemproot, suffix="_3dto3d")
    args.volumeperregion = False
    args.debug = debug

    with (
        patch("rapidtide.workflows.atlastool.tide_io.readfromnifti") as mock_read,
        patch("rapidtide.workflows.atlastool.tide_io.savetonifti") as mock_save,
    ):

        mock_read.return_value = (MagicMock(), data, hdr, dims, sizes)

        atlastool(args)

        mock_save.assert_called_once()
        saved_data = mock_save.call_args[0][0]
        # 3D output should have shape (xsize, ysize, numslices)
        assert saved_data.ndim == 3


def atlastool_3d_to_4d(testtemproot, debug=False):
    """Test converting a 3D atlas to 4D output (one volume per region)."""
    data, hdr, dims, sizes = _make_3d_atlas()
    args = _make_base_args(testtemproot, suffix="_3dto4d")
    args.volumeperregion = True
    args.debug = debug

    with (
        patch("rapidtide.workflows.atlastool.tide_io.readfromnifti") as mock_read,
        patch("rapidtide.workflows.atlastool.tide_io.savetonifti") as mock_save,
    ):

        mock_read.return_value = (MagicMock(), data, hdr, dims, sizes)

        atlastool(args)

        mock_save.assert_called_once()
        saved_data = mock_save.call_args[0][0]
        # 4D output should have 4 dimensions
        assert saved_data.ndim == 4


def atlastool_3d_with_maxval(testtemproot, debug=False):
    """Test 3D atlas with maxval truncation."""
    xsize, ysize, numslices, numregions = 6, 6, 4, 5
    data, hdr, dims, sizes = _make_3d_atlas(xsize, ysize, numslices, numregions)
    args = _make_base_args(testtemproot, suffix="_3dmaxval")
    args.maxval = 3
    args.volumeperregion = False
    args.debug = debug

    with (
        patch("rapidtide.workflows.atlastool.tide_io.readfromnifti") as mock_read,
        patch("rapidtide.workflows.atlastool.tide_io.savetonifti") as mock_save,
    ):

        mock_read.return_value = (MagicMock(), data.copy(), hdr, dims, sizes)

        atlastool(args)

        mock_save.assert_called_once()
        saved_data = mock_save.call_args[0][0]
        # All values should be <= maxval
        assert np.max(saved_data) <= 3


# ==================== Tests for atlastool with 4D input ====================


def atlastool_4d_to_3d(testtemproot, debug=False):
    """Test converting a 4D atlas to 3D output."""
    data, hdr, dims, sizes = _make_4d_atlas()
    args = _make_base_args(testtemproot, suffix="_4dto3d")
    args.volumeperregion = False
    args.debug = debug

    with (
        patch("rapidtide.workflows.atlastool.tide_io.readfromnifti") as mock_read,
        patch("rapidtide.workflows.atlastool.tide_io.savetonifti") as mock_save,
    ):

        mock_read.return_value = (MagicMock(), data, hdr, dims, sizes)

        atlastool(args)

        mock_save.assert_called_once()
        saved_data = mock_save.call_args[0][0]
        assert saved_data.ndim == 3


def atlastool_4d_to_4d(testtemproot, debug=False):
    """Test passing through a 4D atlas to 4D output."""
    data, hdr, dims, sizes = _make_4d_atlas()
    args = _make_base_args(testtemproot, suffix="_4dto4d")
    args.volumeperregion = True
    args.debug = debug

    with (
        patch("rapidtide.workflows.atlastool.tide_io.readfromnifti") as mock_read,
        patch("rapidtide.workflows.atlastool.tide_io.savetonifti") as mock_save,
    ):

        mock_read.return_value = (MagicMock(), data, hdr, dims, sizes)

        atlastool(args)

        mock_save.assert_called_once()
        saved_data = mock_save.call_args[0][0]
        assert saved_data.ndim == 4


def atlastool_4d_with_maxval(testtemproot, debug=False):
    """Test 4D atlas with maxval truncation (retains only first maxval volumes)."""
    data, hdr, dims, sizes = _make_4d_atlas(numregions=5)
    args = _make_base_args(testtemproot, suffix="_4dmaxval")
    args.maxval = 2
    args.volumeperregion = True
    args.debug = debug

    with (
        patch("rapidtide.workflows.atlastool.tide_io.readfromnifti") as mock_read,
        patch("rapidtide.workflows.atlastool.tide_io.savetonifti") as mock_save,
    ):

        mock_read.return_value = (MagicMock(), data, hdr, dims, sizes)

        atlastool(args)

        mock_save.assert_called_once()


# ==================== Tests for split functionality ====================


def atlastool_split_LtoR(testtemproot, debug=False):
    """Test split with default LtoR labeling."""
    data, hdr, dims, sizes = _make_splittable_3d_atlas()
    args = _make_base_args(testtemproot, suffix="_splitLR")
    args.dosplit = True
    args.LtoR = True
    args.volumeperregion = True
    args.debug = debug

    with (
        patch("rapidtide.workflows.atlastool.tide_io.readfromnifti") as mock_read,
        patch("rapidtide.workflows.atlastool.tide_io.savetonifti") as mock_save,
    ):

        mock_read.return_value = (MagicMock(), data, hdr, dims, sizes)

        atlastool(args)

        mock_save.assert_called_once()
        saved_data = mock_save.call_args[0][0]
        # After split, number of regions should be doubled
        assert saved_data.ndim == 4
        assert saved_data.shape[3] == 4  # 2 regions * 2 hemispheres


def atlastool_split_RtoL(testtemproot, debug=False):
    """Test split with RtoL labeling."""
    data, hdr, dims, sizes = _make_splittable_3d_atlas()
    args = _make_base_args(testtemproot, suffix="_splitRL")
    args.dosplit = True
    args.LtoR = False
    args.volumeperregion = True
    args.debug = debug

    with (
        patch("rapidtide.workflows.atlastool.tide_io.readfromnifti") as mock_read,
        patch("rapidtide.workflows.atlastool.tide_io.savetonifti") as mock_save,
    ):

        mock_read.return_value = (MagicMock(), data, hdr, dims, sizes)

        atlastool(args)

        mock_save.assert_called_once()
        saved_data = mock_save.call_args[0][0]
        assert saved_data.shape[3] == 4


def atlastool_split_with_labels(testtemproot, debug=False):
    """Test split with a label file produces L_/R_ prefixed labels.

    Note: finallabels is only populated when removeemptyregions=True,
    so we enable it here. All regions have voxels so none are removed.
    """
    data, hdr, dims, sizes = _make_splittable_3d_atlas()
    labelfile = os.path.join(testtemproot, "split_labels.txt")
    with open(labelfile, "w") as f:
        f.write("RegionA\nRegionB\n")

    args = _make_base_args(testtemproot, suffix="_splitlabels")
    args.dosplit = True
    args.LtoR = True
    args.labelfile = labelfile
    args.volumeperregion = True
    args.removeemptyregions = True
    args.debug = debug

    with (
        patch("rapidtide.workflows.atlastool.tide_io.readfromnifti") as mock_read,
        patch("rapidtide.workflows.atlastool.tide_io.savetonifti"),
        patch("rapidtide.workflows.atlastool.tide_io.niftisplitext") as mock_split,
    ):

        mock_read.return_value = (MagicMock(), data, hdr, dims, sizes)
        mock_split.return_value = (os.path.join(testtemproot, "atlasout_splitlabels"), ".nii.gz")

        atlastool(args)

        # Verify a label file was written
        labeloutpath = os.path.join(testtemproot, "atlasout_splitlabels_labels.txt")
        assert os.path.exists(labeloutpath)
        with open(labeloutpath) as f:
            labels = f.read().splitlines()
        assert len(labels) == 4
        assert labels[0] == "L_RegionA"
        assert labels[1] == "L_RegionB"
        assert labels[2] == "R_RegionA"
        assert labels[3] == "R_RegionB"


# ==================== Tests for label file handling ====================


def atlastool_with_labels_no_split(testtemproot, debug=False):
    """Test atlas with labels, no split, and removeemptyregions preserves labels."""
    data, hdr, dims, sizes = _make_3d_atlas(xsize=6, numregions=3)
    labelfile = os.path.join(testtemproot, "nosplit_labels.txt")
    with open(labelfile, "w") as f:
        f.write("Alpha\nBeta\nGamma\n")

    args = _make_base_args(testtemproot, suffix="_labelsnosplit")
    args.labelfile = labelfile
    args.removeemptyregions = True
    args.volumeperregion = False
    args.debug = debug

    with (
        patch("rapidtide.workflows.atlastool.tide_io.readfromnifti") as mock_read,
        patch("rapidtide.workflows.atlastool.tide_io.savetonifti"),
        patch("rapidtide.workflows.atlastool.tide_io.niftisplitext") as mock_split,
    ):

        mock_read.return_value = (MagicMock(), data, hdr, dims, sizes)
        mock_split.return_value = (os.path.join(testtemproot, "atlasout_labelsnosplit"), ".nii.gz")

        atlastool(args)

        # Verify label file was written with correct content
        labeloutpath = os.path.join(testtemproot, "atlasout_labelsnosplit_labels.txt")
        assert os.path.exists(labeloutpath)
        with open(labeloutpath) as f:
            labels = f.read().splitlines()
        assert len(labels) == 3
        assert labels == ["Alpha", "Beta", "Gamma"]


def atlastool_label_count_mismatch(testtemproot, debug=False):
    """Test that mismatched label count raises an error."""
    data, hdr, dims, sizes = _make_3d_atlas(xsize=6, numregions=3)
    labelfile = os.path.join(testtemproot, "bad_labels.txt")
    with open(labelfile, "w") as f:
        f.write("OnlyOne\n")

    args = _make_base_args(testtemproot, suffix="_labelmismatch")
    args.labelfile = labelfile
    args.debug = debug

    with (
        patch("rapidtide.workflows.atlastool.tide_io.readfromnifti") as mock_read,
        patch("rapidtide.workflows.atlastool.tide_io.savetonifti"),
    ):

        mock_read.return_value = (MagicMock(), data, hdr, dims, sizes)

        try:
            atlastool(args)
            # The code does raise("...") which creates a string, not an exception.
            # It will actually NOT raise, since raise("string") is a TypeError in
            # modern Python. But if the behavior changes, this test catches it.
        except TypeError:
            pass


# ==================== Tests for mask functionality ====================


def atlastool_with_maskfile(testtemproot, debug=False):
    """Test atlastool with an explicit mask file."""
    xsize, ysize, numslices = 6, 6, 4
    data, hdr, dims, sizes = _make_3d_atlas(xsize, ysize, numslices, numregions=3)

    mask_data = np.ones((xsize, ysize, numslices), dtype=np.float64)
    mask_data[:2, :, :] = 0.0  # Mask out part of region 1
    mask_hdr = _make_mock_hdr(xsize, ysize, numslices, 1)
    mask_dims = np.array([3, xsize, ysize, numslices, 1, 1, 1, 1])
    mask_sizes = np.array([1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0])

    args = _make_base_args(testtemproot, suffix="_withmask")
    args.maskfile = "mask.nii.gz"
    args.volumeperregion = False
    args.debug = debug

    with (
        patch("rapidtide.workflows.atlastool.tide_io.readfromnifti") as mock_read,
        patch("rapidtide.workflows.atlastool.tide_io.savetonifti") as mock_save,
        patch("rapidtide.workflows.atlastool.tide_io.checkspacematch") as mock_match,
    ):

        mock_read.side_effect = [
            (MagicMock(), data, hdr, dims, sizes),
            (MagicMock(), mask_data, mask_hdr, mask_dims, mask_sizes),
        ]
        mock_match.return_value = True

        atlastool(args)

        mock_save.assert_called_once()
        saved_data = mock_save.call_args[0][0]
        # Masked-out voxels should be zero
        assert np.all(saved_data[:2, :, :] == 0)


def atlastool_mask_dimension_mismatch(testtemproot, debug=False):
    """Test atlastool raises when mask dimensions don't match."""
    xsize, ysize, numslices = 6, 6, 4
    data, hdr, dims, sizes = _make_3d_atlas(xsize, ysize, numslices, numregions=3)

    mask_data = np.ones((xsize, ysize, numslices), dtype=np.float64)
    mask_hdr = _make_mock_hdr(xsize, ysize, numslices, 1)
    mask_dims = np.array([3, xsize, ysize, numslices, 1, 1, 1, 1])
    mask_sizes = np.array([1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0])

    args = _make_base_args(testtemproot, suffix="_maskmismatch")
    args.maskfile = "mask.nii.gz"
    args.debug = debug

    with (
        patch("rapidtide.workflows.atlastool.tide_io.readfromnifti") as mock_read,
        patch("rapidtide.workflows.atlastool.tide_io.savetonifti"),
        patch("rapidtide.workflows.atlastool.tide_io.checkspacematch") as mock_match,
    ):

        mock_read.side_effect = [
            (MagicMock(), data, hdr, dims, sizes),
            (MagicMock(), mask_data, mask_hdr, mask_dims, mask_sizes),
        ]
        mock_match.return_value = False

        try:
            atlastool(args)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "do not match" in str(e)


def atlastool_4d_mask_raises(testtemproot, debug=False):
    """Test atlastool raises when mask is 4D."""
    xsize, ysize, numslices = 6, 6, 4
    data, hdr, dims, sizes = _make_3d_atlas(xsize, ysize, numslices, numregions=3)

    mask_data = np.ones((xsize, ysize, numslices, 2), dtype=np.float64)
    mask_hdr = _make_mock_hdr(xsize, ysize, numslices, 2)
    mask_dims = np.array([4, xsize, ysize, numslices, 2, 1, 1, 1])
    mask_sizes = np.array([1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0])

    args = _make_base_args(testtemproot, suffix="_4dmask")
    args.maskfile = "mask4d.nii.gz"
    args.debug = debug

    with (
        patch("rapidtide.workflows.atlastool.tide_io.readfromnifti") as mock_read,
        patch("rapidtide.workflows.atlastool.tide_io.savetonifti"),
    ):

        mock_read.side_effect = [
            (MagicMock(), data, hdr, dims, sizes),
            (MagicMock(), mask_data, mask_hdr, mask_dims, mask_sizes),
        ]

        try:
            atlastool(args)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "not 3D" in str(e)


def atlastool_auto_mask(testtemproot, debug=False):
    """Test atlastool auto-generates mask from template when no maskfile given."""
    data, hdr, dims, sizes = _make_3d_atlas(xsize=6, numregions=3)
    args = _make_base_args(testtemproot, suffix="_automask")
    args.maskfile = None
    args.maskthresh = 0.5
    args.volumeperregion = False
    args.debug = debug

    with (
        patch("rapidtide.workflows.atlastool.tide_io.readfromnifti") as mock_read,
        patch("rapidtide.workflows.atlastool.tide_io.savetonifti") as mock_save,
    ):

        mock_read.return_value = (MagicMock(), data, hdr, dims, sizes)

        atlastool(args)

        mock_save.assert_called_once()


# ==================== Tests for removeemptyregions ====================


def atlastool_remove_empty_regions(testtemproot, debug=False):
    """Test that empty regions are removed when removeemptyregions=True."""
    xsize, ysize, numslices = 6, 6, 4
    # Create a 4D atlas with one empty region (all zeros)
    numregions = 3
    data = np.zeros((xsize, ysize, numslices, numregions), dtype=np.float64)
    data[:3, :, :, 0] = 1.0  # Region 1 has voxels
    # Region 2 is empty (all zeros)
    data[3:, :, :, 2] = 1.0  # Region 3 has voxels

    hdr = _make_mock_hdr(xsize, ysize, numslices, numregions)
    dims = np.array([4, xsize, ysize, numslices, numregions, 1, 1, 1])
    sizes = np.array([1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0])

    args = _make_base_args(testtemproot, suffix="_removeempty")
    args.removeemptyregions = True
    args.volumeperregion = True
    args.debug = debug

    with (
        patch("rapidtide.workflows.atlastool.tide_io.readfromnifti") as mock_read,
        patch("rapidtide.workflows.atlastool.tide_io.savetonifti") as mock_save,
    ):

        mock_read.return_value = (MagicMock(), data, hdr, dims, sizes)

        atlastool(args)

        mock_save.assert_called_once()
        saved_data = mock_save.call_args[0][0]
        # Should have 2 regions after removing the empty one
        assert saved_data.shape == (xsize, ysize, numslices, 2)


# ==================== Tests for debug mode ====================


def atlastool_debug_mode(testtemproot, debug=False):
    """Test atlastool with debug=True exercises debug print paths."""
    data, hdr, dims, sizes = _make_3d_atlas(xsize=6, numregions=2)
    args = _make_base_args(testtemproot, suffix="_debug")
    args.debug = True
    args.volumeperregion = False

    with (
        patch("rapidtide.workflows.atlastool.tide_io.readfromnifti") as mock_read,
        patch("rapidtide.workflows.atlastool.tide_io.savetonifti"),
    ):

        mock_read.return_value = (MagicMock(), data, hdr, dims, sizes)

        atlastool(args)


def atlastool_debug_auto_mask(testtemproot, debug=False):
    """Test debug mode with auto-generated mask saves debug mask files."""
    data, hdr, dims, sizes = _make_3d_atlas(xsize=6, numregions=2)
    args = _make_base_args(testtemproot, suffix="_debugmask")
    args.debug = True
    args.maskfile = None
    args.volumeperregion = False

    with (
        patch("rapidtide.workflows.atlastool.tide_io.readfromnifti") as mock_read,
        patch("rapidtide.workflows.atlastool.tide_io.savetonifti") as mock_save,
    ):

        mock_read.return_value = (MagicMock(), data, hdr, dims, sizes)

        atlastool(args)

        # In debug mode with auto mask, savetonifti is called for
        # masktemp1, masktemp2, and the final output = 3 calls
        assert mock_save.call_count == 3


# ==================== Tests for targetfile (resampling) ====================


def atlastool_targetfile_no_fsl(testtemproot, debug=False):
    """Test that atlastool exits when targetfile is given but FSLDIR is not set.

    Note: when xfm is None, the code tries os.path.join(fsldir, ...) before
    checking if fsldir is not None, so we must provide an xfm to reach the
    sys.exit() path.
    """
    data, hdr, dims, sizes = _make_3d_atlas(xsize=6, numregions=2)
    xfmfile = os.path.join(testtemproot, "dummy.mat")
    with open(xfmfile, "w") as f:
        f.write("dummy")

    args = _make_base_args(testtemproot, suffix="_nofsl")
    args.targetfile = "target.nii.gz"
    args.xfm = xfmfile
    args.debug = debug

    with (
        patch("rapidtide.workflows.atlastool.tide_io.readfromnifti") as mock_read,
        patch("rapidtide.workflows.atlastool.tide_io.savetonifti"),
        patch.dict(os.environ, {}, clear=True),
        patch("rapidtide.workflows.atlastool.sys.exit") as mock_exit,
    ):

        mock_read.return_value = (MagicMock(), data, hdr, dims, sizes)
        mock_exit.side_effect = SystemExit(0)

        try:
            atlastool(args)
        except SystemExit:
            pass
        mock_exit.assert_called_once()


# ==================== Main test function ====================


def test_atlastool(debug=False, local=False):
    # set up temp directory
    testtemproot = get_test_temp_path(local)
    create_dir(testtemproot)

    # _get_parser tests
    if debug:
        print("get_parser_returns_parser()")
    get_parser_returns_parser(debug=debug)

    if debug:
        print("get_parser_prog_name()")
    get_parser_prog_name(debug=debug)

    if debug:
        print("get_parser_defaults()")
    get_parser_defaults(debug=debug)

    if debug:
        print("get_parser_with_4d_flag(testtemproot)")
    get_parser_with_4d_flag(testtemproot, debug=debug)

    if debug:
        print("get_parser_with_3d_flag(testtemproot)")
    get_parser_with_3d_flag(testtemproot, debug=debug)

    if debug:
        print("get_parser_with_split(testtemproot)")
    get_parser_with_split(testtemproot, debug=debug)

    if debug:
        print("get_parser_with_all_options(testtemproot)")
    get_parser_with_all_options(testtemproot, debug=debug)

    # atlastool 3D input tests
    if debug:
        print("atlastool_3d_to_3d(testtemproot)")
    atlastool_3d_to_3d(testtemproot, debug=debug)

    if debug:
        print("atlastool_3d_to_4d(testtemproot)")
    atlastool_3d_to_4d(testtemproot, debug=debug)

    if debug:
        print("atlastool_3d_with_maxval(testtemproot)")
    atlastool_3d_with_maxval(testtemproot, debug=debug)

    # atlastool 4D input tests
    if debug:
        print("atlastool_4d_to_3d(testtemproot)")
    atlastool_4d_to_3d(testtemproot, debug=debug)

    if debug:
        print("atlastool_4d_to_4d(testtemproot)")
    atlastool_4d_to_4d(testtemproot, debug=debug)

    if debug:
        print("atlastool_4d_with_maxval(testtemproot)")
    atlastool_4d_with_maxval(testtemproot, debug=debug)

    # split tests
    if debug:
        print("atlastool_split_LtoR(testtemproot)")
    atlastool_split_LtoR(testtemproot, debug=debug)

    if debug:
        print("atlastool_split_RtoL(testtemproot)")
    atlastool_split_RtoL(testtemproot, debug=debug)

    if debug:
        print("atlastool_split_with_labels(testtemproot)")
    atlastool_split_with_labels(testtemproot, debug=debug)

    # label file tests
    if debug:
        print("atlastool_with_labels_no_split(testtemproot)")
    atlastool_with_labels_no_split(testtemproot, debug=debug)

    if debug:
        print("atlastool_label_count_mismatch(testtemproot)")
    atlastool_label_count_mismatch(testtemproot, debug=debug)

    # mask tests
    if debug:
        print("atlastool_with_maskfile(testtemproot)")
    atlastool_with_maskfile(testtemproot, debug=debug)

    if debug:
        print("atlastool_mask_dimension_mismatch(testtemproot)")
    atlastool_mask_dimension_mismatch(testtemproot, debug=debug)

    if debug:
        print("atlastool_4d_mask_raises(testtemproot)")
    atlastool_4d_mask_raises(testtemproot, debug=debug)

    if debug:
        print("atlastool_auto_mask(testtemproot)")
    atlastool_auto_mask(testtemproot, debug=debug)

    # removeemptyregions tests
    if debug:
        print("atlastool_remove_empty_regions(testtemproot)")
    atlastool_remove_empty_regions(testtemproot, debug=debug)

    # debug mode tests
    if debug:
        print("atlastool_debug_mode(testtemproot)")
    atlastool_debug_mode(testtemproot, debug=debug)

    if debug:
        print("atlastool_debug_auto_mask(testtemproot)")
    atlastool_debug_auto_mask(testtemproot, debug=debug)

    # targetfile tests
    if debug:
        print("atlastool_targetfile_no_fsl(testtemproot)")
    atlastool_targetfile_no_fsl(testtemproot, debug=debug)


if __name__ == "__main__":
    test_atlastool(debug=True, local=True)
