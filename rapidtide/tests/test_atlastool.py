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


def _outputsaves(mock_save):
    """The savetonifti calls that produced real output.

    Under --debug atlastool also dumps two masktemp volumes for inspection, so a
    bare call count is only correct when debug is off.  Filtering them out is what
    lets these tests be run either way - which matters, because the __main__ entry
    point at the bottom of this file runs the whole suite with debug=True.

    Parameters
    ----------
    mock_save : MagicMock
        The patched tide_io.savetonifti.

    Returns
    -------
    list
        The non-debug calls, in call order.
    """
    return [
        thecall
        for thecall in mock_save.call_args_list
        if not str(thecall[0][2]).endswith(("masktemp1", "masktemp2"))
    ]


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

        assert len(_outputsaves(mock_save)) == 1
        saved_data = _outputsaves(mock_save)[0][0][0]
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

        assert len(_outputsaves(mock_save)) == 1
        saved_data = _outputsaves(mock_save)[0][0][0]
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

        assert len(_outputsaves(mock_save)) == 1
        saved_data = _outputsaves(mock_save)[0][0][0]
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

        assert len(_outputsaves(mock_save)) == 1
        saved_data = _outputsaves(mock_save)[0][0][0]
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

        assert len(_outputsaves(mock_save)) == 1
        saved_data = _outputsaves(mock_save)[0][0][0]
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

        assert len(_outputsaves(mock_save)) == 1


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

        assert len(_outputsaves(mock_save)) == 1
        saved_data = _outputsaves(mock_save)[0][0][0]
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

        assert len(_outputsaves(mock_save)) == 1
        saved_data = _outputsaves(mock_save)[0][0][0]
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
    """A label file that does not match the atlas must raise a real exception.

    This used to be `raise ("...")`, which raises a string and so produces
    "TypeError: exceptions must derive from BaseException" - a message about
    Python, telling the user nothing about their label file.  The earlier version
    of this test caught TypeError with a note that it would notice if the behavior
    changed, which is what happened.
    """
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
        except ValueError as theerror:
            themessage = str(theerror)
        except TypeError as theerror:
            raise AssertionError(f"raised a bare string rather than an exception: {theerror}")
        else:
            raise AssertionError("a mismatched label file was accepted")

    if debug:
        print(f"error message: {themessage}")
    # the message has to name both counts, or it does not help anyone
    assert "1" in themessage and "3" in themessage
    assert labelfile in themessage


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

        assert len(_outputsaves(mock_save)) == 1
        saved_data = _outputsaves(mock_save)[0][0][0]
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

        assert len(_outputsaves(mock_save)) == 1


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

        assert len(_outputsaves(mock_save)) == 1
        saved_data = _outputsaves(mock_save)[0][0][0]
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

    This covers the case where an xfm IS supplied.  The FSLDIR check used to sit
    below the point where the identity transform path was built, so with xfm None
    the run died on a TypeError before ever reaching it, and this test had to pass
    an xfm to get here at all.  The check is hoisted now;
    atlastool_resample_aborts_when_fsldir_is_unset covers the xfm None case.
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


# ==================== Tests for the --targetfile resampling block ====================
#
# This whole block used to be unreachable: a typo left `aligntype` unassigned when no
# --xfm was given, and the extension test compared against "mat" when os.path.splitext
# returns ".mat", so an explicit FLIRT transform was routed to ANTs instead.  These
# tests pin the command lines that get built, which is the only place those mistakes
# were ever visible - nothing downstream inspects them.


def _make_resample_mocks(data, hdr, dims, sizes, numregions=3, emptyregions=()):
    """Build the readfromnifti side effect for a resampling run.

    atlastool reads twice when resampling: once for the input template, and once to
    read back whatever the aligner wrote.  The second read is what sets the post
    resample dimensions, so it has to be a self consistent 4D volume.

    Parameters
    ----------
    data, hdr, dims, sizes : various
        The input template as returned by the _make_*_atlas helpers.
    numregions : int
        Number of regions in the resampled readback.
    emptyregions : iterable of int
        Region indices to leave empty in the readback, simulating a region that
        alignment resampled out of existence.

    Returns
    -------
    list
        The two return values for readfromnifti, in call order.
    """
    xsize, ysize, numslices = int(dims[1]), int(dims[2]), int(dims[3])
    resampled = np.zeros((xsize, ysize, numslices, numregions), dtype=np.float64)
    for theregion in range(numregions):
        if theregion in emptyregions:
            continue
        resampled[theregion :: max(numregions, 1), :, :, theregion] = 1.0
    resampleddims = np.array([4, xsize, ysize, numslices, numregions, 1, 1, 1])
    resampledhdr = _make_mock_hdr(xsize, ysize, numslices, numregions)
    return [
        (MagicMock(), data, hdr, dims, sizes),
        (MagicMock(), resampled, resampledhdr, resampleddims, sizes),
    ]


def atlastool_resample_debug_adds_the_verbose_flag(testtemproot, debug=False):
    """--debug is not only chatter: it adds -v to the flirt command line, so the
    aligner's own output ends up in the log alongside everything else."""
    data, hdr, dims, sizes = _make_3d_atlas()
    args = _make_base_args(testtemproot, suffix="_flirtverbose")
    args.targetfile = os.path.join(testtemproot, "target.nii.gz")
    args.xfm = None
    args.debug = True

    with (
        patch("rapidtide.workflows.atlastool.tide_io.readfromnifti") as mock_read,
        patch("rapidtide.workflows.atlastool.tide_io.savetonifti"),
        patch("rapidtide.workflows.atlastool.subprocess.call") as mock_call,
        patch("rapidtide.workflows.atlastool.glob") as mock_glob,
        patch.dict(os.environ, {"FSLDIR": "/fake/fsl"}),
    ):
        mock_read.side_effect = _make_resample_mocks(data, hdr, dims, sizes)
        mock_glob.return_value = ["/tmp/temppre_0000.nii.gz"]
        atlastool(args)

    theflirt = [c[0][0] for c in mock_call.call_args_list if c[0][0][0].endswith("flirt")][0]
    if debug:
        print(" ".join(theflirt))
    assert "-v" in theflirt, "debug did not put flirt into verbose mode"


def atlastool_resample_debug_runs_the_ants_path(testtemproot, debug=False):
    """The ANTs branch has its own debug reporting, and antsapply is handed the
    debug flag so its own commands are logged too."""
    data, hdr, dims, sizes = _make_3d_atlas()
    args = _make_base_args(testtemproot, suffix="_antsverbose")
    args.targetfile = os.path.join(testtemproot, "target.nii.gz")
    args.xfm = os.path.join(testtemproot, "mywarp.h5")
    args.debug = True

    with (
        patch("rapidtide.workflows.atlastool.tide_io.readfromnifti") as mock_read,
        patch("rapidtide.workflows.atlastool.tide_io.savetonifti"),
        patch("rapidtide.workflows.atlastool.subprocess.call"),
        patch("rapidtide.workflows.atlastool.tide_exttools.antsapply") as mock_ants,
        patch("rapidtide.workflows.atlastool.glob") as mock_glob,
        patch.dict(os.environ, {"FSLDIR": "/fake/fsl"}),
    ):
        mock_read.side_effect = _make_resample_mocks(data, hdr, dims, sizes)
        mock_glob.return_value = ["/tmp/temppost_0000.nii.gz"]
        atlastool(args)

    if debug:
        print(f"antsapply kwargs: {[c[1] for c in mock_ants.call_args_list]}")
    assert mock_ants.call_count > 0
    for thecall in mock_ants.call_args_list:
        assert thecall[1].get("debug") is True, "the debug flag was not passed to antsapply"


def atlastool_resample_drops_regions_emptied_by_alignment(testtemproot, debug=False):
    """Alignment can resample a small region out of existence.  With
    --removeemptyregions the second sweep has to notice and drop it, and the label
    list has to shrink with it or labels and regions fall out of step.
    """
    data, hdr, dims, sizes = _make_3d_atlas(numregions=3)
    thelabelfile = os.path.join(testtemproot, "resample_labels.txt")
    with open(thelabelfile, "w") as f:
        f.write("Alpha\nBeta\nGamma\n")

    args = _make_base_args(testtemproot, suffix="_resampleempty")
    args.targetfile = os.path.join(testtemproot, "target.nii.gz")
    args.xfm = None
    args.labelfile = thelabelfile
    args.removeemptyregions = True
    args.volumeperregion = True
    # run this one with debug on: it is the only path that has both a label file and
    # the resampling block, so it is where the debug reporting in both gets exercised
    args.debug = True

    with (
        patch("rapidtide.workflows.atlastool.tide_io.readfromnifti") as mock_read,
        patch("rapidtide.workflows.atlastool.tide_io.savetonifti") as mock_save,
        patch("rapidtide.workflows.atlastool.subprocess.call"),
        patch("rapidtide.workflows.atlastool.glob") as mock_glob,
        patch.dict(os.environ, {"FSLDIR": "/fake/fsl"}),
    ):
        # region 1 comes back from the aligner with nothing in it
        mock_read.side_effect = _make_resample_mocks(
            data, hdr, dims, sizes, numregions=3, emptyregions=(1,)
        )
        mock_glob.return_value = ["/tmp/temppre_0000.nii.gz"]
        atlastool(args)

    thesaved = _outputsaves(mock_save)[-1][0][0]
    if debug:
        print(f"final saved shape {thesaved.shape}")
    # three regions in, one emptied by alignment, two out
    assert thesaved.shape[-1] == 2, f"expected 2 surviving regions, got {thesaved.shape[-1]}"

    thelabeloutput = os.path.join(testtemproot, "atlasout_resampleempty_labels.txt")
    assert os.path.exists(thelabeloutput)
    with open(thelabeloutput) as f:
        thefinallabels = f.read().splitlines()
    if debug:
        print(f"final labels {thefinallabels}")
    assert len(thefinallabels) == 2, "the label list did not shrink with the regions"


def atlastool_resample_with_flirt_and_default_xfm(testtemproot, debug=False):
    """--targetfile with no --xfm must use FSL's identity transform and run flirt.

    This is the default resampling path, and before the alignttype typo was fixed it
    raised UnboundLocalError for every single caller.
    """
    data, hdr, dims, sizes = _make_3d_atlas()
    args = _make_base_args(testtemproot, suffix="_flirtdefault")
    args.targetfile = os.path.join(testtemproot, "target.nii.gz")
    args.xfm = None
    args.debug = debug

    with (
        patch("rapidtide.workflows.atlastool.tide_io.readfromnifti") as mock_read,
        patch("rapidtide.workflows.atlastool.tide_io.savetonifti") as mock_save,
        patch("rapidtide.workflows.atlastool.subprocess.call") as mock_call,
        patch("rapidtide.workflows.atlastool.glob") as mock_glob,
        patch.dict(os.environ, {"FSLDIR": "/fake/fsl"}),
    ):
        mock_read.side_effect = _make_resample_mocks(data, hdr, dims, sizes)
        mock_glob.return_value = ["/tmp/temppre_0001.nii.gz", "/tmp/temppre_0000.nii.gz"]

        atlastool(args)

    thecommands = [thecall[0][0] for thecall in mock_call.call_args_list]
    if debug:
        for thecommand in thecommands:
            print(" ".join(thecommand))

    assert len(thecommands) == 2, "expected an fslmerge and a flirt call"
    themerge, theflirt = thecommands

    # the merge collects the per region temp files in sorted order, not glob order
    assert themerge[0].endswith(os.path.join("bin", "fslmerge"))
    assert themerge[1] == "-t"
    assert themerge[3:] == sorted(mock_glob.return_value)

    # flirt applies the identity transform that ships inside FSLDIR
    assert theflirt[0].endswith(os.path.join("bin", "flirt"))
    assert "-applyxfm" in theflirt
    assert args.targetfile == theflirt[theflirt.index("-ref") + 1]
    theusedxfm = theflirt[theflirt.index("-init") + 1]
    assert theusedxfm == os.path.join("/fake/fsl", "data", "atlases", "bin", "eye.mat")

    # one temp volume written per region, then the final output
    thesavednames = [str(thecall[0][2]) for thecall in _outputsaves(mock_save)]
    thetempsaves = [thename for thename in thesavednames if "temppre" in thename]
    assert len(thetempsaves) == 3, f"expected one temp volume per region, got {thetempsaves}"
    assert args.outputtemplatename in thesavednames


def atlastool_resample_with_flirt_and_explicit_mat(testtemproot, debug=False):
    """A .mat transform is a FLIRT transform.

    os.path.splitext returns the extension WITH its dot, so the old comparison
    against "mat" never matched and every .mat file was handed to ANTs.
    """
    data, hdr, dims, sizes = _make_3d_atlas()
    args = _make_base_args(testtemproot, suffix="_flirtmat")
    args.targetfile = os.path.join(testtemproot, "target.nii.gz")
    args.xfm = os.path.join(testtemproot, "mytransform.mat")
    args.debug = debug

    with (
        patch("rapidtide.workflows.atlastool.tide_io.readfromnifti") as mock_read,
        patch("rapidtide.workflows.atlastool.tide_io.savetonifti"),
        patch("rapidtide.workflows.atlastool.subprocess.call") as mock_call,
        patch("rapidtide.workflows.atlastool.tide_exttools.antsapply") as mock_ants,
        patch("rapidtide.workflows.atlastool.glob") as mock_glob,
        patch.dict(os.environ, {"FSLDIR": "/fake/fsl"}),
    ):
        mock_read.side_effect = _make_resample_mocks(data, hdr, dims, sizes)
        mock_glob.return_value = ["/tmp/temppre_0000.nii.gz"]

        atlastool(args)

    thecommands = [thecall[0][0] for thecall in mock_call.call_args_list]
    if debug:
        print(f"ants calls {mock_ants.call_count}, commands {[c[0] for c in thecommands]}")

    assert mock_ants.call_count == 0, "a .mat transform was sent to ANTs"
    assert any(thecommand[0].endswith("flirt") for thecommand in thecommands)
    theflirt = [c for c in thecommands if c[0].endswith("flirt")][0]
    assert theflirt[theflirt.index("-init") + 1] == args.xfm


def atlastool_resample_with_ants(testtemproot, debug=False):
    """A non .mat transform goes to ANTs, one call per region, then a merge."""
    data, hdr, dims, sizes = _make_3d_atlas()
    numregions = 3
    args = _make_base_args(testtemproot, suffix="_ants")
    args.targetfile = os.path.join(testtemproot, "target.nii.gz")
    args.xfm = os.path.join(testtemproot, "mywarp.h5")
    args.debug = debug

    with (
        patch("rapidtide.workflows.atlastool.tide_io.readfromnifti") as mock_read,
        patch("rapidtide.workflows.atlastool.tide_io.savetonifti"),
        patch("rapidtide.workflows.atlastool.subprocess.call") as mock_call,
        patch("rapidtide.workflows.atlastool.tide_exttools.antsapply") as mock_ants,
        patch("rapidtide.workflows.atlastool.glob") as mock_glob,
        patch.dict(os.environ, {"FSLDIR": "/fake/fsl"}),
    ):
        mock_read.side_effect = _make_resample_mocks(data, hdr, dims, sizes)
        mock_glob.return_value = [f"/tmp/temppost_{i:04d}.nii.gz" for i in range(numregions)]

        atlastool(args)

    if debug:
        print(f"ants calls {mock_ants.call_count}")
        for thecall in mock_ants.call_args_list:
            print(f"  {thecall[0]}")

    # one alignment per region, each carrying the supplied warp
    assert mock_ants.call_count == numregions
    for thecall in mock_ants.call_args_list:
        theinfile, thetarget, theoutfile, thexfmlist = thecall[0][:4]
        assert thetarget == args.targetfile
        assert thexfmlist == [args.xfm]
        assert "temppre" in theinfile and "temppost" in theoutfile

    # then a single merge of the aligned volumes, and no flirt
    thecommands = [thecall[0][0] for thecall in mock_call.call_args_list]
    assert len(thecommands) == 1
    assert thecommands[0][0].endswith(os.path.join("bin", "fslmerge"))
    assert not any(thecommand[0].endswith("flirt") for thecommand in thecommands)


def atlastool_resample_aborts_when_fsldir_is_unset(testtemproot, debug=False):
    """Without FSLDIR there is no aligner and no identity transform to point at, so
    the run has to stop with its own message rather than a TypeError from trying to
    build a path out of None."""
    data, hdr, dims, sizes = _make_3d_atlas()
    args = _make_base_args(testtemproot, suffix="_nofsl")
    args.targetfile = os.path.join(testtemproot, "target.nii.gz")
    args.xfm = None
    args.debug = debug

    theenv = {k: v for k, v in os.environ.items() if k != "FSLDIR"}
    with (
        patch("rapidtide.workflows.atlastool.tide_io.readfromnifti") as mock_read,
        patch("rapidtide.workflows.atlastool.tide_io.savetonifti"),
        patch("rapidtide.workflows.atlastool.subprocess.call") as mock_call,
        patch.dict(os.environ, theenv, clear=True),
    ):
        mock_read.return_value = (MagicMock(), data, hdr, dims, sizes)
        try:
            atlastool(args)
        except SystemExit:
            thereason = "SystemExit"
        except TypeError as theerror:
            thereason = f"TypeError: {theerror}"
        else:
            thereason = "returned normally"

    if debug:
        print(f"outcome without FSLDIR: {thereason}")
    assert thereason == "SystemExit", f"expected a clean abort, got {thereason}"
    assert mock_call.call_count == 0, "an aligner was invoked with no FSL present"


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

    # --targetfile resampling tests
    if debug:
        print("atlastool_resample_with_flirt_and_default_xfm(testtemproot)")
    atlastool_resample_with_flirt_and_default_xfm(testtemproot, debug=debug)

    if debug:
        print("atlastool_resample_with_flirt_and_explicit_mat(testtemproot)")
    atlastool_resample_with_flirt_and_explicit_mat(testtemproot, debug=debug)

    if debug:
        print("atlastool_resample_with_ants(testtemproot)")
    atlastool_resample_with_ants(testtemproot, debug=debug)

    if debug:
        print("atlastool_resample_aborts_when_fsldir_is_unset(testtemproot)")
    atlastool_resample_aborts_when_fsldir_is_unset(testtemproot, debug=debug)

    if debug:
        print("atlastool_resample_debug_adds_the_verbose_flag(testtemproot)")
    atlastool_resample_debug_adds_the_verbose_flag(testtemproot, debug=debug)

    if debug:
        print("atlastool_resample_debug_runs_the_ants_path(testtemproot)")
    atlastool_resample_debug_runs_the_ants_path(testtemproot, debug=debug)

    if debug:
        print("atlastool_resample_drops_regions_emptied_by_alignment(testtemproot)")
    atlastool_resample_drops_regions_emptied_by_alignment(testtemproot, debug=debug)


if __name__ == "__main__":
    test_atlastool(debug=True, local=True)
