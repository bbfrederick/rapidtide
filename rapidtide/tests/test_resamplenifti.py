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
import copy
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from rapidtide.workflows.resamplenifti import _get_parser, resamplenifti

# ============================================================================
# Helpers
# ============================================================================


def _make_default_args(**overrides):
    defaults = dict(
        inputfile="input.nii.gz",
        outputfile="output.nii.gz",
        outputtr=2.0,
        antialias=True,
        normalize=False,
        debug=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _make_mock_header(pixdim=None):
    """Create a mock NIfTI header that supports item access and copy()."""
    if pixdim is None:
        pixdim = [0.0, 2.0, 2.0, 2.0, 1.0, 0.0, 0.0, 0.0]
    storage = {"pixdim": list(pixdim)}

    hdr = MagicMock()
    hdr.__getitem__ = lambda self, key: storage[key]
    hdr.__setitem__ = lambda self, key, val: storage.__setitem__(key, val)
    hdr.copy_fn = lambda: _make_mock_header(pixdim=list(storage["pixdim"]))

    def _copy():
        new_pixdim = list(storage["pixdim"])
        return _make_mock_header(pixdim=new_pixdim)

    hdr.copy = _copy
    return hdr


def _make_mock_nifti_data(xsize=2, ysize=2, numslices=1, timepoints=10, inputtr=1.0):
    """Create synthetic 4D data with a known signal."""
    data = np.zeros((xsize, ysize, numslices, timepoints), dtype=np.float64)
    t = np.arange(timepoints) * inputtr
    for x in range(xsize):
        for y in range(ysize):
            for z in range(numslices):
                data[x, y, z, :] = np.sin(2 * np.pi * 0.1 * t) + x + y + z
    return data


# ============================================================================
# Parser tests
# ============================================================================


def parser_basic(debug=False):
    if debug:
        print("parser_basic")
    parser = _get_parser()
    assert parser is not None
    assert parser.prog == "resamplenifti"


def parser_required_args(debug=False):
    if debug:
        print("parser_required_args")
    parser = _get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def parser_defaults(debug=False):
    if debug:
        print("parser_defaults")
    parser = _get_parser()
    args = parser.parse_args(["input.nii.gz", "output.nii.gz", "2.0"])
    assert args.inputfile == "input.nii.gz"
    assert args.outputfile == "output.nii.gz"
    assert args.outputtr == 2.0
    assert args.antialias is True
    assert args.normalize is False
    assert args.debug is False


def parser_noantialias_flag(debug=False):
    if debug:
        print("parser_noantialias_flag")
    parser = _get_parser()
    args = parser.parse_args(["in.nii", "out.nii", "1.0", "--noantialias"])
    assert args.antialias is False


def parser_normalize_flag(debug=False):
    if debug:
        print("parser_normalize_flag")
    parser = _get_parser()
    args = parser.parse_args(["in.nii", "out.nii", "1.0", "--normalize"])
    assert args.normalize is True


def parser_debug_flag(debug=False):
    if debug:
        print("parser_debug_flag")
    parser = _get_parser()
    args = parser.parse_args(["in.nii", "out.nii", "1.0", "--debug"])
    assert args.debug is True


def parser_all_flags(debug=False):
    if debug:
        print("parser_all_flags")
    parser = _get_parser()
    args = parser.parse_args(["in.nii", "out.nii", "0.5", "--noantialias", "--normalize", "--debug"])
    assert args.antialias is False
    assert args.normalize is True
    assert args.debug is True
    assert args.outputtr == 0.5


def parser_outputtr_type(debug=False):
    """Verify outputtr is parsed as float."""
    if debug:
        print("parser_outputtr_type")
    parser = _get_parser()
    args = parser.parse_args(["in.nii", "out.nii", "3"])
    assert isinstance(args.outputtr, float)
    assert args.outputtr == 3.0


# ============================================================================
# resamplenifti function tests
# ============================================================================


def resamplenifti_basic_downsample(debug=False):
    """Test basic downsampling (outputtr > inputtr): antialias stays on."""
    if debug:
        print("resamplenifti_basic_downsample")

    inputtr = 1.0
    numinputtrs = 10
    xsize, ysize, numslices = 2, 2, 1
    outputtr = 2.0

    input_data = _make_mock_nifti_data(xsize, ysize, numslices, numinputtrs, inputtr)
    input_hdr = _make_mock_header(pixdim=[0.0, 2.0, 2.0, 2.0, inputtr, 0.0, 0.0, 0.0])
    thedims = np.array([4, xsize, ysize, numslices, numinputtrs, 0, 0, 0])
    thesizes = np.array([0.0, 2.0, 2.0, 2.0, inputtr, 0.0, 0.0, 0.0])
    mock_img = MagicMock()

    captured = {}

    def _mock_doresample(orig_x, orig_y, new_x, antialias=False, **kwargs):
        captured.setdefault("calls", []).append(
            {"orig_x": orig_x.copy(), "orig_y": orig_y.copy(), "new_x": new_x.copy(), "antialias": antialias}
        )
        return np.interp(new_x, orig_x, orig_y)

    def _mock_savetonifti(thearray, theheader, thename, **kwargs):
        captured["saved_array"] = thearray.copy()
        captured["saved_header"] = theheader
        captured["saved_name"] = thename

    args = _make_default_args(outputtr=outputtr)

    with (
        patch(
            "rapidtide.workflows.resamplenifti.tide_io.fmritimeinfo",
            return_value=(inputtr, numinputtrs),
        ),
        patch(
            "rapidtide.workflows.resamplenifti.tide_io.readfromnifti",
            return_value=(mock_img, input_data, input_hdr, thedims, thesizes),
        ),
        patch(
            "rapidtide.workflows.resamplenifti.tide_resample.doresample",
            side_effect=_mock_doresample,
        ),
        patch(
            "rapidtide.workflows.resamplenifti.tide_io.niftisplitext",
            return_value=("output", ".nii.gz"),
        ),
        patch(
            "rapidtide.workflows.resamplenifti.tide_io.savetonifti",
            side_effect=_mock_savetonifti,
        ),
    ):
        resamplenifti(args)

    # Verify doresample was called for every voxel
    num_voxels = xsize * ysize * numslices
    assert len(captured["calls"]) == num_voxels

    # Verify antialias was True for downsampling (inputtr < outputtr)
    for c in captured["calls"]:
        assert c["antialias"] is True

    # Verify output was saved
    assert captured["saved_name"] == "output"

    # Verify output header has updated TR
    assert captured["saved_header"]["pixdim"][4] == outputtr


def resamplenifti_upsample_disables_antialias(debug=False):
    """Test that upsampling (inputtr > outputtr) disables antialias."""
    if debug:
        print("resamplenifti_upsample_disables_antialias")

    inputtr = 2.0
    numinputtrs = 10
    xsize, ysize, numslices = 1, 1, 1
    outputtr = 0.5

    input_data = _make_mock_nifti_data(xsize, ysize, numslices, numinputtrs, inputtr)
    input_hdr = _make_mock_header(pixdim=[0.0, 2.0, 2.0, 2.0, inputtr, 0.0, 0.0, 0.0])
    thedims = np.array([4, xsize, ysize, numslices, numinputtrs, 0, 0, 0])
    thesizes = np.array([0.0, 2.0, 2.0, 2.0, inputtr, 0.0, 0.0, 0.0])
    mock_img = MagicMock()

    captured = {}

    def _mock_doresample(orig_x, orig_y, new_x, antialias=False, **kwargs):
        captured["antialias"] = antialias
        return np.interp(new_x, orig_x, orig_y)

    args = _make_default_args(outputtr=outputtr, antialias=True)

    with (
        patch(
            "rapidtide.workflows.resamplenifti.tide_io.fmritimeinfo",
            return_value=(inputtr, numinputtrs),
        ),
        patch(
            "rapidtide.workflows.resamplenifti.tide_io.readfromnifti",
            return_value=(mock_img, input_data, input_hdr, thedims, thesizes),
        ),
        patch(
            "rapidtide.workflows.resamplenifti.tide_resample.doresample",
            side_effect=_mock_doresample,
        ),
        patch(
            "rapidtide.workflows.resamplenifti.tide_io.niftisplitext",
            return_value=("output", ".nii.gz"),
        ),
        patch("rapidtide.workflows.resamplenifti.tide_io.savetonifti"),
    ):
        resamplenifti(args)

    # Upsampling should force antialias off
    assert captured["antialias"] is False


def resamplenifti_noantialias_flag(debug=False):
    """Test that --noantialias flag is respected during downsampling."""
    if debug:
        print("resamplenifti_noantialias_flag")

    inputtr = 1.0
    numinputtrs = 10
    xsize, ysize, numslices = 1, 1, 1
    outputtr = 2.0

    input_data = _make_mock_nifti_data(xsize, ysize, numslices, numinputtrs, inputtr)
    input_hdr = _make_mock_header(pixdim=[0.0, 2.0, 2.0, 2.0, inputtr, 0.0, 0.0, 0.0])
    thedims = np.array([4, xsize, ysize, numslices, numinputtrs, 0, 0, 0])
    thesizes = np.array([0.0, 2.0, 2.0, 2.0, inputtr, 0.0, 0.0, 0.0])
    mock_img = MagicMock()

    captured = {}

    def _mock_doresample(orig_x, orig_y, new_x, antialias=False, **kwargs):
        captured["antialias"] = antialias
        return np.interp(new_x, orig_x, orig_y)

    args = _make_default_args(outputtr=outputtr, antialias=False)

    with (
        patch(
            "rapidtide.workflows.resamplenifti.tide_io.fmritimeinfo",
            return_value=(inputtr, numinputtrs),
        ),
        patch(
            "rapidtide.workflows.resamplenifti.tide_io.readfromnifti",
            return_value=(mock_img, input_data, input_hdr, thedims, thesizes),
        ),
        patch(
            "rapidtide.workflows.resamplenifti.tide_resample.doresample",
            side_effect=_mock_doresample,
        ),
        patch(
            "rapidtide.workflows.resamplenifti.tide_io.niftisplitext",
            return_value=("output", ".nii.gz"),
        ),
        patch("rapidtide.workflows.resamplenifti.tide_io.savetonifti"),
    ):
        resamplenifti(args)

    assert captured["antialias"] is False


def resamplenifti_output_shape(debug=False):
    """Verify the output array has the correct shape based on outputtr."""
    if debug:
        print("resamplenifti_output_shape")

    inputtr = 1.0
    numinputtrs = 10
    xsize, ysize, numslices = 3, 2, 2
    outputtr = 2.0

    input_data = _make_mock_nifti_data(xsize, ysize, numslices, numinputtrs, inputtr)
    input_hdr = _make_mock_header(pixdim=[0.0, 2.0, 2.0, 2.0, inputtr, 0.0, 0.0, 0.0])
    thedims = np.array([4, xsize, ysize, numslices, numinputtrs, 0, 0, 0])
    thesizes = np.array([0.0, 2.0, 2.0, 2.0, inputtr, 0.0, 0.0, 0.0])
    mock_img = MagicMock()

    captured = {}

    def _mock_doresample(orig_x, orig_y, new_x, antialias=False, **kwargs):
        return np.interp(new_x, orig_x, orig_y)

    def _mock_savetonifti(thearray, theheader, thename, **kwargs):
        captured["shape"] = thearray.shape

    args = _make_default_args(outputtr=outputtr)

    with (
        patch(
            "rapidtide.workflows.resamplenifti.tide_io.fmritimeinfo",
            return_value=(inputtr, numinputtrs),
        ),
        patch(
            "rapidtide.workflows.resamplenifti.tide_io.readfromnifti",
            return_value=(mock_img, input_data, input_hdr, thedims, thesizes),
        ),
        patch(
            "rapidtide.workflows.resamplenifti.tide_resample.doresample",
            side_effect=_mock_doresample,
        ),
        patch(
            "rapidtide.workflows.resamplenifti.tide_io.niftisplitext",
            return_value=("output", ".nii.gz"),
        ),
        patch(
            "rapidtide.workflows.resamplenifti.tide_io.savetonifti",
            side_effect=_mock_savetonifti,
        ),
    ):
        resamplenifti(args)

    # Calculate expected number of output timepoints using the same formula as the code
    inputendtime = inputtr * (numinputtrs - 1)
    numoutputtrs = int(np.ceil(inputendtime / outputtr) + 1)
    assert captured["shape"] == (xsize, ysize, numslices, numoutputtrs)


def resamplenifti_header_updated(debug=False):
    """Verify the output header pixdim[4] is set to the new TR."""
    if debug:
        print("resamplenifti_header_updated")

    inputtr = 1.0
    numinputtrs = 10
    xsize, ysize, numslices = 1, 1, 1
    outputtr = 0.5

    input_data = _make_mock_nifti_data(xsize, ysize, numslices, numinputtrs, inputtr)
    input_hdr = _make_mock_header(pixdim=[0.0, 2.0, 2.0, 2.0, inputtr, 0.0, 0.0, 0.0])
    thedims = np.array([4, xsize, ysize, numslices, numinputtrs, 0, 0, 0])
    thesizes = np.array([0.0, 2.0, 2.0, 2.0, inputtr, 0.0, 0.0, 0.0])
    mock_img = MagicMock()

    captured = {}

    def _mock_doresample(orig_x, orig_y, new_x, antialias=False, **kwargs):
        return np.interp(new_x, orig_x, orig_y)

    def _mock_savetonifti(thearray, theheader, thename, **kwargs):
        captured["pixdim4"] = theheader["pixdim"][4]

    args = _make_default_args(outputtr=outputtr)

    with (
        patch(
            "rapidtide.workflows.resamplenifti.tide_io.fmritimeinfo",
            return_value=(inputtr, numinputtrs),
        ),
        patch(
            "rapidtide.workflows.resamplenifti.tide_io.readfromnifti",
            return_value=(mock_img, input_data, input_hdr, thedims, thesizes),
        ),
        patch(
            "rapidtide.workflows.resamplenifti.tide_resample.doresample",
            side_effect=_mock_doresample,
        ),
        patch(
            "rapidtide.workflows.resamplenifti.tide_io.niftisplitext",
            return_value=("output", ".nii.gz"),
        ),
        patch(
            "rapidtide.workflows.resamplenifti.tide_io.savetonifti",
            side_effect=_mock_savetonifti,
        ),
    ):
        resamplenifti(args)

    assert captured["pixdim4"] == outputtr


def resamplenifti_header_not_mutated(debug=False):
    """Verify the original header is not mutated (copy is used)."""
    if debug:
        print("resamplenifti_header_not_mutated")

    inputtr = 1.0
    numinputtrs = 10
    xsize, ysize, numslices = 1, 1, 1
    outputtr = 2.0

    input_data = _make_mock_nifti_data(xsize, ysize, numslices, numinputtrs, inputtr)
    input_hdr = _make_mock_header(pixdim=[0.0, 2.0, 2.0, 2.0, inputtr, 0.0, 0.0, 0.0])
    thedims = np.array([4, xsize, ysize, numslices, numinputtrs, 0, 0, 0])
    thesizes = np.array([0.0, 2.0, 2.0, 2.0, inputtr, 0.0, 0.0, 0.0])
    mock_img = MagicMock()

    def _mock_doresample(orig_x, orig_y, new_x, antialias=False, **kwargs):
        return np.interp(new_x, orig_x, orig_y)

    args = _make_default_args(outputtr=outputtr)

    with (
        patch(
            "rapidtide.workflows.resamplenifti.tide_io.fmritimeinfo",
            return_value=(inputtr, numinputtrs),
        ),
        patch(
            "rapidtide.workflows.resamplenifti.tide_io.readfromnifti",
            return_value=(mock_img, input_data, input_hdr, thedims, thesizes),
        ),
        patch(
            "rapidtide.workflows.resamplenifti.tide_resample.doresample",
            side_effect=_mock_doresample,
        ),
        patch(
            "rapidtide.workflows.resamplenifti.tide_io.niftisplitext",
            return_value=("output", ".nii.gz"),
        ),
        patch("rapidtide.workflows.resamplenifti.tide_io.savetonifti"),
    ):
        resamplenifti(args)

    # Original header should still have the original TR
    assert input_hdr["pixdim"][4] == inputtr


def resamplenifti_niftisplitext_called(debug=False):
    """Verify niftisplitext is called with the output filename."""
    if debug:
        print("resamplenifti_niftisplitext_called")

    inputtr = 1.0
    numinputtrs = 10
    xsize, ysize, numslices = 1, 1, 1

    input_data = _make_mock_nifti_data(xsize, ysize, numslices, numinputtrs, inputtr)
    input_hdr = _make_mock_header(pixdim=[0.0, 2.0, 2.0, 2.0, inputtr, 0.0, 0.0, 0.0])
    thedims = np.array([4, xsize, ysize, numslices, numinputtrs, 0, 0, 0])
    thesizes = np.array([0.0, 2.0, 2.0, 2.0, inputtr, 0.0, 0.0, 0.0])
    mock_img = MagicMock()

    def _mock_doresample(orig_x, orig_y, new_x, antialias=False, **kwargs):
        return np.interp(new_x, orig_x, orig_y)

    args = _make_default_args(outputfile="myoutput.nii.gz")

    mock_splitext = MagicMock(return_value=("myoutput", ".nii.gz"))

    with (
        patch(
            "rapidtide.workflows.resamplenifti.tide_io.fmritimeinfo",
            return_value=(inputtr, numinputtrs),
        ),
        patch(
            "rapidtide.workflows.resamplenifti.tide_io.readfromnifti",
            return_value=(mock_img, input_data, input_hdr, thedims, thesizes),
        ),
        patch(
            "rapidtide.workflows.resamplenifti.tide_resample.doresample",
            side_effect=_mock_doresample,
        ),
        patch(
            "rapidtide.workflows.resamplenifti.tide_io.niftisplitext",
            mock_splitext,
        ),
        patch("rapidtide.workflows.resamplenifti.tide_io.savetonifti"),
    ):
        resamplenifti(args)

    mock_splitext.assert_called_once_with("myoutput.nii.gz")


def resamplenifti_doresample_receives_correct_data(debug=False):
    """Verify that doresample receives correct input time axis and voxel data."""
    if debug:
        print("resamplenifti_doresample_receives_correct_data")

    inputtr = 1.0
    numinputtrs = 8
    xsize, ysize, numslices = 1, 1, 1
    outputtr = 2.0

    input_data = np.ones((xsize, ysize, numslices, numinputtrs), dtype=np.float64)
    input_data[0, 0, 0, :] = np.arange(numinputtrs, dtype=np.float64)

    input_hdr = _make_mock_header(pixdim=[0.0, 2.0, 2.0, 2.0, inputtr, 0.0, 0.0, 0.0])
    thedims = np.array([4, xsize, ysize, numslices, numinputtrs, 0, 0, 0])
    thesizes = np.array([0.0, 2.0, 2.0, 2.0, inputtr, 0.0, 0.0, 0.0])
    mock_img = MagicMock()

    captured = {}

    def _mock_doresample(orig_x, orig_y, new_x, antialias=False, **kwargs):
        captured["orig_x"] = orig_x.copy()
        captured["orig_y"] = orig_y.copy()
        captured["new_x"] = new_x.copy()
        return np.interp(new_x, orig_x, orig_y)

    args = _make_default_args(outputtr=outputtr)

    with (
        patch(
            "rapidtide.workflows.resamplenifti.tide_io.fmritimeinfo",
            return_value=(inputtr, numinputtrs),
        ),
        patch(
            "rapidtide.workflows.resamplenifti.tide_io.readfromnifti",
            return_value=(mock_img, input_data, input_hdr, thedims, thesizes),
        ),
        patch(
            "rapidtide.workflows.resamplenifti.tide_resample.doresample",
            side_effect=_mock_doresample,
        ),
        patch(
            "rapidtide.workflows.resamplenifti.tide_io.niftisplitext",
            return_value=("output", ".nii.gz"),
        ),
        patch("rapidtide.workflows.resamplenifti.tide_io.savetonifti"),
    ):
        resamplenifti(args)

    # Input x should be linspace(0, inputtr * numinputtrs, numinputtrs, endpoint=False)
    expected_orig_x = np.linspace(0.0, inputtr * numinputtrs, num=numinputtrs, endpoint=False)
    np.testing.assert_allclose(captured["orig_x"], expected_orig_x)

    # Input y should be the voxel's time series
    np.testing.assert_allclose(captured["orig_y"], np.arange(numinputtrs, dtype=np.float64))

    # Output x should start from 0 and be spaced by outputtr
    assert captured["new_x"][0] == 0.0
    assert np.allclose(np.diff(captured["new_x"]), outputtr)


def resamplenifti_debug_mode(debug=False):
    """Verify that debug mode runs without error."""
    if debug:
        print("resamplenifti_debug_mode")

    inputtr = 1.0
    numinputtrs = 5
    xsize, ysize, numslices = 1, 1, 1
    outputtr = 2.0

    input_data = _make_mock_nifti_data(xsize, ysize, numslices, numinputtrs, inputtr)
    input_hdr = _make_mock_header(pixdim=[0.0, 2.0, 2.0, 2.0, inputtr, 0.0, 0.0, 0.0])
    thedims = np.array([4, xsize, ysize, numslices, numinputtrs, 0, 0, 0])
    thesizes = np.array([0.0, 2.0, 2.0, 2.0, inputtr, 0.0, 0.0, 0.0])
    mock_img = MagicMock()

    def _mock_doresample(orig_x, orig_y, new_x, antialias=False, **kwargs):
        return np.interp(new_x, orig_x, orig_y)

    args = _make_default_args(outputtr=outputtr, debug=True)

    with (
        patch(
            "rapidtide.workflows.resamplenifti.tide_io.fmritimeinfo",
            return_value=(inputtr, numinputtrs),
        ),
        patch(
            "rapidtide.workflows.resamplenifti.tide_io.readfromnifti",
            return_value=(mock_img, input_data, input_hdr, thedims, thesizes),
        ),
        patch(
            "rapidtide.workflows.resamplenifti.tide_resample.doresample",
            side_effect=_mock_doresample,
        ),
        patch(
            "rapidtide.workflows.resamplenifti.tide_io.niftisplitext",
            return_value=("output", ".nii.gz"),
        ),
        patch("rapidtide.workflows.resamplenifti.tide_io.savetonifti"),
    ):
        # Should not raise
        resamplenifti(args)


def resamplenifti_multi_slice(debug=False):
    """Test with multiple slices to verify all voxels are processed."""
    if debug:
        print("resamplenifti_multi_slice")

    inputtr = 1.0
    numinputtrs = 10
    xsize, ysize, numslices = 2, 3, 4
    outputtr = 2.0

    input_data = _make_mock_nifti_data(xsize, ysize, numslices, numinputtrs, inputtr)
    input_hdr = _make_mock_header(pixdim=[0.0, 2.0, 2.0, 2.0, inputtr, 0.0, 0.0, 0.0])
    thedims = np.array([4, xsize, ysize, numslices, numinputtrs, 0, 0, 0])
    thesizes = np.array([0.0, 2.0, 2.0, 2.0, inputtr, 0.0, 0.0, 0.0])
    mock_img = MagicMock()

    call_count = {"n": 0}

    def _mock_doresample(orig_x, orig_y, new_x, antialias=False, **kwargs):
        call_count["n"] += 1
        return np.interp(new_x, orig_x, orig_y)

    args = _make_default_args(outputtr=outputtr)

    with (
        patch(
            "rapidtide.workflows.resamplenifti.tide_io.fmritimeinfo",
            return_value=(inputtr, numinputtrs),
        ),
        patch(
            "rapidtide.workflows.resamplenifti.tide_io.readfromnifti",
            return_value=(mock_img, input_data, input_hdr, thedims, thesizes),
        ),
        patch(
            "rapidtide.workflows.resamplenifti.tide_resample.doresample",
            side_effect=_mock_doresample,
        ),
        patch(
            "rapidtide.workflows.resamplenifti.tide_io.niftisplitext",
            return_value=("output", ".nii.gz"),
        ),
        patch("rapidtide.workflows.resamplenifti.tide_io.savetonifti"),
    ):
        resamplenifti(args)

    expected_calls = xsize * ysize * numslices
    assert call_count["n"] == expected_calls


def resamplenifti_same_tr(debug=False):
    """Test resampling to the same TR (should still work, no antialias override)."""
    if debug:
        print("resamplenifti_same_tr")

    inputtr = 1.0
    numinputtrs = 10
    xsize, ysize, numslices = 1, 1, 1
    outputtr = 1.0

    input_data = _make_mock_nifti_data(xsize, ysize, numslices, numinputtrs, inputtr)
    input_hdr = _make_mock_header(pixdim=[0.0, 2.0, 2.0, 2.0, inputtr, 0.0, 0.0, 0.0])
    thedims = np.array([4, xsize, ysize, numslices, numinputtrs, 0, 0, 0])
    thesizes = np.array([0.0, 2.0, 2.0, 2.0, inputtr, 0.0, 0.0, 0.0])
    mock_img = MagicMock()

    captured = {}

    def _mock_doresample(orig_x, orig_y, new_x, antialias=False, **kwargs):
        captured["antialias"] = antialias
        return np.interp(new_x, orig_x, orig_y)

    args = _make_default_args(outputtr=outputtr, antialias=True)

    with (
        patch(
            "rapidtide.workflows.resamplenifti.tide_io.fmritimeinfo",
            return_value=(inputtr, numinputtrs),
        ),
        patch(
            "rapidtide.workflows.resamplenifti.tide_io.readfromnifti",
            return_value=(mock_img, input_data, input_hdr, thedims, thesizes),
        ),
        patch(
            "rapidtide.workflows.resamplenifti.tide_resample.doresample",
            side_effect=_mock_doresample,
        ),
        patch(
            "rapidtide.workflows.resamplenifti.tide_io.niftisplitext",
            return_value=("output", ".nii.gz"),
        ),
        patch("rapidtide.workflows.resamplenifti.tide_io.savetonifti"),
    ):
        resamplenifti(args)

    # Same TR: inputtr is NOT > outputtr, so antialias should remain True
    assert captured["antialias"] is True


# ============================================================================
# Main test entry point
# ============================================================================


def test_resamplenifti(debug=False):
    # Parser tests
    parser_basic(debug=debug)
    parser_required_args(debug=debug)
    parser_defaults(debug=debug)
    parser_noantialias_flag(debug=debug)
    parser_normalize_flag(debug=debug)
    parser_debug_flag(debug=debug)
    parser_all_flags(debug=debug)
    parser_outputtr_type(debug=debug)

    # resamplenifti function tests
    resamplenifti_basic_downsample(debug=debug)
    resamplenifti_upsample_disables_antialias(debug=debug)
    resamplenifti_noantialias_flag(debug=debug)
    resamplenifti_output_shape(debug=debug)
    resamplenifti_header_updated(debug=debug)
    resamplenifti_header_not_mutated(debug=debug)
    resamplenifti_niftisplitext_called(debug=debug)
    resamplenifti_doresample_receives_correct_data(debug=debug)
    resamplenifti_debug_mode(debug=debug)
    resamplenifti_multi_slice(debug=debug)
    resamplenifti_same_tr(debug=debug)


if __name__ == "__main__":
    test_resamplenifti(debug=True)
