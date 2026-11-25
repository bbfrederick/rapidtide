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

from rapidtide.workflows.resamplenifti import _get_parser, resamplenifti

# ============================================================================
# Helpers
# ============================================================================


def _make_default_args(**overrides):
    """
    Create default args.

    Parameters
    ----------
    **overrides : object
        Keyword settings that replace the test defaults.

    Returns
    -------
    argparse.Namespace
        Arguments configured for the test case.
    """
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
    """
    Create a mock NIfTI header that supports item access and copy().

    Parameters
    ----------
    pixdim : sequence of float or None, optional
        Values for the NIfTI pixel dimension field.

    Returns
    -------
    MagicMock
        Mock object configured for the test case.
    """
    if pixdim is None:
        pixdim = [0.0, 2.0, 2.0, 2.0, 1.0, 0.0, 0.0, 0.0]
    storage = {"pixdim": list(pixdim)}

    hdr = MagicMock()
    hdr.__getitem__ = lambda self, key: storage[key]
    hdr.__setitem__ = lambda self, key, val: storage.__setitem__(key, val)
    hdr.copy_fn = lambda: _make_mock_header(pixdim=list(storage["pixdim"]))

    def _copy():
        """
        Provide the ``_copy`` test substitute.

        Returns
        -------
        object
            Value returned to the caller in this test case.
        """
        new_pixdim = list(storage["pixdim"])
        return _make_mock_header(pixdim=new_pixdim)

    hdr.copy = _copy
    return hdr


def _make_mock_nifti_data(xsize=2, ysize=2, numslices=1, timepoints=10, inputtr=1.0):
    """
    Create synthetic 4D data with a known signal.

    Parameters
    ----------
    xsize : int, optional
        Number of voxels along the x axis.
    ysize : int, optional
        Number of voxels along the y axis.
    numslices : int, optional
        Number of slices in the test image.
    timepoints : int, optional
        Number of time samples.
    inputtr : float, optional
        Input repetition time in seconds.

    Returns
    -------
    numpy.ndarray
        Output values supplied to the caller.
    """
    data = np.zeros((xsize, ysize, numslices, timepoints), dtype=np.float64)
    t = np.arange(timepoints) * inputtr
    for x in range(xsize):
        for y in range(ysize):
            for z in range(numslices):
                data[x, y, z, :] = np.sin(2 * np.pi * 0.1 * t) + x + y + z
    return data


def _run_resamplenifti_with_mocks(
    args,
    inputtr,
    numinputtrs,
    input_data,
    input_hdr,
    thedims,
    thesizes,
    doresample_side_effect,
    savetonifti_side_effect=None,
    niftisplitext_return=("output", ".nii.gz"),
    niftisplitext_mock=None,
):
    """
    Run resamplenifti with the common mock stack used in function tests.

    Parameters
    ----------
    args : object
        Arguments passed to the test helper.
    inputtr : float
        Input repetition time in seconds.
    numinputtrs : object
        Value supplied as ``numinputtrs`` by the test case.
    input_data : object
        Value supplied as ``input_data`` by the test case.
    input_hdr : object
        Value supplied as ``input_hdr`` by the test case.
    thedims : object
        Value supplied as ``thedims`` by the test case.
    thesizes : object
        Value supplied as ``thesizes`` by the test case.
    doresample_side_effect : object
        Value supplied as ``doresample_side_effect`` by the test case.
    savetonifti_side_effect : object or None, optional
        Value supplied as ``savetonifti_side_effect`` by the test case.
    niftisplitext_return : tuple, optional
        Value supplied as ``niftisplitext_return`` by the test case.
    niftisplitext_mock : object or None, optional
        Value supplied as ``niftisplitext_mock`` by the test case.

    Returns
    -------
    object
        The ``splitext_patch_target`` value supplied to the caller.
    """
    mock_img = MagicMock()
    splitext_patch_target = (
        niftisplitext_mock
        if niftisplitext_mock is not None
        else MagicMock(return_value=niftisplitext_return)
    )
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
            side_effect=doresample_side_effect,
        ),
        patch(
            "rapidtide.workflows.resamplenifti.tide_io.niftisplitext",
            splitext_patch_target,
        ),
        patch(
            "rapidtide.workflows.resamplenifti.tide_io.savetonifti",
            side_effect=savetonifti_side_effect,
        ),
    ):
        resamplenifti(args)
    return splitext_patch_target


# ============================================================================
# Parser tests
# ============================================================================


def parser_basic(debug=False):
    """
    Check that the command-line parser initializes.

    Parameters
    ----------
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("parser_basic")
    parser = _get_parser()
    assert parser is not None
    assert parser.prog == "resamplenifti"


def parser_required_args(debug=False):
    """
    Check that the parser requires the mandatory arguments.

    Parameters
    ----------
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("parser_required_args")
    parser = _get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def parser_defaults(debug=False):
    """
    Check the parser defaults.

    Parameters
    ----------
    debug : bool, optional
        Enable diagnostic output for the test.
    """
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
    """
    Check that the parser accepts ``--noantialias``.

    Parameters
    ----------
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("parser_noantialias_flag")
    parser = _get_parser()
    args = parser.parse_args(["in.nii", "out.nii", "1.0", "--noantialias"])
    assert args.antialias is False


def parser_normalize_flag(debug=False):
    """
    Check that the parser accepts ``--normalize``.

    Parameters
    ----------
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("parser_normalize_flag")
    parser = _get_parser()
    args = parser.parse_args(["in.nii", "out.nii", "1.0", "--normalize"])
    assert args.normalize is True


def parser_debug_flag(debug=False):
    """
    Check that the parser accepts ``--debug``.

    Parameters
    ----------
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("parser_debug_flag")
    parser = _get_parser()
    args = parser.parse_args(["in.nii", "out.nii", "1.0", "--debug"])
    assert args.debug is True


def parser_all_flags(debug=False):
    """
    Parse a command line with all optional flags enabled.

    Parameters
    ----------
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("parser_all_flags")
    parser = _get_parser()
    args = parser.parse_args(
        ["in.nii", "out.nii", "0.5", "--noantialias", "--normalize", "--debug"]
    )
    assert args.antialias is False
    assert args.normalize is True
    assert args.debug is True
    assert args.outputtr == 0.5


def parser_outputtr_type(debug=False):
    """
    Verify outputtr is parsed as float.

    Parameters
    ----------
    debug : bool, optional
        Enable diagnostic output for the test.
    """
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
    """
    Test basic downsampling (outputtr > inputtr): antialias stays on.

    Parameters
    ----------
    debug : bool, optional
        Enable diagnostic output for the test.
    """
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
    captured = {}

    def _mock_doresample(orig_x, orig_y, new_x, antialias=False, **kwargs):
        """
        Simulate ``doresample`` for the test.

        Parameters
        ----------
        orig_x : numpy.ndarray
            Sample coordinates on the original grid.
        orig_y : numpy.ndarray
            Values on the original grid.
        new_x : numpy.ndarray
            Sample coordinates on the destination grid.
        antialias : bool, optional
            Apply antialiasing when downsampling.
        **kwargs : object
            Additional keyword arguments accepted by the stub.

        Returns
        -------
        numpy.ndarray
            Array supplied to the test or its caller.
        """
        captured.setdefault("calls", []).append(
            {
                "orig_x": orig_x.copy(),
                "orig_y": orig_y.copy(),
                "new_x": new_x.copy(),
                "antialias": antialias,
            }
        )
        return np.interp(new_x, orig_x, orig_y)

    def _mock_savetonifti(thearray, theheader, thename, **kwargs):
        """
        Simulate ``savetonifti`` for the test.

        Parameters
        ----------
        thearray : object
            Value supplied as ``thearray`` by the test case.
        theheader : object
            Value supplied as ``theheader`` by the test case.
        thename : object
            Value supplied as ``thename`` by the test case.
        **kwargs : object
            Additional keyword arguments accepted by the stub.
        """
        captured["saved_array"] = thearray.copy()
        captured["saved_header"] = theheader
        captured["saved_name"] = thename

    args = _make_default_args(outputtr=outputtr)

    _run_resamplenifti_with_mocks(
        args=args,
        inputtr=inputtr,
        numinputtrs=numinputtrs,
        input_data=input_data,
        input_hdr=input_hdr,
        thedims=thedims,
        thesizes=thesizes,
        doresample_side_effect=_mock_doresample,
        savetonifti_side_effect=_mock_savetonifti,
    )

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
    """
    Test that upsampling (inputtr > outputtr) disables antialias.

    Parameters
    ----------
    debug : bool, optional
        Enable diagnostic output for the test.
    """
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
    captured = {}

    def _mock_doresample(orig_x, orig_y, new_x, antialias=False, **kwargs):
        """
        Simulate ``doresample`` for the test.

        Parameters
        ----------
        orig_x : numpy.ndarray
            Sample coordinates on the original grid.
        orig_y : numpy.ndarray
            Values on the original grid.
        new_x : numpy.ndarray
            Sample coordinates on the destination grid.
        antialias : bool, optional
            Apply antialiasing when downsampling.
        **kwargs : object
            Additional keyword arguments accepted by the stub.

        Returns
        -------
        numpy.ndarray
            Array supplied to the test or its caller.
        """
        captured["antialias"] = antialias
        return np.interp(new_x, orig_x, orig_y)

    args = _make_default_args(outputtr=outputtr, antialias=True)

    _run_resamplenifti_with_mocks(
        args=args,
        inputtr=inputtr,
        numinputtrs=numinputtrs,
        input_data=input_data,
        input_hdr=input_hdr,
        thedims=thedims,
        thesizes=thesizes,
        doresample_side_effect=_mock_doresample,
    )

    # Upsampling should force antialias off
    assert captured["antialias"] is False


def resamplenifti_noantialias_flag(debug=False):
    """
    Test that --noantialias flag is respected during downsampling.

    Parameters
    ----------
    debug : bool, optional
        Enable diagnostic output for the test.
    """
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
    captured = {}

    def _mock_doresample(orig_x, orig_y, new_x, antialias=False, **kwargs):
        """
        Simulate ``doresample`` for the test.

        Parameters
        ----------
        orig_x : numpy.ndarray
            Sample coordinates on the original grid.
        orig_y : numpy.ndarray
            Values on the original grid.
        new_x : numpy.ndarray
            Sample coordinates on the destination grid.
        antialias : bool, optional
            Apply antialiasing when downsampling.
        **kwargs : object
            Additional keyword arguments accepted by the stub.

        Returns
        -------
        numpy.ndarray
            Array supplied to the test or its caller.
        """
        captured["antialias"] = antialias
        return np.interp(new_x, orig_x, orig_y)

    args = _make_default_args(outputtr=outputtr, antialias=False)

    _run_resamplenifti_with_mocks(
        args=args,
        inputtr=inputtr,
        numinputtrs=numinputtrs,
        input_data=input_data,
        input_hdr=input_hdr,
        thedims=thedims,
        thesizes=thesizes,
        doresample_side_effect=_mock_doresample,
    )

    assert captured["antialias"] is False


def resamplenifti_output_shape(debug=False):
    """
    Verify the output array has the correct shape based on outputtr.

    Parameters
    ----------
    debug : bool, optional
        Enable diagnostic output for the test.
    """
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
    captured = {}

    def _mock_doresample(orig_x, orig_y, new_x, antialias=False, **kwargs):
        """
        Simulate ``doresample`` for the test.

        Parameters
        ----------
        orig_x : numpy.ndarray
            Sample coordinates on the original grid.
        orig_y : numpy.ndarray
            Values on the original grid.
        new_x : numpy.ndarray
            Sample coordinates on the destination grid.
        antialias : bool, optional
            Apply antialiasing when downsampling.
        **kwargs : object
            Additional keyword arguments accepted by the stub.

        Returns
        -------
        numpy.ndarray
            Array supplied to the test or its caller.
        """
        return np.interp(new_x, orig_x, orig_y)

    def _mock_savetonifti(thearray, theheader, thename, **kwargs):
        """
        Simulate ``savetonifti`` for the test.

        Parameters
        ----------
        thearray : object
            Value supplied as ``thearray`` by the test case.
        theheader : object
            Value supplied as ``theheader`` by the test case.
        thename : object
            Value supplied as ``thename`` by the test case.
        **kwargs : object
            Additional keyword arguments accepted by the stub.
        """
        captured["shape"] = thearray.shape

    args = _make_default_args(outputtr=outputtr)

    _run_resamplenifti_with_mocks(
        args=args,
        inputtr=inputtr,
        numinputtrs=numinputtrs,
        input_data=input_data,
        input_hdr=input_hdr,
        thedims=thedims,
        thesizes=thesizes,
        doresample_side_effect=_mock_doresample,
        savetonifti_side_effect=_mock_savetonifti,
    )

    # Calculate expected number of output timepoints using the same formula as the code
    inputendtime = inputtr * (numinputtrs - 1)
    numoutputtrs = int(np.ceil(inputendtime / outputtr) + 1)
    assert captured["shape"] == (xsize, ysize, numslices, numoutputtrs)


def resamplenifti_header_updated(debug=False):
    """
    Verify the output header pixdim[4] is set to the new TR.

    Parameters
    ----------
    debug : bool, optional
        Enable diagnostic output for the test.
    """
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
    captured = {}

    def _mock_doresample(orig_x, orig_y, new_x, antialias=False, **kwargs):
        """
        Simulate ``doresample`` for the test.

        Parameters
        ----------
        orig_x : numpy.ndarray
            Sample coordinates on the original grid.
        orig_y : numpy.ndarray
            Values on the original grid.
        new_x : numpy.ndarray
            Sample coordinates on the destination grid.
        antialias : bool, optional
            Apply antialiasing when downsampling.
        **kwargs : object
            Additional keyword arguments accepted by the stub.

        Returns
        -------
        numpy.ndarray
            Array supplied to the test or its caller.
        """
        return np.interp(new_x, orig_x, orig_y)

    def _mock_savetonifti(thearray, theheader, thename, **kwargs):
        """
        Simulate ``savetonifti`` for the test.

        Parameters
        ----------
        thearray : object
            Value supplied as ``thearray`` by the test case.
        theheader : object
            Value supplied as ``theheader`` by the test case.
        thename : object
            Value supplied as ``thename`` by the test case.
        **kwargs : object
            Additional keyword arguments accepted by the stub.
        """
        captured["pixdim4"] = theheader["pixdim"][4]

    args = _make_default_args(outputtr=outputtr)

    _run_resamplenifti_with_mocks(
        args=args,
        inputtr=inputtr,
        numinputtrs=numinputtrs,
        input_data=input_data,
        input_hdr=input_hdr,
        thedims=thedims,
        thesizes=thesizes,
        doresample_side_effect=_mock_doresample,
        savetonifti_side_effect=_mock_savetonifti,
    )

    assert captured["pixdim4"] == outputtr


def resamplenifti_header_not_mutated(debug=False):
    """
    Verify the original header is not mutated (copy is used).

    Parameters
    ----------
    debug : bool, optional
        Enable diagnostic output for the test.
    """
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

    def _mock_doresample(orig_x, orig_y, new_x, antialias=False, **kwargs):
        """
        Simulate ``doresample`` for the test.

        Parameters
        ----------
        orig_x : numpy.ndarray
            Sample coordinates on the original grid.
        orig_y : numpy.ndarray
            Values on the original grid.
        new_x : numpy.ndarray
            Sample coordinates on the destination grid.
        antialias : bool, optional
            Apply antialiasing when downsampling.
        **kwargs : object
            Additional keyword arguments accepted by the stub.

        Returns
        -------
        numpy.ndarray
            Array supplied to the test or its caller.
        """
        return np.interp(new_x, orig_x, orig_y)

    args = _make_default_args(outputtr=outputtr)

    _run_resamplenifti_with_mocks(
        args=args,
        inputtr=inputtr,
        numinputtrs=numinputtrs,
        input_data=input_data,
        input_hdr=input_hdr,
        thedims=thedims,
        thesizes=thesizes,
        doresample_side_effect=_mock_doresample,
    )

    # Original header should still have the original TR
    assert input_hdr["pixdim"][4] == inputtr


def resamplenifti_niftisplitext_called(debug=False):
    """
    Verify niftisplitext is called with the output filename.

    Parameters
    ----------
    debug : bool, optional
        Enable diagnostic output for the test.
    """
    if debug:
        print("resamplenifti_niftisplitext_called")

    inputtr = 1.0
    numinputtrs = 10
    xsize, ysize, numslices = 1, 1, 1

    input_data = _make_mock_nifti_data(xsize, ysize, numslices, numinputtrs, inputtr)
    input_hdr = _make_mock_header(pixdim=[0.0, 2.0, 2.0, 2.0, inputtr, 0.0, 0.0, 0.0])
    thedims = np.array([4, xsize, ysize, numslices, numinputtrs, 0, 0, 0])
    thesizes = np.array([0.0, 2.0, 2.0, 2.0, inputtr, 0.0, 0.0, 0.0])

    def _mock_doresample(orig_x, orig_y, new_x, antialias=False, **kwargs):
        """
        Simulate ``doresample`` for the test.

        Parameters
        ----------
        orig_x : numpy.ndarray
            Sample coordinates on the original grid.
        orig_y : numpy.ndarray
            Values on the original grid.
        new_x : numpy.ndarray
            Sample coordinates on the destination grid.
        antialias : bool, optional
            Apply antialiasing when downsampling.
        **kwargs : object
            Additional keyword arguments accepted by the stub.

        Returns
        -------
        numpy.ndarray
            Array supplied to the test or its caller.
        """
        return np.interp(new_x, orig_x, orig_y)

    args = _make_default_args(outputfile="myoutput.nii.gz")

    mock_splitext = MagicMock(return_value=("myoutput", ".nii.gz"))

    _run_resamplenifti_with_mocks(
        args=args,
        inputtr=inputtr,
        numinputtrs=numinputtrs,
        input_data=input_data,
        input_hdr=input_hdr,
        thedims=thedims,
        thesizes=thesizes,
        doresample_side_effect=_mock_doresample,
        niftisplitext_mock=mock_splitext,
    )

    mock_splitext.assert_called_once_with("myoutput.nii.gz")


def resamplenifti_doresample_receives_correct_data(debug=False):
    """
    Verify that doresample receives correct input time axis and voxel data.

    Parameters
    ----------
    debug : bool, optional
        Enable diagnostic output for the test.
    """
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
    captured = {}

    def _mock_doresample(orig_x, orig_y, new_x, antialias=False, **kwargs):
        """
        Simulate ``doresample`` for the test.

        Parameters
        ----------
        orig_x : numpy.ndarray
            Sample coordinates on the original grid.
        orig_y : numpy.ndarray
            Values on the original grid.
        new_x : numpy.ndarray
            Sample coordinates on the destination grid.
        antialias : bool, optional
            Apply antialiasing when downsampling.
        **kwargs : object
            Additional keyword arguments accepted by the stub.

        Returns
        -------
        numpy.ndarray
            Array supplied to the test or its caller.
        """
        captured["orig_x"] = orig_x.copy()
        captured["orig_y"] = orig_y.copy()
        captured["new_x"] = new_x.copy()
        return np.interp(new_x, orig_x, orig_y)

    args = _make_default_args(outputtr=outputtr)

    _run_resamplenifti_with_mocks(
        args=args,
        inputtr=inputtr,
        numinputtrs=numinputtrs,
        input_data=input_data,
        input_hdr=input_hdr,
        thedims=thedims,
        thesizes=thesizes,
        doresample_side_effect=_mock_doresample,
    )

    # Input x should be linspace(0, inputtr * numinputtrs, numinputtrs, endpoint=False)
    expected_orig_x = np.linspace(0.0, inputtr * numinputtrs, num=numinputtrs, endpoint=False)
    np.testing.assert_allclose(captured["orig_x"], expected_orig_x)

    # Input y should be the voxel's time series
    np.testing.assert_allclose(captured["orig_y"], np.arange(numinputtrs, dtype=np.float64))

    # Output x should start from 0 and be spaced by outputtr
    assert captured["new_x"][0] == 0.0
    assert np.allclose(np.diff(captured["new_x"]), outputtr)


def resamplenifti_debug_mode(debug=False):
    """
    Verify that debug mode runs without error.

    Parameters
    ----------
    debug : bool, optional
        Enable diagnostic output for the test.
    """
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

    def _mock_doresample(orig_x, orig_y, new_x, antialias=False, **kwargs):
        """
        Simulate ``doresample`` for the test.

        Parameters
        ----------
        orig_x : numpy.ndarray
            Sample coordinates on the original grid.
        orig_y : numpy.ndarray
            Values on the original grid.
        new_x : numpy.ndarray
            Sample coordinates on the destination grid.
        antialias : bool, optional
            Apply antialiasing when downsampling.
        **kwargs : object
            Additional keyword arguments accepted by the stub.

        Returns
        -------
        numpy.ndarray
            Array supplied to the test or its caller.
        """
        return np.interp(new_x, orig_x, orig_y)

    args = _make_default_args(outputtr=outputtr, debug=True)

    # Should not raise
    _run_resamplenifti_with_mocks(
        args=args,
        inputtr=inputtr,
        numinputtrs=numinputtrs,
        input_data=input_data,
        input_hdr=input_hdr,
        thedims=thedims,
        thesizes=thesizes,
        doresample_side_effect=_mock_doresample,
    )


def resamplenifti_multi_slice(debug=False):
    """
    Test with multiple slices to verify all voxels are processed.

    Parameters
    ----------
    debug : bool, optional
        Enable diagnostic output for the test.
    """
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
    call_count = {"n": 0}

    def _mock_doresample(orig_x, orig_y, new_x, antialias=False, **kwargs):
        """
        Simulate ``doresample`` for the test.

        Parameters
        ----------
        orig_x : numpy.ndarray
            Sample coordinates on the original grid.
        orig_y : numpy.ndarray
            Values on the original grid.
        new_x : numpy.ndarray
            Sample coordinates on the destination grid.
        antialias : bool, optional
            Apply antialiasing when downsampling.
        **kwargs : object
            Additional keyword arguments accepted by the stub.

        Returns
        -------
        numpy.ndarray
            Array supplied to the test or its caller.
        """
        call_count["n"] += 1
        return np.interp(new_x, orig_x, orig_y)

    args = _make_default_args(outputtr=outputtr)

    _run_resamplenifti_with_mocks(
        args=args,
        inputtr=inputtr,
        numinputtrs=numinputtrs,
        input_data=input_data,
        input_hdr=input_hdr,
        thedims=thedims,
        thesizes=thesizes,
        doresample_side_effect=_mock_doresample,
    )

    expected_calls = xsize * ysize * numslices
    assert call_count["n"] == expected_calls


def resamplenifti_same_tr(debug=False):
    """
    Test resampling to the same TR (should still work, no antialias override).

    Parameters
    ----------
    debug : bool, optional
        Enable diagnostic output for the test.
    """
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
    captured = {}

    def _mock_doresample(orig_x, orig_y, new_x, antialias=False, **kwargs):
        """
        Simulate ``doresample`` for the test.

        Parameters
        ----------
        orig_x : numpy.ndarray
            Sample coordinates on the original grid.
        orig_y : numpy.ndarray
            Values on the original grid.
        new_x : numpy.ndarray
            Sample coordinates on the destination grid.
        antialias : bool, optional
            Apply antialiasing when downsampling.
        **kwargs : object
            Additional keyword arguments accepted by the stub.

        Returns
        -------
        numpy.ndarray
            Array supplied to the test or its caller.
        """
        captured["antialias"] = antialias
        return np.interp(new_x, orig_x, orig_y)

    args = _make_default_args(outputtr=outputtr, antialias=True)

    _run_resamplenifti_with_mocks(
        args=args,
        inputtr=inputtr,
        numinputtrs=numinputtrs,
        input_data=input_data,
        input_hdr=input_hdr,
        thedims=thedims,
        thesizes=thesizes,
        doresample_side_effect=_mock_doresample,
    )

    # Same TR: inputtr is NOT > outputtr, so antialias should remain True
    assert captured["antialias"] is True


# ============================================================================
# Main test entry point
# ============================================================================


PARSER_CASES = [
    parser_basic,
    parser_required_args,
    parser_defaults,
    parser_noantialias_flag,
    parser_normalize_flag,
    parser_debug_flag,
    parser_all_flags,
    parser_outputtr_type,
]


WORKFLOW_CASES = [
    resamplenifti_basic_downsample,
    resamplenifti_upsample_disables_antialias,
    resamplenifti_noantialias_flag,
    resamplenifti_output_shape,
    resamplenifti_header_updated,
    resamplenifti_header_not_mutated,
    resamplenifti_niftisplitext_called,
    resamplenifti_doresample_receives_correct_data,
    resamplenifti_debug_mode,
    resamplenifti_multi_slice,
    resamplenifti_same_tr,
]


@pytest.mark.parametrize("case_func", PARSER_CASES, ids=lambda func: func.__name__)
@pytest.mark.unit
def test_resamplenifti_parser_cases(case_runner, case_func):
    """
    Verify resamplenifti parser cases.

    Parameters
    ----------
    case_runner : callable
        Run the selected test case with the test fixtures.
    case_func : callable
        Test case to execute.
    """
    case_runner(case_func)


@pytest.mark.parametrize("case_func", WORKFLOW_CASES, ids=lambda func: func.__name__)
@pytest.mark.slow
def test_resamplenifti_workflow_cases(case_runner, case_func):
    """
    Verify resamplenifti workflow cases.

    Parameters
    ----------
    case_runner : callable
        Run the selected test case with the test fixtures.
    case_func : callable
        Test case to execute.
    """
    case_runner(case_func)


if __name__ == "__main__":
    for case in PARSER_CASES + WORKFLOW_CASES:
        case(debug=True)
