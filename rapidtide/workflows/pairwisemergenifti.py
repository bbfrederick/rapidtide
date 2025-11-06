#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2016-2025 Blaise Frederick
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
from argparse import Namespace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

import rapidtide.io as tide_io


def _get_parser() -> Any:
    """
    Create and configure argument parser for pairwisemergenifti tool.

    This function initializes an ArgumentParser object with specific arguments
    required for merging adjacent timepoints in NIfTI files.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser with the following arguments:
        - inputfile: The name of the input NIfTI file, including extension
        - inputmask: The name of the mask NIfTI file, including extension
        - outputfile: The name of the output NIfTI file, including extension
        - --maskmerge: Input is a mask (boolean flag)
        - --debug: Print debugging information (boolean flag)

    Notes
    -----
    The parser is configured with:
    - prog="pairwisemergenifti"
    - description="Merges adjacent timepoints in a nifti file."
    - allow_abbrev=False

    Examples
    --------
    >>> parser = _get_parser()
    >>> args = parser.parse_args(['input.nii', 'mask.nii', 'output.nii'])
    >>> print(args.inputfile)
    'input.nii'
    """
    parser = argparse.ArgumentParser(
        prog="pairwisemergenifti",
        description="Merges adjacent timepoints in a nifti file.",
        allow_abbrev=False,
    )

    parser.add_argument("inputfile", help="The name of the input nifti file, including extension")
    parser.add_argument("inputmask", help="The name of the mask nifti file, including extension")
    parser.add_argument(
        "outputfile", help="The name of the output nifti file, including extension"
    )
    parser.add_argument(
        "--maskmerge",
        action="store_true",
        help="Input is a mask",
        default=False,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debugging information",
        default=False,
    )
    return parser


def pairwisemergenifti(args: Any) -> None:
    """
    Pairwise merge NIfTI time series data based on a mask.

    This function reads input NIfTI image and mask files, performs pairwise
    averaging of time series data, and saves the result as a new NIfTI file.
    The input image is expected to have an even number of time points, which
    are merged in pairs. The mask is used to determine which voxels contribute
    to the averaging.

    Parameters
    ----------
    args : Any
        An object containing the following attributes:
        - inputfile : str
            Path to the input NIfTI image file.
        - inputmask : str
            Path to the input NIfTI mask file.
        - outputfile : str
            Path to the output NIfTI file.
        - maskmerge : bool
            If True, the mask is binarized before merging. If False, the
            average of the input data is computed using the mask.

    Returns
    -------
    None
        The function does not return a value but saves the merged data to a
        NIfTI file.

    Notes
    -----
    - The function requires that the spatial dimensions of the input image
      and mask match.
    - The time dimension of the input image must be even.
    - If `maskmerge` is True, the mask is converted to binary (0 or 1) before
      merging.
    - If `maskmerge` is False, the average of the input data is computed
      using the mask to weight the voxels.

    Examples
    --------
    >>> import argparse
    >>> args = argparse.Namespace(
    ...     inputfile='input.nii.gz',
    ...     inputmask='mask.nii.gz',
    ...     outputfile='output.nii.gz',
    ...     maskmerge=False
    ... )
    >>> pairwisemergenifti(args)
    """
    print("reading input data")
    input_img, input_data, input_hdr, thedims, thesizes = tide_io.readfromnifti(args.inputfile)
    print("reading input mask")
    mask_img, mask_data, mask_hdr, themaskdims, themasksizes = tide_io.readfromnifti(
        args.inputmask
    )

    xsize, ysize, numslices, timepoints = tide_io.parseniftidims(thedims)
    numspatiallocs = int(xsize) * int(ysize) * int(numslices)

    # check dimensions
    if not tide_io.checkspacedimmatch(thedims, themaskdims):
        print("input mask spatial dimensions do not match image")
        exit()
    if not tide_io.checktimematch(thedims, themaskdims):
        print("input mask time dimension does not match image")
        exit()
    if timepoints % 2 != 0:
        print("input file must have an even number in the time dimension")
        exit()

    # sanitize input
    print("sanitizing input data")
    input_data = np.nan_to_num(input_data)
    print("sanitizing mask data")
    mask_data = np.nan_to_num(mask_data)

    # make the output array
    output_data = np.zeros((numspatiallocs, timepoints // 2), dtype="float")

    # cycle over timepoints
    print("now cycling over all images")
    for thepair in range(0, timepoints // 2):
        print(f"processing pair {thepair}")
        masksum = (mask_data[:, :, :, thepair * 2] + mask_data[:, :, :, thepair * 2 + 1]).reshape(
            numspatiallocs
        )
        if args.maskmerge:
            masksum[np.where(masksum > 0)] = 1
            output_data[:, thepair] = masksum + 0.0
        else:
            dataavg = (
                input_data[:, :, :, thepair * 2] + input_data[:, :, :, thepair * 2 + 1]
            ).reshape(numspatiallocs)
            dataavg = np.divide(dataavg, masksum, out=np.zeros_like(dataavg), where=masksum > 0)
            output_data[:, thepair] = np.nan_to_num(dataavg)

    # now do the ones with other numbers of time points
    output_hdr = input_hdr.copy()
    output_hdr["pixdim"][4] = timepoints // 2
    outputroot, dummy = tide_io.niftisplitext(args.outputfile)
    tide_io.savetonifti(
        output_data.reshape((xsize, ysize, numslices, timepoints // 2)), output_hdr, outputroot
    )
