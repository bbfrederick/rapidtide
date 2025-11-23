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
from scipy.stats import rankdata

import rapidtide.io as tide_io


def _get_parser() -> Any:
    """
    Argument parser for rankimage.

    Creates and configures an argument parser for the rankimage tool that converts
    3D or 4D nifti images into percentile maps.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser object with all required and optional arguments
        for the rankimage tool.

    Notes
    -----
    The parser is configured with:
    - Required arguments: inputfilename, maskfilename, outputroot
    - Optional argument: --debug flag for additional debugging information

    Examples
    --------
    >>> parser = _get_parser()
    >>> args = parser.parse_args(['input.nii', 'mask.nii', 'output_root'])
    >>> print(args.inputfilename)
    'input.nii'
    """
    parser = argparse.ArgumentParser(
        prog="rankimage",
        description=("Convert a 3D or 4D nifti image into a percentile map."),
        allow_abbrev=False,
    )

    # Required arguments
    parser.add_argument("inputfilename", type=str, help="The name of the input image nifti file.")
    parser.add_argument("maskfilename", type=str, help="The name of the input mask nifti file.")
    parser.add_argument("outputroot", type=str, help="The root name of the output files.")

    # add optional arguments
    parser.add_argument(
        "--debug",
        action="store_true",
        help=("Output additional debugging information."),
        default=False,
    )

    return parser


def imtopercentile(image: Any, mask: Any, debug: bool = False) -> NDArray:
    """
    Convert image values to percentile scores within masked region.

    This function computes percentile rankings for all voxels within a specified
    mask region of a 3D image. The percentile scores are calculated using the
    'dense' ranking method, where tied values receive the same rank.

    Parameters
    ----------
    image : array-like
        Input 3D image data array
    mask : array-like
        Binary mask defining the region of interest (values > 0 are considered valid)
    debug : bool, optional
        If True, print debugging information about processing (default is False)

    Returns
    -------
    NDArray
        Array of same shape as input image containing percentile scores for
        valid voxels, with zeros for masked-out voxels

    Notes
    -----
    - Only voxels where mask > 0 are processed
    - Percentile calculation uses 'dense' ranking method
    - The percentile score ranges from 0 to 100
    - Invalid voxels (where mask <= 0) are set to 0 in output

    Examples
    --------
    >>> import numpy as np
    >>> image = np.random.rand(10, 10, 10)
    >>> mask = np.ones((10, 10, 10))
    >>> mask[0:5, :, :] = 0  # Mask out first 5 slices
    >>> result = imtopercentile(image, mask)
    """
    outmaparray = np.zeros_like(image)
    nativespaceshape = image.shape
    validvoxels = np.where(mask > 0)
    numvalidspatiallocs = np.shape(validvoxels[0])[0]
    input_data_valid = image[validvoxels].copy()

    if debug:
        print(
            f"Processing {numvalidspatiallocs} of {nativespaceshape[0] * nativespaceshape[1] * nativespaceshape[2]} voxels"
        )
        print(f"{np.shape(input_data_valid)=}")
    percentilescore = (
        100.0 * (rankdata(input_data_valid, method="dense") - 1) / (numvalidspatiallocs - 1)
    )
    outmaparray[validvoxels] = percentilescore[:]
    return outmaparray.reshape(nativespaceshape)


def rankimage(args: Any) -> None:
    """
    Convert input NIfTI image data to percentile-ranked values using a mask.

    This function reads an input NIfTI image and a corresponding mask, checks
    that their spatial dimensions match, and computes percentile-ranked values
    for each voxel within the mask. The result is saved as a new NIfTI file with
    associated BIDS metadata.

    Parameters
    ----------
    args : Any
        An object containing the following attributes:
        - inputfilename : str
            Path to the input NIfTI image file.
        - maskfilename : str
            Path to the mask NIfTI file.
        - outputroot : str
            Root name for the output NIfTI file (without extension).
        - debug : bool, optional
            If True, enables debug output. Default is False.

    Returns
    -------
    None
        The function writes the percentile-ranked image to a NIfTI file and
        a JSON metadata file, but does not return any value.

    Notes
    -----
    - The function supports both 3D and 4D NIfTI files.
    - For 4D files, the percentile ranking is computed separately for each
      timepoint.
    - The output file is saved with the extension `.nii.gz`.
    - BIDS metadata is saved in a `.json` file with the same root name.

    Examples
    --------
    >>> class Args:
    ...     inputfilename = "input.nii.gz"
    ...     maskfilename = "mask.nii.gz"
    ...     outputroot = "output"
    ...     debug = False
    >>> args = Args()
    >>> rankimage(args)
    """
    input_img, input_data, input_hdr, thedims, thesizes = tide_io.readfromnifti(args.inputfilename)
    (
        mask_img,
        mask_data,
        mask_hdr,
        themaskdims,
        themasksizes,
    ) = tide_io.readfromnifti(args.maskfilename)

    if not tide_io.checkspacedimmatch(thedims, themaskdims):
        print("input image 1 dimensions do not match mask")
        exit()

    # select the valid voxels
    xsize, ysize, numslices, timepoints = tide_io.parseniftisizes(thedims)
    if timepoints > 1:
        is4d = True
    else:
        is4d = False

    if is4d:
        print("processing 4D nifti file")
        percentiles = np.zeros_like(input_data)
        for i in range(timepoints):
            percentiles[:, :, :, i] = imtopercentile(
                input_data[:, :, :, i], mask_data[:, :, :, i], debug=args.debug
            )
    else:
        print("processing 3D nifti file")
        percentiles = imtopercentile(input_data, mask_data, debug=args.debug)

    savename = args.outputroot
    bidsdict = {
        "RawSources": [args.inputfilename, args.maskfilename],
        "Units": "percentile",
    }
    tide_io.writedicttojson(bidsdict, f"{savename}.json")
    tide_io.savetonifti(percentiles, input_hdr, savename)
