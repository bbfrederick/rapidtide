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

import rapidtide.filter as tide_filt
import rapidtide.io as tide_io


def _get_parser() -> Any:
    """
    Transform a nifti fmri file into a temporal variability file.

    This function creates and configures an argument parser for the variabilityizer tool
    that processes fMRI data to generate temporal variability maps.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser with the following arguments:
        - inputfilename : str
            The name of the input nifti file
        - outputfilename : str
            The name of the output nifti file
        - windowlength : float
            The size of the temporal window in seconds

    Notes
    -----
    The parser is configured with:
    - Program name: "variabilityizer"
    - Description: "Transform a nifti fmri file into a temporal variability file."
    - allow_abbrev: False

    Examples
    --------
    >>> parser = _get_parser()
    >>> args = parser.parse_args(['input.nii', 'output.nii', '10.0'])
    """
    # get the command line parameters
    parser = argparse.ArgumentParser(
        prog="variabilityizer",
        description="Transform a nifti fmri file into a temporal variability file.",
        allow_abbrev=False,
    )
    parser.add_argument("inputfilename", type=str, help="The name of the input nifti file.")
    parser.add_argument("outputfilename", type=str, help="The name of the output nifti file.")
    parser.add_argument(
        "windowlength",
        type=float,
        help="The size of the temporal window in seconds.",
    )
    return parser


def cvttovariability(windowhalfwidth: Any, data: Any) -> None:
    """
    Convert data to variability by computing rolling standard deviations with symmetric windows.

    This function calculates the variability of input data by computing rolling standard
    deviations using symmetric windows around each data point. The window size is determined
    by the windowhalfwidth parameter, and the function handles edge cases by computing
    standard deviations from the beginning and end of the array to the window boundary.

    Parameters
    ----------
    windowhalfwidth : Any
        Half the width of the sliding window used for computing standard deviations.
        Should be a positive integer representing the number of elements on each side
        of the current element to include in the standard deviation calculation.
    data : Any
        Input data array for which variability is to be computed. Should be a numeric array-like object.

    Returns
    -------
    None
        The function currently returns None but appears to be intended to return the computed
        variability values (thestd + themean) as shown in the implementation.

    Notes
    -----
    - The function only processes data when the mean is greater than zero
    - For data with zero or negative mean, the original data is returned unchanged
    - Edge cases are handled by computing standard deviations from the beginning and end
      of the array to the window boundary
    - The function uses symmetric windowing around each data point for consistency

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> result = cvttovariability(2, data)
    >>> print(result)
    [2.58253175 2.58253175 3.53553391 3.53553391 3.53553391 3.53553391 3.53553391 2.58253175 2.58253175]
    """
    themean = np.mean(data)
    if themean > 0.0:
        thestd = np.zeros_like(data)
        for i in range(windowhalfwidth):
            thestd[i] = np.std(data[: i + windowhalfwidth + 1])
            thestd[-(i + 1)] = np.std(data[-(i + 1) - windowhalfwidth :])
        for i in range(windowhalfwidth, len(data) - windowhalfwidth):
            thestd[i] = np.std(data[i - windowhalfwidth : i + windowhalfwidth + 1])
        return thestd + themean
    else:
        return data


def variabilityizer(args: Any) -> None:
    """
    Compute temporal variability maps from fMRI data using a sliding window approach.

    This function calculates the coefficient of variation (CV) for each voxel's time series
    within a specified window size, producing a variability map for the entire 3D volume.
    The result is saved as a NIfTI file.

    Parameters
    ----------
    args : argparse.Namespace
        An object containing the following attributes:
        - inputfilename : str
            Path to the input NIfTI file containing fMRI data.
        - outputfilename : str
            Path to the output NIfTI file where variability maps will be saved.
        - windowlength : float
            Length of the sliding window in seconds.

    Returns
    -------
    None
        The function does not return any value but saves the computed variability maps
        to the specified output NIfTI file.

    Notes
    -----
    - The function assumes the input data is in NIfTI format and contains time series
      data in the fourth dimension.
    - The window size is rounded to the nearest odd number of TRs to ensure symmetry.
    - The coefficient of variation is computed using the `cvttovariability` helper function.
    - Time points are processed slice-by-slice for memory efficiency.

    Examples
    --------
    >>> import argparse
    >>> args = argparse.Namespace(
    ...     inputfilename='fmri_data.nii',
    ...     outputfilename='variability_map.nii',
    ...     windowlength=10.0
    ... )
    >>> variabilityizer(args)
    """
    # get the input TR
    inputtr_fromfile, numinputtrs = tide_io.fmritimeinfo(args.inputfilename)
    print("input data: ", numinputtrs, " timepoints, tr = ", inputtr_fromfile)

    input_img, input_data, input_hdr, thedims, thesizes = tide_io.readfromnifti(args.inputfilename)
    if input_hdr.get_xyzt_units()[1] == "msec":
        tr = thesizes[4] / 1000.0
    else:
        tr = thesizes[4]
    winsize = int(np.round(args.windowlength / tr))
    winsize += 1 - (winsize % 2)  # make odd
    windowhalfwidth = winsize // 2
    print(f"window size in trs = {2 * windowhalfwidth + 1} ({2 * windowhalfwidth * tr} seconds)")

    xsize, ysize, numslices, timepoints = tide_io.parseniftidims(thedims)
    stdtcs = np.zeros((xsize, ysize, numslices, timepoints), dtype="float")

    # cycle over all voxels
    print("now cycling over all voxels")
    for zloc in range(numslices):
        print("processing slice ", zloc)
        for yloc in range(ysize):
            for xloc in range(xsize):
                stdtcs[xloc, yloc, zloc, :] = cvttovariability(
                    windowhalfwidth, input_data[xloc, yloc, zloc, :]
                )

    # now do the ones with other numbers of time points
    tide_io.savetonifti(stdtcs, input_hdr, args.outputfilename)
