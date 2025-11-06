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
import warnings

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        import pyfftw
    except ImportError:
        pyfftwpresent = False
    else:
        pyfftwpresent = True

from argparse import Namespace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from numpy.typing import NDArray
from scipy import fftpack

import rapidtide.io as tide_io
import rapidtide.resample as tide_resample

if pyfftwpresent:
    fftpack = pyfftw.interfaces.scipy_fftpack
    pyfftw.interfaces.cache.enable()


def _get_parser() -> Any:
    """
    Argument parser for resamplenifti.

    Creates and configures an argument parser for the resamplenifti command-line tool
    that resamples NIfTI files to a different temporal resolution (TR).

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser object with all required and optional arguments
        for the resamplenifti tool.

    Notes
    -----
    The returned parser includes the following positional arguments:
    - inputfile: Path to the input NIfTI file
    - outputfile: Path to the output NIfTI file
    - outputtr: Target temporal resolution in seconds

    And the following optional arguments:
    - --noantialias: Disable antialiasing filter (enabled by default)
    - --normalize: Normalize data and save as UINT16 (disabled by default)
    - --debug: Print debugging information (disabled by default)

    Examples
    --------
    >>> parser = _get_parser()
    >>> args = parser.parse_args(['input.nii', 'output.nii', '2.0'])
    >>> print(args.inputfile)
    'input.nii'
    """
    parser = argparse.ArgumentParser(
        prog="resamplenifti",
        description="Resamples a nifti file to a different TR.",
        allow_abbrev=False,
    )

    parser.add_argument("inputfile", help="The name of the input nifti file, including extension")
    parser.add_argument(
        "outputfile", help="The name of the output nifti file, including extension"
    )
    parser.add_argument("outputtr", type=float, help="The target TR, in seconds")
    parser.add_argument(
        "--noantialias",
        dest="antialias",
        action="store_false",
        help="Disable antialiasing filter",
        default=True,
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize data and save as UINT16",
        default=False,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debugging information",
        default=False,
    )

    return parser


def resamplenifti(args: Any) -> None:
    """
    Resample a 4D NIfTI file to a specified temporal resolution.

    This function reads a 4D NIfTI file, resamples its time series data to a
    new temporal resolution specified by `args.outputtr`, and saves the
    resampled data to a new NIfTI file. The resampling is performed using
    spline interpolation, and optional antialiasing is applied based on the
    input and output temporal resolutions.

    Parameters
    ----------
    args : argparse.Namespace
        An object containing the following attributes:
        - inputfile : str
            Path to the input NIfTI file.
        - outputfile : str
            Path to the output NIfTI file.
        - outputtr : float
            Desired output temporal resolution (TR) in seconds.
        - antialias : bool, optional
            Whether to apply antialiasing during resampling. Default is True.
        - debug : bool, optional
            If True, print debugging information. Default is False.

    Returns
    -------
    None
        This function does not return a value but saves the resampled NIfTI
        file to the specified output path.

    Notes
    -----
    - If the input TR is greater than the output TR (i.e., upsampling), antialiasing
      is automatically disabled.
    - The function processes each voxel individually, which may be time-consuming
      for large datasets.
    - The output NIfTI header is updated to reflect the new temporal resolution.

    Examples
    --------
    >>> import argparse
    >>> args = argparse.Namespace(
    ...     inputfile='input.nii.gz',
    ...     outputfile='output.nii.gz',
    ...     outputtr=1.0,
    ...     antialias=True,
    ...     debug=False
    ... )
    >>> resamplenifti(args)
    """
    # get the input TR
    inputtr, numinputtrs = tide_io.fmritimeinfo(args.inputfile)
    if args.debug:
        print("input data: ", numinputtrs, " timepoints, tr = ", inputtr)

    # check to see if we are upsampling or downsampling
    if inputtr > args.outputtr:  # we are upsampling - antialiasing is unnecessary
        args.antialias = False
        print("upsampling - antialiasing disabled")

    # prepare the input timepoint list
    inputstarttime = 0.0
    inputendtime = inputstarttime + inputtr * (numinputtrs - 1)
    if args.debug:
        print(
            "input start,end,tr,numtrs",
            inputstarttime,
            inputendtime,
            inputtr,
            numinputtrs,
        )
    input_x = (
        np.linspace(0.0, inputtr * numinputtrs, num=numinputtrs, endpoint=False) - inputstarttime
    )

    # prepare the output timepoint list
    outputstarttime = inputstarttime
    outputendtime = inputendtime
    numoutputtrs = int(np.ceil((outputendtime - outputstarttime) / args.outputtr) + 1)
    if args.debug:
        print(
            "output start,end,tr,numtrs",
            outputstarttime,
            outputendtime,
            args.outputtr,
            numoutputtrs,
        )
    output_x = (
        np.linspace(0.0, args.outputtr * numoutputtrs, num=numoutputtrs, endpoint=False)
        - outputstarttime
    )

    input_img, input_data, input_hdr, thedims, thesizes = tide_io.readfromnifti(args.inputfile)
    xsize = thedims[1]
    ysize = thedims[2]
    numslices = thedims[3]
    timepoints = thedims[4]
    numspatiallocs = xsize * ysize * numslices

    # mask_data = tide_mask.makeepimask(np.mean(input_data, axis=3))

    # make the output array
    resampledtcs = np.zeros((xsize, ysize, numslices, len(output_x)), dtype="float")

    # cycle over all voxels
    print("now cycling over all voxels")
    for zloc in range(0, numslices):
        print("processing slice ", zloc)
        for yloc in range(0, ysize):
            for xloc in range(0, xsize):
                resampledtcs[xloc, yloc, zloc, :] = tide_resample.doresample(
                    input_x,
                    input_data[xloc, yloc, zloc, :],
                    output_x,
                    antialias=args.antialias,
                )

    # now do the ones with other numbers of time points
    resampled_hdr = input_hdr.copy()
    resampled_hdr["pixdim"][4] = args.outputtr
    outputroot, dummy = tide_io.niftisplitext(args.outputfile)
    tide_io.savetonifti(resampledtcs, resampled_hdr, outputroot)
