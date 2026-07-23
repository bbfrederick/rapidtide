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
import sys
from typing import Any, Optional

import numpy as np
from tqdm import tqdm

import rapidtide.fit as tide_fit
import rapidtide.io as tide_io
from rapidtide.workflows.parser_funcs import is_valid_file


def _get_parser() -> Any:
    """
    Argument parser for slopefit.

    This function constructs and returns an `argparse.ArgumentParser` object configured
    to parse command-line arguments for the `slopefit` tool. It is designed to handle
    the fitting of a spatial template to 3D or 4D NIFTI files, with optional region-specific
    fitting using an atlas file.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser for `slopefit` command-line interface.

    Notes
    -----
    The parser expects two required NIFTI files:
    - First input file (`infile1`)
    - Second input file (`infile2`)

    Optional arguments include:
    - `--maskfile`: 3D NIFTI mask file
    - `--order`: Polynomial order for the fit (default is 1).

    Examples
    --------
    >>> parser = _get_parser()
    >>> args = parser.parse_args(['data.nii', 'mask.nii', 'template.nii', 'template_mask.nii', 'output'])
    >>> print(args.datafile)
    'data.nii'
    """
    parser = argparse.ArgumentParser(
        prog="slopefit",
        description="Fit a voxelwise linear relationship between the values in two, matched 4D NIFTI files.",
        allow_abbrev=False,
    )

    # Required arguments
    parser.add_argument(
        "inputfile1",
        type=lambda x: is_valid_file(parser, x),
        help="The name of the first 4 dimensional nifti file (containing the independent variable).",
    )
    parser.add_argument(
        "inputfile2",
        type=lambda x: is_valid_file(parser, x),
        help="The name of the second 4 dimensional nifti file (containing the independent variable - all dimensions must match the first file).",
    )
    parser.add_argument(
        "outputroot",
        type=str,
        help="The root name for output nifti files.",
    )
    parser.add_argument(
        "--maskfile",
        metavar="MASKFILE",
        dest="maskfile",
        type=lambda x: is_valid_file(parser, x),
        help="The name of a 3D nifti mask file (spatial dimensions must match 4D input files).",
        default=None,
    )
    parser.add_argument(
        "--order",
        metavar="ORDER",
        dest="order",
        type=int,
        help="Perform fit to ORDERth order (default (and minimum) is 1)",
        default=1,
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Print additional internal information.",
        default=False,
    )
    return parser


def slopefit(
    inputfile1: Any,
    inputfile2: Any,
    outputroot: Any,
    maskfile: Optional[Any] = None,
    order: int = 1,
    debug: bool = False,
) -> None:
    """
    Fit polynomial models to time series data within regions defined by masks.

    This function performs polynomial fitting of specified order to time series data
    within spatial regions defined by a template mask. It supports both global and
    region-specific fitting, and outputs fitted time series, polynomial coefficients,
    and R² values.

    Parameters
    ----------
    datafile : Any
        Path to the input NIfTI file containing the time series data.
    datamask : Any
        Path to the NIfTI file containing the data mask. Can be 3D or 4D.
    templatefile : Any
        Path to the NIfTI file containing the template (e.g., a regressor) for fitting.
    templatemask : Any
        Path to the NIfTI file containing the mask for the template.
    outputroot : Any
        Root name for output files (e.g., 'output' will produce 'output_fit.nii.gz').
    maskfile : Any, optional
        Path to the NIfTI file containing region labels. If provided, fitting is
        performed separately for each region. Default is None.
    order : int, optional
        Order of the polynomial to fit. Must be >= 1. Default is 1 (linear fit).

    Returns
    -------
    None
        Function writes multiple output files:
        - Fitted time series (NIfTI format)
        - Residuals (NIfTI format)
        - R² values (text file)
        - Polynomial coefficients (text files)

    Notes
    -----
    - The function assumes that all input files are in NIfTI format.
    - If `datamask` is 4D, it is treated as a time-varying mask.
    - If `maskfile` is provided, fitting is performed only within the specified voxels.
    - The function uses `tide_io` for reading/writing NIfTI files and `tide_fit.mlregress`
      for polynomial regression.
    """
    # check the order
    if order < 1:
        print("order must be >= 1")
        sys.exit()

    if debug:
        print(f"{inputfile1=}")
        print(f"{inputfile2=}")
        print(f"{outputroot=}")
        print(f"{maskfile=}")
        print(f"{order=}")

    # read in data
    print("reading in data arrays")
    (
        inputfile1_img,
        inputfile1_data,
        inputfile1_hdr,
        inputfile1dims,
        inputfile1sizes,
    ) = tide_io.readfromnifti(inputfile1)
    (
        inputfile2_img,
        inputfile2_data,
        inputfile2_hdr,
        inputfile2dims,
        inputfile2sizes,
    ) = tide_io.readfromnifti(inputfile2)
    if maskfile is not None:
        (
            maskfile_img,
            maskfile_data,
            maskfile_hdr,
            maskfiledims,
            maskfilesizes,
        ) = tide_io.readfromnifti(maskfile)

    xsize = inputfile1dims[1]
    ysize = inputfile1dims[2]
    numslices = inputfile1dims[3]
    timepoints = inputfile1dims[4]

    # check dimensions
    print("checking dimensions")
    if not tide_io.checkspacedimmatch(inputfile1dims, inputfile2dims):
        print("input file spatial dimensions do not match")
        exit()
    if not tide_io.checktimematch(inputfile1dims, inputfile2dims):
        print("input mask time dimension does not match")
        exit()
    if maskfile is not None:
        if not tide_io.checkspacedimmatch(inputfile1dims, maskfiledims):
            print("Mask spatial dimensions do not match images")
            exit()
        if not maskfiledims[4] == 1:
            print("mask file time dimension is not equal to 1")
            exit()

    # allocating arrays
    print("allocating arrays")
    numspatiallocs = int(xsize) * int(ysize) * int(numslices)
    rs_inputfile1 = inputfile1_data.reshape((numspatiallocs, timepoints))
    rs_inputfile2 = inputfile2_data.reshape((numspatiallocs, timepoints))
    if maskfile is not None:
        rs_maskfile = maskfile_data.reshape(numspatiallocs)
    else:
        rs_maskfile = np.ones_like(inputfile1_data[:, :, :, 0], dtype=np.float32).reshape(
            numspatiallocs
        )
    rs_maskfile_bin = np.where(rs_maskfile > 0.9, 1.0, 0.0)

    if debug:
        print(f"{inputfile1_data.shape=}, {rs_inputfile1.shape=}")
        print(f"{inputfile2_data.shape=}, {rs_inputfile2.shape=}")
        if maskfile is not None:
            print(f"{maskfile_data.shape=}")
        print(f"{rs_maskfile.shape=}, {rs_maskfile_bin.shape=}")

    polycoffs = np.zeros((numspatiallocs, order + 1), dtype="float")
    r2vals = np.zeros((numspatiallocs), dtype="float")

    # cycle over all voxels
    voxelstofit = np.where(rs_maskfile_bin > 0.0)
    print("now cycling over all voxels")
    for thevoxel in tqdm(
        range(voxelstofit),
        desc="Voxel",
        unit="voxels",
    ):
        # get the appropriate mask
        if rs_maskfile_bin[thevoxel] > 0.0:
            evlist = [np.ones((timepoints), dtype=np.float32)]
            for i in range(1, order + 1):
                evlist.append((rs_inputfile1[thevoxel, :]) ** i)
                if debug:
                    print(f"{evlist[-1].shape=}")
            if debug:
                print(f"{len(evlist)=}")
            thefit, R2 = tide_fit.mlregress(
                evlist,
                rs_inputfile2[thevoxel, :],
            )
            if debug:
                print(f"{thefit=}, {R2=}")
            for i in range(order + 1):
                polycoffs[thevoxel, i] = thefit[0, i]
            r2vals[thevoxel] = R2

    print("writing nifti files")
    theheader = copy.copy(inputfile1_hdr)
    theheader["dim"][0] = 4
    theheader["dim"][4] = order + 1
    tide_io.savetonifti(
        polycoffs.reshape((xsize, ysize, numslices, order + 1)),
        theheader,
        outputroot + "_coffs",
    )
    theheader["dim"][0] = 3
    theheader["dim"][4] = 1
    tide_io.savetonifti(
        r2vals.reshape((xsize, ysize, numslices)),
        theheader,
        outputroot + "_r2vals",
    )


def main(args: Any) -> None:
    """
    Main function to perform polynomial fitting on imaging data.

    This function serves as the entry point for polynomial fitting operations
    on medical imaging data using template-based registration and fitting.

    Parameters
    ----------
    args : Any
        Namespace object containing all required arguments for the polynomial fitting
        operation. Expected attributes include:

        - datafile : str
            Path to the input data file to be fitted
        - datamask : str
            Path to the data mask file for region of interest specification
        - templatefile : str
            Path to the template file for reference
        - templatemask : str
            Path to the template mask file for reference region specification
        - outputroot : str
            Root path for output files
        - maskfile : str, optional
            Path to region atlas file for anatomical labeling
        - order : int, optional
            Order of the polynomial to fit (default is typically 1)

    Returns
    -------
    None
        This function does not return any value. It performs the polynomial fitting
        operation and saves results to the specified output directory.

    Notes
    -----
    The function internally calls `slopefit` with the provided arguments to
    perform the actual polynomial fitting computation. All input files must be
    properly formatted and accessible.

    """
    slopefit(
        args.inputfile1,
        args.inputfile2,
        args.outputroot,
        maskfile=args.maskfile,
        order=args.order,
        debug=args.debug,
    )
