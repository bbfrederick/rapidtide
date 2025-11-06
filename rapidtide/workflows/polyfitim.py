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
import sys
from argparse import Namespace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

import rapidtide.fit as tide_fit
import rapidtide.io as tide_io
from rapidtide.workflows.parser_funcs import is_valid_file


def _get_parser() -> Any:
    """
    Argument parser for polyfitim.

    This function constructs and returns an `argparse.ArgumentParser` object configured
    to parse command-line arguments for the `polyfitim` tool. It is designed to handle
    the fitting of a spatial template to 3D or 4D NIFTI files, with optional region-specific
    fitting using an atlas file.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser for `polyfitim` command-line interface.

    Notes
    -----
    The parser expects several required NIFTI files:
    - Data file (`datafile`)
    - Data mask file (`datamask`)
    - Template file (`templatefile`)
    - Template mask file (`templatemask`)

    Optional arguments include:
    - `--regionatlas`: File containing a 3D NIFTI atlas for region-specific fitting.
    - `--order`: Polynomial order for the fit (default is 1).

    Examples
    --------
    >>> parser = _get_parser()
    >>> args = parser.parse_args(['data.nii', 'mask.nii', 'template.nii', 'template_mask.nii', 'output'])
    >>> print(args.datafile)
    'data.nii'
    """
    parser = argparse.ArgumentParser(
        prog="polyfitim",
        description="Fit a spatial template to a 3D or 4D NIFTI file.",
        allow_abbrev=False,
    )

    # Required arguments
    parser.add_argument(
        "datafile",
        type=lambda x: is_valid_file(parser, x),
        help="The name of the 3 or 4 dimensional nifti file to fit.",
    )
    parser.add_argument(
        "datamask",
        type=lambda x: is_valid_file(parser, x),
        help="The name of the 3 or 4 dimensional nifti file valid voxel mask (must match datafile).",
    )
    parser.add_argument(
        "templatefile",
        type=lambda x: is_valid_file(parser, x),
        help="The name of the 3D nifti template file (must match datafile).",
    )
    parser.add_argument(
        "templatemask",
        type=lambda x: is_valid_file(parser, x),
        help="The name of the 3D nifti template mask (must match datafile).",
    )
    parser.add_argument("outputroot", type=str, help="The root name for all output files.")
    parser.add_argument(
        "--regionatlas",
        metavar="ATLASFILE",
        dest="regionatlas",
        type=lambda x: is_valid_file(parser, x),
        help="Do individual fits to every region in ATLASFILE (3D NIFTI file).",
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
    return parser


def polyfitim(
    datafile: Any,
    datamask: Any,
    templatefile: Any,
    templatemask: Any,
    outputroot: Any,
    regionatlas: Optional[Any] = None,
    order: int = 1,
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
    regionatlas : Any, optional
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
    - If `regionatlas` is provided, fitting is performed separately for each region.
    - The function uses `tide_io` for reading/writing NIfTI files and `tide_fit.mlregress`
      for polynomial regression.

    Examples
    --------
    >>> polyfitim(
    ...     datafile='data.nii.gz',
    ...     datamask='mask.nii.gz',
    ...     templatefile='template.nii.gz',
    ...     templatemask='template_mask.nii.gz',
    ...     outputroot='output',
    ...     regionatlas='atlas.nii.gz',
    ...     order=2
    ... )
    """
    # check the order
    if order < 1:
        print("order must be >= 1")
        sys.exit()

    # read in data
    print("reading in data arrays")
    (
        datafile_img,
        datafile_data,
        datafile_hdr,
        datafiledims,
        datafilesizes,
    ) = tide_io.readfromnifti(datafile)
    (
        datamask_img,
        datamask_data,
        datamask_hdr,
        datamaskdims,
        datamasksizes,
    ) = tide_io.readfromnifti(datamask)
    (
        templatefile_img,
        templatefile_data,
        templatefile_hdr,
        templatefiledims,
        templatefilesizes,
    ) = tide_io.readfromnifti(templatefile)
    (
        templatemask_img,
        templatemask_data,
        templatemask_hdr,
        templatemaskdims,
        templatemasksizes,
    ) = tide_io.readfromnifti(templatemask)

    if regionatlas is not None:
        (
            regionatlas_img,
            regionatlas_data,
            regionatlas_hdr,
            regionatlasdims,
            regionatlassizes,
        ) = tide_io.readfromnifti(regionatlas)

    xsize = datafiledims[1]
    ysize = datafiledims[2]
    numslices = datafiledims[3]
    timepoints = datafiledims[4]

    # check dimensions
    print("checking dimensions")
    if not tide_io.checkspacedimmatch(datafiledims, datamaskdims):
        print("input mask spatial dimensions do not match image")
        exit()
    if datamaskdims[4] == 1:
        print("using 3d data mask")
        datamask3d = True
    else:
        datamask3d = False
        if not tide_io.checktimematch(datafiledims, datamaskdims):
            print("input mask time dimension does not match image")
            exit()
    if not tide_io.checkspacedimmatch(datafiledims, templatefiledims):
        print(templatefiledims, "template file spatial dimensions do not match image")
        exit()
    if not templatefiledims[4] == 1:
        print("template file time dimension is not equal to 1")
        exit()
    if not tide_io.checkspacedimmatch(datafiledims, templatemaskdims):
        print("template mask spatial dimensions do not match image")
        exit()
    if not templatemaskdims[4] == 1:
        print("template mask time dimension is not equal to 1")
        exit()
    if regionatlas is not None:
        if not tide_io.checkspacedimmatch(datafiledims, regionatlasdims):
            print("template mask spatial dimensions do not match image")
            exit()
        if not regionatlasdims[4] == 1:
            print("regionatlas time dimension is not equal to 1")
            exit()

    # allocating arrays
    print("allocating arrays")
    numspatiallocs = int(xsize) * int(ysize) * int(numslices)
    rs_datafile = datafile_data.reshape((numspatiallocs, timepoints))
    if datamask3d:
        rs_datamask = datamask_data.reshape(numspatiallocs)
    else:
        rs_datamask = datamask_data.reshape((numspatiallocs, timepoints))
    rs_datamask_bin = np.where(rs_datamask > 0.9, 1.0, 0.0)
    rs_templatefile = templatefile_data.reshape(numspatiallocs)
    rs_templatemask = templatemask_data.reshape(numspatiallocs)
    rs_templatemask_bin = np.where(rs_templatemask > 0.1, 1.0, 0.0)
    if regionatlas is not None:
        rs_regionatlas = regionatlas_data.reshape(numspatiallocs)
        numregions = int(np.max(rs_regionatlas))
        print(f"atlas file {regionatlas} has {numregions} regions.")

    fitdata = np.zeros((numspatiallocs, timepoints), dtype="float")
    if regionatlas is not None:
        polycoffs = np.zeros((numregions, order + 1, timepoints), dtype="float")
        r2vals = np.zeros((numregions, timepoints), dtype="float")
    else:
        polycoffs = np.zeros((order + 1, timepoints), dtype="float")
        r2vals = np.zeros(timepoints, dtype="float")

    if regionatlas is not None:
        print("making region masks")
        regionvoxels = np.zeros((numspatiallocs, numregions), dtype="float")
        for region in range(0, numregions):
            thisregion = np.where((rs_regionatlas * rs_templatemask_bin) == (region + 1))
            regionvoxels[thisregion, region] = 1.0

    # cycle over all images
    print("now cycling over all images")
    for thetime in range(0, timepoints):
        print("fitting timepoint", thetime)

        # get the appropriate mask
        if datamask3d:
            for i in range(timepoints):
                thisdatamask = rs_datamask_bin
        else:
            thisdatamask = rs_datamask_bin[:, thetime]
        if regionatlas is not None:
            for region in range(0, numregions):
                voxelstofit = np.where(regionvoxels[:, region] * thisdatamask > 0.5)
                voxelstoreconstruct = np.where(regionvoxels[:, region] > 0.5)
                evlist = []
                for i in range(1, order + 1):
                    evlist.append((rs_templatefile[voxelstofit]) ** i)
                thefit, R2 = tide_fit.mlregress(
                    evlist,
                    rs_datafile[voxelstofit, thetime][0],
                )
                for i in range(order + 1):
                    polycoffs[region, i, thetime] = thefit[0, i]
                fitdata[voxelstoreconstruct, thetime] = polycoffs[region, 0, thetime]
                for i in range(1, order + 1):
                    fitdata[voxelstoreconstruct, thetime] += polycoffs[region, i, thetime] * (
                        rs_templatefile[voxelstoreconstruct] ** i
                    )
                r2vals[region, thetime] = R2
        else:
            voxelstofit = np.where(thisdatamask > 0.5)
            voxelstoreconstruct = np.where(rs_templatemask > 0.5)
            evlist = []
            for i in range(1, order + 1):
                evlist.append((rs_templatefile[voxelstofit]) ** i)
            thefit, R2 = tide_fit.mlregress(evlist, rs_datafile[voxelstofit, thetime][0])
            for i in range(order + 1):
                polycoffs[i, thetime] = thefit[0, i]
            fitdata[voxelstoreconstruct, thetime] = polycoffs[0, thetime]
            for i in range(1, order + 1):
                fitdata[voxelstoreconstruct, thetime] += polycoffs[i, thetime] * (
                    rs_templatefile[voxelstoreconstruct] ** i
                )
            r2vals[thetime] = R2
    residuals = rs_datafile - fitdata

    # write out the data files
    print("writing time series")

    tide_io.writenpvecs(r2vals, outputroot + "_r2vals.txt")
    if regionatlas is not None:
        for region in range(0, numregions):
            outstring = f"region {region + 1}:"
            for i in range(order + 1):
                tide_io.writenpvecs(polycoffs[i, :], outputroot + "_O_i_coffs.txt")
                outstring += f" O_{i} mean, std {np.mean(polycoffs[region, i, :])}, {np.std(polycoffs[region, i, :])}"
            print(outstring)
    else:
        outstring = ""
        for i in range(order + 1):
            tide_io.writenpvecs(polycoffs[:], outputroot + f"_O_{i}_coffs.txt")
            outstring += f" O_{i} mean, std {np.mean(polycoffs[i, :])}, {np.std(polycoffs[i, :])}"
        print(outstring)

    print("writing nifti series")
    tide_io.savetonifti(
        fitdata.reshape((xsize, ysize, numslices, timepoints)),
        datafile_hdr,
        outputroot + "_fit",
    )
    tide_io.savetonifti(
        residuals.reshape((xsize, ysize, numslices, timepoints)),
        datafile_hdr,
        outputroot + "_residuals",
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
        - regionatlas : str, optional
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
    The function internally calls `polyfitim` with the provided arguments to
    perform the actual polynomial fitting computation. All input files must be
    properly formatted and accessible.

    Examples
    --------
    >>> import argparse
    >>> args = argparse.Namespace(
    ...     datafile='data.nii.gz',
    ...     datamask='data_mask.nii.gz',
    ...     templatefile='template.nii.gz',
    ...     templatemask='template_mask.nii.gz',
    ...     outputroot='output',
    ...     regionatlas='atlas.nii.gz',
    ...     order=2
    ... )
    >>> main(args)
    """
    polyfitim(
        args.datafile,
        args.datamask,
        args.templatefile,
        args.templatemask,
        args.outputroot,
        regionatlas=args.regionatlas,
        order=args.order,
    )
