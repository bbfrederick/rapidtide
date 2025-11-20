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
    Argument parser for linfitfilt.

    This function constructs and returns an `argparse.ArgumentParser` object configured
    for the `linfitfilt` command-line tool. It defines all required and optional arguments
    needed to specify input files, regression parameters, and output options.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser for the linfitfilt tool.

    Notes
    -----
    The parser is designed to handle:
    - Input NIfTI files (3D or 4D)
    - EV (event-related) files for regression
    - Data masking options
    - Timecourse skipping for fitting
    - Output filtering and saving options

    Examples
    --------
    >>> parser = _get_parser()
    >>> args = parser.parse_args()
    """
    parser = argparse.ArgumentParser(
        prog="linfitfilt",
        description="Fits and removes the effect of voxel specific and/or global regressors.",
        allow_abbrev=False,
    )

    # Required arguments
    parser.add_argument(
        "inputfile",
        type=lambda x: is_valid_file(parser, x),
        help="The name of the 3 or 4 dimensional nifti file to fit.",
    )
    parser.add_argument("outputroot", type=str, help="The root name for all output files.")
    parser.add_argument(
        "--numskip",
        dest="numskip",
        type=int,
        help="The number of points to skip at the beginning of the timecourse when fitting.  Default is 0.",
        default=0,
    )
    parser.add_argument(
        "--evfile",
        dest="evfile",
        type=lambda x: is_valid_file(parser, x),
        nargs="+",
        help="One or more files (text timecourse or 4D NIFTI) containing signals to regress out.",
    )
    parser.add_argument(
        "--dmask",
        dest="datamaskname",
        type=lambda x: is_valid_file(parser, x),
        action="store",
        metavar="DATAMASK",
        help=("Use DATAMASK to specify which voxels in the data to use."),
        default=None,
    )
    parser.add_argument(
        "--limitoutput",
        dest="saveall",
        action="store_false",
        help=("Only save the filtered data and the R value."),
        default=True,
    )
    return parser


def linfitfilt(
    inputfile: Any,
    numskip: Any,
    outputroot: Any,
    evfilename: Any,
    datamaskname: Any,
    saveall: bool = True,
) -> None:
    """
    Perform linear regression fitting on 4D NIfTI data using specified regressors.

    This function fits a linear model to each voxel's time series using the provided
    regressors and saves the resulting fit coefficients, R-squared values, and filtered
    data. It supports both NIfTI and text-based regressor files, as well as FSL par files
    for global regressors.

    Parameters
    ----------
    inputfile : str or pathlib.Path
        Path to the input 4D NIfTI file containing the time series data.
    numskip : int
        Number of initial time points to skip when fitting the model.
    outputroot : str
        Root name for output NIfTI files (e.g., '_mean', '_fit0', '_R2', etc.).
    evfilename : list of str or pathlib.Path
        List of paths to files containing regressor time series. Can be NIfTI files,
        text files, or FSL par files.
    datamaskname : str or pathlib.Path, optional
        Path to a NIfTI file defining the mask for data analysis. If None, a full mask
        is assumed.
    saveall : bool, optional
        If True, save all intermediate and final output files. Default is True.

    Returns
    -------
    None
        This function does not return a value but saves multiple NIfTI files to disk.

    Notes
    -----
    - The function performs ordinary least squares (OLS) regression for each voxel.
    - The regressors are applied to the data after skipping the first `numskip` time points.
    - Output files include:
        - `<outputroot>_mean.nii.gz`: Mean of the data.
        - `<outputroot>_fit{j}.nii.gz`: Fitted coefficients for regressor j.
        - `<outputroot>_R2.nii.gz`: R-squared values for the model.
        - `<outputroot>_totaltoremove.nii.gz`: Regressor contributions to be removed.
        - `<outputroot>_trimmed.nii.gz`: Trimmed input data.
        - `<outputroot>_filtered.nii.gz`: Final filtered data.

    Examples
    --------
    >>> linfitfilt(
    ...     inputfile='data.nii.gz',
    ...     numskip=5,
    ...     outputroot='output',
    ...     evfilename=['regressor1.txt', 'regressor2.nii.gz'],
    ...     datamaskname='mask.nii.gz',
    ...     saveall=True
    ... )
    """
    # initialize some variables
    evdata = []
    evisnifti = []
    thedims_in = []
    thedims_ev = []
    thesizes_ev = []

    # read the datafile and the evfiles
    nim_input, nim_data, nim_header, thedims_in, thesizes_in = tide_io.readfromnifti(inputfile)
    xdim, ydim, slicedim, tr = tide_io.parseniftisizes(thesizes_in)
    print(xdim, ydim, slicedim, tr)
    xsize, ysize, numslices, timepoints = tide_io.parseniftidims(thedims_in)
    print(xsize, ysize, numslices, timepoints)

    if datamaskname is not None:
        print("reading in mask array")
        (
            datamask_img,
            datamask_data,
            datamask_hdr,
            datamaskdims,
            datamasksizes,
        ) = tide_io.readfromnifti(datamaskname)
        if not tide_io.checkspacematch(nim_header, datamask_hdr):
            print("Data and mask dimensions do not match - exiting.")
            sys.exit()
        print("done reading in mask array")
    else:
        datamask_data = np.ones_like(nim_data[:, :, :, 0])

    numregressors = 0
    for i in range(0, len(evfilename)):
        print("file ", i, " has name ", evfilename[i])
        # check to see if file is nifti or text
        fileisnifti = tide_io.checkifnifti(evfilename[i])
        fileisparfile = tide_io.checkifparfile(evfilename[i])
        if fileisnifti:
            # if file is nifti
            print("reading voxel specific regressor from ", evfilename[i])
            (
                nim_evinput,
                ev_data,
                ev_header,
                thedims_evinput,
                thesizes_evinput,
            ) = tide_io.readfromnifti(evfilename[i])
            evisnifti.append(True)
            evdata.append(1.0 * ev_data)
            thedims_ev.append(thedims_evinput)
            thesizes_ev.append(thesizes_evinput)
            numregressors += 1
        elif fileisparfile:
            # check to see if file a par file
            print("reading 6 global regressors from an FSL parfile")
            evtimeseries = tide_io.readvecs(evfilename[i])
            print("timeseries length = ", len(evtimeseries[0, :]))
            for j in range(0, 6):
                evisnifti.append(False)
                evdata.append(1.0 * evtimeseries[j, :])
                thedims_evinput = 1.0 * thedims_in
                thesizes_evinput = 1.0 * thesizes_in
                thedims_ev.append(thedims_evinput)
                thesizes_ev.append(thesizes_evinput)
                numregressors += 1
        else:
            # if file is text
            print("reading global regressor from ", evfilename[i])
            evtimeseries = tide_io.readvec(evfilename[i])
            print("timeseries length = ", len(evtimeseries))
            evisnifti.append(False)
            evdata.append(1.0 * evtimeseries)
            thedims_evinput = 1.0 * thedims_in
            thesizes_evinput = 1.0 * thesizes_in
            thedims_ev.append(thedims_evinput)
            thesizes_ev.append(thesizes_evinput)
            numregressors += 1

    for j in range(0, numregressors):
        for i in range(0, 4):
            if thedims_in[i] != thedims_ev[j][i]:
                print("Input file and ev file ", j, " dimensions do not match")
                print("dimension ", i, ":", thedims_in[i], " != ", thedims_ev[j][i])
                exit()
        if timepoints - numskip != thedims_ev[j][4]:
            print("Input file and ev file ", j, " dimensions do not match")
            print("dimension ", 4, ":", timepoints, "!= ", thedims_ev[j][4], "+", numskip)
            exit()

    print("will perform GLM with ", numregressors, " regressors")
    meandata = np.zeros((xsize, ysize, numslices), dtype="float")
    fitdata = np.zeros((xsize, ysize, numslices, numregressors), dtype="float")
    R2data = np.zeros((xsize, ysize, numslices), dtype="float")
    trimmeddata = 1.0 * nim_data[:, :, :, numskip:]

    for z in range(0, numslices):
        print("processing slice ", z)
        for y in range(0, ysize):
            for x in range(0, xsize):
                if datamask_data[x, y, z] > 0:
                    regressorvec = []
                    for j in range(0, numregressors):
                        if evisnifti[j]:
                            regressorvec.append(evdata[j][x, y, z, :])
                        else:
                            regressorvec.append(evdata[j])
                    if np.max(trimmeddata[x, y, z, :]) - np.min(trimmeddata[x, y, z, :]) > 0.0:
                        thefit, R2 = tide_fit.mlregress(regressorvec, trimmeddata[x, y, z, :])
                        meandata[x, y, z] = thefit[0, 0]
                        R2data[x, y, z] = R2
                        for j in range(0, numregressors):
                            fitdata[x, y, z, j] = thefit[0, j + 1]
                            # datatoremove[x, y, z, :, j] = thefit[0, j + 1] * regressorvec[j]
                    else:
                        meandata[x, y, z] = 0.0
                        R2data[x, y, z] = 0.0
                        for j in range(0, numregressors):
                            fitdata[x, y, z, j] = 0.0
                            # datatoremove[x, y, z, :, j] = 0.0 * regressorvec[j]
                    # totaltoremove[x, y, z, :] = np.sum(datatoremove[x, y, z, :, :], axis=1)
                    # filtereddata[x, y, z, :] = trimmeddata[x, y, z, :] - totaltoremove[x, y, z, :]

    # first save the things with a small numbers of timepoints
    print("fitting complete: about to save the fit data")
    theheader = nim_header
    theheader["dim"][4] = 1
    if saveall:
        tide_io.savetonifti(meandata, theheader, outputroot + "_mean")
        for j in range(0, numregressors):
            tide_io.savetonifti(fitdata[:, :, :, j], theheader, outputroot + "_fit" + str(j))
    tide_io.savetonifti(R2data, theheader, outputroot + "_R2")
    R2data = None

    print()
    print("Now constructing the array of data to remove")
    # datatoremove = np.zeros((xsize, ysize, numslices, timepoints - numskip, numregressors), dtype='float')
    totaltoremove = np.zeros((xsize, ysize, numslices, timepoints - numskip), dtype="float")
    # filtereddata = 1.0 * totaltoremove
    for z in range(0, numslices):
        print("processing slice ", z)
        for y in range(0, ysize):
            for x in range(0, xsize):
                if np.max(trimmeddata[x, y, z, :]) - np.min(trimmeddata[x, y, z, :]) > 0.0:
                    for j in range(0, numregressors):
                        totaltoremove[x, y, z, :] += fitdata[x, y, z, j] * regressorvec[j]
                else:
                    totaltoremove[x, y, z, :] = 0.0
    print("Array construction done.  Saving files")

    # now save the things with full timecourses
    theheader = nim_header
    theheader["dim"][4] = timepoints - numskip
    if saveall:
        tide_io.savetonifti(totaltoremove, theheader, outputroot + "_totaltoremove")
    filtereddata = trimmeddata - totaltoremove
    totaltoremove = None
    if saveall:
        tide_io.savetonifti(trimmeddata, theheader, outputroot + "_trimmed")
    trimmeddata = None
    tide_io.savetonifti(filtereddata, theheader, outputroot + "_filtered")


def main() -> None:
    """
    Main function to execute the linear fit and filter pipeline.

    This function parses command line arguments and executes the linear fit and
    filter process on the input data file. It handles argument parsing exceptions
    and provides help output when needed.

    Parameters
    ----------
    None

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    The function relies on `_get_parser()` to create and return an argument parser
    with the necessary command line options. The parsed arguments are then passed
    to the `linfitfilt` function for processing.

    Examples
    --------
    >>> main()
    # Executes the linear fit and filter pipeline with arguments from command line
    """
    try:
        args = vars(_get_parser().parse_args())
    except SystemExit:
        _get_parser().print_help()
        raise

    linfitfilt(
        args["inputfile"],
        args["numskip"],
        args["outputroot"],
        args["evfile"],
        args["datamaskname"],
        saveall=args["saveall"],
    )


if __name__ == "__main__":
    main()
