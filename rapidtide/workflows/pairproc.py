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
from scipy.stats import pearsonr
from tqdm import tqdm

import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math
from rapidtide.workflows.parser_funcs import is_valid_file


def _get_parser() -> Any:
    """
    Create and configure argument parser for pairproc command-line tool.

    This function sets up an argument parser for comparing even and odd volumes
    of 4D nifti files. It defines required and optional command-line arguments
    needed for the pairproc workflow.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser object with all required and optional
        arguments for the pairproc tool.

    Notes
    -----
    The parser is configured with:
    - Required positional arguments: inputfile and outputroot
    - Optional arguments for data masking, distribution analysis, demeaning, and debugging

    Examples
    --------
    >>> parser = _get_parser()
    >>> args = parser.parse_args(['input.nii.gz', 'output_root'])
    >>> print(args.inputfile)
    'input.nii.gz'
    """
    parser = argparse.ArgumentParser(
        prog="pairproc",
        description="Compare the even and odd volumes of a 4D nifti file.",
        allow_abbrev=False,
    )

    parser.add_argument("inputfile", help="The name of the input nifti file, including extension")
    parser.add_argument("outputroot", help="The base name of the output files")

    # Optional arguments
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
        "--getdist",
        action="store_true",
        help="Get the distribution of false correlations",
        default=False,
    )
    parser.add_argument(
        "--demean",
        action="store_true",
        help="Remove the mean from each image prior to processing",
        default=False,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debugging information",
        default=False,
    )
    return parser


def pairproc(args: Any) -> None:
    """
    Process paired time series data from NIfTI files and compute temporal and spatial correlations.

    This function reads input NIfTI data, splits the time series into pairs (even and odd indices),
    and computes Pearson correlations between the paired time series both temporally and spatially.
    It supports optional demeaning and masking of data, and can output correlation maps and
    statistics for real and shifted (rolled) odd time series.

    Parameters
    ----------
    args : Any
        An object containing the following attributes:
        - inputfile : str
            Path to the input NIfTI file.
        - datamaskname : str, optional
            Path to the data mask NIfTI file. If provided, only voxels with mask values > 0.5
            will be processed.
        - outputroot : str
            Root name for output files.
        - demean : bool
            If True, demean the time series before computing correlations.
        - getdist : bool
            If True, compute correlations for both real and shifted (rolled) odd time series.
        - debug : bool
            If True, print additional debug information.

    Returns
    -------
    None
        This function does not return a value but saves multiple NIfTI files and text files
        containing correlation maps and statistics.

    Notes
    -----
    - The input time series must have an even number of time points.
    - The function assumes that the input data is organized as (x, y, z, t) where t is the time dimension.
    - Temporal correlations are computed between even and odd time series for each voxel.
    - Spatial correlations are computed between even and odd time series for each subject.
    - Output files include:
        - Temporal correlation maps (NIfTI)
        - Temporal p-values (NIfTI)
        - Spatial correlation values (text file)
        - Spatial p-values (text file)

    Examples
    --------
    >>> import argparse
    >>> args = argparse.Namespace(
    ...     inputfile="data.nii.gz",
    ...     datamaskname="mask.nii.gz",
    ...     outputroot="output",
    ...     demean=True,
    ...     getdist=False,
    ...     debug=False
    ... )
    >>> pairproc(args)
    """
    # read in the data files
    print("reading input file")
    input_img, input_data, input_hdr, thedims, thesizes = tide_io.readfromnifti(args.inputfile)
    xsize, ysize, numslices, timepoints = tide_io.parseniftidims(thedims)
    if args.debug:
        print(f"inputshape: {input_data.shape}")
    if timepoints % 2 != 0:
        raise ValueError("pairproc requires an even number of points in the time dimension")

    if args.datamaskname is not None:
        (
            datamask_img,
            datamask_data,
            datamask_hdr,
            datamaskdims,
            datamasksizes,
        ) = tide_io.readfromnifti(args.datamaskname)

    # check dimensions
    if args.datamaskname is not None:
        print("checking mask dimensions")
        if not tide_io.checkspacedimmatch(thedims, datamaskdims):
            print("input mask spatial dimensions do not match image")
            exit()
        if datamaskdims[4] != 1:
            print("input mask must have a time dimension of 1")
            exit()

    # allocating arrays
    print("reshaping arrays")
    numspatiallocs = int(xsize) * int(ysize) * int(numslices)
    rs_datafile = input_data.reshape((numspatiallocs, timepoints))

    print("masking arrays")
    if args.datamaskname is not None:
        proclocs = np.where(datamask_data.reshape(numspatiallocs) > 0.5)
    else:
        themaxes = np.max(rs_datafile, axis=1)
        themins = np.min(rs_datafile, axis=1)
        thediffs = (themaxes - themins).reshape(numspatiallocs)
        proclocs = np.where(thediffs > 0.0)
    procdata = rs_datafile[proclocs, :][0]
    numvalid = procdata.shape[0]
    print(rs_datafile.shape, procdata.shape)

    # split the pairs
    numsubjects = timepoints // 2
    evenims = np.zeros((numvalid, numsubjects), dtype=np.double)
    oddims = np.zeros((numvalid, numsubjects), dtype=np.double)
    if args.debug:
        print(f"evenshape: {evenims.shape}")
        print(f"oddshape: {oddims.shape}")
    for subject in range(numsubjects):
        if args.demean:
            evenims[:, subject] = procdata[:, 2 * subject] - np.mean(procdata[:, 2 * subject])
            oddims[:, subject] = procdata[:, 2 * subject + 1] - np.mean(
                procdata[:, 2 * subject + 1]
            )
        else:
            evenims[:, subject] = procdata[:, 2 * subject]
            oddims[:, subject] = procdata[:, 2 * subject + 1]

    if args.getdist:
        runlist = ["real", "shifted"]
    else:
        runlist = ["real"]

    for therun in runlist:
        if therun == "shifted":
            oddims = np.roll(oddims, 1, axis=1)

        # cycle over all voxels
        print("Calculating temporal correlation over all voxels")
        temporalcorrelations = np.zeros((numvalid), dtype=np.double)
        temporalpvalues = np.zeros((numvalid), dtype=np.double)
        for vox in tqdm(
            range(0, numvalid),
            desc="Voxel",
            unit="voxels",
        ):
            thepearsonresult = pearsonr(
                tide_math.stdnormalize(evenims[vox, :]), tide_math.stdnormalize(oddims[vox, :])
            )
            temporalcorrelations[vox], temporalpvalues[vox] = (
                thepearsonresult.statistic,
                thepearsonresult.pvalue,
            )
        print()

        outarray = np.zeros((xsize, ysize, numslices), dtype=np.double)
        temporal_hdr = input_hdr.copy()
        temporal_hdr["pixdim"][4]
        outarray.reshape((numspatiallocs))[proclocs] = temporalcorrelations
        tide_io.savetonifti(
            outarray, temporal_hdr, f"{args.outputroot}_temporalcorrelations_{therun}"
        )
        outarray.reshape((numspatiallocs))[proclocs] = temporalpvalues
        tide_io.savetonifti(outarray, temporal_hdr, f"{args.outputroot}_temporalpvalues_{therun}")

        # cycle over all timepoints
        print("Calculating spatial correlation over all subjects")
        spatialcorrelations = np.zeros((numsubjects), dtype=np.double)
        spatialpvalues = np.zeros((numsubjects), dtype=np.double)
        for subject in tqdm(
            range(0, numsubjects),
            desc="Subject",
            unit="subjects",
        ):
            thepearsonresult = pearsonr(
                tide_math.stdnormalize(evenims[:, subject]),
                tide_math.stdnormalize(oddims[:, subject]),
            )
            spatialcorrelations[subject], spatialpvalues[subject] = (
                thepearsonresult.statistic,
                thepearsonresult.pvalue,
            )
        print()

        tide_io.writenpvecs(
            spatialcorrelations, f"{args.outputroot}_r1r2spatialcorrelations_{therun}.txt"
        )
        tide_io.writenpvecs(spatialpvalues, f"{args.outputroot}_r1r2spatialpvalues_{therun}.txt")
