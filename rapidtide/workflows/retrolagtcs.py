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
import copy
import logging
import os
import sys
from argparse import Namespace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

import rapidtide.io as tide_io
import rapidtide.linfitfiltpass as tide_linfitfiltpass
import rapidtide.makelaggedtcs as tide_makelagged
import rapidtide.multiproc as tide_multiproc
import rapidtide.resample as tide_resample
import rapidtide.util as tide_util
import rapidtide.workflows.parser_funcs as pf

LGR = logging.getLogger("GENERAL")
ErrorLGR = logging.getLogger("ERROR")
TimingLGR = logging.getLogger("TIMING")

DEFAULT_REGRESSIONFILTDERIVS = 0


def _get_parser() -> Any:
    """
    Argument parser for retrolagtcs.

    This function constructs and returns an `argparse.ArgumentParser` object configured
    for the `retrolagtcs` command-line tool. It defines all required and optional
    arguments needed to generate voxel-specific lagged timecourses from rapidtide
    analysis maps.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser for retrolagtcs.

    Notes
    -----
    The parser expects several input files and parameters to define the processing
    pipeline. The function uses `pf.is_valid_file` to validate the existence of
    the input 4D NIfTI file.

    Examples
    --------
    >>> parser = _get_parser()
    >>> args = parser.parse_args()
    >>> print(args.fmrifile)
    'sub-01_task-rest_bold.nii.gz'
    """
    parser = argparse.ArgumentParser(
        prog="retrolagtcs",
        description="Generate voxel specific lagged timecourses using the maps generated from a previous rapidtide analysis.",
        allow_abbrev=False,
    )

    # Required arguments
    parser.add_argument(
        "fmrifile",
        type=lambda x: pf.is_valid_file(parser, x),
        help="The name of 4D nifti fmri target file.",
    )
    parser.add_argument(
        "maskfile",
        type=str,
        help="The mask file to use (usually called XXX_desc-corrfit_mask.nii.gz)",
    )
    parser.add_argument(
        "lagtimesfile",
        type=str,
        help="The  name of the lag times file (usually called XXX_desc-maxtime_map.nii.gz)",
    )
    parser.add_argument(
        "lagtcgeneratorfile",
        type=str,
        help="The root name of the lagtc generator file (usually called XXX_desc-lagtcgenerator_timeseries)",
    )
    parser.add_argument(
        "outputroot",
        type=str,
        help="Output root.",
    )
    parser.add_argument(
        "--regressderivs",
        dest="regressderivs",
        action="store",
        type=int,
        metavar="NDERIVS",
        help=(
            f"When doing final GLM, include derivatives up to NDERIVS order. Default is {DEFAULT_REGRESSIONFILTDERIVS}"
        ),
        default=DEFAULT_REGRESSIONFILTDERIVS,
    )
    parser.add_argument(
        "--nprocs",
        dest="nprocs",
        action="store",
        type=int,
        metavar="NPROCS",
        help=(
            "Use NPROCS worker processes for multiprocessing. "
            "Setting NPROCS to less than 1 sets the number of "
            "worker processes to n_cpus."
        ),
        default=1,
    )
    parser.add_argument(
        "--numskip",
        dest="numskip",
        action="store",
        type=int,
        metavar="NUMSKIP",
        help=("Skip NUMSKIP points at the beginning of the fmri file."),
        default=0,
    )
    parser.add_argument(
        "--noprogressbar",
        dest="showprogressbar",
        action="store_false",
        help=("Will disable showing progress bars (helpful if stdout is going to a file)."),
        default=True,
    )

    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help=("Output lots of helpful information."),
        default=False,
    )
    return parser


def retrolagtcs(args: Any) -> None:
    """
    Generate lagged time series regressors from fMRI data using a lag-time map and a generator file.

    This function reads fMRI data, a mask, and a lag-time map to compute lagged time series
    regressors for each voxel in the mask. It supports both single-process and multi-process
    execution using shared memory. The computed regressors are saved as NIfTI files.

    Parameters
    ----------
    args : Any
        An object containing the following attributes:
        - fmrifile : str
            Path to the input fMRI NIfTI file.
        - maskfile : str
            Path to the processing mask NIfTI file.
        - lagtimesfile : str
            Path to the lag times NIfTI file.
        - lagtcgeneratorfile : str
            Path to the lagtc generator file (used for resampling).
        - outputroot : str
            Root path for output files.
        - nprocs : int
            Number of processes to use for parallel execution. If less than 1, defaults to max CPU count.
        - numskip : int
            Number of initial time points to skip.
        - regressderivs : int
            Number of time derivatives to include in the regressor set.
        - showprogressbar : bool
            Whether to display a progress bar during processing.
        - debug : bool
            Whether to enable debug mode for additional logging.

    Returns
    -------
    None
        This function does not return a value but writes output files to disk.

    Notes
    -----
    - The function requires the input files to have matching spatial dimensions.
    - Shared memory is used for multi-process execution to improve performance.
    - Output files include:
        - Regressor time series (4D NIfTI)
        - Optional mask and lag time maps (3D NIfTI)

    Examples
    --------
    >>> import argparse
    >>> args = argparse.Namespace(
    ...     fmrifile='fmri.nii.gz',
    ...     maskfile='mask.nii.gz',
    ...     lagtimesfile='lagtimes.nii.gz',
    ...     lagtcgeneratorfile='generator.txt',
    ...     outputroot='output',
    ...     nprocs=4,
    ...     numskip=5,
    ...     regressderivs=2,
    ...     showprogressbar=True,
    ...     debug=False
    ... )
    >>> retrolagtcs(args)
    """
    rt_floattype = np.float64
    rt_outfloattype = np.float64

    # get the pid of the parent process
    args.pid = os.getpid()

    thecommandline = " ".join(sys.argv[1:])

    if args.nprocs < 1:
        args.nprocs = tide_multiproc.maxcpus()
    # don't use shared memory if there is only one process
    if args.nprocs == 1:
        usesharedmem = False
    else:
        usesharedmem = True

    # read the fmri input files
    print("reading fmrifile header")
    fmri_input, dummy, fmri_header, fmri_dims, fmri_sizes = tide_io.readfromnifti(
        args.fmrifile,
        headeronly=True,
    )
    xdim, ydim, slicedim, fmritr = tide_io.parseniftisizes(fmri_sizes)
    xsize, ysize, numslices, timepoints = tide_io.parseniftidims(fmri_dims)
    numspatiallocs = int(xsize) * int(ysize) * int(numslices)

    # read the processed mask
    print("reading procfit maskfile")
    (
        procmask_input,
        procmask,
        procmask_header,
        procmask_dims,
        procmask_sizes,
    ) = tide_io.readfromnifti(args.maskfile)
    if not tide_io.checkspacematch(fmri_header, procmask_header):
        raise ValueError("procmask dimensions do not match fmri dimensions")
    procmask_spacebytime = procmask.reshape((numspatiallocs))
    if args.debug:
        print(f"{procmask_spacebytime.shape=}")

    print("reading lagtimes")
    (
        lagtimes_input,
        lagtimes,
        lagtimes_header,
        lagtimes_dims,
        lagtimes_sizes,
    ) = tide_io.readfromnifti(args.lagtimesfile)
    if not tide_io.checkspacematch(fmri_header, lagtimes_header):
        raise ValueError("lagtimes dimensions do not match fmri dimensions")
    if args.debug:
        print(f"{lagtimes.shape=}")
    lagtimes_spacebytime = lagtimes.reshape((numspatiallocs))
    if args.debug:
        print(f"{lagtimes_spacebytime.shape=}")

    startpt = args.numskip
    endpt = timepoints - 1
    validtimepoints = endpt - startpt + 1
    skiptime = startpt * fmritr
    initial_fmri_x = (
        np.linspace(0.0, validtimepoints * fmritr, num=validtimepoints, endpoint=False) + skiptime
    )

    # read the lagtc generator file
    print("reading lagtc generator")
    genlagtc = tide_resample.FastResamplerFromFile(args.lagtcgeneratorfile)

    # select the voxels in the mask
    print("figuring out valid voxels")
    validvoxels = np.where(procmask_spacebytime > 0)[0]
    if args.debug:
        print(f"{validvoxels.shape=}")
    numvalidspatiallocs = np.shape(validvoxels)[0]
    if args.debug:
        print(f"{numvalidspatiallocs=}")
    internalvalidspaceshape = numvalidspatiallocs
    internalvalidspaceshapederivs = (
        internalvalidspaceshape,
        args.regressderivs + 1,
    )
    internalvalidfmrishape = (numvalidspatiallocs, np.shape(initial_fmri_x)[0])
    if args.debug:
        print(f"validvoxels shape = {numvalidspatiallocs}")
        print(f"internalvalidfmrishape shape = {internalvalidfmrishape}")

    # slicing to valid voxels
    print("selecting valid voxels")
    lagtimes_valid = lagtimes_spacebytime[validvoxels]
    procmask_valid = procmask_spacebytime[validvoxels]
    if args.debug:
        print(f"{lagtimes_valid.shape=}")

    if usesharedmem:
        if args.debug:
            print("allocating shared memory")
    else:
        if args.debug:
            print("allocating memory")
    fitNorm, fitNorm_shm = tide_util.allocarray(
        internalvalidspaceshapederivs, rt_outfloattype, shared=usesharedmem
    )
    fitcoeff, fitcoeff_shm = tide_util.allocarray(
        internalvalidspaceshapederivs, rt_outfloattype, shared=usesharedmem
    )
    lagtc, lagtc_shm = tide_util.allocarray(
        internalvalidfmrishape, rt_floattype, shared=usesharedmem
    )

    outputpath = os.path.dirname(args.outputroot)
    rawsources = [
        os.path.relpath(args.fmrifile, start=outputpath),
        os.path.relpath(args.lagtimesfile, start=outputpath),
        os.path.relpath(args.maskfile, start=outputpath),
        os.path.relpath(args.lagtcgeneratorfile, start=outputpath),
    ]

    bidsbasedict = {
        "RawSources": rawsources,
        "Units": "arbitrary",
        "CommandLineArgs": thecommandline,
    }

    print("calling makelaggedtcs")
    voxelsprocessed_makelagged = tide_makelagged.makelaggedtcs(
        genlagtc,
        initial_fmri_x,
        procmask_valid,
        lagtimes_valid,
        lagtc,
        LGR=LGR,
        nprocs=args.nprocs,
        showprogressbar=args.showprogressbar,
    )

    print(f"generated regressors for {voxelsprocessed_makelagged} voxels")

    theheader = copy.deepcopy(lagtimes_header)
    bidsdict = bidsbasedict.copy()

    if args.debug:
        maplist = [
            (
                lagtimes_valid,
                "maxtimeREAD",
                "map",
                "second",
                "Lag time in seconds used for calculation",
            ),
            (procmask_valid, "maskREAD", "mask", None, "Mask used for calculation"),
        ]

        # write the 3D maps
        tide_io.savemaplist(
            args.outputroot,
            maplist,
            validvoxels,
            (xsize, ysize, numslices),
            theheader,
            bidsdict,
            debug=args.debug,
        )

    # write the 4D maps
    theheader = copy.deepcopy(fmri_header)
    maplist = []

    if args.regressderivs > 0:
        if args.debug:
            print(f"adding derivatives up to order {args.regressderivs} prior to regression")
        regressorset = tide_linfitfiltpass.makevoxelspecificderivs(lagtc, args.regressderivs)

        if args.debug:
            print("going down the multiple EV path")
            print(f"{regressorset[:, :, 0].shape=}")
        maplist += [
            (
                regressorset[:, :, 0],
                "lfofilterEV",
                "bold",
                None,
                "Shifted sLFO regressor to filter",
            ),
        ]
        for thederiv in range(1, args.regressderivs + 1):
            if args.debug:
                print(f"{regressorset[:, :, thederiv].shape=}")
            maplist += [
                (
                    regressorset[:, :, thederiv],
                    f"lfofilterEVDeriv{thederiv}",
                    "bold",
                    None,
                    f"Time derivative {thederiv} of shifted sLFO regressor",
                ),
            ]
    else:
        regressorset = lagtc
        if args.debug:
            print("going down the single EV path")
        maplist += [
            (
                regressorset,
                "lfofilterEV",
                "bold",
                None,
                "Shifted sLFO regressor to filter",
            ),
        ]
    tide_io.savemaplist(
        args.outputroot,
        maplist,
        validvoxels,
        (xsize, ysize, numslices, validtimepoints),
        theheader,
        bidsdict,
        debug=args.debug,
    )

    # clean up shared memory
    if usesharedmem:
        tide_util.cleanup_shm(fitNorm_shm)
        tide_util.cleanup_shm(fitcoeff_shm)
        tide_util.cleanup_shm(lagtc_shm)
