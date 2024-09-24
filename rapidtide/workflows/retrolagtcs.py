#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2016-2024 Blaise Frederick
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

import numpy as np

import rapidtide.glmpass as tide_glmpass
import rapidtide.io as tide_io
import rapidtide.makelaggedtcs as tide_makelagged
import rapidtide.multiproc as tide_multiproc
import rapidtide.resample as tide_resample
import rapidtide.util as tide_util
import rapidtide.workflows.parser_funcs as pf

LGR = logging.getLogger("GENERAL")
ErrorLGR = logging.getLogger("ERROR")
TimingLGR = logging.getLogger("TIMING")

DEFAULT_GLMDERIVS = 0


def _get_parser():
    """
    Argument parser for retrolagtcs
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
        "procmaskfile",
        type=str,
        help="The name of the processed mask file (usually called XXX_desc-processed_mask.nii.gz)",
    )
    parser.add_argument(
        "corrmaskfile",
        type=str,
        help="The root name of the correlation fit mask file (usually called XXX_desc-corrfit_mask.nii.gz)",
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
        "--glmderivs",
        dest="glmderivs",
        action="store",
        type=int,
        metavar="NDERIVS",
        help=(
            f"When doing final GLM, include derivatives up to NDERIVS order. Default is {DEFAULT_GLMDERIVS}"
        ),
        default=DEFAULT_GLMDERIVS,
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


def retrolagtcs(args):
    rt_floatset = np.float64
    rt_floattype = "float64"
    rt_outfloatset = np.float64
    rt_outfloattype = "float64"

    thecommandline = " ".join(sys.argv[1:])

    if args.nprocs < 1:
        args.nprocs = tide_multiproc.maxcpus()
    # don't use shared memory if there is only one process
    if args.nprocs == 1:
        usesharedmem = False
    else:
        usesharedmem = True

    # read the fmri input files
    print("reading fmrifile")
    fmri_input, fmri_data, fmri_header, fmri_dims, fmri_sizes = tide_io.readfromnifti(
        args.fmrifile
    )
    if args.debug:
        print(f"{fmri_data.shape=}")
    xdim, ydim, slicedim, fmritr = tide_io.parseniftisizes(fmri_sizes)
    xsize, ysize, numslices, timepoints = tide_io.parseniftidims(fmri_dims)
    numspatiallocs = int(xsize) * int(ysize) * int(numslices)
    fmri_data_spacebytime = fmri_data.reshape((numspatiallocs, timepoints))
    if args.debug:
        print(f"{fmri_data_spacebytime.shape=}")

    # read the processed mask
    print("reading procfit maskfile")
    (
        procmask_input,
        procmask,
        procmask_header,
        procmask_dims,
        procmask_sizes,
    ) = tide_io.readfromnifti(args.procmaskfile)
    if not tide_io.checkspacematch(fmri_header, procmask_header):
        raise ValueError("procmask dimensions do not match fmri dimensions")
    procmask_spacebytime = procmask.reshape((numspatiallocs))
    if args.debug:
        print(f"{procmask_spacebytime.shape=}")

    # read the corrfit mask
    print("reading corrfit maskfile")
    (
        corrmask_input,
        corrmask,
        corrmask_header,
        corrmask_dims,
        corrmask_sizes,
    ) = tide_io.readfromnifti(args.corrmaskfile)
    if not tide_io.checkspacematch(fmri_header, corrmask_header):
        raise ValueError("corrmask dimensions do not match fmri dimensions")
    corrmask_spacebytime = corrmask.reshape((numspatiallocs))
    if args.debug:
        print(f"{corrmask_spacebytime.shape=}")

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
        args.glmderivs + 1,
    )
    internalvalidfmrishape = (numvalidspatiallocs, np.shape(initial_fmri_x)[0])
    if args.debug:
        print(f"validvoxels shape = {numvalidspatiallocs}")
        print(f"internalvalidfmrishape shape = {internalvalidfmrishape}")

    # slicing to valid voxels
    print("selecting valid voxels")
    fmri_data_valid = fmri_data_spacebytime[validvoxels, :]
    lagtimes_valid = lagtimes_spacebytime[validvoxels]
    corrmask_valid = corrmask_spacebytime[validvoxels]
    procmask_valid = procmask_spacebytime[validvoxels]
    if args.debug:
        print(f"{fmri_data_valid.shape=}")

    if usesharedmem:
        if args.debug:
            print("allocating shared memory")
        fitNorm, dummy, dummy = tide_util.allocshared(
            internalvalidspaceshapederivs, rt_outfloatset
        )
        fitcoeff, dummy, dummy = tide_util.allocshared(
            internalvalidspaceshapederivs, rt_outfloatset
        )
        lagtc, dummy, dummy = tide_util.allocshared(internalvalidfmrishape, rt_floatset)
    else:
        if args.debug:
            print("allocating memory")
        lagtc = np.zeros(internalvalidfmrishape, dtype=rt_floattype)

    outputpath = os.path.dirname(args.outputroot)
    rawsources = [
        os.path.relpath(args.fmrifile, start=outputpath),
        os.path.relpath(args.lagtimesfile, start=outputpath),
        os.path.relpath(args.corrmaskfile, start=outputpath),
        os.path.relpath(args.procmaskfile, start=outputpath),
        os.path.relpath(args.lagtcgeneratorfile, start=outputpath),
    ]

    bidsbasedict = {
        "RawSources": rawsources,
        "Units": "arbitrary",
        "CommandLineArgs": thecommandline,
    }

    if args.debug:
        # dump the fmri input file going to glm
        theheader = copy.deepcopy(fmri_header)
        theheader["dim"][4] = validtimepoints
        theheader["pixdim"][4] = fmritr

        maplist = [
            (
                fmri_data_valid,
                "datatofilter",
                "bold",
                None,
                "fMRI data that will be subjected to GLM filtering",
            ),
        ]
        tide_io.savemaplist(
            args.outputroot,
            maplist,
            validvoxels,
            (xsize, ysize, numslices, validtimepoints),
            theheader,
            bidsbasedict,
            textio=False,
            fileiscifti=False,
            rt_floattype=rt_floattype,
            cifti_hdr=None,
        )

    print("calling glmmfrommaps")
    voxelsprocessed_makelagged = tide_makelagged.makelaggedtcs(
        genlagtc,
        initial_fmri_x,
        corrmask_valid,
        lagtimes_valid,
        lagtc,
        nprocs=args.nprocs,
        showprogressbar=args.showprogressbar,
    )

    print(f"generated regressors for {voxelsprocessed_makelagged} voxels")

    theheader = copy.deepcopy(lagtimes_header)
    bidsdict = bidsbasedict.copy()

    if args.debug:
        maplist += [
            (
                lagtimes_valid,
                "maxtimeREAD",
                "map",
                "second",
                "Lag time in seconds used for calculation",
            ),
            (corrmask_valid, "corrfitREAD", "mask", None, "Correlation mask used for calculation"),
            (procmask_valid, "procfitREAD", "mask", None, "Processed mask used for calculation"),
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

    if args.glmderivs > 0:
        if args.debug:
            print(f"adding derivatives up to order {args.glmderivs} prior to regression")
        regressorset = tide_glmpass.makevoxelspecificderivs(lagtc, args.glmderivs)

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
        for thederiv in range(1, args.glmderivs + 1):
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
    if args.debug:
        maplist.append((fmri_data_valid, "inputdata", "bold", None, None))
    tide_io.savemaplist(
        args.outputroot,
        maplist,
        validvoxels,
        (xsize, ysize, numslices, validtimepoints),
        theheader,
        bidsdict,
        debug=args.debug,
    )
