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

import numpy as np

import rapidtide.io as tide_io
from rapidtide.workflows.parser_funcs import is_valid_file
import rapidtide.workflows.glmfrommaps as tide_glmfrommaps
from rapidtide.RapidtideDataset import RapidtideDataset
import rapidtide.util as tide_util
import rapidtide.resample as tide_resample
import logging

LGR = logging.getLogger("GENERAL")
ErrorLGR = logging.getLogger("ERROR")
TimingLGR = logging.getLogger("TIMING")

DEFAULT_GLMDERIVS = 0


def _get_parser():
    """
    Argument parser for glmfilt
    """
    parser = argparse.ArgumentParser(
        prog="retroglm",
        description="Do the rapidtide GLM filtering using the maps generated from a previous analysis.",
        allow_abbrev=False,
    )

    # Required arguments
    parser.add_argument(
        "fmrifile",
        type=lambda x: is_valid_file(parser, x),
        help="The name of 4D nifti fmri file to filter.",
    )
    parser.add_argument(
        "datafileroot",
        type=str,
        help="The root name of the previously run rapidtide dataset (everything up to but not including the underscore.)",
    )
    parser.add_argument(
        "--alternateoutput",
        dest="alternateoutput",
        type=str,
        help="Alternate output root (if not specified, will use the same root as the previous dataset).",
        default=None,
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
        "--limitoutput",
        dest="saveall",
        action="store_false",
        help=("Only save the filtered data and the R value."),
        default=True,
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help=("Output lots of helpful information."),
        default=False,
    )
    return parser


def retroglm(args):
    rt_floatset = np.float64
    rt_floattype = "float64"
    rt_outfloatset = np.float64
    rt_outfloattype = "float64"

    # don't use shared memory if there is only one process
    if args.nprocs == 1:
        usesharedmem = False
    else:
        usesharedmem = True

    # read the datafile and the evfiles
    nim_input, fmri_data, nim_header, thedims_in, thesizes_in = tide_io.readfromnifti(
        args.fmrifile
    )
    xdim, ydim, slicedim, fmritr = tide_io.parseniftisizes(thesizes_in)
    print(xdim, ydim, slicedim, fmritr)
    xsize, ysize, numslices, timepoints = tide_io.parseniftidims(thedims_in)
    print(xsize, ysize, numslices, timepoints)
    startpt = args.numskip
    endpt = timepoints - 1
    validtimepoints = endpt - startpt + 1
    skiptime = startpt * fmritr
    initial_fmri_x = (
        np.linspace(0.0, validtimepoints * fmritr, num=validtimepoints, endpoint=False) + skiptime
    )

    inputdataset = RapidtideDataset(
        "main",
        args.datafileroot + "_",
        verbose=args.verbose,
        init_LUT=False,
    )

    # get the fitmask and make sure the dimensions match
    fitmask = (inputdataset.overlays["lagmask"]).data
    maskshape = fitmask.shape
    if (maskshape[0] != xsize) or (maskshape[1] != ysize) or (maskshape[2] != numslices):
        raise ValueError(
            f"fmri data {xsize, ysize, numslices}does not have the dimensions of rapidtide dataset {maskshape}"
        )

    validvoxels = np.where(fitmask > 0)[0]
    numvalidspatiallocs = np.shape(validvoxels)[0]
    print(f"validvoxels shape = {numvalidspatiallocs}")
    fmri_data_valid = fmri_data[validvoxels, :] + 0.0
    internalvalidspaceshape = numvalidspatiallocs
    internalvalidspaceshapederivs = (
        internalvalidspaceshape,
        args.glmderivs + 1,
    )
    internalvalidfmrishape = (numvalidspatiallocs, np.shape(initial_fmri_x)[0])

    if usesharedmem:
        glmmean, dummy, dummy = tide_util.allocshared(internalvalidspaceshape, rt_outfloatset)
        rvalue, dummy, dummy = tide_util.allocshared(internalvalidspaceshape, rt_outfloatset)
        r2value, dummy, dummy = tide_util.allocshared(internalvalidspaceshape, rt_outfloatset)
        fitNorm, dummy, dummy = tide_util.allocshared(
            internalvalidspaceshapederivs, rt_outfloatset
        )
        fitcoeff, dummy, dummy = tide_util.allocshared(
            internalvalidspaceshapederivs, rt_outfloatset
        )
        movingsignal, dummy, dummy = tide_util.allocshared(internalvalidfmrishape, rt_outfloatset)
        lagtc, dummy, dummy = tide_util.allocshared(internalvalidfmrishape, rt_floatset)
        filtereddata, dummy, dummy = tide_util.allocshared(internalvalidfmrishape, rt_outfloatset)
    else:
        glmmean = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)
        rvalue = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)
        r2value = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)
        fitNorm = np.zeros(internalvalidspaceshapederivs, dtype=rt_outfloattype)
        fitcoeff = np.zeros(internalvalidspaceshapederivs, dtype=rt_outfloattype)
        movingsignal = np.zeros(internalvalidfmrishape, dtype=rt_outfloattype)
        lagtc = np.zeros(internalvalidfmrishape, dtype=rt_floattype)
        filtereddata = np.zeros(internalvalidfmrishape, dtype=rt_outfloattype)

    lagtimes = (inputdataset.overlays["lagtimes"]).data
    oversampfactor = int(inputdataset.therunoptions["oversampfactor"])
    if args.alternateoutput is None:
        outputname = inputdataset.therunoptions["outputname"]
    else:
        outputname = args.alternateoutput
    oversamptr = fmritr / oversampfactor
    threshval = 0.0
    mode = "glm"
    genlagtc = tide_resample.FastResamplerFromFile(f"{outputname}_desc-lagtcgenerator_timeseries")
    voxelsprocessed_glm = tide_glmfrommaps.glmfrommaps(
        fmri_data_valid,
        glmmean,
        rvalue,
        r2value,
        fitNorm,
        fitcoeff,
        movingsignal,
        lagtc,
        filtereddata,
        lagtimes,
        fitmask,
        genlagtc,
        mode,
        outputname,
        oversamptr,
        LGR,
        TimingLGR,
        validvoxels,
        initial_fmri_x,
        threshval,
        nprocs_makelaggedtcs=args.nprocs,
        nprocs_glm=args.nprocs,
        glmderivs=0,
        mp_chunksize=50000,
        showprogressbar=True,
        alwaysmultiproc=False,
        memprofile=False,
        debug=True,
    )
