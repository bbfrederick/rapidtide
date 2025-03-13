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
import time
from pathlib import Path

import numpy as np

import rapidtide.filter as tide_filt
import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math
import rapidtide.multiproc as tide_multiproc
import rapidtide.refinedelay as tide_refinedelay
import rapidtide.resample as tide_resample
import rapidtide.stats as tide_stats
import rapidtide.util as tide_util
import rapidtide.workflows.glmfrommaps as tide_glmfrommaps
import rapidtide.workflows.parser_funcs as pf

from .utils import setup_logger


# Create a sentinel.
# from https://stackoverflow.com/questions/58594956/find-out-which-arguments-were-passed-explicitly-in-argparse
class _Sentinel:
    pass


sentinel = _Sentinel()
LGR = logging.getLogger(__name__)
ErrorLGR = logging.getLogger("ERROR")
TimingLGR = logging.getLogger("TIMING")

DEFAULT_GLMDERIVS = 0
DEFAULT_PATCHTHRESH = 3.0
DEFAULT_REFINEDELAYMINDELAY = -5.0
DEFAULT_REFINEDELAYMAXDELAY = 5.0
DEFAULT_REFINEDELAYNUMPOINTS = 501
DEFAULT_DELAYOFFSETSPATIALFILT = -1
DEFAULT_REFINEGLMDERIVS = 1


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
        type=lambda x: pf.is_valid_file(parser, x),
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
        type=lambda x: pf.is_int(parser, x, minval=0),
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
        type=lambda x: pf.is_int(parser, x, minval=0),
        metavar="NUMSKIP",
        help=("Skip NUMSKIP points at the beginning of the fmri file."),
        default=0,
    )
    parser.add_argument(
        "--outputlevel",
        dest="outputlevel",
        action="store",
        type=str,
        choices=["min", "less", "normal", "more", "max"],
        help=(
            "The level of file output produced.  'min' produces only absolutely essential files, 'less' adds in "
            "the GLM filtered data (rather than just filter efficacy metrics), 'normal' saves what you "
            "would typically want around for interactive data exploration, "
            "'more' adds files that are sometimes useful, and 'max' outputs anything you might possibly want. "
            "Selecting 'max' will produce ~3x your input datafile size as output.  "
            f'Default is "normal".'
        ),
        default="normal",
    )
    parser.add_argument(
        "--noprogressbar",
        dest="showprogressbar",
        action="store_false",
        help=("Will disable showing progress bars (helpful if stdout is going to a file)."),
        default=True,
    )
    parser.add_argument(
        "--makepseudofile",
        dest="makepseudofile",
        action="store_true",
        help=("Make a simulated input file from the mean and the movingsignal."),
        default=False,
    )
    parser.add_argument(
        "--norefinedelay",
        dest="refinedelay",
        action="store_false",
        help=("Do not calculate a refined delay map using GLM information."),
        default=True,
    )
    parser.add_argument(
        "--refinecorr",
        dest="refinecorr",
        action="store_true",
        help=(
            "Recalculate the maxcorr map using GLM coefficient of determination from bandpassed data."
        ),
        default=False,
    )
    parser.add_argument(
        "--nofilterwithrefineddelay",
        dest="filterwithrefineddelay",
        action="store_false",
        help=("Do not use the refined delay in GLM filter."),
        default=True,
    )
    parser.add_argument(
        "--delaypatchthresh",
        dest="delaypatchthresh",
        action="store",
        type=float,
        metavar="NUMMADs",
        help=(
            "Maximum number of robust standard deviations to permit in the offset delay refine map. "
            f"Default is {DEFAULT_PATCHTHRESH}"
        ),
        default=DEFAULT_PATCHTHRESH,
    )
    parser.add_argument(
        "--delayoffsetspatialfilt",
        dest="delayoffsetgausssigma",
        action="store",
        type=float,
        metavar="GAUSSSIGMA",
        help=(
            "Spatially filter fMRI data prior to calculating delay offsets "
            "using GAUSSSIGMA in mm.  Set GAUSSSIGMA negative "
            "to have rapidtide set it to half the mean voxel "
            "dimension (a rule of thumb for a good value)."
        ),
        default=DEFAULT_DELAYOFFSETSPATIALFILT,
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help=("Output lots of helpful information."),
        default=False,
    )
    parser.add_argument(
        "--focaldebug",
        dest="focaldebug",
        action="store_true",
        help=("Output lots of helpful information on a limited subset of operations."),
        default=False,
    )
    experimental = parser.add_argument_group(
        "Experimental options (not fully tested, or not tested at all, may not work).  Beware!"
    )
    experimental.add_argument(
        "--refineglmderivs",
        dest="refineglmderivs",
        action="store",
        type=lambda x: pf.is_int(parser, x, minval=1),
        metavar="NDERIVS",
        help=(
            f"When doing GLM for delay refinement, include derivatives up to NDERIVS order. Must be 1 or more.  "
            f"Default is {DEFAULT_REFINEGLMDERIVS}"
        ),
        default=DEFAULT_REFINEGLMDERIVS,
    )

    return parser


def retroglm(args):
    # get the pid of the parent process
    args.pid = os.getpid()

    # specify the output name
    if args.alternateoutput is None:
        outputname = args.datafileroot
    else:
        outputname = args.alternateoutput

    # start the loggers low that we know the output name
    sh = logging.StreamHandler()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, handlers=[sh])
    else:
        logging.basicConfig(level=logging.INFO, handlers=[sh])
    # Set up loggers for workflow
    setup_logger(
        logger_filename=f"{outputname}_retrolog.txt",
        timing_filename=f"{outputname}_retroruntimings.tsv",
        error_filename=f"{outputname}_retroerrorlog.txt",
        verbose=False,
        debug=args.debug,
    )
    TimingLGR.info("Start")
    LGR.info(f"starting retroglm")

    # set some global values
    args.mindelay = DEFAULT_REFINEDELAYMINDELAY
    args.maxdelay = DEFAULT_REFINEDELAYMAXDELAY
    args.numpoints = DEFAULT_REFINEDELAYNUMPOINTS

    if args.outputlevel == "min":
        args.saveminimumglmfiles = False
        args.savenormalglmfiles = False
        args.savemovingsignal = False
        args.saveallglmfiles = False
    elif args.outputlevel == "less":
        args.saveminimumglmfiles = True
        args.savenormalglmfiles = False
        args.savemovingsignal = False
        args.saveallglmfiles = False
    elif args.outputlevel == "normal":
        args.saveminimumglmfiles = True
        args.savenormalglmfiles = True
        args.savemovingsignal = False
        args.saveallglmfiles = False
    elif args.outputlevel == "more":
        args.saveminimumglmfiles = True
        args.savenormalglmfiles = True
        args.savemovingsignal = True
        args.saveallglmfiles = False
    elif args.outputlevel == "max":
        args.saveminimumglmfiles = True
        args.savenormalglmfiles = True
        args.savemovingsignal = True
        args.saveallglmfiles = True
    else:
        print(f"illegal output level {args['outputlevel']}")
        sys.exit()

    thecommandline = " ".join(sys.argv[1:])

    if args.nprocs < 1:
        args.nprocs = tide_multiproc.maxcpus()
    # don't use shared memory if there is only one process
    if args.nprocs == 1:
        usesharedmem = False
    else:
        usesharedmem = True

    # read the runoptions file
    print("reading runoptions")
    runoptionsfile = f"{args.datafileroot}_desc-runoptions_info"
    therunoptions = tide_io.readoptionsfile(runoptionsfile)
    try:
        candoretroglm = therunoptions["retroglmcompatible"]
    except KeyError:
        print(
            f"based on {runoptionsfile}, this rapidtide dataset does not support retrospective GLM calculation"
        )
        sys.exit()

    if therunoptions["internalprecision"] == "double":
        rt_floattype = "float64"
        rt_floatset = np.float64
    else:
        rt_floattype = "float32"
        rt_floatset = np.float32

    # set the output precision
    if therunoptions["outputprecision"] == "double":
        rt_outfloattype = "float64"
        rt_outfloatset = np.float64
    else:
        rt_outfloattype = "float32"
        rt_outfloatset = np.float32
    therunoptions["saveminimumglmfiles"] = args.saveminimumglmfiles

    # read the fmri input files
    print("reading fmrifile")
    fmri_input, fmri_data, fmri_header, fmri_dims, fmri_sizes = tide_io.readfromnifti(
        args.fmrifile
    )

    # create the canary file
    Path(f"{outputname}_RETROISRUNNING.txt").touch()

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
    procmaskfile = f"{args.datafileroot}_desc-processed_mask.nii.gz"
    (
        procmask_input,
        procmask,
        procmask_header,
        procmask_dims,
        procmask_sizes,
    ) = tide_io.readfromnifti(procmaskfile)
    if not tide_io.checkspacematch(fmri_header, procmask_header):
        raise ValueError("procmask dimensions do not match fmri dimensions")
    procmask_spacebytime = procmask.reshape((numspatiallocs))
    if args.debug or args.focaldebug:
        print(f"{procmask_spacebytime.shape=}")
        print(f"{tide_stats.getmasksize(procmask_spacebytime)=}")

    # read the corrfit mask
    print("reading corrfit maskfile")
    corrmaskfile = f"{args.datafileroot}_desc-corrfit_mask.nii.gz"
    (
        corrmask_input,
        corrmask,
        corrmask_header,
        corrmask_dims,
        corrmask_sizes,
    ) = tide_io.readfromnifti(corrmaskfile)
    if not tide_io.checkspacematch(fmri_header, corrmask_header):
        raise ValueError("corrmask dimensions do not match fmri dimensions")
    corrmask_spacebytime = corrmask.reshape((numspatiallocs))
    if args.debug or args.focaldebug:
        print(f"{corrmask_spacebytime.shape=}")
        print(f"{tide_stats.getmasksize(corrmask_spacebytime)=}")

    print("reading lagtimes")
    lagtimesfile = f"{args.datafileroot}_desc-maxtime_map.nii.gz"
    (
        lagtimes_input,
        lagtimes,
        lagtimes_header,
        lagtimes_dims,
        lagtimes_sizes,
    ) = tide_io.readfromnifti(lagtimesfile)
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

    if therunoptions["arbvec"] is not None:
        # NOTE - this vector is LOWERPASS, UPPERPASS, LOWERSTOP, UPPERSTOP
        # setfreqs expects LOWERSTOP, LOWERPASS, UPPERPASS, UPPERSTOP
        theprefilter = tide_filt.NoncausalFilter(
            "arb",
            transferfunc=therunoptions["filtertype"],
        )
        theprefilter.setfreqs(
            therunoptions["arbvec"][2],
            therunoptions["arbvec"][0],
            therunoptions["arbvec"][1],
            therunoptions["arbvec"][3],
        )
    else:
        theprefilter = tide_filt.NoncausalFilter(
            therunoptions["filterband"],
            transferfunc=therunoptions["filtertype"],
            padtime=therunoptions["padseconds"],
        )

    # read the lagtc generator file
    print("reading lagtc generator")
    lagtcgeneratorfile = f"{args.datafileroot}_desc-lagtcgenerator_timeseries"
    try:
        thepadtime = therunoptions["fastresamplerpadtime"]
    except KeyError:
        thepadtime = therunoptions["padseconds"]
    genlagtc = tide_resample.FastResamplerFromFile(lagtcgeneratorfile, padtime=thepadtime)

    # select the voxels in the mask
    print("figuring out valid voxels")
    validvoxels = np.where(procmask_spacebytime > 0)[0]
    numvalidspatiallocs = np.shape(validvoxels)[0]
    if args.debug or args.focaldebug:
        print(f"{numvalidspatiallocs=}")
    internalvalidspaceshape = numvalidspatiallocs
    if args.refinedelay:
        derivaxissize = np.max([args.refineglmderivs + 1, args.glmderivs + 1])
    else:
        derivaxissize = args.glmderivs + 1
    internalvalidspaceshapederivs = (
        internalvalidspaceshape,
        derivaxissize,
    )
    internalvalidfmrishape = (numvalidspatiallocs, np.shape(initial_fmri_x)[0])
    if args.debug or args.focaldebug:
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
        glmmean, glmmean_shm = tide_util.allocshared(internalvalidspaceshape, rt_outfloatset)
        rvalue, rvalue_shm = tide_util.allocshared(internalvalidspaceshape, rt_outfloatset)
        r2value, r2value_shm = tide_util.allocshared(internalvalidspaceshape, rt_outfloatset)
        fitNorm, fitNorm_shm = tide_util.allocshared(internalvalidspaceshapederivs, rt_outfloatset)
        fitcoeff, fitcoeff_shm = tide_util.allocshared(
            internalvalidspaceshapederivs, rt_outfloatset
        )
        movingsignal, movingsignal_shm = tide_util.allocshared(
            internalvalidfmrishape, rt_outfloatset
        )
        lagtc, lagtc_shm = tide_util.allocshared(internalvalidfmrishape, rt_floatset)
        filtereddata, filtereddata_shm = tide_util.allocshared(
            internalvalidfmrishape, rt_outfloatset
        )
        ramlocation = "in shared memory"
    else:
        if args.debug:
            print("allocating memory")
        glmmean = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)
        rvalue = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)
        r2value = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)
        fitNorm = np.zeros(internalvalidspaceshapederivs, dtype=rt_outfloattype)
        fitcoeff = np.zeros(internalvalidspaceshapederivs, dtype=rt_outfloattype)
        movingsignal = np.zeros(internalvalidfmrishape, dtype=rt_outfloattype)
        lagtc = np.zeros(internalvalidfmrishape, dtype=rt_floattype)
        filtereddata = np.zeros(internalvalidfmrishape, dtype=rt_outfloattype)
        ramlocation = "locally"

    totalbytes = (
        glmmean.nbytes
        + rvalue.nbytes
        + r2value.nbytes
        + fitNorm.nbytes
        + fitcoeff.nbytes
        + movingsignal.nbytes
        + lagtc.nbytes
        + filtereddata.nbytes
    )
    thesize, theunit = tide_util.format_bytes(totalbytes)
    ramstring = f"allocated {thesize:.3f} {theunit} {ramlocation}"
    print(ramstring)
    therunoptions["totalretrobytes"] = totalbytes

    oversampfactor = int(therunoptions["oversampfactor"])
    if args.debug:
        print(f"{outputname=}")
    oversamptr = fmritr / oversampfactor
    try:
        threshval = therunoptions["glmthreshval"]
    except KeyError:
        threshval = 0.0
        therunoptions["glmthreshval"] = threshval
    mode = "glm"

    if args.debug or args.focaldebug:
        print(f"{validvoxels.shape=}")
        np.savetxt(f"{outputname}_validvoxels.txt", validvoxels)

    outputpath = os.path.dirname(outputname)
    rawsources = [
        os.path.relpath(args.fmrifile, start=outputpath),
        os.path.relpath(lagtimesfile, start=outputpath),
        os.path.relpath(corrmaskfile, start=outputpath),
        os.path.relpath(procmaskfile, start=outputpath),
        os.path.relpath(runoptionsfile, start=outputpath),
        os.path.relpath(lagtcgeneratorfile, start=outputpath),
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
                "inputdata",
                "bold",
                None,
                "fMRI data that will be subjected to GLM filtering",
            ),
        ]
        tide_io.savemaplist(
            outputname,
            maplist,
            validvoxels,
            (xsize, ysize, numslices, validtimepoints),
            theheader,
            bidsbasedict,
            textio=therunoptions["textio"],
            fileiscifti=False,
            rt_floattype=rt_floattype,
            cifti_hdr=None,
        )

    # refine the delay value prior to calculating the GLM
    if args.refinedelay:
        print("\n\nDelay refinement")
        TimingLGR.info("Delay refinement start")
        LGR.info("\n\nDelay refinement")

        if args.delayoffsetgausssigma < 0.0:
            # set gausssigma automatically
            args.delayoffsetgausssigma = np.mean([xdim, ydim, slicedim]) / 2.0

        TimingLGR.info("Refinement calibration start")
        glmderivratios = tide_refinedelay.getderivratios(
            fmri_data_valid,
            validvoxels,
            initial_fmri_x,
            lagtimes_valid,
            corrmask_valid,
            genlagtc,
            mode,
            outputname,
            oversamptr,
            glmmean,
            rvalue,
            r2value,
            fitNorm[:, : (args.refineglmderivs + 1)],
            fitcoeff[:, : (args.refineglmderivs + 1)],
            movingsignal,
            lagtc,
            filtereddata,
            LGR,
            TimingLGR,
            therunoptions,
            glmderivs=args.refineglmderivs,
            debug=args.debug,
        )

        if args.refineglmderivs == 1:
            medfiltglmderivratios, filteredglmderivratios, delayoffsetMAD = (
                tide_refinedelay.filterderivratios(
                    glmderivratios,
                    (xsize, ysize, numslices),
                    validvoxels,
                    (xdim, ydim, slicedim),
                    gausssigma=args.delayoffsetgausssigma,
                    patchthresh=args.delaypatchthresh,
                    fileiscifti=False,
                    textio=False,
                    rt_floattype=rt_floattype,
                    debug=args.debug,
                )
            )

            # find the mapping of glm ratios to delays
            tide_refinedelay.trainratiotooffset(
                genlagtc,
                initial_fmri_x,
                outputname,
                args.outputlevel,
                mindelay=args.mindelay,
                maxdelay=args.maxdelay,
                numpoints=args.numpoints,
                debug=args.debug,
            )
            TimingLGR.info("Refinement calibration end")

            # now calculate the delay offsets
            TimingLGR.info("Calculating delay offsets")
            delayoffset = np.zeros_like(filteredglmderivratios)
            if args.focaldebug:
                print(f"calculating delayoffsets for {filteredglmderivratios.shape[0]} voxels")
            for i in range(filteredglmderivratios.shape[0]):
                delayoffset[i] = tide_refinedelay.ratiotodelay(filteredglmderivratios[i])
                """delayoffset[i] = tide_refinedelay.coffstodelay(
                    np.asarray([filteredglmderivratios[i]]),
                    mindelay=args.mindelay,
                    maxdelay=args.maxdelay,
                )"""

            refinedvoxelstoreport = filteredglmderivratios.shape[0]
        else:
            medfiltglmderivratios = np.zeros_like(glmderivratios)
            filteredglmderivratios = np.zeros_like(glmderivratios)
            delayoffsetMAD = np.zeros(args.refineglmderivs, dtype=float)
            for i in range(args.refineglmderivs):
                medfiltglmderivratios[i, :], filteredglmderivratios[i, :], delayoffsetMAD[i] = (
                    tide_refinedelay.filterderivratios(
                        glmderivratios[i, :],
                        (xsize, ysize, numslices),
                        validvoxels,
                        (xdim, ydim, slicedim),
                        gausssigma=args.delayoffsetgausssigma,
                        patchthresh=args.delaypatchthresh,
                        fileiscifti=False,
                        textio=False,
                        rt_floattype=rt_floattype,
                        debug=args.debug,
                    )
                )

            # now calculate the delay offsets
            delayoffset = np.zeros_like(filteredglmderivratios[0, :])
            if args.debug:
                print(f"calculating delayoffsets for {filteredglmderivratios.shape[1]} voxels")
            for i in range(filteredglmderivratios.shape[1]):
                delayoffset[i] = tide_refinedelay.coffstodelay(
                    filteredglmderivratios[:, i],
                    mindelay=args.mindelay,
                    maxdelay=args.maxdelay,
                )
            refinedvoxelstoreport = filteredglmderivratios.shape[1]

        namesuffix = "_desc-delayoffset_hist"
        tide_stats.makeandsavehistogram(
            delayoffset,
            therunoptions["histlen"],
            1,
            outputname + namesuffix,
            displaytitle="Histogram of delay offsets calculated from GLM",
            dictvarname="delayoffsethist",
            thedict=None,
        )
        lagtimesrefined_valid = lagtimes_valid + delayoffset

        TimingLGR.info(
            "Delay offset calculation done",
            {
                "message2": refinedvoxelstoreport,
                "message3": "voxels",
            },
        )
        ####################################################
        #  Delay refinement end
        ####################################################

    # initialrawvariance = tide_math.imagevariance(fmri_data_valid, None, 1.0 / fmritr)
    initialvariance = tide_math.imagevariance(fmri_data_valid, theprefilter, 1.0 / fmritr)

    print("calling glmmfrommaps")
    TimingLGR.info("Starting GLM filtering")
    if args.refinedelay and args.filterwithrefineddelay:
        lagstouse_valid = lagtimesrefined_valid
    else:
        lagstouse_valid = lagtimes_valid
    voxelsprocessed_glm, regressorset, evset = tide_glmfrommaps.glmfrommaps(
        fmri_data_valid,
        validvoxels,
        initial_fmri_x,
        lagstouse_valid,
        corrmask_valid,
        genlagtc,
        mode,
        outputname,
        oversamptr,
        glmmean,
        rvalue,
        r2value,
        fitNorm[:, : args.glmderivs + 1],
        fitcoeff[:, : args.glmderivs + 1],
        movingsignal,
        lagtc,
        filtereddata,
        LGR,
        TimingLGR,
        threshval,
        args.saveminimumglmfiles,
        nprocs_makelaggedtcs=args.nprocs,
        nprocs_glm=args.nprocs,
        glmderivs=args.glmderivs,
        showprogressbar=args.showprogressbar,
        debug=args.debug,
    )
    print(f"filtered {voxelsprocessed_glm} voxels")
    TimingLGR.info(
        "GLM filtering done",
        {
            "message2": voxelsprocessed_glm,
            "message3": "voxels",
        },
    )
    # finalrawvariance = tide_math.imagevariance(filtereddata, None, 1.0 / fmritr)
    finalvariance = tide_math.imagevariance(filtereddata, theprefilter, 1.0 / fmritr)

    divlocs = np.where(finalvariance > 0.0)
    varchange = initialvariance * 0.0
    varchange[divlocs] = 100.0 * (finalvariance[divlocs] / initialvariance[divlocs] - 1.0)

    """divlocs = np.where(finalrawvariance > 0.0)
    rawvarchange = initialrawvariance * 0.0
    rawvarchange[divlocs] = 100.0 * (finalrawvariance[divlocs] / initialrawvariance[divlocs] - 1.0)"""

    # save outputs
    TimingLGR.info("Starting output save")
    theheader = copy.deepcopy(lagtimes_header)
    if mode == "glm":
        maplist = [
            (
                initialvariance,
                "lfofilterInbandVarianceBefore",
                "map",
                None,
                "Inband variance prior to filtering",
            ),
            (
                finalvariance,
                "lfofilterInbandVarianceAfter",
                "map",
                None,
                "Inband variance after filtering",
            ),
            (
                varchange,
                "lfofilterInbandVarianceChange",
                "map",
                "percent",
                "Change in inband variance after filtering, in percent",
            ),
            # (
            #   initialrawvariance,
            #    "lfofilterTotalVarianceBefore",
            #    "map",
            #    None,
            #    "Total variance prior to filtering",
            # ),
            # (
            #    finalrawvariance,
            #    "lfofilterTotalVarianceAfter",
            #    "map",
            #    None,
            #    "Total variance after filtering",
            # ),
            # (
            #    rawvarchange,
            #    "lfofilterTotalVarianceChange",
            #    "map",
            #    "percent",
            #    "Change in total variance after filtering, in percent",
            # ),
        ]
        if args.saveminimumglmfiles:
            maplist += [
                (
                    r2value,
                    "lfofilterR2",
                    "map",
                    None,
                    "Squared R value of the GLM fit (proportion of variance explained)",
                ),
            ]
        if args.savenormalglmfiles:
            maplist += [
                (np.fabs(rvalue), "lfofilterR", "map", None, "R value of the GLM fit"),
                (glmmean, "lfofilterMean", "map", None, "Intercept from GLM fit"),
            ]
    else:
        maplist = [
            (initialvariance, "lfofilterInbandVarianceBefore", "map", None),
            (finalvariance, "lfofilterInbandVarianceAfter", "map", None),
            (varchange, "CVRVariance", "map", None),
        ]
        if args.savenormalglmfiles:
            maplist += [
                (rvalue, "CVRR", "map", None),
                (r2value, "CVRR2", "map", None),
                (fitcoeff, "CVR", "map", "percent"),
            ]
    bidsdict = bidsbasedict.copy()

    if args.debug or args.focaldebug:
        maplist += [
            (
                lagtimes_valid,
                "maxtimeREAD",
                "map",
                "second",
                "Lag time in seconds used for calculation",
            ),
            (corrmask_valid, "corrfitREAD", "mask", None, "Correlation mask used for calculation"),
            (procmask_valid, "processedREAD", "mask", None, "Processed mask used for calculation"),
        ]
    if args.savenormalglmfiles:
        if args.glmderivs > 0 or args.refinedelay:
            maplist += [
                (fitcoeff[:, 0], "lfofilterCoeff", "map", None, "Fit coefficient"),
                (fitNorm[:, 0], "lfofilterNorm", "map", None, "Normalized fit coefficient"),
            ]
            for thederiv in range(1, args.glmderivs + 1):
                maplist += [
                    (
                        fitcoeff[:, thederiv],
                        f"lfofilterCoeffDeriv{thederiv}",
                        "map",
                        None,
                        f"Fit coefficient for temporal derivative {thederiv}",
                    ),
                    (
                        fitNorm[:, thederiv],
                        f"lfofilterNormDeriv{thederiv}",
                        "map",
                        None,
                        f"Normalized fit coefficient for temporal derivative {thederiv}",
                    ),
                ]
        else:
            maplist += [
                (fitcoeff, "lfofilterCoeff", "map", None, "Fit coefficient"),
                (fitNorm, "lfofilterNorm", "map", None, "Normalized fit coefficient"),
            ]

    if args.refinedelay:
        if args.refineglmderivs > 1:
            for i in range(args.refineglmderivs):
                maplist += [
                    (
                        glmderivratios[i, :],
                        f"glmderivratios_{i}",
                        "map",
                        None,
                        f"Ratio of derivative {i+1} of delayed sLFO to the delayed sLFO",
                    ),
                    (
                        medfiltglmderivratios[i, :],
                        f"medfiltglmderivratios_{i}",
                        "map",
                        None,
                        f"Median filtered version of the glmderivratios_{i} map",
                    ),
                    (
                        filteredglmderivratios[i, :],
                        f"filteredglmderivratios_{i}",
                        "map",
                        None,
                        f"glmderivratios_{i}, with outliers patched using median filtered data",
                    ),
                ]
        else:
            maplist += [
                (
                    glmderivratios,
                    "glmderivratios",
                    "map",
                    None,
                    "Ratio of the first derivative of delayed sLFO to the delayed sLFO",
                ),
                (
                    medfiltglmderivratios,
                    "medfiltglmderivratios",
                    "map",
                    None,
                    "Median filtered version of the glmderivratios map",
                ),
                (
                    filteredglmderivratios,
                    "filteredglmderivratios",
                    "map",
                    None,
                    "glmderivratios, with outliers patched using median filtered data",
                ),
            ]
        maplist += [
            (
                delayoffset,
                "delayoffset",
                "map",
                "second",
                "Delay offset correction from delay refinement",
            ),
            (
                lagtimesrefined_valid,
                "maxtimerefined",
                "map",
                "second",
                "Lag time in seconds, refined",
            ),
        ]

    # write the 3D maps
    tide_io.savemaplist(
        outputname,
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
    if args.saveminimumglmfiles:
        maplist = [
            (
                filtereddata,
                "lfofilterCleaned",
                "bold",
                None,
                "fMRI data with sLFO signal filtered out",
            ),
        ]
    if args.savemovingsignal:
        maplist += [
            (
                movingsignal,
                "lfofilterRemoved",
                "bold",
                None,
                "sLFO signal filtered out of this voxel",
            )
        ]

    if args.saveallglmfiles:
        if args.glmderivs > 0:
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
    if args.makepseudofile:
        print("reading mean image")
        meanfile = f"{args.datafileroot}_desc-mean_map.nii.gz"
        (
            mean_input,
            mean,
            mean_header,
            mean_dims,
            mean_sizes,
        ) = tide_io.readfromnifti(meanfile)
        if not tide_io.checkspacematch(fmri_header, mean_header):
            raise ValueError("mean dimensions do not match fmri dimensions")
        if args.debug:
            print(f"{mean.shape=}")
        mean_spacebytime = mean.reshape((numspatiallocs))
        if args.debug:
            print(f"{mean_spacebytime.shape=}")
        pseudofile = mean_spacebytime[validvoxels, None] + movingsignal[:, :]
        maplist.append((pseudofile, "pseudofile", "bold", None, None))
    tide_io.savemaplist(
        outputname,
        maplist,
        validvoxels,
        (xsize, ysize, numslices, validtimepoints),
        theheader,
        bidsdict,
        debug=args.debug,
    )
    TimingLGR.info("Finishing output save")

    if args.refinecorr:
        TimingLGR.info("Filtering for maxcorralt calculation start")
        for thevoxel in range(fmri_data_valid.shape[0]):
            fmri_data_valid[thevoxel, :] = theprefilter.apply(
                1.0 / fmritr, fmri_data_valid[thevoxel, :]
            )
        TimingLGR.info("Filtering for maxcorralt calculation complete")
        TimingLGR.info("GLM for maxcorralt calculation start")
        voxelsprocessed_glm, regressorset, evset = tide_glmfrommaps.glmfrommaps(
            fmri_data_valid,
            validvoxels,
            initial_fmri_x,
            lagstouse_valid,
            corrmask_valid,
            genlagtc,
            mode,
            outputname,
            oversamptr,
            glmmean,
            rvalue,
            r2value,
            fitNorm[:, : args.glmderivs + 1],
            fitcoeff[:, : args.glmderivs + 1],
            movingsignal,
            lagtc,
            filtereddata,
            LGR,
            TimingLGR,
            threshval,
            args.saveminimumglmfiles,
            nprocs_makelaggedtcs=args.nprocs,
            nprocs_glm=args.nprocs,
            glmderivs=args.glmderivs,
            showprogressbar=args.showprogressbar,
            debug=args.debug,
        )
        TimingLGR.info(
            "GLM for maxcorralt calculation done",
            {
                "message2": voxelsprocessed_glm,
                "message3": "voxels",
            },
        )

        maplist = [
            (
                rvalue,
                "maxcorralt",
                "map",
                None,
                "R value for the lfo component of the delayed regressor, with sign",
            ),
        ]
        theheader = copy.deepcopy(lagtimes_header)
        tide_io.savemaplist(
            outputname,
            maplist,
            validvoxels,
            (xsize, ysize, numslices),
            theheader,
            bidsdict,
            debug=args.debug,
        )
        if args.debug:
            # dump the fmri input file going to glm
            theheader = copy.deepcopy(fmri_header)
            theheader["dim"][4] = validtimepoints
            theheader["pixdim"][4] = fmritr

            maplist = [
                (
                    fmri_data_valid,
                    "prefilteredinputdata",
                    "bold",
                    None,
                    "fMRI data after temporal filtering",
                ),
            ]
            tide_io.savemaplist(
                outputname,
                maplist,
                validvoxels,
                (xsize, ysize, numslices, validtimepoints),
                theheader,
                bidsbasedict,
                textio=therunoptions["textio"],
                fileiscifti=False,
                rt_floattype=rt_floattype,
                cifti_hdr=None,
            )

    # read the runoptions file
    print("writing runoptions")
    if args.refinedelay:
        therunoptions["retroglm_delayoffsetMAD"] = delayoffsetMAD
    therunoptions["retroglm_runtime"] = time.strftime(
        "%a, %d %b %Y %H:%M:%S %Z", time.localtime(time.time())
    )

    # clean up shared memory
    if usesharedmem:
        TimingLGR.info("Shared memory cleanup start")
        tide_util.cleanup_shm(glmmean_shm)
        tide_util.cleanup_shm(rvalue_shm)
        tide_util.cleanup_shm(r2value_shm)
        tide_util.cleanup_shm(fitNorm_shm)
        tide_util.cleanup_shm(fitcoeff_shm)
        tide_util.cleanup_shm(movingsignal_shm)
        tide_util.cleanup_shm(lagtc_shm)
        tide_util.cleanup_shm(filtereddata_shm)
        TimingLGR.info("Shared memory cleanup complete")

    # shut down logging
    TimingLGR.info("Done")
    logging.shutdown()

    # reformat timing information and delete the unformatted version
    timingdata, therunoptions["totalretroruntime"] = tide_util.proctiminglogfile(
        f"{outputname}_retroruntimings.tsv"
    )
    tide_io.writevec(
        timingdata,
        f"{outputname}_desc-formattedretroruntimings_info.tsv",
    )
    Path(f"{outputname}_retroruntimings.tsv").unlink(missing_ok=True)

    # save the modified runoptions file
    tide_io.writedicttojson(therunoptions, f"{outputname}_desc-runoptions_info.json")

    # shut down the loggers
    for thelogger in [LGR, ErrorLGR, TimingLGR]:
        handlers = thelogger.handlers[:]
        for handler in handlers:
            thelogger.removeHandler(handler)
            handler.close()

    # delete the canary file
    Path(f"{outputname}_RETROISRUNNING.txt").unlink()

    # create the finished file
    Path(f"{outputname}_RETRODONE.txt").touch()


def process_args(inputargs=None):
    """
    Compile arguments for retroglm workflow.
    """
    args, argstowrite = pf.setargs(_get_parser, inputargs=inputargs)
    return args
