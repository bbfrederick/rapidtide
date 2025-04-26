#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2025 Blaise Frederick
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
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from tf_keras.src.dtensor.integration_test_utils import train_step

import rapidtide.filter as tide_filt
import rapidtide.io as tide_io
import rapidtide.multiproc as tide_multiproc
import rapidtide.refinedelay as tide_refinedelay
import rapidtide.resample as tide_resample
import rapidtide.stats as tide_stats
import rapidtide.util as tide_util
import rapidtide.voxelData as tide_voxelData
import rapidtide.workflows.parser_funcs as pf
import rapidtide.workflows.regressfrommaps as tide_regressfrommaps

from .utils import setup_logger


# Create a sentinel.
# from https://stackoverflow.com/questions/58594956/find-out-which-arguments-were-passed-explicitly-in-argparse
class _Sentinel:
    pass


sentinel = _Sentinel()
LGR = logging.getLogger(__name__)
ErrorLGR = logging.getLogger("ERROR")
TimingLGR = logging.getLogger("TIMING")

DEFAULT_REGRESSIONFILTDERIVS = 0
DEFAULT_PATCHTHRESH = 3.0
DEFAULT_REFINEDELAYMINDELAY = -2.5
DEFAULT_REFINEDELAYMAXDELAY = 2.5
DEFAULT_REFINEDELAYNUMPOINTS = 201
DEFAULT_DELAYOFFSETSPATIALFILT = -1
DEFAULT_WINDOWSIZE = 30.0
DEFAULT_SYSTEMICFITTYPE = "pca"
DEFAULT_PCACOMPONENTS = 1
DEFAULT_LAGMIN = 0.0
DEFAULT_LAGMAX = 0.0
DEFAULT_TRAINSTEP = 0.5


def _get_parser():
    """
    Argument parser for glmfilt
    """
    parser = argparse.ArgumentParser(
        prog="delayvar",
        description="Calculate variation in delay time over the course of an acquisition.",
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
            "the sLFO filtered data (rather than just filter efficacy metrics), 'normal' saves what you "
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
        "--nohpfilter",
        dest="hpf",
        action="store_false",
        help=("Disable highpass filtering on data and regressor."),
        default=True,
    )
    parser.add_argument(
        "--trainrange",
        dest="lag_extrema",
        action=pf.IndicateSpecifiedAction,
        nargs=2,
        type=float,
        metavar=("LAGMIN", "LAGMAX"),
        help=(
            "Set the range of delay offset center frequencies to span LAGMIN to LAGMAX. The derivative "
            "ratio calculation only works over a narrow range, so if the static offset is large, "
            "you need to train the ratio calculation with a central delay close to that value. "
            f"LAGMAX.  Default is {DEFAULT_LAGMIN} to {DEFAULT_LAGMAX} seconds. "
        ),
        default=(DEFAULT_LAGMIN, DEFAULT_LAGMAX),
    )
    parser.add_argument(
        "--trainstep",
        dest="trainstep",
        action="store",
        type=float,
        metavar="STEP",
        help=(
            "Use this step size (in seconds) to span the training width.  The derivative "
            "ratio calculation only works over a narrow range, so if the static offset is large, "
            "you need to train the ratio calculation with a central delay close to that value. "
            f"Default is {DEFAULT_TRAINSTEP}"
        ),
        default=DEFAULT_TRAINSTEP,
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
        "--systemicfittype",
        dest="systemicfittype",
        action="store",
        type=str,
        choices=[
            "mean",
            "pca",
        ],
        help=(
            f"Use mean or pca to fit the systemic variation in delay offset. "
            f'Default is "{DEFAULT_SYSTEMICFITTYPE}".'
        ),
        default=DEFAULT_SYSTEMICFITTYPE,
    )
    parser.add_argument(
        "--pcacomponents",
        metavar="NCOMP",
        dest="pcacomponents",
        type=float,
        help="Use NCOMP components for PCA fit of delay offset.",
        default=DEFAULT_PCACOMPONENTS,
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help=("Be wicked chatty."),
        default=False,
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
        "--windowsize",
        dest="windowsize",
        action="store",
        type=lambda x: pf.is_float(parser, x, minval=10.0),
        metavar="SIZE",
        help=(
            f"Set segmented delay analysis window size to SIZE seconds. Default is {DEFAULT_WINDOWSIZE}."
        ),
        default=DEFAULT_WINDOWSIZE,
    )
    experimental.add_argument(
        "--windelayoffsetspatialfilt",
        dest="windelayoffsetgausssigma",
        action="store",
        type=float,
        metavar="GAUSSSIGMA",
        help=(
            "Spatially filter fMRI data prior to calculating windowed delay offsets "
            "using GAUSSSIGMA in mm.  Set GAUSSSIGMA negative "
            "to have rapidtide set it to half the mean voxel "
            "dimension (a rule of thumb for a good value)."
        ),
        default=DEFAULT_DELAYOFFSETSPATIALFILT,
    )

    return parser


def delayvar(args):
    # get the pid of the parent process
    args.pid = os.getpid()

    args.lagmin = args.lag_extrema[0]
    args.lagmax = args.lag_extrema[1]

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
    LGR.info(f"starting delayvar")

    # set some global values
    args.mindelay = DEFAULT_REFINEDELAYMINDELAY
    args.maxdelay = DEFAULT_REFINEDELAYMAXDELAY
    args.numpoints = DEFAULT_REFINEDELAYNUMPOINTS

    if args.outputlevel == "min":
        args.saveminimumsLFOfiltfiles = False
        args.savenormalsLFOfiltfiles = False
        args.savemovingsignal = False
        args.saveallsLFOfiltfiles = False
    elif args.outputlevel == "less":
        args.saveminimumsLFOfiltfiles = True
        args.savenormalsLFOfiltfiles = False
        args.savemovingsignal = False
        args.saveallsLFOfiltfiles = False
    elif args.outputlevel == "normal":
        args.saveminimumsLFOfiltfiles = True
        args.savenormalsLFOfiltfiles = True
        args.savemovingsignal = False
        args.saveallsLFOfiltfiles = False
    elif args.outputlevel == "more":
        args.saveminimumsLFOfiltfiles = True
        args.savenormalsLFOfiltfiles = True
        args.savemovingsignal = True
        args.saveallsLFOfiltfiles = False
    elif args.outputlevel == "max":
        args.saveminimumsLFOfiltfiles = True
        args.savenormalsLFOfiltfiles = True
        args.savemovingsignal = True
        args.saveallsLFOfiltfiles = True
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

    # read the runoptions file, update if necessary
    print("reading runoptions")
    runoptionsfile = f"{args.datafileroot}_desc-runoptions_info"
    therunoptions = tide_io.readoptionsfile(runoptionsfile)
    sublist = (
        ("retroglmcompatible", "retroregresscompatible"),
        ("glmthreshval", "regressfiltthreshval"),
    )
    therunoptions["singleproc_regressionfilt"] = False
    therunoptions["nprocs_regressionfilt"] = args.nprocs
    for subpair in sublist:
        try:
            therunoptions[subpair[1]] = therunoptions[subpair[0]]
            print(f"substituting {subpair[1]} for {subpair[0]} in runoptions")
        except KeyError:
            pass

    try:
        candoretroregress = therunoptions["retroregresscompatible"]
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
    therunoptions["saveminimumsLFOfiltfiles"] = args.saveminimumsLFOfiltfiles

    # read the fmri input files
    print("reading fmrifile")
    theinputdata = tide_voxelData.VoxelData(args.fmrifile)
    xsize, ysize, numslices, timepoints = theinputdata.getdims()
    xdim, ydim, slicethickness, fmritr = theinputdata.getsizes()
    fmri_header = theinputdata.copyheader()
    fmri_data = theinputdata.byvol()
    numspatiallocs = theinputdata.numspatiallocs

    # create the canary file
    Path(f"{outputname}_DELAYVARISRUNNING.txt").touch()

    if args.debug:
        print(f"{fmri_data.shape=}")
    fmri_data_spacebytime = theinputdata.byvoxel()
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
    if args.debug:
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
    if args.debug:
        print(f"{corrmask_spacebytime.shape=}")
        print(f"{tide_stats.getmasksize(corrmask_spacebytime)=}")

    print("reading lagtimes")
    lagtimesfile = f"{args.datafileroot}_desc-maxtimerefined_map.nii.gz"
    if not os.path.exists(lagtimesfile):
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

    # read the lagtc generator file
    print("reading lagtc generator")
    lagtcgeneratorfile = f"{args.datafileroot}_desc-lagtcgenerator_timeseries"
    thepadtime = therunoptions["padseconds"]
    genlagtc = tide_resample.FastResamplerFromFile(lagtcgeneratorfile, padtime=thepadtime)

    # select the voxels in the mask
    print("figuring out valid voxels")
    validvoxels = np.where(procmask_spacebytime > 0)[0]
    numvalidspatiallocs = np.shape(validvoxels)[0]
    if args.debug:
        print(f"{numvalidspatiallocs=}")

    # slicing to valid voxels
    print("selecting valid voxels")
    fmri_data_valid = fmri_data_spacebytime[validvoxels, :]
    lagtimes_valid = lagtimes_spacebytime[validvoxels]
    corrmask_valid = corrmask_spacebytime[validvoxels]
    procmask_valid = procmask_spacebytime[validvoxels]
    if args.debug:
        print(f"{fmri_data_valid.shape=}")

    oversampfactor = int(therunoptions["oversampfactor"])
    if args.debug:
        print(f"{outputname=}")
    oversamptr = fmritr / oversampfactor
    try:
        threshval = therunoptions["regressfiltthreshval"]
    except KeyError:
        threshval = 0.0
        therunoptions["regressfiltthreshval"] = threshval
    mode = "glm"

    if args.debug:
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

    # windowed delay deviation estimation
    lagstouse_valid = lagtimes_valid

    # find the robust range of the static delays
    (
        pct02,
        pct98,
    ) = tide_stats.getfracvals(lagstouse_valid, [0.02, 0.98], debug=args.debug)
    if args.lagmin == -999:
        args.lagmin = np.round(pct02 / args.trainstep, 0) * args.trainstep
    if args.lagmax == -999:
        args.lagmax = np.round(pct98 / args.trainstep, 0) * args.trainstep

    print("\n\nWindowed delay estimation")
    TimingLGR.info("Windowed delay estimation start")
    LGR.info("\n\nWindowed delay estimation")

    if args.windelayoffsetgausssigma < 0.0:
        # set gausssigma automatically
        args.windelayoffsetgausssigma = np.mean([xdim, ydim, slicethickness]) / 2.0

    wintrs = int(np.round(args.windowsize / fmritr, 0))
    wintrs += wintrs % 2
    winskip = wintrs // 2
    numtrs = fmri_data_valid.shape[1]
    numwins = (numtrs // winskip) - 2
    winspace = winskip * fmritr
    winwidth = wintrs * fmritr

    # make a highpass filter
    if args.hpf:
        hpfcutoff = 1.0 / winwidth
        thehpf = tide_filt.NoncausalFilter(
            "arb",
            transferfunc="trapezoidal",
            padtime=30.0,
            padtype="reflect",
        )
        thehpf.setfreqs(hpfcutoff * 0.95, hpfcutoff, 0.15, 0.15)

    # make a filtered lagtc generator if necessary
    if args.hpf:
        reference_x, reference_y, dummy, dummy, genlagsamplerate = genlagtc.getdata()
        genlagtc = tide_resample.FastResampler(
            reference_x,
            thehpf.apply(genlagsamplerate, reference_y),
            padtime=thepadtime,
        )
        genlagtc.save(f"{outputname}_desc-hpflagtcgenerator_timeseries")

    # and filter the data if necessary
    if args.hpf:
        Fs = 1.0 / fmritr
        print("highpass filtering fmri data")
        themean = fmri_data_valid.mean(axis=1)
        for vox in range(fmri_data_valid.shape[0]):
            fmri_data_valid[vox, :] = thehpf.apply(Fs, fmri_data_valid[vox, :]) + themean[vox]
        if args.focaldebug:
            # dump the filtered fmri input file
            theheader = copy.deepcopy(fmri_header)
            theheader["dim"][4] = validtimepoints
            theheader["pixdim"][4] = fmritr

            maplist = [
                (
                    fmri_data_valid,
                    "hpfinputdata",
                    "bold",
                    None,
                    "fMRI data after highpass filtering",
                ),
            ]
            tide_io.savemaplist(
                outputname,
                maplist,
                validvoxels,
                (xsize, ysize, numslices, validtimepoints),
                theheader,
                bidsbasedict,
                filetype=theinputdata.filetype,
                rt_floattype=rt_floattype,
                cifti_hdr=None,
            )

    # allocate destination arrays
    internalwinspaceshape = (numvalidspatiallocs, numwins)
    internalwinspaceshapederivs = (
        numvalidspatiallocs,
        2,
        numwins,
    )
    internalwinfmrishape = (numvalidspatiallocs, wintrs)
    if args.debug:
        print(f"window space shape = {internalwinspaceshape}")
        print(f"internalwindowfmrishape shape = {internalwinfmrishape}")

    windowedregressderivratios = np.zeros(internalwinspaceshape, dtype=float)
    windowedregressrvalues = np.zeros(internalwinspaceshape, dtype=float)
    windowedmedfiltregressderivratios = np.zeros(internalwinspaceshape, dtype=float)
    windowedfilteredregressderivratios = np.zeros(internalwinspaceshape, dtype=float)
    windoweddelayoffset = np.zeros(internalwinspaceshape, dtype=float)
    windowedclosestoffset = np.zeros(internalwinspaceshape, dtype=float)
    if usesharedmem:
        if args.debug:
            print("allocating shared memory")
        winsLFOfitmean, winsLFOfitmean_shm = tide_util.allocshared(
            internalwinspaceshape, rt_outfloatset
        )
        winrvalue, winrvalue_shm = tide_util.allocshared(internalwinspaceshape, rt_outfloatset)
        winr2value, winr2value_shm = tide_util.allocshared(internalwinspaceshape, rt_outfloatset)
        winfitNorm, winfitNorm_shm = tide_util.allocshared(
            internalwinspaceshapederivs, rt_outfloatset
        )
        winfitcoeff, winitcoeff_shm = tide_util.allocshared(
            internalwinspaceshapederivs, rt_outfloatset
        )
        winmovingsignal, winmovingsignal_shm = tide_util.allocshared(
            internalwinfmrishape, rt_outfloatset
        )
        winlagtc, winlagtc_shm = tide_util.allocshared(internalwinfmrishape, rt_floatset)
        winfiltereddata, winfiltereddata_shm = tide_util.allocshared(
            internalwinfmrishape, rt_outfloatset
        )
    else:
        if args.debug:
            print("allocating memory")
        winsLFOfitmean = np.zeros(internalwinspaceshape, dtype=rt_outfloattype)
        winrvalue = np.zeros(internalwinspaceshape, dtype=rt_outfloattype)
        winr2value = np.zeros(internalwinspaceshape, dtype=rt_outfloattype)
        winfitNorm = np.zeros(internalwinspaceshapederivs, dtype=rt_outfloattype)
        winfitcoeff = np.zeros(internalwinspaceshapederivs, dtype=rt_outfloattype)
        winmovingsignal = np.zeros(internalwinfmrishape, dtype=rt_outfloattype)
        winlagtc = np.zeros(internalwinfmrishape, dtype=rt_floattype)
        winfiltereddata = np.zeros(internalwinfmrishape, dtype=rt_outfloattype)
    if args.debug:
        print(f"wintrs={wintrs}, winskip={winskip}, numtrs={numtrs}, numwins={numwins}")
    thewindowprocoptions = therunoptions
    if args.verbose:
        thewindowprocoptions["showprogressbar"] = True
    else:
        thewindowprocoptions["showprogressbar"] = False
    if args.focaldebug:
        thewindowprocoptions["saveminimumsLFOfiltfiles"] = True
        winoutputlevel = "max"
    else:
        thewindowprocoptions["saveminimumsLFOfiltfiles"] = False
        winoutputlevel = "min"

    # Now get the derivative ratios the individual windows
    print("Finding derivative ratios:")
    for thewin in range(numwins):
        print(f"\tProcessing window {thewin + 1} of {numwins}")
        starttr = thewin * winskip
        endtr = starttr + wintrs
        winlabel = f"_win-{str(thewin + 1).zfill(3)}"
        if args.verbose:
            thisLGR = LGR
            thisTimingLGR = TimingLGR
        else:
            thisLGR = None
            thisTimingLGR = None

        windowedregressderivratios[:, thewin], windowedregressrvalues[:, thewin] = (
            tide_refinedelay.getderivratios(
                fmri_data_valid,
                validvoxels,
                initial_fmri_x,
                lagstouse_valid,
                corrmask_valid,
                genlagtc,
                mode,
                outputname + winlabel,
                oversamptr,
                winsLFOfitmean[:, thewin],
                winrvalue[:, thewin],
                winr2value[:, thewin],
                winfitNorm[:, :, thewin],
                winfitcoeff[:, :, thewin],
                winmovingsignal,
                winlagtc,
                winfiltereddata,
                thisLGR,
                thisTimingLGR,
                thewindowprocoptions,
                regressderivs=1,
                starttr=starttr,
                endtr=endtr,
                debug=args.debug,
            )
        )
        if args.focaldebug:
            theheader = copy.deepcopy(fmri_header)
            theheader["dim"][4] = wintrs
            theheader["toffset"] = winwidth / 2.0
            maplist = [
                (
                    winlagtc,
                    "windowedlagtcs",
                    "bold",
                    None,
                    f"Lagtcs in each {winspace} second window",
                ),
            ]
            tide_io.savemaplist(
                outputname + winlabel,
                maplist,
                validvoxels,
                (xsize, ysize, numslices, wintrs),
                theheader,
                bidsbasedict,
                debug=args.debug,
            )

    # Filter the derivative ratios
    print("Filtering derivative ratios:")
    for thewin in range(numwins):
        print(f"\tProcessing window {thewin + 1} of {numwins}")
        (
            windowedmedfiltregressderivratios[:, thewin],
            windowedfilteredregressderivratios[:, thewin],
            windoweddelayoffsetMAD,
        ) = tide_refinedelay.filterderivratios(
            windowedregressderivratios[:, thewin],
            (xsize, ysize, numslices),
            validvoxels,
            (xdim, ydim, slicethickness),
            gausssigma=args.windelayoffsetgausssigma,
            patchthresh=args.delaypatchthresh,
            rt_floattype=rt_floattype,
            verbose=args.verbose,
            debug=args.debug,
        )

    # Train the ratio offsets
    print("Training ratio offsets:")
    for thewin in range(numwins):
        print(f"\tProcessing window {thewin + 1} of {numwins}")
        starttr = thewin * winskip
        endtr = starttr + wintrs
        winlabel = f"_win-{str(thewin + 1).zfill(3)}"
        # find the mapping of glm ratios to delays
        tide_refinedelay.trainratiotooffset(
            genlagtc,
            initial_fmri_x[starttr:endtr],
            outputname + winlabel,
            winoutputlevel,
            trainlagmin=args.lagmin,
            trainlagmax=args.lagmax,
            trainlagstep=args.trainstep,
            mindelay=args.mindelay,
            maxdelay=args.maxdelay,
            numpoints=args.numpoints,
            verbose=args.verbose,
            debug=args.focaldebug,
        )
        TimingLGR.info("Refinement calibration end")

    # now calculate the delay offsets
    print("Calculating delay offsets:")
    for thewin in range(numwins):
        print(f"\tProcessing window {thewin + 1} of {numwins}")
        winlabel = f"_win-{str(thewin + 1).zfill(3)}"
        TimingLGR.info("Calculating delay offsets")
        if args.debug:
            print(
                f"calculating delayoffsets for {windowedfilteredregressderivratios.shape[0]} voxels"
            )
        for i in range(windowedfilteredregressderivratios.shape[0]):
            (windoweddelayoffset[i, thewin], windowedclosestoffset[i, thewin]) = (
                tide_refinedelay.ratiotodelay(
                    windowedfilteredregressderivratios[i, thewin],
                    offset=lagstouse_valid[i],
                    debug=args.focaldebug,
                )
            )
        namesuffix = "_desc-delayoffset_hist"
        tide_stats.makeandsavehistogram(
            windoweddelayoffset[:, thewin],
            therunoptions["histlen"],
            1,
            outputname + winlabel + namesuffix,
            displaytitle="Histogram of delay offsets calculated from GLM",
            dictvarname="delayoffsethist",
            thedict=None,
        )

    # now see if there are common timecourses in the delay offsets
    themean = np.mean(windoweddelayoffset, axis=1)
    thevar = np.var(windoweddelayoffset, axis=1)
    scaledvoxels = windoweddelayoffset * 0.0
    for vox in range(0, windoweddelayoffset.shape[0]):
        scaledvoxels[vox, :] = windoweddelayoffset[vox, :] - themean[vox]
        if thevar[vox] > 0.0:
            scaledvoxels[vox, :] = scaledvoxels[vox, :] / thevar[vox]
    if args.systemicfittype == "pca":
        if args.pcacomponents < 0.0:
            pcacomponents = "mle"
        elif args.pcacomponents >= 1.0:
            pcacomponents = int(np.round(args.pcacomponents))
        elif args.pcacomponents == 0.0:
            print("0.0 is not an allowed value for pcacomponents")
            sys.exit()
        else:
            pcacomponents = args.pcacomponents

        # use the method of "A novel perspective to calibrate temporal delays in cerebrovascular reactivity
        # using hypercapnic and hyperoxic respiratory challenges". NeuroImage 187, 154?165 (2019).
        print(f"performing pca refinement with pcacomponents set to {pcacomponents}")
        try:
            thefit = PCA(n_components=pcacomponents).fit(scaledvoxels)
        except ValueError:
            if pcacomponents == "mle":
                print("mle estimation failed - falling back to pcacomponents=0.8")
                thefit = PCA(n_components=0.8).fit(scaledvoxels)
            else:
                print("unhandled math exception in PCA refinement - exiting")
                sys.exit()
        print(
            f"Using {len(thefit.components_)} component(s), accounting for "
            + f"{100.0 * np.cumsum(thefit.explained_variance_ratio_)[len(thefit.components_) - 1]}% of the variance"
        )
        reduceddata = thefit.inverse_transform(thefit.transform(scaledvoxels))
        # unscale the PCA cleaned data
        for vox in range(0, windoweddelayoffset.shape[0]):
            reduceddata[vox, :] = reduceddata[vox, :] * thevar[vox] + themean[vox]
        if args.debug:
            print("complex processing: reduceddata.shape =", scaledvoxels.shape)
        # pcadata = np.mean(reduceddata, axis=0)
        pcadata = thefit.components_[0]
        averagedata = np.mean(windoweddelayoffset, axis=0)
        thepxcorr = pearsonr(averagedata, pcadata)[0]
        LGR.info(f"pca/avg correlation = {thepxcorr}")
        if thepxcorr > 0.0:
            systemiccomp = 1.0 * pcadata
        else:
            systemiccomp = -1.0 * pcadata
        thecomponents = thefit.components_[:]
        tide_io.writebidstsv(
            f"{outputname}_desc-pcacomponents_timeseries",
            thecomponents,
            1.0 / winspace,
        )
        tide_io.writevec(
            100.0 * thefit.explained_variance_ratio_,
            f"{outputname}_desc-pcaexplainedvarianceratio_info.tsv",
        )
    elif args.systemicfittype == "mean":
        systemiccomp = np.mean(scaledvoxels, axis=0)
        reduceddata = None
    else:
        print("unhandled systemic filter type")
        sys.exit(0)
    tide_io.writebidstsv(
        f"{outputname}_desc-systemiccomponent_timeseries",
        systemiccomp,
        1.0 / winspace,
    )

    doregress = False
    if doregress:
        if usesharedmem:
            if args.debug:
                print("allocating shared memory")
            systemicsLFOfitmean, systemicsLFOfitmean_shm = tide_util.allocshared(
                internalwinspaceshape, rt_outfloatset
            )
            systemicrvalue, systemicrvalue_shm = tide_util.allocshared(
                internalwinspaceshape, rt_outfloatset
            )
            systemicr2value, systemicr2value_shm = tide_util.allocshared(
                internalwinspaceshape, rt_outfloatset
            )
            systemicfitNorm, systemicfitNorm_shm = tide_util.allocshared(
                internalwinspaceshapederivs, rt_outfloatset
            )
            systemicfitcoeff, systemicitcoeff_shm = tide_util.allocshared(
                internalwinspaceshapederivs, rt_outfloatset
            )
            systemicmovingsignal, systemicmovingsignal_shm = tide_util.allocshared(
                internalwinspaceshape, rt_outfloatset
            )
            systemiclagtc, systemiclagtc_shm = tide_util.allocshared(
                internalwinspaceshape, rt_floatset
            )
            systemicfiltereddata, systemicfiltereddata_shm = tide_util.allocshared(
                internalwinspaceshape, rt_outfloatset
            )
        else:
            if args.debug:
                print("allocating memory")
            systemicsLFOfitmean = np.zeros(internalwinspaceshape, dtype=rt_outfloattype)
            systemicrvalue = np.zeros(internalwinspaceshape, dtype=rt_outfloattype)
            systemicr2value = np.zeros(internalwinspaceshape, dtype=rt_outfloattype)
            systemicfitNorm = np.zeros(internalwinspaceshapederivs, dtype=rt_outfloattype)
            systemicfitcoeff = np.zeros(internalwinspaceshapederivs, dtype=rt_outfloattype)
            systemicmovingsignal = np.zeros(internalwinspaceshape, dtype=rt_outfloattype)
            systemiclagtc = np.zeros(internalwinspaceshape, dtype=rt_floattype)
            systemicfiltereddata = np.zeros(internalwinspaceshape, dtype=rt_outfloattype)

        windowlocs = np.linspace(0.0, winspace * numwins, num=numwins, endpoint=False) + skiptime
        voxelsprocessed_regressionfilt, regressorset, evset = tide_regressfrommaps.regressfrommaps(
            windoweddelayoffset,
            validvoxels,
            windowlocs,
            0.0 * lagstouse_valid,
            corrmask_valid,
            genlagtc,
            mode,
            outputname,
            oversamptr,
            systemicsLFOfitmean,
            systemicrvalue,
            systemicr2value,
            systemicfitNorm[:, :],
            systemicfitcoeff[:, :],
            systemicmovingsignal,
            systemiclagtc,
            systemicfiltereddata,
            LGR,
            TimingLGR,
            threshval,
            False,
            nprocs_makelaggedtcs=args.nprocs,
            nprocs_regressionfilt=args.nprocs,
            regressderivs=1,
            showprogressbar=args.showprogressbar,
            debug=args.debug,
        )

    theheader = copy.deepcopy(fmri_header)
    theheader["dim"][4] = numwins
    theheader["pixdim"][4] = winspace
    theheader["toffset"] = winwidth / 2.0
    maplist = [
        (
            windoweddelayoffset,
            "windoweddelayoffset",
            "info",
            None,
            f"Delay offsets in each {winspace} second window",
        ),
        (
            windowedclosestoffset,
            "windowedclosestoffset",
            "info",
            None,
            f"Closest delay offsets in each {winspace} second window",
        ),
        (
            np.square(windowedregressrvalues),
            "windowedregressr2values",
            "info",
            None,
            f"R2 values for regression in each {winspace} second window",
        ),
    ]
    if doregress:
        maplist += [
            (
                systemicfiltereddata,
                "systemicfiltereddata",
                "info",
                None,
                f"Systemic filtered delay offsets in each {winspace} second window",
            ),
            (
                np.square(systemicr2value),
                "systemicr2value",
                "info",
                None,
                f"R2 values for systemic regression in each {winspace} second window",
            ),
        ]
        if args.focaldebug:
            maplist += [
                (
                    systemicsLFOfitmean,
                    "systemicsLFOfitmean",
                    "info",
                    None,
                    f"Constant coefficient for systemic filter",
                ),
                (
                    systemicfitcoeff[:, 0],
                    "systemiccoffEV0",
                    "info",
                    None,
                    f"Coefficient 0 for systemic filter",
                ),
                (
                    systemicfitcoeff[:, 1],
                    "systemiccoffEV1",
                    "info",
                    None,
                    f"Coefficient 1 for systemic filter",
                ),
            ]
    if reduceddata is not None:
        maplist += (
            (
                reduceddata,
                "windoweddelayoffsetPCA",
                "info",
                None,
                f"PCA cleaned delay offsets in each {winspace} second window",
            ),
        )

    """(
        filtwindoweddelayoffset,
        "filtwindoweddelayoffset",
        "info",
        None,
        f"Delay offsets in each {winspace} second window with the systemic component removed",
    ),"""
    if args.focaldebug:
        maplist += [
            (
                windowedmedfiltregressderivratios,
                "windowedmedfiltregressderivratios",
                "info",
                None,
                f"Mediean filtered derivative ratios in each {winspace} second window",
            ),
            (
                windowedfilteredregressderivratios,
                "windowedfilteredregressderivratios",
                "info",
                None,
                f"Filtered derivative ratios in each {winspace} second window",
            ),
            (
                windowedregressderivratios,
                "windowedregressderivratios",
                "info",
                None,
                f"Raw derivative ratios in each {winspace} second window",
            ),
        ]
    tide_io.savemaplist(
        outputname,
        maplist,
        validvoxels,
        (xsize, ysize, numslices, numwins),
        theheader,
        bidsbasedict,
        debug=args.debug,
    )
    #########################
    # End window processing
    #########################

    # save outputs
    TimingLGR.info("Starting output save")
    bidsdict = bidsbasedict.copy()

    # read the runoptions file
    print("writing runoptions")
    therunoptions["delayvar_runtime"] = time.strftime(
        "%a, %d %b %Y %H:%M:%S %Z", time.localtime(time.time())
    )

    # clean up shared memory
    if usesharedmem:
        tide_util.cleanup_shm(winsLFOfitmean_shm)
        tide_util.cleanup_shm(winrvalue_shm)
        tide_util.cleanup_shm(winr2value_shm)
        tide_util.cleanup_shm(winfitNorm_shm)
        tide_util.cleanup_shm(winitcoeff_shm)
        tide_util.cleanup_shm(winmovingsignal_shm)
        tide_util.cleanup_shm(winlagtc_shm)
        tide_util.cleanup_shm(winfiltereddata_shm)
        if doregress:
            tide_util.cleanup_shm(systemicsLFOfitmean_shm)
            tide_util.cleanup_shm(systemicrvalue_shm)
            tide_util.cleanup_shm(systemicr2value_shm)
            tide_util.cleanup_shm(systemicfitNorm_shm)
            tide_util.cleanup_shm(systemicitcoeff_shm)
            tide_util.cleanup_shm(systemicmovingsignal_shm)
            tide_util.cleanup_shm(systemiclagtc_shm)
            tide_util.cleanup_shm(systemicfiltereddata_shm)
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
    Path(f"{outputname}_DELAYVARISRUNNING.txt").unlink()

    # create the finished file
    Path(f"{outputname}_DELAYVARDONE.txt").touch()


def process_args(inputargs=None):
    """
    Compile arguments for delayvar workflow.
    """
    args, argstowrite = pf.setargs(_get_parser, inputargs=inputargs)
    return args
