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

from matplotlib.pyplot import *

import rapidtide.filter as tide_filt
import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math
import rapidtide.resample as tide_resample
import rapidtide.voxelData as tide_voxelData
import rapidtide.workflows.parser_funcs as pf


def _get_parser():
    """
    Argument parser for simdata
    """
    parser = argparse.ArgumentParser(
        prog="simdata",
        description=("Generate simulated fMRI data with known correlation parameters"),
        allow_abbrev=False,
    )

    # Required arguments
    parser.add_argument(
        "fmritr",
        type=lambda x: pf.is_float(parser, x, minval=0.0),
        help="TR of the simulated data, in seconds.",
    )
    parser.add_argument(
        "numtrs",
        type=lambda x: pf.is_int(parser, x, minval=1),
        help="Number of TRs in the simulated data.",
    )
    pf.addreqinputniftifile(
        parser, "immeanfilename", addedtext="3D file with the mean value for each voxel"
    )
    parser.add_argument("outputroot", type=str, help="Root name for the output files.")

    for band in ["lfo", "resp", "cardiac"]:
        if band == "lfo":
            bandopts = parser.add_argument_group("LFO band options")
        elif band == "resp":
            bandopts = parser.add_argument_group("Resp band options")
        else:
            bandopts = parser.add_argument_group("Cardiac band options")
        strengthopts = bandopts.add_mutually_exclusive_group()
        strengthopts.add_argument(
            f"--{band}pctfile",
            dest=(f"{band}pctfile"),
            action="store",
            type=lambda x: pf.is_valid_file(parser, x),
            metavar="FILE",
            help=(f"3D NIFTI file with the {band} amplitude in percent of mean at every point"),
            default=None,
        )
        strengthopts.add_argument(
            f"--{band}signalfraction",
            dest=(f"{band}sigfracfile"),
            action="store",
            type=lambda x: pf.is_valid_file(parser, x),
            metavar="FILE",
            help=(
                f"3D NIFTI file with the {band} amplitude expressed as the percentage of inband variance accounted for by the regressor"
            ),
            default=None,
        )
        bandopts.add_argument(
            f"--{band}lagfile",
            dest=(f"{band}lagfile"),
            action="store",
            type=lambda x: pf.is_valid_file(parser, x),
            metavar="FILE",
            help=(f"3D NIFTI file with the {band} delay value in seconds at every point"),
            default=None,
        )
        bandopts.add_argument(
            f"--{band}regressor",
            dest=(f"{band}regressor"),
            action="store",
            type=lambda x: pf.is_valid_file(parser, x),
            metavar="FILE",
            help=(f"The {band} regressor text file"),
            default=None,
        )
        bandopts.add_argument(
            f"--{band}samprate",
            dest=(f"{band}samprate"),
            action="store",
            type=float,
            metavar="SAMPRATE",
            help=(f"The sample rate of the {band} regressor file, in Hz"),
            default=None,
        )
        bandopts.add_argument(
            f"--{band}starttime",
            dest=(f"{band}starttime"),
            action="store",
            type=float,
            metavar="STARTTIME",
            help=(
                "The time delay, in seconds, into the "
                + band
                + " regressor file that matches the start time of the fmrifile. Default is 0.0"
            ),
            default=None,
        )

    # optional arguments
    parser.add_argument(
        "--slicetimefile",
        dest="slicetimefile",
        action="store",
        type=str,
        metavar="FILE",
        help="Slice acquisition time file, either FSL format or BIDS sidecar.",
        default=None,
    )

    parser.add_argument(
        "--numskip",
        dest="numskip",
        action="store",
        type=int,
        metavar="SKIP",
        help=("Use to simulate tr periods deleted during preprocessing"),
        default=0,
    )
    parser.add_argument(
        "--globalnoiselevel",
        dest="globalnoiselevel",
        action="store",
        type=float,
        metavar="LEVEL",
        help=("The variance of the noise common to every voxel.  Default is 0.0"),
        default=0.0,
    )
    parser.add_argument(
        "--voxelnoiselevel",
        dest="voxelnoiselevel",
        action="store",
        type=float,
        metavar="LEVEL",
        help=(
            "The variance of the voxel specific noise, as percent of the voxel mean.  Default is 0.0"
        ),
        default=0.0,
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help=("Enable additional debugging output."),
        default=False,
    )

    # Miscellaneous options

    return parser


def prepareband(
    simdatadims,
    pctfile,
    sigfracfile,
    lagfile,
    regressorfile,
    samprate,
    starttime,
    regressorname,
    padtime=30.0,
    debug=False,
):
    if debug:
        print("simdatadims:", simdatadims)
        print("pctfile:", pctfile)
        print("sigfracfile:", sigfracfile)
        print("lagfile:", lagfile)
        print("regressorfile:", regressorfile)
        print("regressorname:", regressorname)
        print("padtime:", padtime)

    (
        insamprate,
        instarttime,
        colnames,
        invec,
        compressed,
        filetype,
    ) = tide_io.readvectorsfromtextfile(regressorfile, onecol=True)
    rawvec = tide_math.stdnormalize(invec)

    if starttime is None:
        if instarttime is None:
            starttime = 0.0
        else:
            starttime = instarttime
    if samprate is None:
        if insamprate is not None:
            samprate = insamprate
    if debug:
        print("samprate:", samprate)
        print("starttime:", starttime)

    # read in the timecourse to resample
    # rawvec = tide_math.stdnormalize(tide_io.readvec(regressorfile))
    regressorpts = len(rawvec)

    print("Input regressor has ", regressorpts, " points")
    regressor_x = (
        np.linspace(0.0, (1.0 / samprate) * regressorpts, num=regressorpts, endpoint=False)
        - starttime
    )
    regressor_y = rawvec[0:regressorpts]
    print(
        regressorname,
        "regressor has length",
        len(regressor_x),
        "and runs from ",
        regressor_x[0],
        " to ",
        regressor_x[-1],
    )

    if pctfile is not None:
        nim_pct, pctdata, pctheader, pctdims, pctsizes = tide_io.readfromnifti(pctfile)
        pctscale = True
    else:
        nim_pct, pctdata, pctheader, pctdims, pctsizes = tide_io.readfromnifti(sigfracfile)
        pctscale = False
        if not tide_io.checkspacedimmatch(pctdims, simdatadims):
            print(regressorname, "pct file does not match fmri")
            exit()
        pctdata /= 100.0
    nim_lag, lagdata, lagheader, lagdims, lagsizes = tide_io.readfromnifti(lagfile)
    if not tide_io.checkspacedimmatch(lagdims, simdatadims):
        print(regressorname, "lag file does not match fmri")
        exit()

    generator = tide_resample.FastResampler(
        regressor_x, regressor_y, padtime=padtime, doplot=False
    )
    return pctdata, pctscale, lagdata, generator


def fmrisignal(
    Fs,
    times,
    meanvalue,
    dolfo=False,
    lfowave=None,
    lfomag=None,
    lfodelay=None,
    lfonoise=0.0,
    lfofilter=None,
    doresp=False,
    respwave=None,
    respmag=None,
    respdelay=None,
    respnoise=0.0,
    respfilter=None,
    docardiac=False,
    cardiacwave=None,
    cardiacmag=None,
    cardiacdelay=None,
    cardiacnoise=0.0,
    cardiacfilter=None,
):
    thesignal = np.zeros((len(times)), dtype=float)
    if dolfo:
        thesignal += meanvalue * (
            lfomag * lfowave.yfromx(times - lfodelay)
            + lfonoise * lfofilter.apply(Fs, np.random.standard_normal(len(times)))
        )
    if doresp:
        thesignal += meanvalue * (
            respmag * respwave.yfromx(times - respdelay)
            + respnoise * respfilter.apply(Fs, np.random.standard_normal(len(times)))
        )
    if docardiac:
        thesignal += meanvalue * (
            cardiacmag * cardiacwave.yfromx(times - cardiacdelay)
            + cardiacnoise * cardiacfilter.apply(Fs, np.random.standard_normal(len(times)))
        )
    return thesignal + meanvalue


def simdata(args):
    # set default variable values
    lfopctdata = None
    lfolagdata = None
    lfogenerator = None
    lfofilter = None

    resppctdata = None
    resplagdata = None
    respgenerator = None
    respfilter = None

    cardiacpctdata = None
    cardiaclagdata = None
    cardiacgenerator = None
    cardiacfilter = None

    # check for complete information
    if (
        ((args.lfopctfile is None) and (args.lfosigfracfile is None))
        or (args.lfolagfile is None)
        or (args.lforegressor is None)
        or ((args.lfosamprate is None) and (tide_io.parsefilespec(args.lforegressor)[1] is None))
    ):
        print("lfopctfile:", args.lfopctfile)
        print("lfosigfracfile:", args.lfosigfracfile)
        print("lfolagfile:", args.lfolagfile)
        print("lforegressor:", args.lforegressor)
        print("lfopctsamprate:", args.lfosamprate)
        dolfo = False
    else:
        dolfo = True
        lfofilter = tide_filt.NoncausalFilter("lfo")
        print("LFO information is complete, will be included.")

    if (
        ((args.resppctfile is None) and (args.respsigfracfile is None))
        or (args.resplagfile is None)
        or (args.respregressor is None)
        or ((args.respsamprate is None) and (tide_io.parsefilespec(args.respregressor)[1] is None))
    ):
        doresp = False
    else:
        doresp = True
        respfilter = tide_filt.NoncausalFilter("resp")
        print("Respiratory information is complete, will be included.")

    if (
        ((args.cardiacpctfile is None) and (args.cardiacsigfracfile is None))
        or (args.cardiaclagfile is None)
        or (args.cardiacregressor is None)
        or (
            (args.cardiacsamprate is None)
            and (tide_io.parsefilespec(args.cardiacregressor)[1] is None)
        )
    ):
        docardiac = False
    else:
        docardiac = True
        cardiacfilter = tide_filt.NoncausalFilter("cardiac")
        print("Cardiac information is complete, will be included.")
    if not (dolfo or doresp or docardiac):
        print(
            "Must specify parameters for at least one of LFO, respiration, or cardiac signals - exiting"
        )
        _get_parser().print_help()
        sys.exit()

    print(f"simulated fmri data: {args.numtrs} timepoints, tr = {args.fmritr}")

    # prepare the output timepoints
    initial_fmri_x = (
        np.linspace(
            0.0,
            args.fmritr * (args.numtrs - args.numskip),
            num=(args.numtrs - args.numskip),
            endpoint=False,
        )
        + args.fmritr * args.numskip
    )
    print("length of fmri after removing skip:", len(initial_fmri_x))
    print(
        f"fmri time has length {len(initial_fmri_x)}",
        f"and runs runs from {initial_fmri_x[0]} to {initial_fmri_x[-1]}",
    )

    # read in the immean file
    print("reading in source files")
    theimmeandata = tide_voxelData.VoxelData(args.immeanfilename, timestep=args.fmritr)
    immeandata = theimmeandata.byvol()

    # now set up the simulated data array
    simdataheader = theimmeandata.copyheader(
        numtimepoints=len(initial_fmri_x), tr=args.fmritr, toffset=args.numskip * args.fmritr
    )
    simdatadims = simdataheader["dim"].copy()
    xsize, ysize, numslices, timepoints = tide_io.parseniftidims(simdatadims)
    simdata = np.zeros((xsize, ysize, numslices, timepoints), dtype="float")

    # read in the slicetimes file if we have one
    if args.slicetimefile is not None:
        sliceoffsettimes, normalizedtotr, fileisjson = tide_io.getslicetimesfromfile(
            args.slicetimefile
        )
    else:
        sliceoffsettimes = np.zeros((numslices), dtype=float)

    # set up fast resampling
    padtime = 60.0
    numpadtrs = int(padtime / args.fmritr)
    padtime = args.fmritr * numpadtrs

    # prepare the input data for interpolation
    if dolfo:
        lfopctdata, lfopctscale, lfolagdata, lfogenerator = prepareband(
            simdatadims,
            args.lfopctfile,
            args.lfosigfracfile,
            args.lfolagfile,
            args.lforegressor,
            args.lfosamprate,
            args.lfostarttime,
            "LFO",
            padtime=padtime,
            debug=args.debug,
        )
    if doresp:
        resppctdata, resppctscale, resplagdata, respgenerator = prepareband(
            simdatadims,
            args.resppctfile,
            args.respsigfracfile,
            args.resplagfile,
            args.respregressor,
            args.respsamprate,
            args.respstarttime,
            "respiratory",
            padtime=padtime,
            debug=args.debug,
        )
    if docardiac:
        cardiacpctdata, cardiacpctscale, cardiaclagdata, cardiacgenerator = prepareband(
            simdatadims,
            args.cardiacpctfile,
            args.cardiacsigfracfile,
            args.cardiaclagfile,
            args.cardiacregressor,
            args.cardiacsamprate,
            args.cardiacstarttime,
            "cardiac",
            padtime=padtime,
            debug=args.debug,
        )

    # loop over space
    theglobalnoise = args.globalnoiselevel * np.random.standard_normal(len(initial_fmri_x))
    for k in range(0, numslices):
        fmri_x_slice = initial_fmri_x - sliceoffsettimes[k]
        print("processing slice ", k, ": sliceoffsettime=", sliceoffsettimes[k])
        for j in range(0, ysize):
            for i in range(0, xsize):
                # generate the noise
                thevoxelnoise = args.voxelnoiselevel * np.random.standard_normal(
                    len(initial_fmri_x)
                )

                # add in the signals
                if dolfo:
                    lfopct = lfopctdata[i, j, k]
                    if lfopctscale:
                        lfonoise = 0.0
                    else:
                        lfonoise = 1.0 - lfopct
                    lfolag = lfolagdata[i, j, k]
                else:
                    lfopct = 0.0
                    lfolag = 0.0
                    lfonoise = 0.0
                if doresp:
                    resppct = resppctdata[i, j, k]
                    if resppctscale:
                        respnoise = 0.0
                    else:
                        respnoise = 1.0 - resppct
                    resplag = resplagdata[i, j, k]
                else:
                    resppct = 0.0
                    resplag = 0.0
                    respnoise = 0.0
                if docardiac:
                    cardiacpct = cardiacpctdata[i, j, k]
                    if cardiacpctscale:
                        cardiacnoise = 0.0
                    else:
                        cardiacnoise = 1.0 - cardiacpct
                    cardiaclag = cardiaclagdata[i, j, k]
                else:
                    cardiacpct = 0.0
                    cardiaclag = 0.0
                    cardiacnoise = 0.0

                simdata[i, j, k, :] = (
                    fmrisignal(
                        (1.0 / args.fmritr),
                        fmri_x_slice,
                        immeandata[i, j, k],
                        dolfo=dolfo,
                        lfowave=lfogenerator,
                        lfomag=lfopct,
                        lfodelay=lfolag,
                        lfonoise=lfonoise,
                        lfofilter=lfofilter,
                        doresp=doresp,
                        respwave=respgenerator,
                        respmag=resppct,
                        respdelay=resplag,
                        respnoise=respnoise,
                        respfilter=respfilter,
                        docardiac=docardiac,
                        cardiacwave=cardiacgenerator,
                        cardiacmag=cardiacpct,
                        cardiacdelay=cardiaclag,
                        cardiacnoise=cardiacnoise,
                        cardiacfilter=cardiacfilter,
                    )
                    + theglobalnoise
                    + (thevoxelnoise) / 100.0 * immeandata[i, j, k]
                )

    tide_io.savetonifti(simdata, simdataheader, args.outputroot)
