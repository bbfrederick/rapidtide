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
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import argparse
import sys

import nibabel as nib
import numpy as np
from scipy.stats import pearsonr

import rapidtide.correlate as tide_corr
import rapidtide.fit as tide_fit
import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math
import rapidtide.resample as tide_resample
import rapidtide.stats as tide_stats
import rapidtide.workflows.parser_funcs as pf

DEFAULT_DETREND_ORDER = 3
DEFAULT_CORRWEIGHTING = "phat"


def _get_parser():
    """
    Argument parser for ccorrica
    """
    parser = argparse.ArgumentParser(
        prog="ccorrica",
        description=("Find temporal crosscorrelations between a set of timecourses"),
        allow_abbrev=False,
    )

    # Required arguments
    pf.addreqinputtextfile(parser, "timecoursefile")
    pf.addreqoutputtextfile(parser, "outputroot", rootname=True)

    # add optional arguments
    freq_group = parser.add_mutually_exclusive_group()
    freq_group.add_argument(
        "--samplerate",
        dest="samplerate",
        action="store",
        type=lambda x: pf.is_float(parser, x),
        metavar="FREQ",
        help=(
            "Timecourses in file have sample "
            "frequency FREQ (default is 1.0Hz) "
            "NB: --samplerate and --sampletstep) "
            "are two ways to specify the same thing."
        ),
        default="auto",
    )
    freq_group.add_argument(
        "--sampletstep",
        dest="samplerate",
        action="store",
        type=lambda x: pf.invert_float(parser, x),
        metavar="TSTEP",
        help=(
            "Timecourses in file have sample "
            "timestep TSTEP (default is 1.0s) "
            "NB: --samplerate and --sampletstep) "
            "are two ways to specify the same thing."
        ),
        default="auto",
    )

    # add window options
    pf.addwindowopts(parser)

    # Search range arguments
    pf.addsearchrangeopts(parser)

    # Filter arguments
    pf.addfilteropts(parser, filtertarget="timecourses")

    parser.add_argument(
        "--corrweighting",
        dest="corrweighting",
        action="store",
        type=str,
        choices=["None", "phat", "liang", "eckart"],
        help=(
            f"Method to use for cross-correlation weighting. Default is {DEFAULT_CORRWEIGHTING}. "
        ),
        default=DEFAULT_CORRWEIGHTING,
    )
    parser.add_argument(
        "--detrendorder",
        dest="detrendorder",
        type=int,
        help=(f"Detrending order (default is {DEFAULT_DETREND_ORDER}).  Set to 0 to disable"),
        default=DEFAULT_DETREND_ORDER,
    )

    parser.add_argument(
        "--oversampfactor",
        dest="oversampfactor",
        type=int,
        help=(
            "Factor by which to oversample timecourses prior to correlation.  Default is 1. If set negative, "
            "factor will be set automatically."
        ),
        default=1,
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


def ccorrica(args):
    args, theprefilter = pf.postprocessfilteropts(args)

    # read in data
    (
        samplerate,
        starttime,
        colnames,
        tcdata,
        compressed,
        filetype,
    ) = tide_io.readvectorsfromtextfile(args.timecoursefile)

    if samplerate is None:
        if args.samplerate == "auto":
            print(
                "sample rate must be specified, either by command line arguments or in the file header."
            )
            sys.exit()
        else:
            Fs = args.samplerate
    else:
        Fs = samplerate

    sampletime = 1.0 / Fs
    thedims = tcdata.shape
    tclen = thedims[1]
    numcomponents = thedims[0]

    # check to see if we need to adjust the oversample factor
    if args.oversampfactor < 0:
        args.oversampfactor = int(np.max([np.ceil(sampletime // 0.5), 1]))
        print("oversample factor set to", args.oversampfactor)

    reformdata = np.reshape(tcdata, (numcomponents, tclen))
    if args.oversampfactor == 1:
        print("data array shape is ", reformdata.shape)
    else:
        resampdata = np.zeros((numcomponents, tclen * args.oversampfactor), dtype=np.float)
        for component in range(0, numcomponents):
            resampdata[component, :] = tide_resample.upsample(
                reformdata[component, :], Fs, Fs * args.oversampfactor, intfac=True
            )
        reformdata = resampdata
        Fs *= args.oversampfactor
        tclen *= args.oversampfactor

    # filter the data
    for component in range(0, numcomponents):
        reformdata[component, :] = tide_math.stdnormalize(
            theprefilter.apply(Fs, reformdata[component, :])
        )

    # save the filtered timecourses
    tide_io.writenpvecs(reformdata, args.outputroot + "_filtereddata.txt")

    # now detrend, window, and normalize the data
    for component in range(0, numcomponents):
        reformdata[component, :] = tide_math.corrnormalize(
            reformdata[component, :],
            detrendorder=args.detrendorder,
            windowfunc=args.windowfunc,
        )

    xcorrlen = 2 * tclen - 1
    sampletime = 1.0 / Fs
    xcorr_x = np.r_[0.0:xcorrlen] * sampletime - (xcorrlen * sampletime) / 2.0 + sampletime / 2.0
    searchrange = 15.0
    widthmax = 15.0

    halfwindow = int(searchrange * Fs)
    corrzero = xcorrlen // 2
    searchstart = corrzero - halfwindow
    searchend = corrzero + halfwindow
    corrwin = searchend - searchstart

    outputdata = np.zeros((numcomponents, numcomponents, 1, xcorrlen), dtype="float")
    outputpdata = np.zeros((numcomponents, numcomponents, 1, tclen), dtype="float")
    outputcorrmax = np.zeros((numcomponents, numcomponents, 1, 1), dtype="float")
    outputcorrlag = np.zeros((numcomponents, numcomponents, 1, 1), dtype="float")
    outputcorrwidth = np.zeros((numcomponents, numcomponents, 1, 1), dtype="float")
    outputcorrmask = np.zeros((numcomponents, numcomponents, 1, 1), dtype="float")
    for component1 in range(0, numcomponents):
        print("correlating with component", component1)
        for component2 in range(0, numcomponents):
            thexcorr = tide_corr.fastcorrelate(
                reformdata[component1, :],
                reformdata[component2, :],
                usefft=True,
                weighting=args.corrweighting,
                zeropadding=0,
                displayplots=args.debug,
            )
            thepxcorr = pearsonr(reformdata[component1, :] / tclen, reformdata[component2, :])
            outputdata[component1, component2, 0, :] = thexcorr
            outputpdata[component1, component2, 0, :] = thepxcorr[0]
            (
                maxindex,
                maxlag,
                maxval,
                maxsigma,
                maskval,
                failreason,
                peakstart,
                peakend,
            ) = tide_fit.findmaxlag_gauss(
                xcorr_x[searchstart:searchend],
                thexcorr[searchstart:searchend],
                -searchrange,
                searchrange,
                widthmax,
                refine=True,
                useguess=False,
                fastgauss=False,
                displayplots=False,
            )
            outputcorrmax[component1, component2, 0, 0] = maxval
            outputcorrlag[component1, component2, 0, 0] = maxlag
            outputcorrwidth[component1, component2, 0, 0] = maxsigma
            outputcorrmask[component1, component2, 0, 0] = maskval

    # symmetrize the matrices
    outputcorrmax[:, :, 0, 0] = tide_stats.symmetrize(outputcorrmax[:, :, 0, 0], zerodiagonal=True)
    outputcorrlag[:, :, 0, 0] = tide_stats.symmetrize(
        outputcorrlag[:, :, 0, 0], antisymmetric=True
    )
    outputcorrwidth[:, :, 0, 0] = tide_stats.symmetrize(outputcorrwidth[:, :, 0, 0])
    outputcorrmask[:, :, 0, 0] = tide_stats.symmetrize(
        outputcorrmask[:, :, 0, 0], zerodiagonal=True
    )

    # show()
    outputaffine = np.eye(4)
    out4d_hdr = nib.Nifti1Image(outputdata[:, :, :, searchstart:searchend], outputaffine).header
    out4d_hdr["pixdim"][4] = sampletime
    out4d_sizes = out4d_hdr["pixdim"]
    tide_io.savetonifti(
        outputdata[:, :, :, searchstart:searchend], out4d_hdr, args.outputroot + "_xcorr"
    )

    outputaffine = np.eye(4)
    out4d_hdr = nib.Nifti1Image(outputpdata, outputaffine).header
    out4d_hdr["pixdim"][4] = sampletime
    out4d_sizes = out4d_hdr["pixdim"]
    tide_io.savetonifti(outputpdata, out4d_hdr, args.outputroot + "_pxcorr")

    out3d_hdr = nib.Nifti1Image(outputcorrmax, outputaffine).header
    out3d_hdr["pixdim"][4] = sampletime
    out3d_sizes = out3d_hdr["pixdim"]
    tide_io.savetonifti(outputcorrmax, out3d_hdr, args.outputroot + "_corrmax")
    tide_io.writenpvecs(
        outputcorrmax.reshape(numcomponents, numcomponents), args.outputroot + "_corrmax.txt"
    )
    tide_io.savetonifti(outputcorrlag, out3d_hdr, args.outputroot + "_corrlag")
    tide_io.writenpvecs(
        outputcorrlag.reshape(numcomponents, numcomponents), args.outputroot + "_corrlag.txt"
    )
    tide_io.savetonifti(outputcorrwidth, out3d_hdr, args.outputroot + "_corrwidth")
    tide_io.writenpvecs(
        outputcorrwidth.reshape(numcomponents, numcomponents),
        args.outputroot + "_corrwidth.txt",
    )
    tide_io.savetonifti(outputcorrmask, out3d_hdr, args.outputroot + "_corrmask")
    tide_io.writenpvecs(
        outputcorrmask.reshape(numcomponents, numcomponents),
        args.outputroot + "_corrmask.txt",
    )

    tide_io.writenpvecs(reformdata, args.outputroot + "_reformdata.txt")
