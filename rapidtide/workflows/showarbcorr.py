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

import numpy as np
from scipy.stats import pearsonr

import rapidtide.correlate as tide_corr
import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math
import rapidtide.simFuncClasses as tide_simFuncClasses
import rapidtide.workflows.parser_funcs as pf


def _get_parser():
    """
    Argument parser for showarbcorr
    """
    parser = argparse.ArgumentParser(
        prog="showarbcorr",
        description=(
            "Calculate and display crosscorrelation between two timeseries. "
            "Timeseries do not have to have the same length or sampling frequency."
        ),
        allow_abbrev=False,
    )

    # Required arguments
    pf.addreqinputtextfile(parser, "infilename1", onecol=True)
    pf.addreqinputtextfile(parser, "infilename2", onecol=True)

    # add optional arguments
    parser.add_argument(
        "--samplerate1",
        type=lambda x: pf.is_float(parser, x),
        help="Sample rate of timecourse 1, in Hz",
        default=None,
    )
    parser.add_argument(
        "--samplerate2",
        type=lambda x: pf.is_float(parser, x),
        help="Sample rate of timecourse 2, in Hz",
        default=None,
    )

    # add optional arguments
    parser.add_argument(
        "--nodisplay",
        dest="display",
        action="store_false",
        help=("Do not plot the data (for noninteractive use)"),
        default=True,
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help=("Enable additional debugging output."),
        default=False,
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help=("Print out more debugging information"),
        default=False,
    )
    pf.addsearchrangeopts(parser, details=True)
    pf.addtimerangeopts(parser)
    parser.add_argument(
        "--trimdata",
        dest="trimdata",
        action="store_true",
        help=("Trimming data to match"),
        default=False,
    )

    preproc = parser.add_argument_group()
    preproc.add_argument(
        "--detrendorder",
        dest="detrendorder",
        action="store",
        type=int,
        metavar="ORDER",
        help=("Set order of trend removal (0 to disable, default is 1 - linear). "),
        default=1,
    )

    # add window options
    pf.addwindowopts(parser)

    # Filter arguments
    pf.addfilteropts(parser, filtertarget="timecourses", details=True)

    # Preprocessing options
    preproc = parser.add_argument_group("Preprocessing options")
    preproc.add_argument(
        "--corrweighting",
        dest="corrweighting",
        action="store",
        type=str,
        choices=["None", "phat", "liang", "eckart"],
        help=("Method to use for cross-correlation " "weighting. Default is  None. "),
        default="None",
    )
    preproc.add_argument(
        "--invert",
        dest="invert",
        action="store_true",
        help=("Invert one timecourse prior to correlation. "),
        default=False,
    )
    preproc.add_argument(
        "--label",
        dest="label",
        metavar="LABEL",
        action="store",
        type=str,
        help=("Label for the delay value. "),
        default="None",
    )

    # Add permutation options
    pf.addpermutationopts(parser, numreps=0)

    # similarity function options
    similarityopts = parser.add_argument_group("Similarity function options")
    pf.addsimilarityopts(similarityopts)

    # fitting options
    fittingopts = parser.add_argument_group("Fitting options")
    fittingopts.add_argument(
        "--bipolar",
        dest="bipolar",
        action="store_true",
        help=("Fit largest magnitude peak regardless of sign."),
        default=False,
    )
    # Output options
    output = parser.add_argument_group("Output options")
    output.add_argument(
        "--outputfile",
        dest="outputfile",
        action="store",
        type=str,
        metavar="FILE",
        help=("Write results to FILE. "),
        default=None,
    )
    output.add_argument(
        "--corroutputfile",
        dest="corroutputfile",
        action="store",
        type=str,
        metavar="FILE",
        help=("Write correlation function to FILE. "),
        default=None,
    )
    output.add_argument(
        "--graphfile",
        dest="graphfile",
        action="store",
        type=str,
        metavar="FILE",
        help=("Output an image of the correlation function to FILE. "),
        default=None,
    )
    output.add_argument(
        "--summarymode",
        dest="summarymode",
        action="store_true",
        help=("Print all results on a single line. "),
        default="False",
    )
    output.add_argument(
        "--labelline",
        dest="labelline",
        action="store_true",
        help=("Print a header line identifying fields in the summary line. "),
        default="False",
    )

    # Miscellaneous options
    misc = parser.add_argument_group("Miscellaneous options")
    misc.add_argument(
        "--noprogressbar",
        dest="showprogressbar",
        action="store_false",
        help="Will disable showing progress bars (helpful if stdout is going to a file). ",
        default=True,
    )
    misc.add_argument(
        "--nonorm",
        dest="minorm",
        action="store_false",
        help="Will disable normalization of the mutual information function. ",
        default=True,
    )
    misc.add_argument(
        "--nprocs",
        dest="nprocs",
        action="store",
        type=int,
        metavar="NPROCS",
        help=(
            "Use NPROCS worker processes for multiprocessing. "
            "Setting NPROCS to less than 1 sets the number of "
            "worker processes to n_cpus - 1. "
        ),
        default=1,
    )

    misc.add_argument(
        "--saveres",
        dest="saveres",
        action="store",
        type=int,
        metavar="SAVERES",
        help=("If saving graphics, use SAVERES ppi."),
        default=1,
    )

    return parser


def printthresholds(pcts, thepercentiles, labeltext):
    print(labeltext)
    for i in range(0, len(pcts)):
        print("\tp <", "{:.3f}".format(1.0 - thepercentiles[i]), ": ", pcts[i])


def showarbcorr(args):
    # set some default values
    absmaxsigma = 1000.0
    absminsigma = 0.25
    zerooutbadfit = False
    peakfittype = "gauss"

    # finish up processing arguments
    args, theprefilter = pf.postprocessfilteropts(args)
    args = pf.postprocesssearchrangeopts(args)
    args = pf.postprocesstimerangeopts(args)

    if args.display:
        import matplotlib as mpl

        mpl.use("TkAgg")
        import matplotlib.pyplot as plt

    Fs1, starttime1, dummy, inputdata1, dummy, dummy = tide_io.readvectorsfromtextfile(
        args.infilename1
    )
    inputdata1 = np.transpose(inputdata1)
    if np.shape(inputdata1)[1] > 1:
        print("specify only one column for input file 1")
        sys.exit()
    else:
        inputdata1 = inputdata1[:, 0]

    Fs2, starttime2, dummy, inputdata2, dummy, dummy = tide_io.readvectorsfromtextfile(
        args.infilename2
    )
    inputdata2 = np.transpose(inputdata2)
    if np.shape(inputdata2)[1] > 1:
        print("specify only one column for input file 2")
        sys.exit()
    else:
        inputdata2 = inputdata2[:, 0]

    if args.debug:
        dumpfiltered = True
    else:
        dumpfiltered = False
    showpearson = True

    if args.samplerate1 is not None:
        Fs1 = args.samplerate1
    if Fs1 is None:
        print("sample rate must be specified for timecourse 1 - exiting")
        sys.exit()
    if starttime1 == None:
        starttime1 = 0.0
    endtime1 = starttime1 + len(inputdata1) / Fs1
    print(f"inputdata1 goes from {starttime1} to {endtime1}")

    if args.samplerate2 is not None:
        Fs2 = args.samplerate2
    if Fs2 is None:
        print("sample rate must be specified for timecourse 2 - exiting")
        sys.exit()
    if starttime2 == None:
        starttime2 = 0.0
    endtime2 = starttime2 + len(inputdata2) / Fs2
    print(f"inputdata2 goes from {starttime2} to {endtime2}")

    matchedinput1, matchedinput2, commonFs = tide_corr.matchsamplerates(
        inputdata1,
        Fs1,
        inputdata2,
        Fs2,
        method="univariate",
        debug=args.debug,
    )
    trimdata1 = matchedinput1
    trimdata2 = matchedinput2

    if args.trimdata:
        minlen = np.min([len(trimdata1), len(trimdata2)])
        trimdata1 = trimdata1[0:minlen]
        trimdata2 = trimdata2[0:minlen]

    if args.invert:
        flipfac = -1.0
    else:
        flipfac = 1.0

    # band limit the regressor if that is needed
    if theprefilter.gettype() != "None":
        if args.verbose:
            print("filtering to ", theprefilter.gettype(), " band")
    filtereddata1 = tide_math.corrnormalize(
        theprefilter.apply(commonFs, trimdata1),
        detrendorder=args.detrendorder,
        windowfunc=args.windowfunc,
    )
    filtereddata2 = tide_math.corrnormalize(
        theprefilter.apply(commonFs, trimdata2),
        detrendorder=args.detrendorder,
        windowfunc=args.windowfunc,
    )
    if dumpfiltered:
        tide_io.writenpvecs(filtereddata1, "filtereddata1.txt")
        tide_io.writenpvecs(filtereddata2, "filtereddata2.txt")

    if args.debug:
        print(f"{Fs1=}, {Fs2=}, {starttime1=}, {starttime2=}, {args.windowfunc=}")
    xcorr_x, thexcorr, corrFs, zeroloc = tide_corr.arbcorr(
        filtereddata1,
        commonFs,
        filtereddata2,
        commonFs,
        start1=starttime1,
        start2=starttime2,
        windowfunc=args.windowfunc,
        method="univariate",
        debug=args.debug,
    )

    if args.display:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(
            xcorr_x,
            thexcorr,
        )
        ax.set_title(f"{args.label} crosscorrelation function")
        if args.graphfile is not None:
            plt.savefig(args.graphfile, bbox_inches="tight")
        else:
            plt.show()

    # do the correlation
    print("Correlator lengths (x, y):", len(xcorr_x), len(thexcorr))
    print("lagmin, lagmax", args.lagmin, args.lagmax)
    lowerlim = int((args.lagmin * corrFs) - 0.5) + zeroloc
    upperlim = int((args.lagmax * corrFs) + 0.5) + zeroloc
    print("Fs1, Fs2, corrFs", Fs1, Fs2, corrFs)
    print("lowerlim, upperlim", lowerlim, upperlim)
    xcorr_x_trim = xcorr_x[lowerlim:upperlim]
    thexcorr_trim = thexcorr[lowerlim:upperlim]
    print("trimmed Correlator lengths (x, y):", len(xcorr_x_trim), len(thexcorr_trim))

    print(f"{len(filtereddata1)=}, {len(filtereddata2)=}")
    thepxcorr = pearsonr(filtereddata1, filtereddata2)

    # initialize the correlation fitter
    thexsimfuncfitter = tide_simFuncClasses.SimilarityFunctionFitter(
        corrtimeaxis=xcorr_x,
        lagmin=args.lagmin,
        lagmax=args.lagmax,
        absmaxsigma=absmaxsigma,
        absminsigma=absminsigma,
        debug=args.debug,
        bipolar=args.bipolar,
        peakfittype=peakfittype,
        functype="correlation",
        zerooutbadfit=zerooutbadfit,
        useguess=False,
    )

    if args.bipolar:
        maxdelay = xcorr_x_trim[np.argmax(np.fabs(thexcorr_trim))]
    else:
        maxdelay = xcorr_x_trim[np.argmax(thexcorr_trim)]
    if args.debug:
        print("\n\nmaxdelay before refinement", maxdelay)

    (
        maxindex,
        maxdelay,
        maxval,
        maxsigma,
        maskval,
        failreason,
        peakstart,
        peakend,
    ) = thexsimfuncfitter.fit(flipfac * thexcorr)
    if failreason > 0:
        print("showarbcorr: FIT FAILED with reason:")
        print(thexsimfuncfitter.diagnosefail(np.uint32(failreason)))
    if args.debug:
        print(maxindex, maxdelay, maxval, maxsigma, maskval, failreason)
    R = maxval
    if args.debug:
        print("maxdelay after refinement", maxdelay)

    if args.summarymode:
        thelabelitems = ["xcorr_R", "xcorr_maxdelay", "failreason"]
        thedataitems = [
            str(flipfac * R),
            str(-maxdelay),
            '"' + thexsimfuncfitter.diagnosefail(np.uint32(failreason)) + '"',
        ]

        if args.label is not None:
            thelabelitems = ["thelabel"] + thelabelitems
            thedataitems = [args.label] + thedataitems
        if args.labelline:
            outputstring = "\t".join(thelabelitems) + "\n" + "\t".join(thedataitems)
        else:
            outputstring = "\t".join(thedataitems)
        if args.outputfile is None:
            print(outputstring)
        else:
            with open(args.outputfile, "w") as text_file:
                text_file.write(outputstring + "\n")
    else:
        # report the pearson correlation
        if showpearson:
            print("Pearson_R:\t", thepxcorr[0])
            print("")
        if args.label is not None:
            print(args.label, ":\t", -maxdelay)
        else:
            print("Crosscorrelation_Rmax:\t", R)
            print("Crosscorrelation_maxdelay:\t", -maxdelay)
            print("Failreason:\t,", thexsimfuncfitter.diagnosefail(np.uint32(failreason)))
            print(
                args.infilename1, "[0 seconds] == ", args.infilename2, "[", -maxdelay, " seconds]"
            )

    if args.corroutputfile is not None:
        tide_io.writenpvecs(np.stack((xcorr_x, thexcorr), axis=0), args.corroutputfile)
