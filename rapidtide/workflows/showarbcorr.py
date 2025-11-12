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
from scipy.stats import pearsonr

import rapidtide.correlate as tide_corr
import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math
import rapidtide.simFuncClasses as tide_simFuncClasses
import rapidtide.workflows.parser_funcs as pf


def _get_parser() -> Any:
    """
    Argument parser for showarbcorr.

    This function constructs and returns an `argparse.ArgumentParser` object configured
    for the `showarbcorr` command-line tool. It defines required and optional arguments
    for calculating and displaying crosscorrelation between two time series, supporting
    variable lengths and sampling frequencies.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser for `showarbcorr`.

    Notes
    -----
    The parser includes groups for:
    - Required input files
    - Optional arguments (e.g., sample rates, display control)
    - Preprocessing options (e.g., detrending, correlation weighting)
    - Filtering and windowing options
    - Output configuration (e.g., files for results, plots)
    - Miscellaneous settings (e.g., multiprocessing, progress bar)

    Examples
    --------
    >>> parser = _get_parser()
    >>> args = parser.parse_args()
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


def printthresholds(pcts: Any, thepercentiles: Any, labeltext: Any) -> None:
    """
    Print thresholds with corresponding percentile labels.

    This function prints a formatted list of thresholds along with their
    corresponding percentile labels for statistical analysis output.

    Parameters
    ----------
    pcts : Any
        Array or list of threshold values to be printed.
    thepercentiles : Any
        Array or list of percentile values corresponding to the thresholds.
    labeltext : Any
        Text label to be printed before the threshold list.

    Returns
    -------
    None
        This function prints to standard output and does not return any value.

    Notes
    -----
    The function formats the percentile values as "1.0 - thepercentiles[i]"
    to show the alpha level (significance threshold) for each percentile.

    Examples
    --------
    >>> pcts = [0.05, 0.01, 0.001]
    >>> thepercentiles = [0.95, 0.99, 0.999]
    >>> labeltext = "Critical Values:"
    >>> printthresholds(pcts, thepercentiles, labeltext)
    Critical Values:
        p < 0.050 : 0.05
        p < 0.010 : 0.01
        p < 0.001 : 0.001
    """
    print(labeltext)
    for i in range(0, len(pcts)):
        print("\tp <", "{:.3f}".format(1.0 - thepercentiles[i]), ": ", pcts[i])


def showarbcorr(args: Any) -> None:
    """
    Compute and display cross-correlation between two time series with optional filtering and plotting.

    This function reads two time series from text files, matches their sampling rates, applies
    optional filtering, and computes the cross-correlation. It supports various options for
    data trimming, inversion, and output formatting, including optional visualization and
    correlation fitting.

    Parameters
    ----------
    args : Any
        An object containing command-line arguments and configuration options. Expected
        attributes include:
        - infilename1, infilename2 : str
            Paths to input text files containing the two time series.
        - samplerate1, samplerate2 : float, optional
            Sampling rates for the two time series. If not provided, they are inferred
            from the input files.
        - start1, start2 : float, optional
            Start times for the two time series.
        - trimdata : bool
            If True, trim the data to the shorter of the two time series.
        - invert : bool
            If True, invert the second time series before correlation.
        - windowfunc : str, optional
            Window function to apply during correlation normalization.
        - detrendorder : int, optional
            Order of detrending to apply before correlation.
        - display : bool
            If True, display the cross-correlation plot.
        - graphfile : str, optional
            Output filename for saving the plot.
        - label : str, optional
            Label to use in output.
        - lagmin, lagmax : float
            Minimum and maximum lags (in seconds) to consider in the correlation.
        - debug : bool
            If True, enable debug output.
        - bipolar : bool
            If True, fit the peak using bipolar symmetry.
        - summarymode : bool
            If True, output results in a tab-separated summary format.
        - outputfile : str, optional
            File to write summary output.
        - corroutputfile : str, optional
            File to write full cross-correlation data.
        - verbose : bool
            If True, print verbose messages.

    Returns
    -------
    None
        This function does not return a value but may produce plots, print outputs,
        and write files depending on the provided arguments.

    Notes
    -----
    - The function uses `tide_io.readvectorsfromtextfile` to read input data.
    - Filtering is applied via `tide_math.corrnormalize` and `theprefilter.apply`.
    - Cross-correlation is computed using `tide_corr.arbcorr`.
    - A peak-fitting procedure is used to refine the maximum correlation lag.
    - If `summarymode` is True, output is written in tab-separated format to stdout or a file.

    Examples
    --------
    >>> import argparse
    >>> args = argparse.Namespace(
    ...     infilename1="data1.txt",
    ...     infilename2="data2.txt",
    ...     samplerate1=10.0,
    ...     samplerate2=10.0,
    ...     display=True,
    ...     lagmin=-5.0,
    ...     lagmax=5.0,
    ...     windowfunc="hanning",
    ...     debug=False
    ... )
    >>> showarbcorr(args)
    """
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
    thepxcorr = pearsonr(filtereddata1, filtereddata2).statistic

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
            print("Pearson_R:\t", thepxcorr)
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
