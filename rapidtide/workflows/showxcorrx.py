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

import matplotlib.cm as cm
import numpy as np
import scipy as sp
from scipy.signal import correlate
from scipy.stats import pearsonr

import rapidtide.calcnullsimfunc as tide_nullsimfunc
import rapidtide.correlate as tide_corr
import rapidtide.fit as tide_fit
import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math
import rapidtide.peakeval as tide_peakeval
import rapidtide.simFuncClasses as tide_simFuncClasses
import rapidtide.stats as tide_stats
import rapidtide.util as tide_util
import rapidtide.workflows.parser_funcs as pf

DEFAULT_SIGMAMAX = 1000.0
DEFAULT_SIGMAMIN = 0.25


def _get_parser():
    """
    Argument parser for showxcorrx
    """
    parser = argparse.ArgumentParser(
        prog="showxcorrx",
        description=("Calculate and display crosscorrelation between two timeseries."),
        allow_abbrev=False,
    )

    # Required arguments
    parser.add_argument(
        "infilename1",
        type=str,
        help="Text file containing a timeseries.  Select column COLNUM if multicolumn file",
    )
    parser.add_argument(
        "infilename2",
        type=str,
        help="Text file containing a timeseries.  Select column COLNUM if multicolumn file",
    )

    # add optional arguments
    general = parser.add_argument_group("General Options")
    sampling = general.add_mutually_exclusive_group()
    sampling.add_argument(
        "--samplerate",
        dest="samplerate",
        action="store",
        metavar="FREQ",
        type=lambda x: pf.is_float(parser, x),
        help=(
            "Set the sample rate of the data file to FREQ. "
            "If neither samplerate or sampletime is specified, sample rate is 1.0."
        ),
        default="auto",
    )
    sampling.add_argument(
        "--sampletime",
        dest="samplerate",
        action="store",
        metavar="TSTEP",
        type=lambda x: pf.invert_float(parser, x),
        help=(
            "Set the sample rate of the data file to 1.0/TSTEP. "
            "If neither samplerate or sampletime is specified, sample rate is 1.0."
        ),
        default="auto",
    )

    pf.addsearchrangeopts(general, details=True)

    pf.addtimerangeopts(general)

    # add window options
    pf.addwindowopts(parser)

    # Filter arguments
    pf.addfilteropts(parser, filtertarget="timecourses", details=True)

    # Preprocessing options
    preproc = parser.add_argument_group("Preprocessing options")
    preproc.add_argument(
        "--detrendorder",
        dest="detrendorder",
        action="store",
        type=int,
        metavar="ORDER",
        help=("Set order of trend removal (0 to disable, default is 1 - linear). "),
        default=1,
    )
    preproc.add_argument(
        "--trimdata",
        dest="trimdata",
        action="store_true",
        help=("Trimming data to match"),
        default=False,
    )
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
        help=("Invert the second timecourse prior to calculating similarity. "),
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
    preproc.add_argument(
        "--partialcorr",
        dest="controlvariablefile",
        action="store",
        type=lambda x: pf.is_valid_file(parser, x),
        metavar="FILE",
        help=(
            "Use the columns of FILE as controlling variables and "
            "return the partial correlation. "
        ),
        default=None,
    )

    additionalcalcs = parser.add_argument_group("Additional calculations")
    additionalcalcs.add_argument(
        "--cepstral",
        dest="cepstral",
        action="store_true",
        help="Check time delay using Choudhary's cepstral technique. ",
        default=False,
    )
    additionalcalcs.add_argument(
        "--calccsd",
        dest="calccsd",
        action="store_true",
        help="Calculate the cross-spectral density. ",
        default=False,
    )
    additionalcalcs.add_argument(
        "--calccoherence",
        dest="calccoherence",
        action="store_true",
        help="Calculate the coherence. ",
        default=False,
    )

    pf.addpermutationopts(preproc, numreps=0)

    # similarity function options
    similarityopts = parser.add_argument_group("Similarity function options")
    similarityopts.add_argument(
        "--similaritymetric",
        dest="similaritymetric",
        action="store",
        type=str,
        choices=["correlation", "mutualinfo", "hybrid"],
        help=(
            "Similarity metric for finding delay values.  "
            "Choices are 'correlation' (default), 'mutualinfo', and 'hybrid'."
        ),
        default="correlation",
    )
    similarityopts.add_argument(
        "--sigmamax",
        dest="absmaxsigma",
        action="store",
        type=float,
        metavar="SIGMAMAX",
        help=(
            "Reject lag fits with linewidth wider than "
            f"SIGMAMAX Hz. Default is {DEFAULT_SIGMAMAX} Hz."
        ),
        default=DEFAULT_SIGMAMAX,
    )
    similarityopts.add_argument(
        "--sigmamin",
        dest="absminsigma",
        action="store",
        type=float,
        metavar="SIGMAMIN",
        help=(
            "Reject lag fits with linewidth narrower than "
            f"SIGMAMIN Hz. Default is {DEFAULT_SIGMAMIN} Hz."
        ),
        default=DEFAULT_SIGMAMIN,
    )

    pf.addsimilarityopts(similarityopts)

    # Output options
    output = parser.add_argument_group("Output options")
    output.add_argument(
        "--outputfile",
        dest="resoutputfile",
        action="store",
        type=str,
        metavar="FILE",
        help=("Save results to FILE. "),
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

    # add plot appearance options
    pf.addplotopts(parser, multiline=False)

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
        "--nodisplay",
        dest="display",
        action="store_false",
        help=("Do not plot the data (for noninteractive use)"),
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

    # debugging options
    debugging = parser.add_argument_group("Debugging options")
    debugging.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help=("Enable additional debugging output."),
        default=False,
    )
    debugging.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help=("Print out more debugging information"),
        default=False,
    )

    return parser


def printthresholds(pcts, thepercentiles, labeltext):
    print(labeltext)
    for i in range(0, len(pcts)):
        print("\tp <", "{:.3f}".format(1.0 - thepercentiles[i]), ": ", pcts[i])


def showxcorrx(args):
    # set some default values
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

    # get the filenames and read in the data
    infilename1, colspec1 = tide_io.parsefilespec(args.infilename1)
    infilename2, colspec2 = tide_io.parsefilespec(args.infilename2)
    (
        samplerate1,
        startoffset1,
        colnames1,
        inputdata1,
        dummy,
        dummy,
    ) = tide_io.readvectorsfromtextfile(args.infilename1, onecol=True, debug=args.debug)
    if colnames1 is not None:
        tcname1 = colnames1
    else:
        if colspec1 is not None:
            tcname1 = infilename1 + "_" + colspec1
        else:
            tcname1 = infilename1

    (
        samplerate2,
        startoffset2,
        colnames2,
        inputdata2,
        dummy,
        dummy,
    ) = tide_io.readvectorsfromtextfile(args.infilename2, onecol=True, debug=args.debug)
    if colnames2 is not None:
        tcname2 = colnames2
    else:
        if colspec2 is not None:
            tcname2 = infilename2 + "_" + colspec2
        else:
            tcname2 = infilename2

    if samplerate1 != samplerate2:
        print("time courses must have the same sample rate")
        sys.exit()
    if samplerate1 is not None:
        args.samplerate = samplerate1
        args.startpoint = startoffset1

    # Set the default samplerate if not set
    if args.samplerate == "auto":
        print("samplerate not set - setting to 1.0 Hz")
        args.samplerate = 1.0

    if args.debug:
        print(args.samplerate)
        print(args.startpoint)
        print(inputdata1)
        print(inputdata2)

    if args.debug:
        dumpfiltered = True
    else:
        dumpfiltered = False
    showpearson = True

    print("startpoint, endpoint:", args.startpoint, args.endpoint)
    print("thetime:", args.startpoint / args.samplerate)

    startpoint1 = np.max([int(args.startpoint / args.samplerate), 0])
    if args.debug:
        print("startpoint set to ", startpoint1)
    endpoint1 = np.min([int(args.endpoint / args.samplerate), int(len(inputdata1))])
    if args.debug:
        print("endpoint set to ", endpoint1)
    endpoint2 = np.min(
        [int(args.endpoint / args.samplerate), int(len(inputdata1)), int(len(inputdata2))]
    )
    trimdata1 = inputdata1[startpoint1:endpoint1]
    trimdata2 = inputdata2[0:endpoint2]

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
        theprefilter.apply(args.samplerate, trimdata1),
        detrendorder=args.detrendorder,
        windowfunc=args.windowfunc,
    )
    filtereddata2 = tide_math.corrnormalize(
        theprefilter.apply(args.samplerate, trimdata2),
        detrendorder=args.detrendorder,
        windowfunc=args.windowfunc,
    )
    filtereddata2 *= flipfac
    if dumpfiltered:
        tide_io.writenpvecs(filtereddata1, "filtereddata1.txt")
        tide_io.writenpvecs(filtereddata2, "filtereddata2.txt")

    if args.controlvariablefile is not None:
        controlvars = tide_io.readnpvecs(args.controlvariablefile)
        regressorvec = []
        for j in range(0, controlvars.shape[0]):
            regressorvec.append(
                tide_math.corrnormalize(
                    theprefilter.apply(args.samplerate, controlvars[j, :]),
                    detrendorder=args.detrendorder,
                    windowfunc=args.windowfunc,
                )
            )
        if (np.max(filtereddata1) - np.min(filtereddata1)) > 0.0:
            thefit, R2 = tide_fit.mlregress(regressorvec, filtereddata1)
        if (np.max(filtereddata2) - np.min(filtereddata2)) > 0.0:
            thefit, R2 = tide_fit.mlregress(regressorvec, filtereddata2)

    # initialize the Correlator and MutualInformationator
    theCorrelator = tide_simFuncClasses.Correlator(
        Fs=args.samplerate,
        ncprefilter=theprefilter,
        detrendorder=args.detrendorder,
        windowfunc=args.windowfunc,
        corrweighting=args.corrweighting,
        corrpadding=args.zeropadding,
        debug=args.debug,
    )
    theCorrelator.setreftc(trimdata2 * flipfac)
    theMutualInformationator = tide_simFuncClasses.MutualInformationator(
        Fs=args.samplerate,
        smoothingtime=args.smoothingtime,
        ncprefilter=theprefilter,
        detrendorder=args.detrendorder,
        norm=args.minorm,
        windowfunc=args.windowfunc,
    )
    theMutualInformationator.setreftc(trimdata2 * flipfac)

    # calculate the similarity
    if args.similaritymetric == "mutualinfo":
        # calculate the MI
        theMI, MI_x, globalmax = theMutualInformationator.run(
            trimdata1, trim=False, gettimeaxis=True
        )
        print("MutualInformationator lengths (x, y):", len(MI_x), len(theMI))
        if dumpfiltered:
            tide_io.writenpvecs(theMutualInformationator.preptesttc, "MI_filtereddata1.txt")
            tide_io.writenpvecs(theMutualInformationator.prepreftc, "MI_filtereddata2.txt")
        theMutualInformationator.setlimits(
            int((-args.lagmin * args.samplerate) - 0.5), int((args.lagmax * args.samplerate) + 0.5)
        )
        theMI_trim, MI_x_trim, globalmax = theMutualInformationator.getfunction(trim=True)
        print("trimmed MutualInformationator lengths (x, y):", len(MI_x_trim), len(theMI_trim))
    else:
        # do the correlation
        thexcorr, xcorr_x, globalmax = theCorrelator.run(trimdata1, trim=False)
        if args.display and args.debug:
            plt.plot(xcorr_x, thexcorr)
            plt.show()
        print("Correlator lengths (x, y):", len(xcorr_x), len(thexcorr))
        if dumpfiltered:
            tide_io.writenpvecs(theCorrelator.preptesttc, "correlator_filtereddata1.txt")
            tide_io.writenpvecs(theCorrelator.prepreftc, "correlator_filtereddata2.txt")
        if args.debug:
            print(f"limits: {args.lagmin, args.lagmax}")
        theCorrelator.setlimits(
            int((-args.lagmin * args.samplerate) - 0.5), int((args.lagmax * args.samplerate) + 0.5)
        )
        thexcorr_trim, xcorr_x_trim, dummy = theCorrelator.getfunction(trim=True)
        if args.display and args.debug:
            plt.plot(xcorr_x_trim, thexcorr_trim)
            plt.show()
        print("trimmed Correlator lengths (x, y):", len(xcorr_x_trim), len(thexcorr_trim))

    if args.cepstral:
        cepdelay = tide_corr.cepstraldelay(
            filtereddata1, filtereddata2, 1.0 / args.samplerate, displayplots=args.display
        )
        cepcoff = tide_corr.delayedcorr(
            filtereddata1, filtereddata2, cepdelay, 1.0 / args.samplerate
        )
        print("cepstral delay time is", cepdelay, ", correlation is", cepcoff)
    thepxcorr = pearsonr(filtereddata1, filtereddata2)

    if args.calccoherence:
        # calculate the coherence
        fC, Cxy = sp.signal.coherence(
            tide_math.corrnormalize(
                theprefilter.apply(args.samplerate, trimdata1),
                detrendorder=args.detrendorder,
                windowfunc=args.windowfunc,
            ),
            tide_math.corrnormalize(
                theprefilter.apply(args.samplerate, trimdata2),
                detrendorder=args.detrendorder,
                windowfunc=args.windowfunc,
            ),
            args.samplerate,
        )

    if args.calccsd:
        # calculate the cross spectral density
        fP, Pxy = sp.signal.csd(
            tide_math.corrnormalize(
                theprefilter.apply(args.samplerate, trimdata1),
                detrendorder=args.detrendorder,
                windowfunc=args.windowfunc,
            ),
            tide_math.corrnormalize(
                theprefilter.apply(args.samplerate, trimdata2),
                detrendorder=args.detrendorder,
                windowfunc=args.windowfunc,
            ),
            args.samplerate,
        )

    if args.similaritymetric == "mutualinfo":
        # initialize the similarity function fitter
        themifitter = tide_simFuncClasses.SimilarityFunctionFitter(
            corrtimeaxis=MI_x_trim,
            lagmin=args.lagmin,
            lagmax=args.lagmax,
            absmaxsigma=args.absmaxsigma,
            absminsigma=args.absminsigma,
            debug=args.debug,
            peakfittype="quad",
            functype="mutualinfo",
            zerooutbadfit=zerooutbadfit,
            useguess=False,
        )
        maxdelaymi = MI_x_trim[np.argmax(theMI_trim)]
    else:
        # initialize the correlation fitter
        thexsimfuncfitter = tide_simFuncClasses.SimilarityFunctionFitter(
            corrtimeaxis=xcorr_x,
            lagmin=args.lagmin,
            lagmax=args.lagmax,
            absmaxsigma=args.absmaxsigma,
            absminsigma=args.absminsigma,
            debug=args.debug,
            peakfittype=peakfittype,
            functype="correlation",
            zerooutbadfit=zerooutbadfit,
            useguess=False,
        )
        maxdelay = xcorr_x_trim[np.argmax(thexcorr_trim)]

    if args.debug:
        print(
            "searching for peak correlation over range ",
            theCorrelator.similarityfuncorigin - theCorrelator.lagmininpts,
            theCorrelator.similarityfuncorigin + theCorrelator.lagmaxinpts,
        )

    if args.debug:
        print("\n\nmaxdelay before refinement", maxdelay)

    timeaxis = np.linspace(0, 1.0, num=len(trimdata1), endpoint=False) / args.samplerate
    thetc = trimdata1 * 0.0

    if args.similaritymetric == "hybrid" or args.similaritymetric == "correlation":
        peakstartind = tide_util.valtoindex(xcorr_x, args.lagmin, discretization="floor")
        peakendind = tide_util.valtoindex(xcorr_x, args.lagmax, discretization="ceiling") + 1
        dummy, thepeaks = tide_peakeval._procOneVoxelPeaks(
            0,
            thetc,
            theMutualInformationator,
            timeaxis,
            trimdata1,
            timeaxis,
            xcorr_x[peakstartind:peakendind],
            thexcorr[peakstartind:peakendind],
            oversampfactor=1,
        )

        thesortedindices = np.argsort(np.asarray(thepeaks)[:, 2])[::-1]
        print("peaklist:")
        print('peak\tloc\tR\tMI\t"R"')
        for i in range(len(thepeaks)):
            print(
                "{0:2d}\t{1:3.2f}\t{2:3.2f}\t{3:3.2f}\t{4:3.2f}".format(
                    i,
                    thepeaks[thesortedindices[i]][0],
                    thepeaks[thesortedindices[i]][1],
                    thepeaks[thesortedindices[i]][2],
                    tide_corr.mutual_info_to_r(thepeaks[thesortedindices[i]][2]),
                )
            )

    if args.similaritymetric == "mutualinfo":
        if args.debug:
            print("\n\nmaxdelaymi before refinement", maxdelaymi)
        (
            maxindexmi,
            maxdelaymi,
            maxvalmi,
            maxsigmami,
            maskvalmi,
            failreasonmi,
            peakstartmi,
            peakendmi,
        ) = themifitter.fit(theMI_trim)
        if failreasonmi > 0:
            print("showxcorrx: FIT FAILED for mutual information with reason:")
            print(themifitter.diagnosefail(np.uint32(failreasonmi)))
        if args.debug:
            print(
                maxindexmi,
                maxdelaymi,
                maxvalmi,
                maxsigmami,
                maskvalmi,
                failreasonmi,
                peakstartmi,
                peakendmi,
            )
        R = maxvalmi
        if args.debug:
            print("maxdelay after refinement", maxdelaymi, "\n\n")
    else:
        (
            maxindex,
            maxdelay,
            maxval,
            maxsigma,
            maskval,
            failreason,
            peakstart,
            peakend,
        ) = thexsimfuncfitter.fit(thexcorr)
        if failreason > 0:
            print("showxcorrx: FIT FAILED with reason:")
            print(thexsimfuncfitter.diagnosefail(np.uint32(failreason)))
        if args.debug:
            print(maxindex, maxdelay, maxval, maxsigma, maskval, failreason)
        R = maxval
        if args.debug:
            print("maxdelay after refinement", maxdelay, "\n\n")

    # set the significance threshold
    if args.numestreps > 0:
        # generate a list of correlations from shuffled data
        print("calculating null crosscorrelations")
        corrlist = tide_nullsimfunc.getNullDistributionData(
            args.samplerate,
            theCorrelator,
            thexsimfuncfitter,
            None,
            numestreps=args.numestreps,
            showprogressbar=args.showprogressbar,
            permutationmethod=args.permutationmethod,
            nprocs=args.nprocs,
        )

        # calculate percentiles for the crosscorrelation from the distribution data
        histlen = 100
        thepercentiles = [0.95, 0.99, 0.995]

        pcts, pcts_fit, histfit = tide_stats.sigFromDistributionData(
            corrlist, histlen, thepercentiles
        )
        if args.debug:
            tide_stats.printthresholds(
                pcts,
                thepercentiles,
                "Crosscorrelation significance thresholds from data:",
            )
            tide_stats.printthresholds(
                pcts_fit,
                thepercentiles,
                "Crosscorrelation significance thresholds from fit:",
            )

        print("calculating null Pearson correlations")
        corrlist_pear = tide_nullsimfunc.getNullDistributionData(
            args.samplerate,
            theCorrelator,
            thexsimfuncfitter,
            None,
            numestreps=args.numestreps,
            showprogressbar=args.showprogressbar,
            permutationmethod=args.permutationmethod,
            nprocs=args.nprocs,
        )

        # calculate significance for the pearson correlation
        pearpcts, pearpcts_fit, histfit = tide_stats.sigFromDistributionData(
            corrlist_pear, histlen, thepercentiles
        )
        if args.debug:
            tide_stats.printthresholds(
                pearpcts,
                thepercentiles,
                "Pearson correlation significance thresholds from data:",
            )
            tide_stats.printthresholds(
                pearpcts_fit,
                thepercentiles,
                "Pearson correlation significance thresholds from fit:",
            )

        if args.debug:
            tide_io.writenpvecs(corrlist, "corrlist.txt")
            tide_io.writenpvecs(corrlist_pear, "corrlist_pear.txt")

    if args.debug:
        print(thepxcorr)

    if args.similaritymetric == "mutualinfo":
        print(f"{tcname1}[0] = {tcname2}[{-maxdelaymi} seconds]")
        if args.summarymode:
            if args.numestreps > 0:
                thelabelitems = [
                    "MI_R",
                    "MI_R(p=0.05)",
                    "MI_maxdelay",
                ]
                thedataitems = [
                    str(maxvalmi),
                    str(pcts_fit[0]),
                    str(-maxdelaymi),
                ]
            else:
                thelabelitems = ["xcorr_R", "xcorr_maxdelay"]
                thedataitems = [
                    str(maxvalmi),
                    str(-maxdelaymi),
                ]
    else:
        print(f"{tcname1}[0] = {tcname2}[{-maxdelay} seconds]")
        if args.summarymode:
            if args.numestreps > 0:
                thelabelitems = [
                    "pearson_R",
                    "pearson_R(p=0.05)",
                    "xcorr_R",
                    "xcorr_R(p=0.05)",
                    "xcorr_maxdelay",
                ]
                thedataitems = [
                    str(thepxcorr[0]),
                    str(pearpcts_fit[0]),
                    str(R),
                    str(pcts_fit[0]),
                    str(-maxdelay),
                ]
            else:
                thelabelitems = ["pearson_R", "pearson_p", "xcorr_R", "xcorr_maxdelay"]
                thedataitems = [
                    str(thepxcorr[0]),
                    str(thepxcorr[1]),
                    str(R),
                    str(-maxdelay),
                ]
        if args.label is not None:
            thelabelitems = ["thelabel"] + thelabelitems
            thedataitems = [args.label] + thedataitems
        if args.labelline:
            outputstring = "\t".join(thelabelitems) + "\n" + "\t".join(thedataitems)
        else:
            outputstring = "\t".join(thedataitems)
        if args.resoutputfile is None:
            print(outputstring)
        else:
            with open(args.resoutputfile, "w") as text_file:
                text_file.write(outputstring + "\n")
    """else:
        # report the pearson correlation
        if showpearson:
            print("Pearson_R:\t", thepxcorr[0])
            if args.numestreps > 0:
                for idx, percentile in enumerate(thepercentiles):
                    print(
                        "    pear_p(",
                        "{:.3f}".format(1.0 - percentile),
                        "):\t",
                        pearpcts[idx],
                    )
            print("")
        if args.label is not None:
            print(args.label, ":\t", -maxdelay)
        else:
            print("Crosscorrelation_Rmax:\t", R)
            print("Crosscorrelation_maxdelay:\t", -maxdelay)
            if args.numestreps > 0:
                for idx, percentile in enumerate(thepercentiles):
                    print(
                        "    xc_p(",
                        "{:.3f}".format(1.0 - percentile),
                        "):\t",
                        pcts[idx],
                    )
            print(infilename1, "[0 seconds] == ", infilename2, "[", -maxdelay, " seconds]")
    """
    thexaxfontsize = 10 * args.fontscalefac
    theyaxfontsize = 10 * args.fontscalefac
    thexlabelfontsize = 10 * args.fontscalefac
    theylabelfontsize = 10 * args.fontscalefac
    thelegendfontsize = 8 * args.fontscalefac
    thetitlefontsize = 10 * args.fontscalefac
    thesuptitlefontsize = 10 * args.fontscalefac

    # set various cosmetic aspects of the plots
    if args.colors is not None:
        colornames = args.colors.split(",")
    else:
        colornames = []
    if len(colornames) > 0:
        colorlist = [colornames[0]]
    else:
        colorlist = [cm.nipy_spectral(float(0))]

    if args.linewidths is not None:
        thelinewidth = []
        for thestring in args.linewidths.split(","):
            thelinewidth.append(float(thestring))
    else:
        thelinewidth = [1.0]

    if 0 <= args.legendloc <= 10:
        pass
    else:
        print("illegal legend location:", args.legendloc)
        sys.exit()

    if args.display:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        thelegend = []
        if args.legends is not None:
            thelegend.append = args.legends
        else:
            if args.similaritymetric == "mutualinfo":
                thelegend.append("Mutual Information")
                ax.plot(
                    MI_x_trim,
                    theMI_trim,
                    color=colorlist[0],
                    label=thelegend[0],
                    linewidth=thelinewidth[0],
                )
            else:
                thelegend.append("Cross correlation")
                ax.plot(
                    xcorr_x_trim,
                    thexcorr_trim,
                    color=colorlist[0],
                    label=thelegend[0],
                    linewidth=thelinewidth[0],
                )
        if args.dolegend:
            ax.legend(thelegend, fontsize=thelegendfontsize, loc=args.legendloc)
        if args.thetitle is not None:
            ax.set_title(args.thetitle, fontsize=thetitlefontsize)
        else:
            ax.set_title("Similarity metric over the search range", fontsize=thetitlefontsize)
        if args.showxax:
            ax.set_xlabel(args.xlabel, fontsize=thexlabelfontsize, fontweight="bold")
            ax.tick_params(axis="x", labelsize=thexlabelfontsize, which="both")
        if args.showyax:
            ax.set_ylabel(args.ylabel, fontsize=theylabelfontsize, fontweight="bold")
            ax.tick_params(axis="y", labelsize=theylabelfontsize, which="both")
        if args.outputfile is not None:
            plt.savefig(args.outputfile, bbox_inches="tight", dpi=args.saveres)
        else:
            plt.show()

    if args.calccoherence:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(fC, np.sqrt(np.abs(Cxy)) / np.max(np.sqrt(np.abs(Cxy))), "b")
        ax.set_title("Coherence")

    if args.calccsd:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(fP, np.sqrt(np.abs(Pxy)) / np.max(np.sqrt(np.abs(Pxy))), "g")
        ax.set_title("Cross-spectral density")

    if args.display and (args.calccoherence or args.calccsd):
        plt.show()

    if args.similaritymetric == "correlation" and args.corroutputfile is not None:
        tide_io.writenpvecs(np.stack((xcorr_x, thexcorr), axis=0), args.corroutputfile)
    if args.similaritymetric == "mutualinfo" and args.debug:
        tide_io.writenpvecs(np.stack((MI_x_trim, theMI_trim), axis=0), "mifunc.txt")
