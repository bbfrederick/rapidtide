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
import getopt
import sys

import numpy as np
from matplotlib.pyplot import figure, plot, show
from numpy import argmax, r_, zeros
from numpy.random import permutation
from scipy.stats import pearsonr

import rapidtide.correlate as tide_corr
import rapidtide.filter as tide_filt
import rapidtide.fit as tide_fit
import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math
import rapidtide.stats as tide_stats


def getNullDistributionData(
    indata,
    xcorr_x,
    thefilter,
    windowfunc,
    detrendorder,
    searchstart,
    searchend,
    numreps=1000,
):
    print("estimating significance distribution using ", numreps, " repetitions")
    corrlist = zeros(numreps, dtype="float")
    corrlist_pear = zeros(numreps, dtype="float")
    xcorr_x_trim = xcorr_x[searchstart : searchend + 1]

    filteredindata = tide_math.corrnormalize(
        thefilter.apply(Fs, indata), windowfunc=windowfunc, detrendorder=detrendorder
    )
    for i in range(0, numreps):
        # make a shuffled copy of the regressors
        shuffleddata = permutation(indata)

        # filter it
        filteredshuffleddata = tide_math.corrnormalize(
            thefilter.apply(Fs, shuffleddata),
            windowfunc=windowfunc,
            detrendorder=detrendorder,
        )

        # crosscorrelate with original
        if gccphat:
            theshuffledxcorr = tide_corr.fastcorrelate(
                filteredindata, filteredshuffleddata, usefft=dofftcorr, weighting="phat"
            )

        else:
            theshuffledxcorr = tide_corr.fastcorrelate(
                filteredindata, filteredshuffleddata, usefft=dofftcorr
            )

        # find and tabulate correlation coefficient at optimal lag
        theshuffledxcorr_trim = theshuffledxcorr[searchstart : searchend + 1]
        maxdelay = xcorr_x_trim[argmax(theshuffledxcorr_trim)]
        corrlist[i] = theshuffledxcorr_trim[argmax(theshuffledxcorr_trim)]

        # find and tabulate correlation coefficient at 0 lag
        corrlist_pear[i] = pearsonr(filteredindata, filteredshuffleddata)[0]

        # progress
        # tide_util.progressbar(i + 1, numreps, label='Percent complete')

    # jump to line after progress bar
    print()

    # return the distribution data
    return corrlist, corrlist_pear


def usage():
    print("showxcorr - calculate and display crosscorrelation between two timeseries")
    print("")
    print(
        "usage: showxcorr timecourse1 timecourse2 samplerate [-l LABEL] [-s STARTTIME] [-D DURATION] [-d] [-F LOWERFREQ,UPPERFREQ[,LOWERSTOP,UPPERSTOP]] [-V] [-L] [-R] [-C] [-t] [-w] [-f] [-z FILENAME] [-N TRIALS]"
    )
    print("")
    print("required arguments:")
    print("	timcoursefile1:	text file containing a timeseries")
    print("	timcoursefile2:	text file containing a timeseries")
    print("	samplerate:	the sample rate of the timecourses, in Hz")
    print("")
    print("optional arguments:")
    print("    -t            - detrend the data")
    print("    -w            - window the data with a hamming function")
    print("    -l LABEL      - label for the delay value")
    print("    -s STARTTIME  - time of first datapoint to use in seconds in the first file")
    print("    -D DURATION   - amount of data to use in seconds")
    print("    -r RANGE      - restrict peak search range to +/- RANGE seconds (default is ")
    print("                    +/-15)")
    print("    -d            - turns off display of graph")
    print("    -F            - filter data and regressors from LOWERFREQ to UPPERFREQ.")
    print("                    LOWERSTOP and UPPERSTOP can be specified, or will be ")
    print("                    calculated automatically")
    print("    -V            - filter data and regressors to VLF band")
    print("    -L            - filter data and regressors to LFO band")
    print("    -R            - filter data and regressors to respiratory band")
    print("    -C            - filter data and regressors to cardiac band")
    print("    -T            - trim data to match")
    print("    -A            - print data on a single summary line")
    print("    -a            - if summary mode is on, add a header line showing what values ")
    print("                    mean")
    print("    -f            - negate (flip) second regressor")
    print("    -z FILENAME   - use the columns of FILENAME as controlling variables and ")
    print("                    return the partial correlation")
    print("    -N TRIALS     - estimate significance thresholds by Monte Carlo with TRIALS ")
    print("                    repetition")
    print("    -Y            - turn on debugging")
    print("")
    return ()


def main():
    # get the command line parameters
    searchrange = 15.0
    uselabel = False
    displayplots = True
    gccphat = False
    windowfunc = "None"
    detrendorder = 0
    dopartial = False
    duration = 1000000.0
    starttime = 0.0
    doplot = False
    thelabel = ""
    trimdata = False
    verbose = True
    summarymode = False
    dofftcorr = True
    labelline = False
    estimate_significance = False
    writecorrlists = False
    flipregressor = False

    debug = False
    if debug:
        dumpfiltered = True
    else:
        dumpfiltered = False
    showpearson = True

    nargs = len(sys.argv)
    if nargs < 4:
        usage()
        exit()
    infilename1 = sys.argv[1]
    infilename2 = sys.argv[2]
    Fs = float(sys.argv[3])

    theprefilter = tide_filt.NoncausalFilter()

    inputdata1 = tide_io.readvec(infilename1)
    inputdata2 = tide_io.readvec(infilename2)
    numpoints = len(inputdata1)

    # now scan for optional arguments
    try:
        opts, args = getopt.getopt(sys.argv[4:], "fN:r:z:aATtVLRCF:dl:s:D:wY", ["help"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(str(err))  # will print something like "option -x not recognized"
        usage()
        sys.exit(2)

    for o, a in opts:
        if o == "-d":
            displayplots = False
            if verbose:
                print("disable display")
        elif o == "-Y":
            debug = True
            dumpfiltered = True
            if verbose:
                print("turning on debugging")
        elif o == "-T":
            trimdata = True
            if verbose:
                print("trimming data to match")
        elif o == "-g":
            gccphat = True
            if verbose:
                print("doing phase alignment transform")
        elif o == "-f":
            flipregressor = True
            if verbose:
                print("negating second regressor")
        elif o == "-a":
            labelline = True
            if verbose:
                print("turning on label line")
        elif o == "-t":
            detrendorder = 1
            if verbose:
                print("enabling linear detrending")
        elif o == "-w":
            windowfunc = "hamming"
            if verbose:
                print("enabling hamming windowing")
        elif o == "-z":
            controlvariablefile = a
            dopartial = True
            if verbose:
                print("performing partial correlations")
        elif o == "-l":
            thelabel = a
            uselabel = True
            if verbose:
                print("label set to", thelabel)
        elif o == "-N":
            numreps = int(a)
            estimate_significance = True
            if verbose:
                print("estimating significance threshold with ", numreps, " trials")
        elif o == "-r":
            searchrange = float(a)
            if verbose:
                print("peak search restricted to +/-", searchrange, " seconds")
        elif o == "-D":
            duration = float(a)
            if verbose:
                print("duration set to", duration)
        elif o == "-s":
            starttime = float(a)
            if verbose:
                print("starttime set to", starttime)
        elif o == "-V":
            theprefilter.settype("vlf")
            if verbose:
                print("prefiltering to vlf band")
        elif o == "-L":
            theprefilter.settype("lfo")
            if verbose:
                print("prefiltering to lfo band")
        elif o == "-R":
            theprefilter.settype("resp")
            if verbose:
                print("prefiltering to respiratory band")
        elif o == "-C":
            theprefilter.settype("cardiac")
            if verbose:
                print("prefiltering to cardiac band")
        elif o == "-A":
            verbose = False
            summarymode = True
        elif o == "-F":
            arbvec = a.split(",")
            if len(arbvec) != 2 and len(arbvec) != 4:
                usage()
                sys.exit()
            if len(arbvec) == 2:
                arb_lower = float(arbvec[0])
                arb_upper = float(arbvec[1])
                arb_lowerstop = 0.9 * float(arbvec[0])
                arb_upperstop = 1.1 * float(arbvec[1])
            if len(arbvec) == 4:
                arb_lower = float(arbvec[0])
                arb_upper = float(arbvec[1])
                arb_lowerstop = float(arbvec[2])
                arb_upperstop = float(arbvec[3])
            theprefilter.settype("arb")
            theprefilter.setfreqs(arb_lowerstop, arb_lower, arb_upper, arb_upperstop)
            if verbose:
                print(
                    "prefiltering to ",
                    arb_lower,
                    arb_upper,
                    "(stops at ",
                    arb_lowerstop,
                    arb_upperstop,
                    ")",
                )
        else:
            assert False, "unhandled option"

    startpoint1 = max([int(starttime * Fs), 0])
    if debug:
        print("startpoint set to ", startpoint1)
    endpoint1 = min([startpoint1 + int(duration * Fs), int(len(inputdata1))])
    if debug:
        print("endpoint set to ", endpoint1)
    endpoint2 = min([int(duration * Fs), int(len(inputdata1)), int(len(inputdata2))])
    trimdata1 = inputdata1[startpoint1:endpoint1]
    trimdata2 = inputdata2[0:endpoint2]

    if trimdata:
        minlen = np.min([len(trimdata1), len(trimdata2)])
        trimdata1 = trimdata1[0:minlen]
        trimdata2 = trimdata2[0:minlen]

    # band limit the regressor if that is needed
    if theprefilter.gettype() != "None":
        if verbose:
            print("filtering to ", theprefilter.gettype(), " band")
    filtereddata1 = tide_math.corrnormalize(
        theprefilter.apply(Fs, trimdata1),
        windowfunc=windowfunc,
        detrendorder=detrendorder,
    )
    filtereddata2 = tide_math.corrnormalize(
        theprefilter.apply(Fs, trimdata2),
        windowfunc=windowfunc,
        detrendorder=detrendorder,
    )
    if flipregressor:
        filtereddata2 *= -1.0
    if dumpfiltered:
        tide_io.writenpvecs(filtereddata1, "filtereddata1.txt")
        tide_io.writenpvecs(filtereddata2, "filtereddata2.txt")

    if dopartial:
        controlvars = tide_io.readvecs(controlvariablefile)
        regressorvec = []
        for j in range(0, controlvars.shape[0]):
            regressorvec.append(
                tide_math.corrnormalize(
                    theprefilter.apply(Fs, controlvars[j, :]),
                    windowfunc=windowfunc,
                    detrendorder=detrendorder,
                )
            )
        if (np.max(filtereddata1) - np.min(filtereddata1)) > 0.0:
            thefit, filtereddata1 = tide_fit.mlregress(regressorvec, filtereddata1)
        if (np.max(filtereddata2) - np.min(filtereddata2)) > 0.0:
            thefit, filtereddata2 = tide_fit.mlregress(regressorvec, filtereddata2)

    if gccphat:
        thexcorr = tide_corr.fastcorrelate(
            filtereddata1, filtereddata2, usefft=dofftcorr, weighting="phat"
        )
    else:
        thexcorr = tide_corr.fastcorrelate(filtereddata1, filtereddata2, usefft=dofftcorr)
    thepxcorr = pearsonr(filtereddata1, filtereddata2)

    xcorrlen = len(thexcorr)
    sampletime = 1.0 / Fs
    xcorr_x = r_[0.0:xcorrlen] * sampletime - (xcorrlen * sampletime) / 2.0 + sampletime / 2.0
    halfwindow = int(searchrange * Fs)
    corrzero = xcorrlen // 2
    searchstart = corrzero - halfwindow
    searchend = corrzero + halfwindow
    xcorr_x_trim = xcorr_x[searchstart : searchend + 1]
    thexcorr_trim = thexcorr[searchstart : searchend + 1]
    if debug:
        print("searching for peak correlation over range ", searchstart, searchend)
    maxdelay = xcorr_x_trim[argmax(thexcorr_trim)]
    if debug:
        print("maxdelay before refinement", maxdelay)
    dofindmaxlag = True
    if dofindmaxlag:
        print("executing findmaxlag")
        (
            maxindex,
            maxdelay,
            maxval,
            maxsigma,
            maskval,
            failreason,
            peakstart,
            peakend,
        ) = tide_fit.findmaxlag_gauss(
            xcorr_x_trim,
            thexcorr_trim,
            -searchrange,
            searchrange,
            1000.0,
            refine=True,
            useguess=False,
            fastgauss=False,
            displayplots=False,
        )
        print(maxindex, maxdelay, maxval, maxsigma, maskval, failreason)
        R = maxval
    if debug:
        print("maxdelay after refinement", maxdelay)
        if failreason > 0:
            print("failreason =", failreason)
    else:
        R = thexcorr_trim[argmax(thexcorr_trim)]

    # set the significance threshold
    if estimate_significance:
        # generate a list of correlations from shuffled data
        corrlist, corrlist_pear = getNullDistributionData(
            trimdata1,
            xcorr_x,
            theprefilter,
            windowfunc,
            detrendorder,
            searchstart,
            searchend,
            numreps=numreps,
        )

        # calculate percentiles for the crosscorrelation from the distribution data
        histlen = 100
        thepercentiles = [0.95, 0.99, 0.995]

        pcts, pcts_fit, histfit = tide_stats.sigFromDistributionData(
            corrlist, histlen, thepercentiles
        )
        if debug:
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

        # calculate significance for the pearson correlation
        pearpcts, pearpcts_fit, histfit = tide_stats.sigFromDistributionData(
            corrlist_pear, histlen, thepercentiles
        )
        if debug:
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

        if writecorrlists:
            tide_io.writenpvecs(corrlist, "corrlist.txt")
            tide_io.writenpvecs(corrlist_pear, "corrlist_pear.txt")

    # report the pearson correlation
    if showpearson and verbose:
        print("Pearson_R:\t", thepxcorr[0])
        if estimate_significance:
            for idx, percentile in enumerate(thepercentiles):
                print(
                    "    pear_p(",
                    "{:.3f}".format(1.0 - percentile),
                    "):\t",
                    pearpcts[idx],
                )
        print("")

    if debug:
        print(thepxcorr)

    if verbose:
        if uselabel:
            print(thelabel, ":\t", maxdelay)
        else:
            print("Crosscorrelation_Rmax:\t", R)
            print("Crosscorrelation_maxdelay:\t", maxdelay)
            if estimate_significance:
                for idx, percentile in enumerate(thepercentiles):
                    print(
                        "    xc_p(",
                        "{:.3f}".format(1.0 - percentile),
                        "):\t",
                        pcts[idx],
                    )
            print(infilename1, "[0 seconds] == ", infilename2, "[", -maxdelay, " seconds]")

    if summarymode:
        if estimate_significance:
            if uselabel:
                if labelline:
                    print(
                        "thelabel",
                        "pearson_R",
                        "pearson_R(p=0.05)",
                        "xcorr_R",
                        "xcorr_R(P=0.05)",
                        "xcorr_maxdelay",
                    )
                print(thelabel, thepxcorr[0], pearpcts_fit[0], R, pcts_fit[0], -maxdelay)
            else:
                if labelline:
                    print(
                        "pearson_R",
                        "pearson_R(p=0.05)",
                        "xcorr_R",
                        "xcorr_R(P=0.05)",
                        "xcorr_maxdelay",
                    )
                print(thepxcorr[0], pearpcts_fit[0], R, pcts_fit[0], -maxdelay)
        else:
            if uselabel:
                if labelline:
                    print(
                        "thelabel",
                        "pearson_r",
                        "pearson_p",
                        "xcorr_R",
                        "xcorr_maxdelay",
                    )
                print(thelabel, thepxcorr[0], thepxcorr[1], R, -maxdelay)
            else:
                if labelline:
                    print("pearson_r\tpearson_p\txcorr_R\txcorr_t\txcorr_maxdelay")
                print(thepxcorr[0], "\t", thepxcorr[1], "\t", R, "\t", -maxdelay)

    if displayplots:
        fig = figure()
        ax = fig.add_subplot(111)
        # ax.set_title('GCC')
        plot(xcorr_x, thexcorr, "k")
        show()


if __name__ == "__main__":
    main()
