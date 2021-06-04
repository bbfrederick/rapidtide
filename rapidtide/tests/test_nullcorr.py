#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
#   Copyright 2016-2021 Blaise Frederick
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
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import rapidtide.calcnullsimfunc as tide_nullsimfunc
import rapidtide.correlate as tide_corr
import rapidtide.filter as tide_filt
import rapidtide.helper_classes as tide_classes
import rapidtide.io as tide_io
import rapidtide.stats as tide_stats
from rapidtide.tests.utils import get_test_data_path, get_test_temp_path


def test_nullsimfunc(debug=False, display=False):
    # make the lfo filter
    lfofilter = tide_filt.NoncausalFilter(filtertype="lfo")

    # make the starting regressor
    timestep = 1.5
    Fs = 1.0 / timestep
    # sourcelen = 1200
    # sourcedata = lfofilter.apply(Fs, np.random.rand(sourcelen))
    sourcedata = tide_io.readvecs(os.path.join(get_test_data_path(), "fmri_globalmean.txt"))[0]
    sourcelen = len(sourcedata)
    numpasses = 1

    if display:
        plt.figure()
        plt.plot(sourcedata)
        plt.show()

    thexcorr = tide_corr.fastcorrelate(sourcedata, sourcedata)
    xcorrlen = len(thexcorr)
    xcorr_x = (
        np.linspace(0.0, xcorrlen, xcorrlen, endpoint=False) * timestep
        - (xcorrlen * timestep) / 2.0
        + timestep / 2.0
    )

    if display:
        plt.figure()
        plt.plot(xcorr_x, thexcorr)
        plt.show()

    corrzero = xcorrlen // 2
    lagmin = -10
    lagmax = 10
    lagmininpts = int((-lagmin / timestep) - 0.5)
    lagmaxinpts = int((lagmax / timestep) + 0.5)

    searchstart = int(np.round(corrzero + lagmin / timestep))
    searchend = int(np.round(corrzero + lagmax / timestep))

    optiondict = {
        "numestreps": 10000,
        "showprogressbar": debug,
        "detrendorder": 3,
        "windowfunc": "hamming",
        "corrweighting": "None",
        "nprocs": 1,
        "widthlimit": 1000.0,
        "bipolar": False,
        "fixdelay": False,
        "peakfittype": "gauss",
        "lagmin": lagmin,
        "lagmax": lagmax,
        "absminsigma": 0.25,
        "absmaxsigma": 25.0,
        "edgebufferfrac": 0.0,
        "lthreshval": 0.0,
        "uthreshval": 1.0,
        "debug": False,
        "enforcethresh": True,
        "lagmod": 1000.0,
        "searchfrac": 0.5,
        "permutationmethod": "shuffle",
        "hardlimit": True,
    }
    theprefilter = tide_filt.NoncausalFilter("lfo")
    theCorrelator = tide_classes.Correlator(
        Fs=Fs,
        ncprefilter=theprefilter,
        detrendorder=optiondict["detrendorder"],
        windowfunc=optiondict["windowfunc"],
        corrweighting=optiondict["corrweighting"],
    )

    thefitter = tide_classes.SimilarityFunctionFitter(
        lagmod=optiondict["lagmod"],
        lthreshval=optiondict["lthreshval"],
        uthreshval=optiondict["uthreshval"],
        bipolar=optiondict["bipolar"],
        lagmin=optiondict["lagmin"],
        lagmax=optiondict["lagmax"],
        absmaxsigma=optiondict["absmaxsigma"],
        absminsigma=optiondict["absminsigma"],
        debug=optiondict["debug"],
        peakfittype=optiondict["peakfittype"],
        searchfrac=optiondict["searchfrac"],
        enforcethresh=optiondict["enforcethresh"],
        hardlimit=optiondict["hardlimit"],
    )

    if debug:
        print(optiondict)

    theCorrelator.setlimits(lagmininpts, lagmaxinpts)
    theCorrelator.setreftc(sourcedata)
    dummy, trimmedcorrscale, dummy = theCorrelator.getfunction()
    thefitter.setcorrtimeaxis(trimmedcorrscale)
    histograms = []
    for thenprocs in [1, -1]:
        for i in range(numpasses):
            corrlist = tide_nullsimfunc.getNullDistributionDatax(
                sourcedata,
                Fs,
                theCorrelator,
                thefitter,
                despeckle_thresh=5.0,
                fixdelay=False,
                fixeddelayvalue=0.0,
                numestreps=optiondict["numestreps"],
                nprocs=thenprocs,
                showprogressbar=optiondict["showprogressbar"],
                chunksize=1000,
                permutationmethod=optiondict["permutationmethod"],
            )
            tide_io.writenpvecs(corrlist, os.path.join(get_test_temp_path(), "corrdistdata.txt"))

            # calculate percentiles for the crosscorrelation from the distribution data
            histlen = 250
            thepercentiles = [0.95, 0.99, 0.995]

            pcts, pcts_fit, histfit = tide_stats.sigFromDistributionData(
                corrlist, histlen, thepercentiles
            )
            if debug:
                tide_stats.printthresholds(
                    pcts, thepercentiles, "Crosscorrelation significance thresholds from data:",
                )
                tide_stats.printthresholds(
                    pcts_fit, thepercentiles, "Crosscorrelation significance thresholds from fit:",
                )

            (thehist, peakheight, peakloc, peakwidth, centerofmass,) = tide_stats.makehistogram(
                np.abs(corrlist), histlen, therange=[0.0, 1.0]
            )
            histograms.append(thehist)
            thestore = np.zeros((2, len(thehist[0])), dtype="float64")
            thestore[0, :] = (thehist[1][1:] + thehist[1][0:-1]) / 2.0
            thestore[1, :] = thehist[0][-histlen:]
            if display:
                plt.figure()
                plt.plot(thestore[0, :], thestore[1, :])
                plt.show()

            # tide_stats.makeandsavehistogram(corrlist, histlen, 0,
            # os.path.join(get_test_temp_path(), 'correlationhist'),
            # displaytitle='Null correlation histogram',
            # displayplots=display, refine=False)
            assert True


if __name__ == "__main__":
    mpl.use("TkAgg")
    test_nullsimfunc(debug=True, display=True)
