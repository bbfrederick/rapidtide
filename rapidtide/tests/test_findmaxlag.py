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
import os.path as op
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import rapidtide.fit as tide_fit
import rapidtide.helper_classes as tide_classes
import rapidtide.io as tide_io
from rapidtide.tests.utils import get_examples_path


def dumplists(results, targets, failflags):
    print("assertion failed - dumping lists")
    if (len(results) != len(targets)) or (len(results) != len(failflags)):
        print("array lengths do not match")
        sys.exit()
    print("results", "targets")
    for i in range(len(results)):
        print(results[i], targets[i], failflags[i])


def eval_fml_result(absmin, absmax, testvalues, foundvalues, failflags, tolerance=0.0001):
    for i in range(len(testvalues)):
        if testvalues[i] < absmin:
            if foundvalues[i] != absmin:
                print(foundvalues[i], " != ", absmin, "for input", testvalues[i])
                dumplists(foundvalues, testvalues, failflags)
                return False
        elif testvalues[i] > absmax:
            if foundvalues[i] != absmax:
                print(foundvalues[i], " != ", absmax, "for input", testvalues[i])
                dumplists(foundvalues, testvalues, failflags)
                return False
        else:
            if np.fabs(foundvalues[i] - testvalues[i]) > tolerance:
                if failflags[i] == 0:
                    print("error found at index", i, failflags[i])
                    print(foundvalues[i], " != ", testvalues[i])
                    dumplists(foundvalues, testvalues, failflags)
                    return False
    return True


def test_findmaxlag(displayplots=False, fittype="gauss", debug=False):
    textfilename = op.join(get_examples_path(), "lt_rt.txt")

    # set default variable values
    searchfrac = 0.75

    indata = tide_io.readvecs(textfilename, debug=debug)
    xvecs = indata[0, :]
    yvecs = indata[1, :]

    # set some fit parameters
    lagmin = -20.0
    lagmax = 20.0
    widthlimit = 1000.0
    absmaxsigma = 1000.0
    absminsigma = 0.1
    absmaxval = 1.0
    absminval = 0.0

    # test over the lag range
    testmaxval = 0.8
    testmaxsigma = 5.0
    testlags = np.linspace(-25.0, 25.0, 50, endpoint=True)
    testsigmas = np.full((len(testlags)), testmaxsigma, dtype=np.float64)
    testvals = np.full((len(testlags)), testmaxval, dtype=np.float64)

    fml_maxlags = np.zeros(len(testlags), dtype=np.float64)
    fml_maxsigmas = np.zeros(len(testlags), dtype=np.float64)
    fml_maxvals = np.zeros(len(testlags), dtype=np.float64)
    fml_lfailreasons = np.zeros(len(testlags), dtype=np.uint16)
    fmlc_maxlags = np.zeros(len(testlags), dtype=np.float64)
    fmlc_maxsigmas = np.zeros(len(testlags), dtype=np.float64)
    fmlc_maxvals = np.zeros(len(testlags), dtype=np.float64)
    fmlc_lfailreasons = np.zeros(len(testlags), dtype=np.uint16)

    # initialize the correlation fitter
    thefitter = tide_classes.SimilarityFunctionFitter(
        corrtimeaxis=xvecs,
        lagmin=lagmin,
        lagmax=lagmax,
        absmaxsigma=absmaxsigma,
        absminsigma=absminsigma,
        peakfittype=fittype,
        debug=debug,
        searchfrac=searchfrac,
        zerooutbadfit=False,
    )

    for i in range(len(testlags)):
        yvecs = tide_fit.gauss_eval(xvecs, np.array([testvals[i], testlags[i], testsigmas[i]]))

        print()
        print()
        print()
        (
            maxindex,
            fml_maxlags[i],
            fml_maxvals[i],
            fml_maxsigmas[i],
            maskval,
            fml_lfailreasons[i],
            peakstart,
            peakend,
        ) = tide_fit.findmaxlag_gauss(
            xvecs,
            yvecs,
            lagmin,
            lagmax,
            widthlimit,
            tweaklims=False,
            refine=True,
            debug=debug,
            searchfrac=searchfrac,
            absmaxsigma=absmaxsigma,
            absminsigma=absminsigma,
            zerooutbadfit=False,
        )

        print()
        print()
        print()
        (
            maxindexc,
            fmlc_maxlags[i],
            fmlc_maxvals[i],
            fmlc_maxsigmas[i],
            maskvalc,
            fmlc_lfailreasons[i],
            peakstartc,
            peakendc,
        ) = thefitter.fit(yvecs)
        print(
            maxindexc,
            fmlc_maxlags[i],
            fmlc_maxvals[i],
            fmlc_maxsigmas[i],
            maskvalc,
            fmlc_lfailreasons[i],
            peakstartc,
            peakendc,
        )

    if debug:
        print("findmaxlag_gauss results over lag range")
        for i in range(len(testlags)):
            print(testlags[i], fml_maxlags[i], fml_lfailreasons[i])

    assert eval_fml_result(lagmin, lagmax, testlags, fml_maxlags, fml_lfailreasons)
    assert eval_fml_result(absminval, absmaxval, testvals, fml_maxvals, fml_lfailreasons)
    assert eval_fml_result(absminsigma, absmaxsigma, testsigmas, fml_maxsigmas, fml_lfailreasons)

    assert eval_fml_result(lagmin, lagmax, testlags, fmlc_maxlags, fmlc_lfailreasons)
    assert eval_fml_result(absminval, absmaxval, testvals, fmlc_maxvals, fmlc_lfailreasons)
    assert eval_fml_result(absminsigma, absmaxsigma, testsigmas, fmlc_maxsigmas, fmlc_lfailreasons)

    if displayplots:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(testlags, fml_maxlags, "r")
        ax.plot(testlags, fmlc_maxlags, "b")
        ax.legend(["findmaxlag_gauss", "classes"])
        plt.show()

    # now test over range of sigmas
    testlag = 5.0
    testsigmas = np.asarray(
        [
            0.01,
            0.02,
            0.05,
            0.1,
            0.2,
            0.5,
            1.0,
            2.0,
            5.0,
            10.0,
            20.0,
            50.0,
            100.0,
            200.0,
            500.0,
            1000.0,
            2000.0,
        ]
    )
    testlags = np.full((len(testsigmas)), testlag, dtype=np.float64)
    testvals = np.full((len(testsigmas)), testmaxval, dtype=np.float64)

    fml_maxlags = np.zeros(len(testsigmas), dtype=np.float64)
    fml_maxsigmas = np.zeros(len(testsigmas), dtype=np.float64)
    fml_maxvals = np.zeros(len(testsigmas), dtype=np.float64)
    fml_wfailreasons = np.zeros(len(testsigmas), dtype=np.uint16)
    fmlc_maxlags = np.zeros(len(testsigmas), dtype=np.float64)
    fmlc_maxsigmas = np.zeros(len(testsigmas), dtype=np.float64)
    fmlc_maxvals = np.zeros(len(testsigmas), dtype=np.float64)
    fmlc_wfailreasons = np.zeros(len(testsigmas), dtype=np.uint16)
    peakstartc = np.zeros(len(testsigmas), dtype=np.int32)
    peakendc = np.zeros(len(testsigmas), dtype=np.int32)

    for i in range(len(testsigmas)):
        yvecs = tide_fit.gauss_eval(xvecs, np.array([testvals[i], testlags[i], testsigmas[i]]))

        print()
        print()
        print()
        (
            maxindex,
            fml_maxlags[i],
            fml_maxvals[i],
            fml_maxsigmas[i],
            maskval,
            fml_wfailreasons[i],
            peakstart,
            peakend,
        ) = tide_fit.findmaxlag_gauss(
            xvecs,
            yvecs,
            lagmin,
            lagmax,
            widthlimit,
            tweaklims=False,
            refine=True,
            debug=debug,
            searchfrac=searchfrac,
            absmaxsigma=absmaxsigma,
            absminsigma=absminsigma,
            zerooutbadfit=False,
        )

        print()
        print()
        print()
        (
            maxindexc,
            fmlc_maxlags[i],
            fmlc_maxvals[i],
            fmlc_maxsigmas[i],
            maskvalc,
            fmlc_wfailreasons[i],
            peakstartc[i],
            peakendc[i],
        ) = thefitter.fit(yvecs)
        print(
            maxindexc,
            fmlc_maxlags[i],
            fmlc_maxvals[i],
            fmlc_maxsigmas[i],
            maskvalc,
            fmlc_wfailreasons[i],
            peakstartc[i],
            peakendc[i],
        )

    if debug:
        print("findmaxlag_gauss results over sigma range")
        for i in range(len(testsigmas)):
            print(
                testsigmas[i],
                fml_maxsigmas[i],
                fmlc_maxlags[i],
                fmlc_maxvals[i],
                fml_wfailreasons[i],
            )

        print("\nfitter class results over lag range")
        for i in range(len(testsigmas)):
            print(
                testsigmas[i],
                fmlc_maxsigmas[i],
                fmlc_maxlags[i],
                fmlc_maxvals[i],
                peakstartc[i],
                peakendc[i],
                fmlc_wfailreasons[i],
                thefitter.diagnosefail(np.uint32(fmlc_wfailreasons[i])),
            )

    if displayplots:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.loglog(testsigmas, fml_maxsigmas, "r")
        ax.loglog(testsigmas, fmlc_maxsigmas, "b")
        ax.legend(["findmaxlag_gauss", "classes"])
        plt.show()

    assert eval_fml_result(lagmin, lagmax, testlags, fml_maxlags, fml_wfailreasons)
    # assert eval_fml_result(absminval, absmaxval, testvals, fml_maxvals, fml_wfailreasons)
    assert eval_fml_result(absminsigma, absmaxsigma, testsigmas, fml_maxsigmas, fml_wfailreasons)

    assert eval_fml_result(lagmin, lagmax, testlags, fmlc_maxlags, fmlc_wfailreasons)
    assert eval_fml_result(absminval, absmaxval, testvals, fmlc_maxvals, fmlc_wfailreasons)
    assert eval_fml_result(absminsigma, absmaxsigma, testsigmas, fmlc_maxsigmas, fmlc_wfailreasons)


if __name__ == "__main__":
    mpl.use("TkAgg")
    test_findmaxlag(displayplots=True, debug=True)
