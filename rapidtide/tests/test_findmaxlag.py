#!/usr/bin/env python
# -*- coding: latin-1 -*-
from __future__ import print_function

import os.path as op

import numpy as np
import matplotlib.pyplot as plt 
from scipy import arange

import rapidtide.io as tide_io
import rapidtide.fit as tide_fit
import rapidtide.helper_classes as tide_classes
from rapidtide.tests.utils import get_test_data_path


def eval_fml_lag(lagmin, lagmax, testlags, foundlags, tolerance=0.0001):
    for i in range(len(testlags)):
        if testlags[i] < lagmin:
            if foundlags[i] != lagmin:
                print(foundlags[i], ' != ', lagmin, 'for input', testlags[i])
                return False
        elif testlags[i] > lagmax:
            if foundlags[i] != lagmax:
                print(foundlags[i], ' != ', lagmax, 'for input', testlags[i])
                return False
        else:
            if np.fabs(foundlags[i] - testlags[i]) > tolerance:
                print(foundlags[i], ' != ', testlags[i])
                return False
    return True


def eval_fml_sigma(sigmamin, sigmamax, testsigmas, foundsigmas, tolerance=0.0001):
    for i in range(len(testsigmas)):
        if testsigmas[i] < sigmamin:
            if foundsigmas[i] != sigmamin:
                print(foundsigmas[i], ' != ', sigmamin, 'for input', testsigmas[i])
                return False
        elif testsigmas[i] > sigmamax:
            if foundsigmas[i] != sigmamax:
                print(foundsigmas[i], ' != ', sigmamax, 'for input', testsigmas[i])
                return False
        else:
            if np.fabs(foundsigmas[i] - testsigmas[i]) > tolerance:
                print(foundsigmas[i], ' != ', testsigmas[i])
                return False
    return True


def test_findmaxlag(display=False, debug=False):
    textfilename = op.join(get_test_data_path(), 'lt_rt.txt')

    # set default variable values
    searchfrac = 0.75
    limitfit = False

    indata = tide_io.readvecs(textfilename)
    xvecs = indata[0, :]
    yvecs = indata[1, :]

    # set some fit parameters
    lagmin = -20.0
    lagmax = 20.0
    widthlimit = 1000.0
    absmaxsigma = 1000.0
    absminsigma = 0.10

    # test over the lag range
    testlags = np.linspace(-25.0,25.0, 50, endpoint=True)
    fml_maxlags = np.zeros(len(testlags), dtype=np.float)
    fml_lfailreasons = np.zeros(len(testlags), dtype=np.int)
    #fmlr_maxlags = np.zeros(len(testlags), dtype=np.float)
    #fmlr_lfailreasons = np.zeros(len(testlags), dtype=np.int)
    fmlc_maxlags = np.zeros(len(testlags), dtype=np.float)
    fmlc_lfailreasons = np.zeros(len(testlags), dtype=np.int)

    testmaxval = 0.8
    testmaxsigma = 5.0

    # initialize the correlation fitter
    thefitter = tide_classes.correlation_fitter(corrtimeaxis=xvecs,
                                                lagmin=lagmin,
                                                lagmax=lagmax,
                                                absmaxsigma=absmaxsigma,
                                                absminsigma=absminsigma,
                                                refine=True, debug=debug,
                                                searchfrac=searchfrac,
                                                zerooutbadfit=False)

    for i in range(len(testlags)):
        yvecs = tide_fit.gauss_eval(xvecs, np.array([testmaxval, testlags[i],
                                                     testmaxsigma]))

        print()
        print()
        print()
        (maxindex, fml_maxlags[i], maxval, maxsigma, maskval,
         fml_lfailreasons[i], peakstart, peakend) = tide_fit.findmaxlag_gauss(
             xvecs,
             yvecs,
             lagmin, lagmax, widthlimit,
             tweaklims=False,
             refine=True,
             debug=debug,
             searchfrac=searchfrac,
             absmaxsigma=absmaxsigma,
             absminsigma=absminsigma,
             zerooutbadfit=False)

        #print()
        #print()
        #print()
        #(maxindexr, fmlr_maxlags[i], maxvalr, maxsigmar, maskvalr,
        # fmlr_lfailreasons[i], peakstartr, peakendr) = tide_fit.findmaxlag_gauss_rev(
        #     xvecs,
        #     yvecs,
        #     lagmin, lagmax, widthlimit,
        #     absmaxsigma=absmaxsigma,
        #     tweaklims=False,
        #     refine=True,
        #     debug=debug,
        #     searchfrac=searchfrac,
        #     zerooutbadfit=False)

        print()
        print()
        print()
        (maxindexc, fmlc_maxlags[i], maxvalc, maxsigmac, maskvalc, fmlc_lfailreasons[i], peakstartc, peakendc) = thefitter.fit(yvecs)
        print(maxindexc, fmlc_maxlags[i], maxvalc, maxsigmac, maskvalc, fmlc_lfailreasons[i], peakstartc, peakendc)


    if debug:
        print('findmaxlag_gauss results over lag range')
        for i in range(len(testlags)):
            print(testlags[i], fml_maxlags[i], fml_lfailreasons[i])

        #print('\nfindmaxlag_gauss_rev results over lag range')
        #for i in range(len(testlags)):
        #    print(testlags[i], fmlr_maxlags[i], fmlr_lfailreasons[i])

    assert eval_fml_lag(lagmin, lagmax, testlags, fml_maxlags)
    #assert eval_fml_lag(lagmin, lagmax, testlags, fmlr_maxlags)
    assert eval_fml_lag(lagmin, lagmax, testlags, fmlc_maxlags)

    if display:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(testlags, fml_maxlags, 'r')
        #ax.plot(testlags, fmlr_maxlags, 'g')
        ax.plot(testlags, fmlc_maxlags, 'b')
        ax.legend(['findmaxlag_gauss', 'classes'])
        #ax.set_xlim((lagmin, lagmax))
        plt.show()

    # now test over range of sigmas
    testlag = 5.0
    testsigmas = np.asarray([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0])
    fml_maxsigmas = np.zeros(len(testsigmas), dtype=np.float)
    fml_wfailreasons = np.zeros(len(testsigmas), dtype=np.int)
    #fmlr_maxsigmas = np.zeros(len(testsigmas), dtype=np.float)
    #fmlr_wfailreasons = np.zeros(len(testsigmas), dtype=np.int)
    fmlc_maxsigmas = np.zeros(len(testsigmas), dtype=np.float)
    fmlc_wfailreasons = np.zeros(len(testsigmas), dtype=np.int)
    peakstartc = np.zeros(len(testsigmas), dtype=np.int)
    peakendc = np.zeros(len(testsigmas), dtype=np.int)

    for i in range(len(testsigmas)):
        testmaxval = 0.8
        yvecs = tide_fit.gauss_eval(xvecs, np.array([testmaxval, testlag,
                                                     testsigmas[i]]))

        print()
        print()
        print()
        (maxindex, maxlag, maxval, fml_maxsigmas[i], maskval,
         fml_wfailreasons[i], peakstart, peakend) = tide_fit.findmaxlag_gauss(
             xvecs,
             yvecs,
             lagmin, lagmax, widthlimit,
             tweaklims=False,
             refine=True,
             debug=debug,
             searchfrac=searchfrac,
             absmaxsigma=absmaxsigma,
             absminsigma=absminsigma,
             zerooutbadfit=False)

        #print()
        #print()
        #print()
        #(maxindexr, maxlagr, maxvalr, fmlr_maxsigmas[i], maskvalr,
        # fmlr_wfailreasons[i], peakstartr, peakendr) = tide_fit.findmaxlag_gauss_rev(
        #     xvecs,
        #     yvecs,
        #     lagmin, lagmax, widthlimit,
        #     absmaxsigma=absmaxsigma,
        #     tweaklims=False,
        #     refine=True,
        #     debug=debug,
        #     searchfrac=searchfrac,
        #     zerooutbadfit=False)

        print()
        print()
        print()
        (maxindexc, maxlagc, maxvalc, fmlc_maxsigmas[i], maskvalc, fmlc_wfailreasons[i], peakstartc[i], peakendc[i]) = thefitter.fit(yvecs)
        print(maxindexc, fmlc_maxlags[i], maxvalc, maxsigmac, maskvalc, fmlc_lfailreasons[i], peakstartc[i], peakendc[i])


    if debug:
        print('findmaxlag_gauss results over sigma range')
        for i in range(len(testsigmas)):
            print(testsigmas[i], fml_maxsigmas[i], maxval, maxlag, fml_wfailreasons[i])

        #print('\nfindmaxlag_gauss_rev results over lag range')
        #for i in range(len(testsigmas)):
        #    print(testsigmas[i], fmlr_maxsigmas[i], maxvalr, maxlagr, fmlr_wfailreasons[i])

        print('\nfitter class results over lag range')
        for i in range(len(testsigmas)):
            print(testsigmas[i], fmlc_maxsigmas[i], maxvalc, maxlagc, peakstartc[i], peakendc[i], thefitter.diagnosefail(fmlc_wfailreasons[i]))

    if display:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.loglog(testsigmas, fml_maxsigmas, 'r')
        #ax.loglog(testsigmas, fmlr_maxsigmas, 'g')
        ax.loglog(testsigmas, fmlc_maxsigmas, 'b')
        ax.legend(['findmaxlag_gauss', 'classes'])
        plt.show()

    assert eval_fml_sigma(absminsigma, absmaxsigma, testsigmas, fml_maxsigmas)
    #assert eval_fml_sigma(absminsigma, absmaxsigma, testsigmas, fmlr_maxsigmas)
    assert eval_fml_sigma(absminsigma, absmaxsigma, testsigmas, fmlc_maxsigmas)

def main():
    test_findmaxlag(display=True, debug=True)


if __name__ == '__main__':
    main()
