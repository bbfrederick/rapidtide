#!/usr/bin/env python
# -*- coding: latin-1 -*-
from __future__ import print_function

import os.path as op

import numpy as np
import matplotlib.pyplot as plt 
from scipy import arange

import rapidtide.io as tide_io
import rapidtide.fit as tide_fit
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


def test_findmaxlag(display=False, debug=False):
    textfilename = op.join(get_test_data_path(), 'lt_rt.txt')

    # set default variable values
    searchfrac = 0.75
    limitfit = False

    indata = tide_io.readvecs(textfilename)
    xvecs = indata[0, :]
    yvecs = indata[1, :]


    # test over the lag range
    testlags = np.linspace(-25.0,25.0, 50, endpoint=True)
    fml_maxlags = np.zeros(len(testlags), dtype=np.float)
    fml_failreasons = np.zeros(len(testlags), dtype=np.int)
    fmlr_maxlags = np.zeros(len(testlags), dtype=np.float)
    fmlr_failreasons = np.zeros(len(testlags), dtype=np.int)

    testmaxval = 0.8
    testmaxsigma = 5.0
    lagmin = -20.0
    lagmax = 20.0
    widthlimit = 1000.0
    absmaxsigma = 1000.0

    for i in range(len(testlags)):
        yvecs = tide_fit.gauss_eval(xvecs, np.array([testmaxval, testlags[i],
                                                     testmaxsigma]))

        (maxindex, fml_maxlags[i], maxval, maxsigma, maskval,
         fml_failreasons[i], peakstart, peakend) = tide_fit.findmaxlag_gauss(
             xvecs,
             yvecs,
             lagmin, lagmax, widthlimit,
             tweaklims=False,
             refine=True,
             debug=debug,
             searchfrac=searchfrac,
             zerooutbadfit=False)

        (maxindexr, fmlr_maxlags[i], maxvalr, maxsigmar, maskvalr,
         fmlr_failreasons[i], peakstartr, peakendr) = tide_fit.findmaxlag_gauss_rev(
             xvecs,
             yvecs,
             lagmin, lagmax, widthlimit,
             absmaxsigma=absmaxsigma,
             tweaklims=False,
             refine=True,
             debug=debug,
             searchfrac=searchfrac,
             zerooutbadfit=False)

    if debug:
        print('findmaxlag_gauss results over lag range')
        for i in range(len(testlags)):
            print(testlags[i], fml_maxlags[i], fml_failreasons[i])

        print('\nfindmaxlag_gauss_rev results over lag range')
        for i in range(len(testlags)):
            print(testlags[i], fmlr_maxlags[i], fmlr_failreasons[i])

    assert eval_fml_lag(lagmin, lagmax, testlags, fml_maxlags)
    assert eval_fml_lag(lagmin, lagmax, testlags, fmlr_maxlags)

    if display:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(testlags, fml_maxlags, 'r')
        ax.plot(testlags, fmlr_maxlags, 'g')
        ax.legend(['findmaxlag_gauss', 'findmaxlag_gauss_rev'])
        #ax.set_xlim((lagmin, lagmax))
        plt.show()


def main():
    test_findmaxlag(display=True, debug=True)


if __name__ == '__main__':
    main()
