#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
#   Copyright 2016-2019 Blaise Frederick
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
from __future__ import print_function

import os.path as op

import numpy as np
import matplotlib.pyplot as plt 

import rapidtide.io as tide_io
import rapidtide.peakeval as tide_peakeval
import rapidtide.filter as tide_filt
import rapidtide.helper_classes as tide_classes
import rapidtide.resample as tide_resample
from rapidtide.tests.utils import get_test_data_path
import rapidtide.calcsimfunc as tide_calcsimfunc

import matplotlib as mpl
mpl.use('Qt5Agg')


def test_peakeval(display=False, debug=False):
    Fs = 10.0
    numpoints = 1000
    thefreq = 0.1
    numlocs = 10
    timestep = 0.25
    oversampfac = 2
    detrendorder = 1
    corrtr = 1.0 / (Fs * oversampfac)
    smoothingtime = 1.0
    bipolar = False
    nprocs = 1
    interptype = 'univariate'
    lagmod = 1000.0
    lagmin = -10.0
    lagmax = 10.0
    lagmininpts = int((-lagmin / corrtr) - 0.5)
    lagmaxinpts = int((lagmax / corrtr) + 0.5)
    peakfittype = 'gauss'
    corrweighting = 'None'


    timepoints = np.linspace(0.0, numpoints / Fs, num=numpoints, endpoint=False)
    oversamptimepoints = np.linspace(0.0, numpoints / (oversampfac * Fs), num=oversampfac * numpoints, endpoint=False)
    waveforms = np.zeros((numlocs, numpoints), dtype=np.float)
    if display:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for i in range(waveforms.shape[0]):
            waveforms[i, :] = np.cos((timepoints + i * timestep) * thefreq * 2.0 * np.pi) * tide_filt.hamming(numpoints)
            ax.plot(timepoints, waveforms[i, :], 'r')
        plt.show()
    referencetc = tide_resample.doresample(timepoints, waveforms[0, :], oversamptimepoints, method=interptype)

    theprefilter = tide_filt.noncausalfilter('arb',
                                             transferfunc='brickwall',
                                             debug=False)
    theprefilter.setfreqs(0.0, 0.0, 1.0, 1.0)

    # set up thecorrelator
    print('\n\nsetting up thecorrelator')
    thecorrelator = tide_classes.correlator(Fs=Fs*oversampfac,
                                            ncprefilter=theprefilter,
                                            detrendorder=detrendorder,
                                            windowfunc='hamming',
                                            corrweighting=corrweighting,
                                            hpfreq=None,
                                            debug=True)
    thecorrelator.setreftc(np.zeros((oversampfac * numpoints), dtype=np.float))
    thecorrelator.setlimits(lagmininpts, lagmaxinpts)
    corrorigin = thecorrelator.similarityfuncorigin
    dummy, trimmedcorrscale, dummy = thecorrelator.getfunction()
    corroutlen = np.shape(trimmedcorrscale)[0]
    internalvalidcorrshape = (numlocs, corroutlen)
    corrout = np.zeros(internalvalidcorrshape, dtype=np.float)
    meanval = np.zeros((numlocs), dtype=np.float)
    print('corrout shape:', corrout.shape)
    print('thecorrelator: corroutlen=', corroutlen, ', corrorigin=', corrorigin)

    # set up themutualinformationator
    print('\n\nsetting up themutualinformationator')
    themutualinformationator = tide_classes.mutualinformationator(Fs=Fs*oversampfac,
                                                                  smoothingtime=smoothingtime,
                                                                  ncprefilter=theprefilter,
                                                                  detrendorder=detrendorder,
                                                                  windowfunc='hamming',
                                                                  madnorm=False,
                                                                  lagmininpts=lagmininpts,
                                                                  lagmaxinpts=lagmaxinpts,
                                                                  debug=False)

    themutualinformationator.setreftc(np.zeros((oversampfac * numpoints), dtype=np.float))
    themutualinformationator.setlimits(lagmininpts, lagmaxinpts)


    # set up thefitter
    print('\n\nsetting up thefitter')
    thefitter = tide_classes.simfunc_fitter(lagmod=lagmod,
                                             lthreshval=0.0,
                                             uthreshval=1.0,
                                             bipolar=bipolar,
                                             lagmin=lagmin,
                                             lagmax=lagmax,
                                             absmaxsigma=10000.0,
                                             absminsigma=0.01,
                                             debug=False,
                                             peakfittype=peakfittype
                                             )

    # call correlationpass
    print('\n\ncalling correlationpass')
    print('waveforms shape:', waveforms.shape)
    voxelsprocessed_cp, theglobalmaxlist, trimmedcorrscale = tide_calcsimfunc.correlationpass(
                waveforms[:, :],
                referencetc,
                thecorrelator,
                timepoints,
                oversamptimepoints,
                lagmininpts,
                lagmaxinpts,
                corrout,
                meanval,
                nprocs=nprocs,
                alwaysmultiproc=False,
                oversampfactor=oversampfac,
                interptype=interptype,
                showprogressbar=False,
                chunksize=100)
    print(voxelsprocessed_cp, len(theglobalmaxlist), len(trimmedcorrscale))

    if display:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for i in range(numlocs):
            ax.plot(trimmedcorrscale, corrout[i, :])
        plt.show()

    # call peakeval
    print('\n\ncalling peakeval')
    voxelsprocessed_pe, thepeakdict = tide_peakeval.peakevalpass(
                waveforms[:, :],
                referencetc,
                timepoints,
                oversamptimepoints,
                themutualinformationator,
                trimmedcorrscale,
                corrout,
                nprocs=nprocs,
                alwaysmultiproc=False,
                bipolar=bipolar,
                oversampfactor=oversampfac,
                interptype=interptype,
                showprogressbar=False,
                chunksize=100)

    print(thepeakdict)
    
    #assert eval_fml_result(lagmin, lagmax, testlags, fml_maxlags, fml_lfailreasons)
    #assert eval_fml_result(absminval, absmaxval, testvals, fml_maxvals, fml_lfailreasons)
    #assert eval_fml_result(absminsigma, absmaxsigma, testsigmas, fml_maxsigmas, fml_lfailreasons)

    #assert eval_fml_result(lagmin, lagmax, testlags, fmlc_maxlags, fmlc_lfailreasons)
    #assert eval_fml_result(absminval, absmaxval, testvals, fmlc_maxvals, fmlc_lfailreasons)
    #assert eval_fml_result(absminsigma, absmaxsigma, testsigmas, fmlc_maxsigmas, fmlc_lfailreasons)

def main():
    test_peakeval(display=True, debug=True)


if __name__ == '__main__':
    main()
