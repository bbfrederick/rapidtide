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

import rapidtide.fit as tide_fit
import rapidtide.peakeval as tide_peakeval
import rapidtide.filter as tide_filt
import rapidtide.helper_classes as tide_classes
import rapidtide.resample as tide_resample
import rapidtide.simfuncfit as tide_simfuncfit
import rapidtide.calcsimfunc as tide_calcsimfunc

import matplotlib as mpl
mpl.use('Qt5Agg')


def gaussianpacket(timepoints, offset, width, frequency):
    return tide_fit.gauss_eval(timepoints, [1.0, offset, width]) * np.cos((timepoints - offset) * frequency * 2.0 * np.pi)


def test_delayestimation(display=False, debug=False):
    Fs = 10.0
    numpoints = 5000
    thefreq = 0.1
    thewidth = 10.0
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
    similaritymetric = 'hybrid'


    timepoints = np.linspace(0.0, numpoints / Fs, num=numpoints, endpoint=False)
    oversamptimepoints = np.linspace(0.0, numpoints / (oversampfac * Fs), num=oversampfac * numpoints, endpoint=False)
    waveforms = np.zeros((numlocs, numpoints), dtype=np.float)
    if display:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for i in range(waveforms.shape[0]):
            waveforms[i, :] = gaussianpacket(timepoints, i * timestep + timepoints[int(len(timepoints) // 2)], thewidth, thefreq)
            ax.plot(timepoints, waveforms[i, :], 'r')
        plt.show()
    referencetc = tide_resample.doresample(timepoints, waveforms[0, :], oversamptimepoints, method=interptype)

    theprefilter = tide_filt.noncausalfilter('arb',
                                             transferfunc='brickwall',
                                             debug=False)
    theprefilter.setfreqs(0.0, 0.0, 1.0, 1.0)

    # set up thecorrelator
    if debug:
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
    if debug:
        print('corrout shape:', corrout.shape)
        print('thecorrelator: corroutlen=', corroutlen, ', corrorigin=', corrorigin)

    # set up themutualinformationator
    if debug:
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
    if debug:
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
    if debug:
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
    if debug:
        print(voxelsprocessed_cp, len(theglobalmaxlist), len(trimmedcorrscale))

    if display:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for i in range(numlocs):
            ax.plot(trimmedcorrscale, corrout[i, :])
        plt.show()

    # call peakeval
    if debug:
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

    if debug:
        for key in thepeakdict:
            print(key, thepeakdict[key])

    # call thefitter
    if debug:
        print('\n\ncalling fitter')
    thefitter.setfunctype(similaritymetric)
    thefitter.setcorrtimeaxis(trimmedcorrscale)
    genlagtc = tide_resample.fastresampler(timepoints, waveforms[0, :])
    lagtc = np.zeros(waveforms.shape, dtype=np.float)
    fitmask = np.zeros((numlocs), dtype='uint16')
    failreason = np.zeros((numlocs), dtype='uint32')
    lagtimes = np.zeros((numlocs), dtype=np.float)
    lagstrengths = np.zeros((numlocs), dtype=np.float)
    lagsigma = np.zeros((numlocs), dtype=np.float)
    gaussout = np.zeros(internalvalidcorrshape, dtype=np.float)
    windowout = np.zeros(internalvalidcorrshape, dtype=np.float)
    R2 = np.zeros((numlocs), dtype=np.float)

    if display:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        legend = []
    for peakfittype in ['gauss', 'fastgauss', 'quad', 'fastquad', 'COM']:
        thefitter.setpeakfittype(peakfittype)
        voxelsprocessed_fc = tide_simfuncfit.fitcorr(
            genlagtc,
            timepoints,
            lagtc,
            trimmedcorrscale,
            thefitter,
            corrout,
            fitmask, failreason, lagtimes, lagstrengths, lagsigma,
            gaussout, windowout, R2,
            peakdict=thepeakdict,
            nprocs=nprocs,
            alwaysmultiproc=False,
            fixdelay=None,
            showprogressbar=False,
            chunksize=1000,
            despeckle_thresh=100.0,
            initiallags=None
        )
        if debug:
            print(voxelsprocessed_fc)

        print('\npeakfittype:', peakfittype)
        for i in range(numlocs):
            print('location', i,':', lagtimes[i], lagstrengths[i], lagsigma[i])
        if display:
            ax.plot(np.linspace(0.0, numlocs, num=numlocs, endpoint=False) * timestep, lagtimes, label=peakfittype)
    if display:
        ax.legend()
        plt.show()

    #assert eval_fml_result(lagmin, lagmax, testlags, fml_maxlags, fml_lfailreasons)
    #assert eval_fml_result(absminval, absmaxval, testvals, fml_maxvals, fml_lfailreasons)
    #assert eval_fml_result(absminsigma, absmaxsigma, testsigmas, fml_maxsigmas, fml_lfailreasons)

    #assert eval_fml_result(lagmin, lagmax, testlags, fmlc_maxlags, fmlc_lfailreasons)
    #assert eval_fml_result(absminval, absmaxval, testvals, fmlc_maxvals, fmlc_lfailreasons)
    #assert eval_fml_result(absminsigma, absmaxsigma, testsigmas, fmlc_maxsigmas, fmlc_lfailreasons)

def main():
    test_delayestimation(display=True, debug=True)


if __name__ == '__main__':
    main()
