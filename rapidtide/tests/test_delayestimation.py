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


def multisine(timepoints, parameterlist):
    output = timepoints * 0.0
    for element in parameterlist:
        amp, freq, phase = element
        output += amp * np.sin(2.0 * np.pi * freq * timepoints + phase)
    return output

def checkfits(foundvalues, testvalues, tolerance=0.001):
    for i in range(len(foundvalues)):
        if np.fabs(foundvalues[i] - testvalues[i]) > tolerance:
            print('error error exceeds', tolerance, 'at', i)
            print(foundvalues[i], ' != ', testvalues[i])
            return False
    return True


def test_delayestimation(display=False, debug=False):

    # set parameters
    Fs = 10.0
    numpoints = 5000
    numlocs = 21
    refnum = int(numlocs // 2)
    timestep = 0.228764
    oversampfac = 2
    detrendorder = 1
    oversampfreq = Fs * oversampfac
    corrtr = 1.0 / oversampfreq
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

    # set up the filter
    theprefilter = tide_filt.noncausalfilter('arb',
                                             transferfunc='brickwall',
                                             debug=False)
    theprefilter.setfreqs(0.009, 0.01, 0.15, 0.16)


    # construct the various test waveforms
    timepoints = np.linspace(0.0, numpoints / Fs, num=numpoints, endpoint=False)
    oversamptimepoints = np.linspace(0.0, numpoints / Fs, num=oversampfac * numpoints, endpoint=False)
    waveforms = np.zeros((numlocs, numpoints), dtype=np.float)
    paramlist = [[1.0, 0.05, 0.0], [0.7, 0.08, np.pi], [0.2, 0.1, 0.0]]
    offsets = np.zeros(numlocs, dtype=np.float)
    amplitudes = np.ones(numlocs, dtype=np.float)
    for i in range(numlocs):
        offsets[i] = timestep * (i - refnum)
        waveforms[i, :] = multisine(timepoints - offsets[i], paramlist)
    if display:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for i in range(numlocs):
            ax.plot(timepoints, waveforms[i, :])
        plt.show()

    referencetc = tide_resample.doresample(timepoints, waveforms[refnum, :], oversamptimepoints, method=interptype)


    # set up thecorrelator
    if debug:
        print('\n\nsetting up thecorrelator')
    thecorrelator = tide_classes.correlator(Fs=oversampfreq,
                                            ncprefilter=theprefilter,
                                            detrendorder=detrendorder,
                                            windowfunc='hamming',
                                            corrweighting=corrweighting,
                                            hpfreq=None,
                                            debug=True)
    thecorrelator.setreftc(np.zeros((oversampfac * numpoints), dtype=np.float))
    thecorrelator.setlimits(lagmininpts, lagmaxinpts)
    dummy, trimmedcorrscale, dummy = thecorrelator.getfunction()
    corroutlen = np.shape(trimmedcorrscale)[0]
    internalvalidcorrshape = (numlocs, corroutlen)
    corrout = np.zeros(internalvalidcorrshape, dtype=np.float)
    meanval = np.zeros((numlocs), dtype=np.float)
    if debug:
        print('corrout shape:', corrout.shape)
        print('thecorrelator: corroutlen=', corroutlen)

    # set up themutualinformationator
    if debug:
        print('\n\nsetting up themutualinformationator')
    themutualinformationator = tide_classes.mutualinformationator(Fs=oversampfreq,
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
            print('location', i,':', offsets[i], lagtimes[i], lagtimes[i] - offsets[i], lagstrengths[i], lagsigma[i])
        if display:
            ax.plot(offsets, lagtimes, label=peakfittype)
        print('for', peakfittype)
        if checkfits(lagtimes, offsets, tolerance=0.001):
            print('\tlagtime: pass')
        else:
            print('\tlagtime: fail')
        if checkfits(lagstrengths, amplitudes, tolerance=0.001):
            print('\tlagstrength: pass')
        else:
            print('\tlagstrength: fail')

    if display:
        ax.legend()
        plt.show()

def main():
    test_delayestimation(display=True, debug=True)


if __name__ == '__main__':
    main()
