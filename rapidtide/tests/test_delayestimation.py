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
import multiprocessing as mp

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import rapidtide.calcsimfunc as tide_calcsimfunc
import rapidtide.filter as tide_filt
import rapidtide.helper_classes as tide_classes
import rapidtide.linfitfiltpass as tide_linfitfiltpass
import rapidtide.miscmath as tide_math
import rapidtide.peakeval as tide_peakeval
import rapidtide.resample as tide_resample
import rapidtide.simfuncfit as tide_simfuncfit
import rapidtide.util as tide_util

try:
    import mkl

    mklexists = True
except ImportError:
    mklexists = False


def multisine(timepoints, parameterlist):
    output = timepoints * 0.0
    for element in parameterlist:
        amp, freq, phase = element
        output += amp * np.sin(2.0 * np.pi * freq * timepoints + phase)
    return output


def checkfits(foundvalues, testvalues, tolerance=0.001):
    for i in range(len(foundvalues)):
        if np.fabs(foundvalues[i] - testvalues[i]) > tolerance:
            print("error error exceeds", tolerance, "at", i)
            print(foundvalues[i], " != ", testvalues[i])
            return False
    return True


def test_delayestimation(displayplots=False, debug=False):
    # set the number of MKL threads to use
    if mklexists:
        print("disabling MKL")
        mkl.set_num_threads(1)

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
    interptype = "univariate"
    lagmod = 1000.0
    lagmin = -20.0
    lagmax = 20.0
    lagmininpts = int((-lagmin / corrtr) - 0.5)
    lagmaxinpts = int((lagmax / corrtr) + 0.5)
    peakfittype = "gauss"
    corrweighting = "None"
    similaritymetric = "hybrid"
    windowfunc = "hamming"
    chunksize = 5
    pedestal = 100.0

    # set up the filter
    theprefilter = tide_filt.NoncausalFilter("arb", transferfunc="brickwall", debug=False)
    theprefilter.setfreqs(0.009, 0.01, 0.15, 0.16)

    # construct the various test waveforms
    timepoints = np.linspace(0.0, numpoints / Fs, num=numpoints, endpoint=False)
    oversamptimepoints = np.linspace(
        0.0, numpoints / Fs, num=oversampfac * numpoints, endpoint=False
    )
    waveforms = np.zeros((numlocs, numpoints), dtype=np.float64)
    paramlist = [
        [0.314, 0.055457, 0.0],
        [-0.723, 0.08347856, np.pi],
        [-0.834, 0.1102947, 0.0],
        [1.0, 0.13425, 0.5],
    ]
    offsets = np.zeros(numlocs, dtype=np.float64)
    amplitudes = np.ones(numlocs, dtype=np.float64)
    for i in range(numlocs):
        offsets[i] = timestep * (i - refnum)
        waveforms[i, :] = multisine(timepoints - offsets[i], paramlist) + pedestal
    if displayplots:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for i in range(numlocs):
            ax.plot(timepoints, waveforms[i, :])
        plt.show()

    threshval = pedestal / 4.0
    waveforms, waveforms_shm = tide_util.numpy2shared(waveforms, np.float64)

    referencetc = tide_resample.doresample(
        timepoints, waveforms[refnum, :], oversamptimepoints, method=interptype
    )
    referencetc = theprefilter.apply(oversampfreq, referencetc)
    referencetc = tide_math.corrnormalize(
        referencetc, detrendorder=detrendorder, windowfunc=windowfunc
    )

    # set up theCorrelator
    if debug:
        print("\n\nsetting up theCorrelator")
    theCorrelator = tide_classes.Correlator(
        Fs=oversampfreq,
        ncprefilter=theprefilter,
        detrendorder=detrendorder,
        windowfunc=windowfunc,
        corrweighting=corrweighting,
        debug=True,
    )
    theCorrelator.setreftc(np.zeros((oversampfac * numpoints), dtype=np.float64))
    theCorrelator.setlimits(lagmininpts, lagmaxinpts)
    dummy, trimmedcorrscale, dummy = theCorrelator.getfunction()
    corroutlen = np.shape(trimmedcorrscale)[0]
    internalvalidcorrshape = (numlocs, corroutlen)
    corrout, corrout_shm = tide_util.allocshared(internalvalidcorrshape, np.float64)
    meanval, meanval_shm = tide_util.allocshared((numlocs), np.float64)
    if debug:
        print("corrout shape:", corrout.shape)
        print("theCorrelator: corroutlen=", corroutlen)

    # set up theMutualInformationator
    if debug:
        print("\n\nsetting up theMutualInformationator")
    theMutualInformationator = tide_classes.MutualInformationator(
        Fs=oversampfreq,
        smoothingtime=smoothingtime,
        ncprefilter=theprefilter,
        detrendorder=detrendorder,
        windowfunc=windowfunc,
        madnorm=False,
        lagmininpts=lagmininpts,
        lagmaxinpts=lagmaxinpts,
        debug=False,
    )

    theMutualInformationator.setreftc(np.zeros((oversampfac * numpoints), dtype=np.float64))
    theMutualInformationator.setlimits(lagmininpts, lagmaxinpts)

    # set up thefitter
    if debug:
        print("\n\nsetting up thefitter")
    thefitter = tide_classes.SimilarityFunctionFitter(
        lagmod=lagmod,
        lthreshval=0.0,
        uthreshval=1.0,
        bipolar=bipolar,
        lagmin=lagmin,
        lagmax=lagmax,
        absmaxsigma=10000.0,
        absminsigma=0.01,
        debug=False,
        peakfittype=peakfittype,
    )

    lagtc, lagtc_shm = tide_util.allocshared(waveforms.shape, np.float64)
    fitmask, fitmask_shm = tide_util.allocshared((numlocs), "uint16")
    failreason, failreason_shm = tide_util.allocshared((numlocs), "uint32")
    lagtimes, lagtimes_shm = tide_util.allocshared((numlocs), np.float64)
    lagstrengths, lagstrengths_shm = tide_util.allocshared((numlocs), np.float64)
    lagsigma, lagsigma_shm = tide_util.allocshared((numlocs), np.float64)
    gaussout, gaussout_shm = tide_util.allocshared(internalvalidcorrshape, np.float64)
    windowout, windowout_shm = tide_util.allocshared(internalvalidcorrshape, np.float64)
    rvalue, rvalue_shm = tide_util.allocshared((numlocs), np.float64)
    r2value, r2value_shm = tide_util.allocshared((numlocs), np.float64)
    fitcoff, fitcoff_shm = tide_util.allocshared((waveforms.shape), np.float64)
    fitNorm, fitNorm_shm = tide_util.allocshared((waveforms.shape), np.float64)
    R2, R2_shm = tide_util.allocshared((numlocs), np.float64)
    movingsignal, movingsignal_shm = tide_util.allocshared(waveforms.shape, np.float64)
    filtereddata, filtereddata_shm = tide_util.allocshared(waveforms.shape, np.float64)

    for nprocs in [4, 1]:
        # call correlationpass
        if debug:
            print("\n\ncalling correlationpass")
            print("waveforms shape:", waveforms.shape)
        (
            voxelsprocessed_cp,
            theglobalmaxlist,
            trimmedcorrscale,
        ) = tide_calcsimfunc.correlationpass(
            waveforms[:, :],
            referencetc,
            theCorrelator,
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
            chunksize=chunksize,
        )
        if debug:
            print(voxelsprocessed_cp, len(theglobalmaxlist), len(trimmedcorrscale))

        if displayplots:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            for i in range(numlocs):
                ax.plot(trimmedcorrscale, corrout[i, :])
            plt.show()

        # call peakeval
        if debug:
            print("\n\ncalling peakeval")
        voxelsprocessed_pe, thepeakdict = tide_peakeval.peakevalpass(
            waveforms[:, :],
            referencetc,
            timepoints,
            oversamptimepoints,
            theMutualInformationator,
            trimmedcorrscale,
            corrout,
            nprocs=nprocs,
            alwaysmultiproc=False,
            bipolar=bipolar,
            oversampfactor=oversampfac,
            interptype=interptype,
            showprogressbar=False,
            chunksize=chunksize,
        )

        if debug:
            for key in thepeakdict:
                print(key, thepeakdict[key])

        # call thefitter
        if debug:
            print("\n\ncalling fitter")
        thefitter.setfunctype(similaritymetric)
        thefitter.setcorrtimeaxis(trimmedcorrscale)

        if displayplots:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        if nprocs == 1:
            proctype = "singleproc"
        else:
            proctype = "multiproc"
        for peakfittype in ["fastgauss", "quad", "fastquad", "gauss"]:
            thefitter.setpeakfittype(peakfittype)
            voxelsprocessed_fc = tide_simfuncfit.fitcorr(
                trimmedcorrscale,
                thefitter,
                corrout,
                fitmask,
                failreason,
                lagtimes,
                lagstrengths,
                lagsigma,
                gaussout,
                windowout,
                R2,
                peakdict=thepeakdict,
                nprocs=nprocs,
                alwaysmultiproc=False,
                fixdelay=None,
                showprogressbar=False,
                chunksize=chunksize,
                despeckle_thresh=100.0,
                initiallags=None,
            )
            if debug:
                print(voxelsprocessed_fc)

            if debug:
                print("\npeakfittype:", peakfittype)
                for i in range(numlocs):
                    print(
                        "location",
                        i,
                        ":",
                        offsets[i],
                        lagtimes[i],
                        lagtimes[i] - offsets[i],
                        lagstrengths[i],
                        lagsigma[i],
                    )
                if displayplots:
                    ax.plot(offsets, lagtimes, label=peakfittype)
            if checkfits(lagtimes, offsets, tolerance=0.01):
                print(proctype, peakfittype, " lagtime: pass")
                assert True
            else:
                print(proctype, peakfittype, " lagtime: fail")
                assert False
            if checkfits(lagstrengths, amplitudes, tolerance=0.05):
                print(proctype, peakfittype, " lagstrength: pass")
                assert True
            else:
                print(proctype, peakfittype, " lagstrength: fail")
                assert False

    if displayplots:
        ax.legend()
        plt.show()

    filteredwaveforms, filteredwaveforms_shm = tide_util.allocshared(waveforms.shape, np.float64)
    for i in range(numlocs):
        filteredwaveforms[i, :] = theprefilter.apply(Fs, waveforms[i, :])

    for nprocs in [4, 1]:
        voxelsprocessed_regressionfilt = tide_linfitfiltpass.linfitfiltpass(
            numlocs,
            waveforms[:, :],
            threshval,
            lagtc,
            meanval,
            rvalue,
            r2value,
            fitcoff,
            fitNorm,
            movingsignal,
            filtereddata,
            nprocs=nprocs,
            alwaysmultiproc=False,
            showprogressbar=False,
            mp_chunksize=chunksize,
        )

        if nprocs == 1:
            proctype = "singleproc"
        else:
            proctype = "multiproc"
        diffsignal = filtereddata
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        # ax.plot(timepoints, filtereddata[refnum, :], label='filtereddata')
        ax.plot(oversamptimepoints, referencetc, label="referencetc")
        ax.plot(timepoints, movingsignal[refnum, :], label="movingsignal")
        ax.legend()
        plt.show()

        print(proctype, "linfitfiltpass", np.mean(diffsignal), np.max(np.fabs(diffsignal)))

    # clean up shared memory
    tide_util.cleanup_shm(waveforms_shm)
    tide_util.cleanup_shm(corrout_shm)
    tide_util.cleanup_shm(meanval_shm)
    tide_util.cleanup_shm(lagtc_shm)
    tide_util.cleanup_shm(fitmask_shm)
    tide_util.cleanup_shm(failreason_shm)
    tide_util.cleanup_shm(lagtimes_shm)
    tide_util.cleanup_shm(lagstrengths_shm)
    tide_util.cleanup_shm(lagsigma_shm)
    tide_util.cleanup_shm(gaussout_shm)
    tide_util.cleanup_shm(windowout_shm)
    tide_util.cleanup_shm(rvalue_shm)
    tide_util.cleanup_shm(r2value_shm)
    tide_util.cleanup_shm(fitcoff_shm)
    tide_util.cleanup_shm(fitNorm_shm)
    tide_util.cleanup_shm(R2_shm)
    tide_util.cleanup_shm(movingsignal_shm)
    tide_util.cleanup_shm(filtereddata_shm)
    tide_util.cleanup_shm(filteredwaveforms_shm)


if __name__ == "__main__":
    mpl.use("TkAgg")
    test_delayestimation(displayplots=False, debug=True)
