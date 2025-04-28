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
import copy
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math
import rapidtide.refinedelay as tide_refinedelay
import rapidtide.resample as tide_resample
from rapidtide.filter import NoncausalFilter
from rapidtide.tests.utils import get_examples_path, get_test_temp_path, mse


def eval_refinedelay(
    sampletime=0.72,
    tclengthinsecs=300.0,
    mindelay=-5.0,
    maxdelay=5.0,
    numpoints=501,
    smoothpts=3,
    nativespaceshape=(10, 10, 10),
    displayplots=False,
    padtime=30.0,
    noiselevel=0.0,
    outputsuffix="",
    debug=False,
):
    np.random.seed(12345)
    tclen = int(tclengthinsecs // sampletime)

    Fs = 1.0 / sampletime
    print("Testing transfer function:")
    lowestfreq = 1.0 / (sampletime * tclen)
    nyquist = 0.5 / sampletime
    print(
        "    sampletime=",
        sampletime,
        ", timecourse length=",
        tclengthinsecs,
        "s,  possible frequency range:",
        lowestfreq,
        nyquist,
    )

    # make an sLFO timecourse
    timeaxis = np.linspace(0.0, sampletime * tclen, num=tclen, endpoint=False)
    rawgms = tide_math.stdnormalize(np.random.normal(size=tclen))
    testfilter = NoncausalFilter(filtertype="lfo")
    sLFO = tide_math.stdnormalize(testfilter.apply(Fs, rawgms))
    if displayplots:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Initial regressor")
        plt.plot(timeaxis, rawgms)
        plt.plot(timeaxis, sLFO)
        plt.show()

    # now turn it into a lagtc generator
    numpadtrs = int(padtime // sampletime)
    padtime = sampletime * numpadtrs
    lagtcgenerator = tide_resample.FastResampler(timeaxis, sLFO, padtime=padtime)

    # find the mapping of derivative ratios to delays
    tide_refinedelay.trainratiotooffset(
        lagtcgenerator,
        timeaxis,
        os.path.join(get_test_temp_path(), "refinedelaytest" + outputsuffix),
        "norm",
        mindelay=mindelay,
        maxdelay=maxdelay,
        numpoints=numpoints,
        smoothpts=smoothpts,
        debug=debug,
    )

    # make a delay map
    numlags = nativespaceshape[0] * nativespaceshape[1] * nativespaceshape[2]
    lagtimes = np.linspace(mindelay, maxdelay, numlags, endpoint=True)
    if debug:
        print("    lagtimes=", lagtimes)

    # now make synthetic fMRI data
    internalvalidfmrishape = (numlags, tclen)
    fmridata = np.zeros(internalvalidfmrishape, dtype=float)
    fmrimask = np.ones(numlags, dtype=float)
    validvoxels = np.where(fmrimask > 0)[0]
    for i in range(numlags):
        noisevec = tide_math.stdnormalize(
            testfilter.apply(Fs, tide_math.stdnormalize(np.random.normal(size=tclen)))
        )
        fmridata[i, :] = lagtcgenerator.yfromx(timeaxis - lagtimes[i]) + noiselevel * noisevec

    """if displayplots:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Timecourses")
        for i in range(0, numlags, 200):
            plt.plot(timeaxis, fmridata[i, :])
        plt.show()"""

    # make a fake header
    nim, nim_data, nim_hdr, thedims, thesizes = tide_io.readfromnifti(
        os.path.join(get_examples_path(), "sub-RAPIDTIDETEST_brainmask.nii.gz")
    )
    xdim, ydim, slicedim, fmritr = tide_io.parseniftisizes(thesizes)
    theheader = copy.copy(nim_hdr)
    theheader["dim"][0] = 3
    theheader["dim"][1] = nativespaceshape[0]
    theheader["dim"][2] = nativespaceshape[1]
    theheader["dim"][3] = nativespaceshape[2]
    theheader["pixdim"][1] = 1.0
    theheader["pixdim"][2] = 1.0
    theheader["pixdim"][3] = 1.0
    theheader["pixdim"][4] = 1.0

    rt_floattype = "float64"
    rt_floatset = np.float64
    sLFOfitmean = np.zeros(numlags, dtype=rt_floattype)
    rvalue = np.zeros(numlags, dtype=rt_floattype)
    r2value = np.zeros(numlags, dtype=rt_floattype)
    fitNorm = np.zeros((numlags, 2), dtype=rt_floattype)
    fitcoeff = np.zeros((numlags, 2), dtype=rt_floattype)
    movingsignal = np.zeros(internalvalidfmrishape, dtype=rt_floattype)
    lagtc = np.zeros(internalvalidfmrishape, dtype=rt_floattype)
    filtereddata = np.zeros(internalvalidfmrishape, dtype=rt_floattype)
    optiondict = {
        "regressfiltthreshval": 0.0,
        "saveminimumsLFOfiltfiles": False,
        "nprocs_makelaggedtcs": 1,
        "nprocs_regressionfilt": 1,
        "mp_chunksize": 1000,
        "showprogressbar": False,
        "alwaysmultiproc": False,
        "focaldebug": debug,
        "fmrifreq": Fs,
    }

    regressderivratios, regressrvalues = tide_refinedelay.getderivratios(
        fmridata,
        validvoxels,
        timeaxis,
        0.0 * lagtimes,
        fmrimask,
        lagtcgenerator,
        "glm",
        "refinedelaytest",
        sampletime,
        sLFOfitmean,
        rvalue,
        r2value,
        fitNorm[:, :2],
        fitcoeff[:, :2],
        movingsignal,
        lagtc,
        filtereddata,
        None,
        None,
        optiondict,
        debug=debug,
    )

    medfilt, filteredregressderivratios, themad = tide_refinedelay.filterderivratios(
        regressderivratios,
        nativespaceshape,
        validvoxels,
        (xdim, ydim, slicedim),
        patchthresh=3.0,
        rt_floattype="float64",
        debug=debug,
    )

    delayoffset = filteredregressderivratios * 0.0
    for i in range(filteredregressderivratios.shape[0]):
        delayoffset[i], closestoffset = tide_refinedelay.ratiotodelay(
            filteredregressderivratios[i]
        )

    # do the tests
    msethresh = 0.1
    aethresh = 2
    print(f"{mse(lagtimes, delayoffset)=}")
    assert mse(lagtimes, delayoffset) < msethresh
    # np.testing.assert_almost_equal(lagtimes, delayoffset, aethresh)

    if displayplots:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Lagtimes")
        plt.plot(lagtimes)
        plt.plot(delayoffset)
        plt.legend(["Target", "Fit"])
        plt.show()


def test_refinedelay(displayplots=False, debug=False):
    for noiselevel in np.linspace(0.0, 0.5, num=5, endpoint=True):
        eval_refinedelay(
            sampletime=0.72,
            tclengthinsecs=300.0,
            mindelay=-3.0,
            maxdelay=3.0,
            numpoints=501,
            smoothpts=9,
            nativespaceshape=(10, 10, 10),
            displayplots=displayplots,
            outputsuffix="_1",
            noiselevel=noiselevel,
            debug=debug,
        )
    eval_refinedelay(
        sampletime=0.72,
        tclengthinsecs=300.0,
        mindelay=-3.0,
        maxdelay=3.0,
        numpoints=501,
        smoothpts=9,
        nativespaceshape=(10, 10, 10),
        displayplots=displayplots,
        outputsuffix="_2",
        debug=debug,
    )
    eval_refinedelay(
        sampletime=0.72,
        tclengthinsecs=300.0,
        mindelay=-3.0,
        maxdelay=3.0,
        numpoints=501,
        smoothpts=5,
        nativespaceshape=(10, 10, 10),
        displayplots=displayplots,
        outputsuffix="_3",
        debug=debug,
    )
    eval_refinedelay(
        sampletime=1.5,
        tclengthinsecs=300.0,
        mindelay=-3.0,
        maxdelay=3.0,
        numpoints=501,
        smoothpts=3,
        nativespaceshape=(10, 10, 10),
        displayplots=displayplots,
        outputsuffix="_1p5_501_3",
        debug=debug,
    )
    eval_refinedelay(
        sampletime=3.0,
        tclengthinsecs=300.0,
        mindelay=-3.0,
        maxdelay=3.0,
        numpoints=501,
        smoothpts=3,
        nativespaceshape=(10, 10, 10),
        displayplots=displayplots,
        outputsuffix="_3p0_501_3",
        debug=debug,
    )


if __name__ == "__main__":
    mpl.use("TkAgg")
    test_refinedelay(displayplots=True, debug=True)
