#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2016-2024 Blaise Frederick
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
import rapidtide.refinedelay as tide_refinedelay
import rapidtide.resample as tide_resample
from rapidtide.filter import NoncausalFilter
from rapidtide.tests.utils import get_examples_path


def eval_refinedelay(
    sampletime=0.72,
    order=1,
    tclengthinsecs=300.0,
    mindelay=-5.0,
    maxdelay=5.0,
    nativespaceshape=(10, 10, 10),
    displayplots=False,
    padtime=30.0,
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
    rawgms = np.random.normal(size=tclen)
    testfilter = NoncausalFilter(filtertype="lfo")
    sLFO = testfilter.apply(Fs, rawgms)
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
        fmridata[i, :] = lagtcgenerator.yfromx(timeaxis - lagtimes[i])

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
    glmmean = np.zeros(numlags, dtype=rt_floattype)
    rvalue = np.zeros(numlags, dtype=rt_floattype)
    r2value = np.zeros(numlags, dtype=rt_floattype)
    fitNorm = np.zeros((numlags, order + 1), dtype=rt_floattype)
    fitcoeff = np.zeros((numlags, order + 1), dtype=rt_floattype)
    movingsignal = np.zeros(internalvalidfmrishape, dtype=rt_floattype)
    lagtc = np.zeros(internalvalidfmrishape, dtype=rt_floattype)
    filtereddata = np.zeros(internalvalidfmrishape, dtype=rt_floattype)
    optiondict = {
        "glmthreshval": 0.0,
        "saveminimumglmfiles": False,
        "nprocs_makelaggedtcs": 1,
        "nprocs_glm": 1,
        "mp_chunksize": 1000,
        "showprogressbar": False,
        "alwaysmultiproc": False,
        "memprofile": False,
        "focaldebug": debug,
        "fmrifreq": Fs,
        "textio": False,
    }
    delayoffset = tide_refinedelay.refinedelay(
        fmridata,
        nativespaceshape,
        validvoxels,
        timeaxis,
        0.0 * lagtimes,
        fmrimask,
        lagtcgenerator,
        "glm",
        "refinedelaytest",
        sampletime,
        glmmean,
        rvalue,
        r2value,
        fitNorm[:, : (order + 1)],
        fitcoeff[:, : (order + 1)],
        movingsignal,
        lagtc,
        filtereddata,
        theheader,
        None,
        None,
        optiondict,
        {
            "Units": "arbitrary",
        },
        None,
        order=order,
        fileiscifti=False,
        textio=False,
        rt_floattype=rt_floattype,
        rt_floatset=rt_floattype,
        debug=debug,
    )

    if displayplots:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Ratio")
        plt.plot(np.nan_to_num(fitcoeff[:, 1] / fitcoeff[:, 0]))
        plt.show()


def test_refinedelay(displayplots=False, debug=False):
    eval_refinedelay(
        sampletime=0.72,
        order=1,
        tclengthinsecs=300.0,
        mindelay=-3.0,
        maxdelay=3.0,
        nativespaceshape=(10, 10, 10),
        displayplots=displayplots,
        debug=debug,
    )
    eval_refinedelay(
        sampletime=0.72,
        order=2,
        tclengthinsecs=300.0,
        mindelay=-3.0,
        maxdelay=3.0,
        nativespaceshape=(10, 10, 10),
        displayplots=displayplots,
        debug=debug,
    )
    eval_refinedelay(
        sampletime=1.5,
        tclengthinsecs=300.0,
        mindelay=-3.0,
        maxdelay=3.0,
        nativespaceshape=(10, 10, 10),
        displayplots=displayplots,
        debug=debug,
    )
    eval_refinedelay(
        sampletime=3.0,
        tclengthinsecs=300.0,
        mindelay=-3.0,
        maxdelay=3.0,
        nativespaceshape=(10, 10, 10),
        displayplots=displayplots,
        debug=debug,
    )


if __name__ == "__main__":
    mpl.use("TkAgg")
    test_refinedelay(displayplots=True, debug=True)
