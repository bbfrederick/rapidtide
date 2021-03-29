#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
#   Copyright 2017-2021 Blaise Frederick
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
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import rapidtide.miscmath as tide_math
import rapidtide.resample as tide_res
from rapidtide.tests.utils import mse


def test_simulate(display=False):
    fmritr = 1.5
    numtrs = 260
    fmriskip = 0

    oversampfac = 10
    inputfreq = oversampfac / fmritr
    inputstarttime = 0.0
    timecourse = np.zeros((oversampfac * numtrs), dtype="float")
    timecourse[500:600] = 1.0
    timecourse[700:750] = 1.0

    # read in the timecourse to resample
    inputvec = tide_math.stdnormalize(timecourse)
    simregressorpts = len(inputvec)

    # prepare the input data for interpolation
    print("Input regressor has ", simregressorpts, " points")
    inputstep = 1.0 / inputfreq
    nirs_x = np.r_[0.0 : 1.0 * simregressorpts] * inputstep - inputstarttime
    nirs_y = inputvec[0:simregressorpts]
    print("nirs regressor runs from ", nirs_x[0], " to ", nirs_x[-1])

    # prepare the output timepoints
    fmrifreq = 1.0 / fmritr
    initial_fmri_x = np.r_[0 : fmritr * (numtrs - fmriskip) : fmritr] + fmritr * fmriskip
    print("length of fmri after removing skip:", len(initial_fmri_x))
    print("fmri time runs from ", initial_fmri_x[0], " to ", initial_fmri_x[-1])

    # set the sim parameters
    immean = 1.0
    boldpc = 1.0
    lag = 10.0 * fmritr
    noiselevel = 0.0

    simdata = np.zeros((len(initial_fmri_x)), dtype="float")

    fmrilcut = 0.0
    fmriucut = fmrifreq / 2.0

    # set up fast resampling
    padtime = 60.0
    numpadtrs = int(padtime / fmritr)
    padtime = fmritr * numpadtrs

    genlagtc = tide_res.FastResampler(nirs_x, nirs_y, padtime=padtime, doplot=False)
    initial_fmri_y = genlagtc.yfromx(initial_fmri_x)

    if display:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("Regressors")
        plt.plot(nirs_x, nirs_y, initial_fmri_x, initial_fmri_y)
        plt.show()

    # loop over space
    sliceoffsettime = 0.0
    fmri_x = initial_fmri_x - lag - sliceoffsettime
    print(fmri_x[0], initial_fmri_x[0], lag, sliceoffsettime)
    fmri_y = genlagtc.yfromx(fmri_x)
    thenoise = noiselevel * np.random.standard_normal(len(fmri_y))
    simdata[:] = immean * (1.0 + (boldpc / 100.0) * fmri_y) + thenoise
    if display:
        plt.plot(initial_fmri_x, simdata, initial_fmri_x, initial_fmri_y)
        plt.show()

    # tests
    msethresh = 1e-6
    aethresh = 2
    assert mse(simdata, initial_fmri_y) < aethresh
    # np.testing.assert_almost_equal(simdata, initial_fmri_y)


def main():
    test_simulate(display=True)


if __name__ == "__main__":
    mpl.use("TkAgg")
    main()
