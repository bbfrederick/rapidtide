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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from rapidtide.correlate import shorttermcorr_1D, shorttermcorr_2D
from rapidtide.filter import NoncausalFilter
from rapidtide.io import writenpvecs
from rapidtide.tests.utils import create_dir, get_test_temp_path


def test_stcorrelate(debug=False):
    tr = 0.72
    testlen = 800
    shiftdist = 5
    windowtime = 30.0
    stepsize = 5.0
    corrweighting = "None"
    outfilename = op.join(get_test_temp_path(), "stcorrtest")

    # create outputdir if it doesn't exist
    create_dir(get_test_temp_path())

    dodetrend = True
    timeaxis = np.arange(0.0, 1.0 * testlen) * tr

    testfilter = NoncausalFilter(filtertype="lfo")
    sig1 = testfilter.apply(1.0 / tr, np.random.random(testlen))
    sig2 = np.float64(np.roll(sig1, int(shiftdist)))

    if debug:
        plt.figure()
        plt.plot(sig1)
        plt.plot(sig2)
        legend = ["Original", "Shifted"]
        plt.show()

    times, corrpertime, ppertime = shorttermcorr_1D(
        sig1, sig2, tr, windowtime, samplestep=int(stepsize // tr), detrendorder=0
    )
    plength = len(times)
    times, xcorrpertime, Rvals, delayvals, valid = shorttermcorr_2D(
        sig1,
        sig2,
        tr,
        windowtime,
        samplestep=int(stepsize // tr),
        weighting=corrweighting,
        detrendorder=0,
        display=False,
    )
    xlength = len(times)
    writenpvecs(corrpertime, outfilename + "_pearson.txt")
    writenpvecs(ppertime, outfilename + "_pvalue.txt")
    writenpvecs(Rvals, outfilename + "_Rvalue.txt")
    writenpvecs(delayvals, outfilename + "_delay.txt")
    writenpvecs(valid, outfilename + "_mask.txt")


def main():
    test_stcorrelate(debug=True)


if __name__ == "__main__":
    mpl.use("TkAgg")
    main()
