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
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from rapidtide.filter import dolpfiltfilt
from rapidtide.resample import timeshift
from rapidtide.tests.utils import mse


def test_timeshift(debug=False):
    tr = 1.0
    padtime = 300.0
    testlen = 1000
    shiftdist = 30
    timeaxis = np.arange(0.0, 1.0 * testlen) * tr
    # timecoursein = np.zeros((testlen), dtype='float64')
    timecoursein = np.float64(timeaxis * 0.0)
    midpoint = int(testlen // 2) + 1
    timecoursein[midpoint - 1] = np.float64(1.0)
    timecoursein[midpoint] = np.float64(1.0)
    timecoursein[midpoint + 1] = np.float64(1.0)
    timecoursein -= 0.5
    butterorder = 4
    timecoursein = 0.5 * dolpfiltfilt(1.0, 0.25, timecoursein, butterorder) + 0.5

    shiftlist = [-30, -20, -10, 0, 10, 20, 30]
    shiftlist = [-100]

    if debug:
        plt.figure()
        plt.ylim([-1.0, 2.0 * len(shiftlist) + 1.0])
        plt.plot(timecoursein)
        legend = ["Original"]
        offset = 0.0

    for shiftdist in shiftlist:
        # generate the ground truth rolled regressor
        tcrolled = np.float64(np.roll(timecoursein, int(shiftdist)))

        # generate the fast resampled regressor
        # tcshifted = genlaggedtc.yfromx(timeaxis - shiftdist, debug=debug)
        tcshifted, weights, alltc, allweights = timeshift(
            timecoursein, 1.0 * shiftdist, int(padtime // tr), doplot=False
        )

        # print out all elements
        print("len tcrolled=", len(tcrolled), "len tcshifted=", len(tcshifted))
        for i in range(0, len(tcrolled)):
            print(i, tcrolled[i], tcshifted[i], tcshifted[i] - tcrolled[i])

        # plot if we are doing that
        if debug:
            offset += 1.0
            plt.plot(tcrolled + offset)
            legend.append("Roll " + str(shiftdist))
            offset += 1.0
            plt.plot(tcshifted + offset)
            legend.append("Timeshift " + str(shiftdist))
            plt.plot(weights + offset)
            legend.append("Weights " + str(shiftdist))

        # do the tests
        msethresh = 1e-6
        aethresh = 2
        assert mse(tcrolled, tcshifted) < msethresh
        np.testing.assert_almost_equal(tcrolled, tcshifted, aethresh)

    if debug:
        plt.legend(legend)
        plt.show()


if __name__ == "__main__":
    mpl.use("TkAgg")
    test_timeshift(debug=True)
