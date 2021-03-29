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

from rapidtide.resample import congrid
from rapidtide.tests.utils import mse


def funcvalue2(x, frequency=1.0, phase=0.0, amplitude=1.5):
    return amplitude * np.sin(2.0 * np.pi * frequency * x + phase)


def test_congrid(debug=False, display=False):
    # make the source axis
    starttime = 0.0
    endtime = 1.0
    sourcelen = 1000
    sourceaxis = np.linspace(starttime, endtime, num=sourcelen, endpoint=False)
    if debug:
        print("sourceaxis range:", sourceaxis[0], sourceaxis[-1])

    # now make the destination
    gridlen = 32
    gridaxis = np.linspace(starttime, endtime, num=gridlen, endpoint=False)
    if debug:
        print("gridaxis range:", gridaxis[0], gridaxis[-1])

    cycles = 1.0
    if debug:
        outputlines = []
    if debug:
        cyclist = [1.0, 2.0, 3.0]
        kernellist = ["gauss", "kaiser"]
        binslist = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    else:
        cyclist = [1.0]
        kernellist = ["gauss", "kaiser"]
        binslist = [1.5, 2.0, 2.5, 3.0]

    for cycles in cyclist:
        timecoursein = np.float64(sourceaxis * 0.0)
        for i in range(len(sourceaxis)):
            timecoursein[i] = funcvalue2(sourceaxis[i], frequency=cycles)

        # define the gridding
        congridbins = 1.5
        gridkernel = "gauss"

        # initialize the test points
        numsamples = 200
        testvals = np.zeros((numsamples), dtype=np.float64)
        for i in range(numsamples):
            testvals[i] = np.random.uniform() * (endtime - starttime) + starttime

        weights = np.zeros((gridlen), dtype=float)
        griddeddata = np.zeros((gridlen), dtype=float)

        for gridkernel in kernellist:
            for congridbins in binslist:
                print("about to grid")

                # reinitialize grid outputs
                weights *= 0.0
                griddeddata *= 0.0

                for i in range(numsamples):
                    thevals, theweights, theindices = congrid(
                        gridaxis,
                        testvals[i],
                        funcvalue2(testvals[i], frequency=cycles),
                        congridbins,
                        kernel=gridkernel,
                        debug=False,
                    )
                    for i in range(len(theindices)):
                        weights[theindices[i]] += theweights[i]
                        griddeddata[theindices[i]] += thevals[i]

                griddeddata = np.where(weights > 0.0, griddeddata / weights, 0.0)

                target = np.float64(gridaxis * 0.0)
                for i in range(len(gridaxis)):
                    target[i] = funcvalue2(gridaxis[i], frequency=cycles)

                print("gridding done")
                print("debug:", debug)

                # plot if we are doing that
                if display:
                    offset = 0.0
                    legend = []
                    plt.plot(sourceaxis, timecoursein)
                    legend.append("Original")
                    # offset += 1.0
                    plt.plot(gridaxis, target + offset)
                    legend.append("Target")
                    # offset += 1.0
                    plt.plot(gridaxis, griddeddata + offset)
                    legend.append("Gridded")
                    plt.plot(gridaxis, weights)
                    legend.append("Weights")
                    plt.legend(legend)
                    plt.show()

                # do the tests
                msethresh = 1.5e-2
                themse = mse(target, griddeddata)
                if debug:
                    if themse >= msethresh:
                        extra = "FAIL"
                    else:
                        extra = ""
                    print(
                        "mse for",
                        cycles,
                        "cycles:",
                        gridkernel,
                        str(congridbins),
                        ":",
                        themse,
                        extra,
                    )
                    outputlines.append(
                        " ".join(
                            [
                                "mse for",
                                str(cycles),
                                "cycles:",
                                gridkernel,
                                str(congridbins),
                                ":",
                                str(themse),
                            ]
                        )
                    )
                if not debug:
                    assert themse < msethresh
    if debug:
        for theline in outputlines:
            print(theline)


def main():
    test_congrid(debug=True, display=True)


if __name__ == "__main__":
    mpl.use("TkAgg")
    main()
