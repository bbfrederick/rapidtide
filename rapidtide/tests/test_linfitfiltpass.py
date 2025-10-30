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
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import rapidtide.linfitfiltpass as tide_linfitfiltpass
from rapidtide.tests.utils import mse


def gen2d(xsize=150, xcycles=11, tsize=200, tcycles=13, mean=10.0):
    thearray = np.zeros((xsize, tsize), dtype=np.float64)
    xwaves = np.zeros((xsize, tsize), dtype=np.float64)
    twaves = np.zeros((xsize, tsize), dtype=np.float64)
    xmax = 2.0 * np.pi * xcycles
    tmax = 2.0 * np.pi * tcycles
    xfreq = xmax / xsize
    tfreq = tmax / tsize
    for i in range(tsize):
        thearray[:, i] = np.sin(np.linspace(0.0, xmax, xsize, endpoint=False))
        xwaves[:, i] = np.sin(np.linspace(0.0, xmax, xsize, endpoint=False))
    for i in range(xsize):
        thearray[i, :] *= np.sin(np.linspace(0.0, tmax, tsize, endpoint=False))
        twaves[i, :] = np.sin(np.linspace(0.0, tmax, tsize, endpoint=False))
    return thearray, xwaves, twaves


def test_linfitfiltpass(debug=True, displayplots=False):
    np.random.seed(12345)
    xsize = 150
    xcycles = 7
    tsize = 200
    tcycles = 23
    mean = 100.0
    noiselevel = 5.0

    targetarray, xwaveforms, twaveforms = gen2d(
        xsize=xsize, xcycles=xcycles, tsize=tsize, tcycles=tcycles
    )
    if debug:
        print(f"{twaveforms.shape=}")
        print(f"{xwaveforms.shape=}")
    testarray = targetarray + np.random.random((xsize, tsize)) + mean
    if displayplots:
        plt.figure()
        plt.imshow(targetarray)
        plt.show()

    filtereddata = 0.0 * testarray
    datatoremove = 0.0 * testarray
    threshval = 0.01
    meanvals_t = np.zeros(tsize, dtype=np.float64)
    rvals_t = np.zeros(tsize, dtype=np.float64)
    r2vals_t = np.zeros(tsize, dtype=np.float64)
    fitcoffs_t = np.zeros((xsize, tsize), dtype=np.float64)
    fitNorm_t = np.zeros((xsize, tsize), dtype=np.float64)

    meanvals_x = np.zeros(xsize, dtype=np.float64)
    rvals_x = np.zeros(xsize, dtype=np.float64)
    r2vals_x = np.zeros(xsize, dtype=np.float64)
    fitcoffs_x = np.zeros((xsize, tsize), dtype=np.float64)
    fitNorm_x = np.zeros((xsize, tsize), dtype=np.float64)

    for confoundregress in [True, False]:
        if confoundregress:
            twaveformrange = np.transpose(twaveforms[:6, :])
            xwaveformrange = xwaveforms[:, :6]
            print(f"{twaveformrange.shape=} - {xwaveformrange.shape=}")
        else:
            twaveformrange = twaveforms
            xwaveformrange = xwaveforms
        for procbyvoxel in [True, False]:
            if procbyvoxel:
                waveforms = twaveformrange
                meanvals = meanvals_x
                rvals = rvals_x
                r2vals = r2vals_x
                fitcoffs = fitcoffs_x
                fitNorm = fitNorm_x
                direction = "space"
            else:
                waveforms = xwaveformrange
                meanvals = meanvals_t
                rvals = rvals_t
                r2vals = r2vals_t
                fitcoffs = fitcoffs_t
                fitNorm = fitNorm_t
                direction = "time"
            for nprocs in [1, 2]:
                if nprocs == 1:
                    procstring = "single"
                else:
                    procstring = "multi"
                for thisthreshval in [threshval, None]:
                    if thisthreshval is None:
                        maskstatus = "no mask"
                    else:
                        maskstatus = f"threshold={threshval}"

                    if debug:
                        print(
                            f"confoundregress={confoundregress}, proc by {direction}, {procstring} proc, {maskstatus}"
                        )
                    tide_linfitfiltpass.linfitfiltpass(
                        xsize,
                        testarray,
                        thisthreshval,
                        waveforms,
                        meanvals,
                        rvals,
                        r2vals,
                        fitcoffs,
                        fitNorm,
                        datatoremove,
                        filtereddata,
                        showprogressbar=False,
                        procbyvoxel=procbyvoxel,
                        nprocs=nprocs,
                        confoundregress=confoundregress,
                    )
                    if displayplots:
                        plt.figure()
                        plt.imshow(datatoremove)
                        plt.show()
                        plt.imshow(filtereddata)
                        plt.show()
                    if debug:
                        print(f"\tMSE: {mse(datatoremove, targetarray)}\n")
                    if not confoundregress:
                        assert mse(datatoremove, targetarray) < 1e-3


if __name__ == "__main__":
    mpl.use("TkAgg")
    test_linfitfiltpass(debug=True, displayplots=True)
