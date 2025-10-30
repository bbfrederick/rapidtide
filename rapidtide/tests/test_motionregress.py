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

import rapidtide.fit as tide_fit
import rapidtide.io as tide_io
import rapidtide.linfitfiltpass as tide_linfitfiltpass
from rapidtide.tests.utils import mse


def genmotions(tsize=200, numcycles=11):
    timeaxis = 2.0 * np.pi * np.linspace(0, 1.0, num=tsize, endpoint=False)

    motiondict = {}

    directions = ["xtrans", "ytrans", "ztrans", "xrot", "yrot", "zrot"]
    for idx, motname in enumerate(directions):
        totalcycles = 1.0 * (numcycles + idx)
        motiondict[motname] = 0.5 * np.sin(totalcycles * timeaxis)

    motiondict["maxtrans"] = np.max(
        [
            np.max(motiondict["xtrans"]),
            np.max(motiondict["ytrans"]),
            np.max(motiondict["ztrans"]),
        ]
    )
    motiondict["mintrans"] = np.min(
        [
            np.min(motiondict["xtrans"]),
            np.min(motiondict["ytrans"]),
            np.min(motiondict["ztrans"]),
        ]
    )
    motiondict["maxrot"] = np.max(
        [
            np.max(motiondict["xrot"]),
            np.max(motiondict["yrot"]),
            np.max(motiondict["zrot"]),
        ]
    )
    motiondict["minrot"] = np.min(
        [
            np.min(motiondict["xrot"]),
            np.min(motiondict["yrot"]),
            np.min(motiondict["zrot"]),
        ]
    )
    return motiondict


def showthetcs(motionregressors, motionregressorlabels):
    plt.figure()
    for whichregressor in range(motionregressors.shape[0]):
        plt.plot(motionregressors[whichregressor, :] + whichregressor)
    plt.legend(motionregressorlabels)
    plt.show()


def makedataarray(motionregressors):
    offsets = [0, 1, 2, 5, 10, 20]
    numoffsets = len(offsets)
    dataarray = np.zeros(
        (motionregressors.shape[0] * numoffsets, motionregressors.shape[1]), dtype=float
    )
    for i in range(numoffsets):
        start = 0 + i * motionregressors.shape[0]
        end = start + motionregressors.shape[0]
        dataarray[start:end, :] = np.roll(motionregressors[:, :], offsets, axis=1)
    return dataarray


def test_motionregress(debug=False, displayplots=False):
    np.random.seed(12345)
    tsize = 200
    startcycles = 11

    thismotiondict = genmotions(tsize=tsize, numcycles=startcycles)

    motionregressors, motionregressorlabels = tide_fit.calcexpandedregressors(
        thismotiondict,
        labels=["xtrans", "ytrans", "ztrans", "xrot", "yrot", "zrot"],
        deriv=False,
        order=1,
    )
    dataarray = makedataarray(motionregressors)
    if displayplots:
        plt.figure()
        plt.imshow(dataarray)
        plt.show()

    for orthogonalize in [False, True]:
        for deriv in [True, False]:
            for order in [2, 1]:
                motionregressors, motionregressorlabels = tide_fit.calcexpandedregressors(
                    thismotiondict,
                    labels=["xtrans", "ytrans", "ztrans", "xrot", "yrot", "zrot"],
                    deriv=deriv,
                    order=order,
                )
                if orthogonalize:
                    motionregressors = tide_fit.gram_schmidt(motionregressors)
                    initregressors = len(motionregressorlabels)
                    motionregressorlabels = []
                    for theregressor in range(motionregressors.shape[0]):
                        motionregressorlabels.append("orthogmotion_{:02d}".format(theregressor))

                if debug:
                    print(f"{order=}, {deriv=}, {orthogonalize=}, {motionregressors.shape=}")
                if displayplots:
                    showthetcs(motionregressors, motionregressorlabels)

                thedataarray = makedataarray(motionregressors)
                numprocitems = thedataarray.shape[0]
                filtereddata = thedataarray * 0.0
                r2value = np.zeros(numprocitems)
                dummy = tide_linfitfiltpass.linfitfiltpass(
                    numprocitems,
                    thedataarray,
                    None,
                    np.transpose(motionregressors),
                    None,
                    None,
                    r2value,
                    None,
                    None,
                    None,
                    filtereddata,
                    confoundregress=True,
                    nprocs=1,
                    showprogressbar=debug,
                    procbyvoxel=True,
                    debug=debug,
                )
                if displayplots:
                    plt.figure()
                    plt.imshow(filtereddata)
                    plt.show()
                if debug:
                    print(f"\tMSE: {mse(filtereddata, 0.0 * thedataarray)}\n")
                if deriv:
                    assert mse(filtereddata, 0.0 * thedataarray) < 1e-2


if __name__ == "__main__":
    mpl.use("TkAgg")
    test_motionregress(debug=True, displayplots=True)
