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
import sys

import numpy as np

import rapidtide.linfitfiltpass as tide_linfitfiltpass
import rapidtide.makelaggedtcs as tide_makelagged
import rapidtide.util as tide_util


def makeRIPTiDeRegressors(
    initial_fmri_x,
    lagmin,
    lagmax,
    lagtcgenerator,
    LGR,
    nprocs=1,
    alwaysmultiproc=False,
    showprogressbar=True,
    chunksize=1000,
    targetstep=2.5,
    edgepad=0,
    rt_floatset=np.float64,
    rt_floattype="float64",
    debug=False,
):
    # make the RIPTiDe evs
    numdelays = int(np.round((lagmax - lagmin) / targetstep, 0))
    numregressors = numdelays + 2 * edgepad
    delaystep = (lagmax - lagmin) / numdelays
    delaystouse = np.linspace(
        lagmin - edgepad * delaystep,
        lagmax + edgepad * delaystep,
        numdelays + 2 * edgepad,
        endpoint=True,
    )
    if debug:
        print(f"{lagmin=}")
        print(f"{lagmax=}")
        print(f"{numdelays=}")
        print(f"{edgepad=}")
        print(f"{numregressors=}")
        print(f"{delaystep=}")
        print(f"{delaystouse=}, {len(delaystouse)}")
        print(f"{len(initial_fmri_x)}")

    regressorset = np.zeros((len(delaystouse), len(initial_fmri_x)), dtype=rt_floatset)

    dummy = tide_makelagged.makelaggedtcs(
        lagtcgenerator,
        initial_fmri_x,
        np.ones_like(delaystouse, dtype=np.float64),
        delaystouse,
        regressorset,
        LGR=LGR,
        nprocs=nprocs,
        alwaysmultiproc=alwaysmultiproc,
        showprogressbar=showprogressbar,
        chunksize=chunksize,
        rt_floatset=rt_floatset,
        rt_floattype=rt_floattype,
        debug=debug,
    )

    if debug:
        print(regressorset)

    return regressorset, delaystouse


def linfitDelay(
    numvalidspatiallocs,
    fmri_data_valid,
    fitmask,
    lagtimes,
    lagstrengths,
    lagsigma,
    corrout,
    regressorset,
    delayvals,
    sLFOfitmean,
    rvalue,
    r2value,
    fitcoeff,
    fitNorm,
    thepass,
    LGR,
    TimingLGR,
    mklthreads=1,
    nprocs=1,
    alwaysmultiproc=False,
    showprogressbar=True,
    chunksize=1000,
    debug=False,
    rt_floatset=np.float64,
    rt_floattype="float64",
):
    LGR.info(f"\n\nRIPTiDe calculation, pass {thepass}")
    TimingLGR.info(f"RIPTiDe calculation start, pass {thepass}")

    if debug:
        print(f"{regressorset.shape=}")
        print(f"{fitcoeff.shape=}")
        print(f"{fmri_data_valid.shape=}")

    # disable mkl
    tide_util.disablemkl(nprocs)

    # do the linear fit to the comb of delayed regressors
    for thedelay in range(len(delayvals)):
        voxelsprocessed_rt = tide_linfitfiltpass.linfitfiltpass(
            numvalidspatiallocs,
            fmri_data_valid,
            0.0,
            regressorset[thedelay, :],
            sLFOfitmean,
            corrout[:, thedelay],
            r2value,
            fitcoeff[:, thedelay],
            fitNorm[:, thedelay],
            None,
            None,
            coefficientsonly=True,
            voxelspecific=False,
            nprocs=nprocs,
            alwaysmultiproc=alwaysmultiproc,
            showprogressbar=showprogressbar,
            verbose=(LGR is not None),
            chunksize=chunksize,
            rt_floatset=rt_floatset,
            rt_floattype=rt_floattype,
            debug=debug,
        )

    # reenable mkl
    tide_util.enablemkl(mklthreads)

    TimingLGR.info(
        f"RIPTiDe calculation end, pass {thepass}",
        {
            "message2": voxelsprocessed_rt,
            "message3": "voxels",
        },
    )

    for thevox in range(numvalidspatiallocs):
        fitmask[thevox] = 1
        thismax = np.argmax(np.fabs(corrout[thevox, :]))
        lagtimes[thevox] = delayvals[thismax]
        lagstrengths[thevox] = corrout[thevox, thismax]
        lagsigma[thevox] = 1.0