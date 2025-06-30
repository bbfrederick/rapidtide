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

import rapidtide.makelaggedtcs as tide_makelagged
import rapidtide.util as tide_util


def linfitDelay(
    fmri_data_valid,
    initial_fmri_x,
    thepass,
    LGR,
    TimingLGR,
    lagmin,
    lagmax,
    lagtcgenerator,
    targetstep=2.0,
    edgepad=1.1,
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

    # disable mkl
    tide_util.disablemkl(nprocs)

    # make the RIPTiDe evs
    numdelays = int(np.round((lagmax - lagmin) / targetstep, 0))
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
        print(f"{delaystep=}")
        print(f"{delaystouse=}, {len(delaystouse)}")
        print(f"{len(initial_fmri_x)}")

    evset = np.zeros((len(delaystouse), len(initial_fmri_x)), dtype=rt_floatset)

    dummy = tide_makelagged.makelaggedtcs(
        lagtcgenerator,
        initial_fmri_x,
        np.ones_like(delaystouse, dtype=np.float64),
        delaystouse,
        evset,
        LGR=LGR,
        nprocs=nprocs,
        alwaysmultiproc=alwaysmultiproc,
        showprogressbar=showprogressbar,
        chunksize=chunksize,
        rt_floatset=rt_floatset,
        rt_floattype=rt_floattype,
        debug=debug,
    )

    print(evset)

    # do the linear fit to the comb of delayed regressors
    voxelsprocessed_rt = 0

    # reenable mkl
    tide_util.enablemkl(mklthreads)

    TimingLGR.info(
        f"RIPTiDe calculation end, pass {thepass}",
        {
            "message2": voxelsprocessed_rt,
            "message3": "voxels",
        },
    )

    sys.exit(1)
