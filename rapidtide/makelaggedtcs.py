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
import gc
import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

import rapidtide.genericmultiproc as tide_genericmultiproc


def _procOneVoxelMakelagtc(
    vox: int,
    voxelargs: list,
    **kwargs: Any,
) -> tuple[int, NDArray]:
    # unpack arguments
    options = {
        "rt_floatset": np.float64,
        "debug": False,
    }
    options.update(kwargs)
    rt_floatset = options["rt_floatset"]
    debug = options["debug"]
    (lagtcgenerator, thelag, timeaxis) = voxelargs
    if debug:
        print(f"{vox=}, {thelag=}, {timeaxis=}")

    # question - should maxlag be added or subtracted?  As of 10/18, it is subtracted
    #  potential answer - tried adding, results are terrible.
    thelagtc = rt_floatset(lagtcgenerator.yfromx(timeaxis - thelag))

    return (
        vox,
        (thelagtc),
    )


def _packvoxeldata(voxnum: int, voxelargs: list) -> list:
    return [voxelargs[0], (voxelargs[1])[voxnum], voxelargs[2]]


def _unpackvoxeldata(retvals: tuple, voxelproducts: list) -> None:
    (voxelproducts[0])[retvals[0], :] = retvals[1]


def makelaggedtcs(
    lagtcgenerator: Any,
    timeaxis: NDArray,
    lagmask: NDArray,
    lagtimes: NDArray,
    lagtc: NDArray,
    LGR: logging.Logger | None = None,
    nprocs: int = 1,
    alwaysmultiproc: bool = False,
    showprogressbar: bool = True,
    chunksize: int = 1000,
    rt_floatset: type = np.float64,
    rt_floattype: str = "float64",
    debug: bool = False,
) -> int:
    if debug:
        print("makelaggedtcs: Starting")
        print(f"\t{lagtc.shape=}")
        print(f"\t{lagtimes.shape=}")
        print(f"\t{timeaxis.shape=}")

    inputshape = lagtc.shape
    voxelargs = [
        lagtcgenerator,
        lagtimes,
        timeaxis,
    ]
    voxelfunc = _procOneVoxelMakelagtc
    packfunc = _packvoxeldata
    unpackfunc = _unpackvoxeldata
    voxeltargets = [lagtc]

    volumetotal = tide_genericmultiproc.run_multiproc(
        voxelfunc,
        packfunc,
        unpackfunc,
        voxelargs,
        voxeltargets,
        inputshape,
        lagmask,
        LGR,
        nprocs,
        alwaysmultiproc,
        showprogressbar,
        chunksize,
        rt_floatset=rt_floatset,
    )
    if LGR is not None:
        LGR.info(f"\nLagged timecourses created for {volumetotal} voxels")

    # garbage collect
    uncollected = gc.collect()
    if uncollected != 0:
        if LGR is not None:
            LGR.info(f"garbage collected - unable to collect {uncollected} objects")
    else:
        if LGR is not None:
            LGR.info("garbage collected")

    if debug:
        print("makelaggedtcs: End\n\n")

    return volumetotal
