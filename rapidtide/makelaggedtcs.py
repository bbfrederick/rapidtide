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

import numpy as np

import rapidtide.genericmultiproc as tide_genericmultiproc


def _procOneVoxelMakelagtc(
    vox,
    voxelargs,
    **kwargs,
):
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


def _packvoxeldata(voxnum, voxelargs):
    return [voxelargs[0], (voxelargs[1])[voxnum], voxelargs[2]]


def _unpackvoxeldata(retvals, voxelproducts):
    (voxelproducts[0])[retvals[0], :] = retvals[1]


def makelaggedtcs(
    lagtcgenerator,
    timeaxis,
    lagmask,
    lagtimes,
    lagtc,
    LGR=None,
    nprocs=1,
    alwaysmultiproc=False,
    showprogressbar=True,
    chunksize=1000,
    rt_floatset=np.float64,
    rt_floattype="float64",
):
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

    return volumetotal
