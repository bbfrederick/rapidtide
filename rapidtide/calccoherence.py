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
import warnings

import numpy as np

import rapidtide.genericmultiproc as tide_genericmultiproc

warnings.simplefilter(action="ignore", category=FutureWarning)
LGR = logging.getLogger("GENERAL")


def _procOneVoxelCoherence(
    vox,
    voxelargs,
    **kwargs,
):
    options = {
        "alt": False,
        "debug": False,
    }
    options.update(kwargs)
    alt = options["alt"]
    debug = options["debug"]
    (theCoherer, fmritc) = voxelargs
    if debug:
        print(f"{alt=}")
    if alt:
        (
            thecoherence_y,
            thecoherence_x,
            globalmaxindex,
            dummy,
            dummy,
            dummy,
        ) = theCoherer.run(fmritc, trim=True, alt=True)
    else:
        thecoherence_y, thecoherence_x, globalmaxindex = theCoherer.run(fmritc, trim=True)
    maxindex = np.argmax(thecoherence_y)
    return (
        vox,
        thecoherence_x,
        thecoherence_y,
        thecoherence_y[maxindex],
        thecoherence_x[maxindex],
    )


def _packvoxeldata(voxnum, voxelargs):
    return [
        voxelargs[0],
        (voxelargs[1])[voxnum, :],
    ]


def _unpackvoxeldata(retvals, voxelproducts):
    (voxelproducts[0])[retvals[0]] = retvals[2]
    (voxelproducts[1])[retvals[0]] = retvals[3]
    (voxelproducts[2])[retvals[0]] = retvals[4]


def coherencepass(
    fmridata,
    theCoherer,
    coherencefunc,
    coherencepeakval,
    coherencepeakfreq,
    alt=False,
    chunksize=1000,
    nprocs=1,
    alwaysmultiproc=False,
    showprogressbar=True,
    debug=False,
):
    """

    Parameters
    ----------
    fmridata
    theCoherer
    coherencefunc
    coherencepeakval
    coherencepeakfreq
    chunksize
    nprocs
    alwaysmultiproc
    showprogressbar
    rt_floatset
    rt_floattype

    Returns
    -------

    """
    inputshape = np.shape(fmridata)
    voxelargs = [theCoherer, fmridata]
    voxelfunc = _procOneVoxelCoherence
    packfunc = _packvoxeldata
    unpackfunc = _unpackvoxeldata
    voxeltargets = [coherencefunc, coherencepeakval, coherencepeakfreq]
    voxelmask = fmridata[:, 0] * 0.0 + 1

    volumetotal = tide_genericmultiproc.run_multiproc(
        voxelfunc,
        packfunc,
        unpackfunc,
        voxelargs,
        voxeltargets,
        inputshape,
        voxelmask,
        LGR,
        nprocs,
        alwaysmultiproc,
        showprogressbar,
        chunksize,
        alt=alt,
        debug=debug,
    )
    LGR.info(f"\nCoherence performed on {volumetotal} voxels")

    # garbage collect
    uncollected = gc.collect()
    if uncollected != 0:
        LGR.info(f"garbage collected - unable to collect {uncollected} objects")
    else:
        LGR.info("garbage collected")

    return volumetotal
