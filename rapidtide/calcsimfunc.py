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
import rapidtide.resample as tide_resample

warnings.simplefilter(action="ignore", category=FutureWarning)
LGR = logging.getLogger("GENERAL")


def _procOneVoxelCorrelation(
    vox,
    voxelargs,
    **kwargs,
):
    options = {
        "oversampfactor": 1,
        "interptype": "univariate",
        "debug": False,
    }
    options.update(kwargs)
    oversampfactor = options["oversampfactor"]
    interptype = options["interptype"]
    debug = options["debug"]
    if debug:
        print(f"{oversampfactor=} {interptype=}")
    (thetc, theCorrelator, fmri_x, fmritc, os_fmri_x, theglobalmaxlist, thexcorr_y) = voxelargs
    if oversampfactor >= 1:
        thetc[:] = tide_resample.doresample(fmri_x, fmritc, os_fmri_x, method=interptype)
    else:
        thetc[:] = fmritc
    thexcorr_y, thexcorr_x, theglobalmax = theCorrelator.run(thetc)
    # print(f"_procOneVoxelCorrelation: {thexcorr_x=}")

    return vox, np.mean(thetc), thexcorr_y, thexcorr_x, theglobalmax, theglobalmaxlist


def _packvoxeldata(voxnum, voxelargs):
    return [
        voxelargs[0],
        voxelargs[1],
        voxelargs[2],
        (voxelargs[3])[voxnum, :],
        voxelargs[4],
        voxelargs[5],
        voxelargs[6],
    ]


def _unpackvoxeldata(retvals, voxelproducts):
    (voxelproducts[0])[retvals[0]] = retvals[1]
    (voxelproducts[1])[retvals[0], :] = retvals[2]
    voxelproducts[2] = retvals[3]
    (voxelproducts[3]).append(retvals[4] + 0)


def correlationpass(
    fmridata,
    referencetc,
    theCorrelator,
    fmri_x,
    os_fmri_x,
    lagmininpts,
    lagmaxinpts,
    corrout,
    meanval,
    nprocs=1,
    alwaysmultiproc=False,
    oversampfactor=1,
    interptype="univariate",
    showprogressbar=True,
    chunksize=1000,
    rt_floatset=np.float64,
    rt_floattype="float64",
    debug=False,
):
    """

    Parameters
    ----------
    fmridata
    referencetc - the reference regressor, already oversampled
    theCorrelator
    fmri_x
    os_fmri_x
    tr
    lagmininpts
    lagmaxinpts
    corrout
    meanval
    nprocs
    oversampfactor
    interptype
    showprogressbar
    chunksize
    rt_floatset
    rt_floattype

    Returns
    -------

    """
    if debug:
        print(f"calling setreftc in calcsimfunc with length {len(referencetc)}")
    theCorrelator.setreftc(referencetc)
    theCorrelator.setlimits(lagmininpts, lagmaxinpts)
    thetc = np.zeros(np.shape(os_fmri_x), dtype=rt_floattype)
    theglobalmaxlist = []

    # generate a corrscale of the correct length
    dummy = np.zeros(100, dtype=rt_floattype)
    dummy, dummy, dummy, thecorrscale, dummy, dummy = _procOneVoxelCorrelation(
        0,
        _packvoxeldata(
            0, [thetc, theCorrelator, fmri_x, fmridata, os_fmri_x, theglobalmaxlist, dummy]
        ),
        oversampfactor=oversampfactor,
        interptype=interptype,
    )

    inputshape = np.shape(fmridata)
    voxelargs = [thetc, theCorrelator, fmri_x, fmridata, os_fmri_x, theglobalmaxlist, thecorrscale]
    voxelfunc = _procOneVoxelCorrelation
    packfunc = _packvoxeldata
    unpackfunc = _unpackvoxeldata
    voxeltargets = [meanval, corrout, thecorrscale, theglobalmaxlist]
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
        oversampfactor=oversampfactor,
        interptype=interptype,
        debug=debug,
    )
    LGR.info(f"\nSimilarity function calculated on {volumetotal} voxels")

    # garbage collect
    uncollected = gc.collect()
    if uncollected != 0:
        LGR.info(f"garbage collected - unable to collect {uncollected} objects")
    else:
        LGR.info("garbage collected")

    return volumetotal, theglobalmaxlist, thecorrscale
