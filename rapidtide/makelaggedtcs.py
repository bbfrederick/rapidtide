#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2016-2024 Blaise Frederick
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
import bisect
import gc

import numpy as np
from tqdm import tqdm

import rapidtide.fit as tide_fit
import rapidtide.multiproc as tide_multiproc


def _procOneVoxelMakelagtc(
    vox,
    lagtcgenerator,
    maxlag,
    timeaxis,
    rt_floatset=np.float64,
    rt_floattype="float64",
):
    # question - should maxlag be added or subtracted?  As of 10/18, it is subtracted
    #  potential answer - tried adding, results are terrible.
    thelagtc = rt_floatset(lagtcgenerator.yfromx(timeaxis - maxlag))

    return (
        vox,
        thelagtc,
    )


def makelaggedtcs(
    lagtcgenerator,
    timeaxis,
    lagmask,
    lagtimes,
    lagtc,
    nprocs=1,
    alwaysmultiproc=False,
    showprogressbar=True,
    chunksize=1000,
    rt_floatset=np.float64,
    rt_floattype="float64",
):
    inputshape = lagtc.shape
    if nprocs > 1 or alwaysmultiproc:
        # define the consumer function here so it inherits most of the arguments
        def makelagtc_consumer(inQ, outQ):
            while True:
                try:
                    # get a new message
                    val = inQ.get()

                    # this is the 'TERM' signal
                    if val is None:
                        break

                    # process and send the data
                    outQ.put(
                        _procOneVoxelMakelagtc(
                            val,
                            lagtcgenerator,
                            lagtimes[val],
                            timeaxis,
                            rt_floatset=rt_floatset,
                            rt_floattype=rt_floattype,
                        )
                    )
                except Exception as e:
                    print("error!", e)
                    break

        data_out = tide_multiproc.run_multiproc(
            makelagtc_consumer,
            inputshape,
            lagmask,
            nprocs=nprocs,
            showprogressbar=showprogressbar,
            chunksize=chunksize,
        )

        # unpack the data
        volumetotal = 0
        for voxel in data_out:
            volumetotal += 1
            lagtc[voxel[0], :] = voxel[1]
        del data_out
    else:
        volumetotal = 0
        for vox in tqdm(
            range(0, inputshape[0]),
            desc="Voxel",
            unit="voxels",
            disable=(not showprogressbar),
        ):
            if lagmask[vox] > 0:
                dothisone = True
            else:
                dothisone = False
            if dothisone:
                (
                    dummy,
                    lagtc[vox, :],
                ) = _procOneVoxelMakelagtc(
                    vox,
                    lagtcgenerator,
                    lagtimes[vox],
                    timeaxis,
                    rt_floatset=rt_floatset,
                    rt_floattype=rt_floattype,
                )
                volumetotal += 1

    print("\nLagged timecourses created for " + str(volumetotal) + " voxels")

    # garbage collect
    collected = gc.collect()
    print("Garbage collector: collected %d objects." % collected)

    return volumetotal
