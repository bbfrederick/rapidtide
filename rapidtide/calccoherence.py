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
# $Author: frederic $
# $Date: 2016/07/11 14:50:43 $
# $Id: rapidtide,v 1.161 2016/07/11 14:50:43 frederic Exp $
#
#
#
import gc
import warnings

import numpy as np

import rapidtide.multiproc as tide_multiproc
import rapidtide.util as tide_util

warnings.simplefilter(action="ignore", category=FutureWarning)


def _procOneVoxelCoherence(
    vox, theCoherer, fmritc, alt=False, rt_floatset=np.float64, rt_floattype="float64"
):
    if alt:
        (thecoherence_y, thecoherence_x, globalmaxindex, dummy, dummy, dummy,) = theCoherer.run(
            fmritc, trim=True, alt=True
        )
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


def coherencepass(
    fmridata,
    theCoherer,
    coherencefunc,
    coherencepeakval,
    coherencepeakfreq,
    reportstep,
    alt=False,
    chunksize=1000,
    nprocs=1,
    alwaysmultiproc=False,
    showprogressbar=True,
    rt_floatset=np.float64,
    rt_floattype="float64",
):
    """

    Parameters
    ----------
    fmridata
    theCoherer
    coherencefunc
    coherencepeakval
    coherencepeakfreq
    reportstep
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
    volumetotal = 0
    if nprocs > 1 or alwaysmultiproc:
        # define the consumer function here so it inherits most of the arguments
        def coherence_consumer(inQ, outQ):
            while True:
                try:
                    # get a new message
                    val = inQ.get()

                    # this is the 'TERM' signal
                    if val is None:
                        break

                    # process and send the data
                    outQ.put(
                        _procOneVoxelCoherence(
                            val,
                            theCoherer,
                            fmridata[val, :],
                            alt=alt,
                            rt_floatset=rt_floatset,
                            rt_floattype=rt_floattype,
                        )
                    )

                except Exception as e:
                    print("error!", e)
                    break

        data_out = tide_multiproc.run_multiproc(
            coherence_consumer,
            inputshape,
            None,
            nprocs=nprocs,
            showprogressbar=showprogressbar,
            chunksize=chunksize,
        )

        # unpack the data
        volumetotal = 0
        for voxel in data_out:
            coherencefunc[voxel[0], :] = voxel[2] + 0.0
            coherencepeakval[voxel[0]] = voxel[3] + 0.0
            coherencepeakfreq[voxel[0]] = voxel[4] + 0.0
            volumetotal += 1
        del data_out
    else:
        for vox in range(0, inputshape[0]):
            if (vox % reportstep == 0 or vox == inputshape[0] - 1) and showprogressbar:
                tide_util.progressbar(vox + 1, inputshape[0], label="Percent complete")
            (
                dummy,
                dummy,
                coherencefunc[vox],
                coherencepeakval[vox],
                coherencepeakfreq[vox],
            ) = _procOneVoxelCoherence(
                vox,
                theCoherer,
                fmridata[vox, :],
                alt=alt,
                rt_floatset=rt_floatset,
                rt_floattype=rt_floattype,
            )
            volumetotal += 1
    print("\nCoherence performed on " + str(volumetotal) + " voxels")

    # garbage collect
    collected = gc.collect()
    print("Garbage collector: collected %d objects." % collected)

    return volumetotal
