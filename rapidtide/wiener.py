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
import numpy as np
from tqdm import tqdm

import rapidtide.fit as tide_fit
import rapidtide.multiproc as tide_multiproc


def _procOneVoxelWiener(vox, lagtc, inittc, rt_floatset=np.float64, rt_floattype="float64"):
    thefit, R = tide_fit.mlregress(lagtc, inittc)
    fitcoff = rt_floatset(thefit[0, 1])
    datatoremove = rt_floatset(fitcoff * lagtc)
    return (
        vox,
        rt_floatset(thefit[0, 0]),
        rt_floatset(R),
        rt_floatset(R * R),
        fitcoff,
        rt_floatset(thefit[0, 1] / thefit[0, 0]),
        datatoremove,
        rt_floatset(inittc - datatoremove),
    )


def wienerpass(
    numspatiallocs,
    fmri_data,
    threshval,
    lagtc,
    optiondict,
    wienerdeconv,
    wpeak,
    resampref_y,
    rt_floatset=np.float64,
    rt_floattype="float64",
):
    rt_floatset = (rt_floatset,)
    rt_floattype = rt_floattype
    inputshape = np.shape(fmri_data)
    themask = np.where(np.mean(fmri_data, axis=1) > threshval, 1, 0)
    if optiondict["nprocs"] > 1:
        # define the consumer function here so it inherits most of the arguments
        def Wiener_consumer(inQ, outQ):
            while True:
                try:
                    # get a new message
                    val = inQ.get()

                    # this is the 'TERM' signal
                    if val is None:
                        break

                    # process and send the data
                    outQ.put(
                        _procOneVoxelWiener(
                            val,
                            lagtc[val, :],
                            fmri_data[val, :],
                            rt_floatset=rt_floatset,
                            rt_floattype=rt_floattype,
                        )
                    )

                except Exception as e:
                    print("error!", e)
                    break

        data_out = tide_multiproc.run_multiproc(
            Wiener_consumer,
            inputshape,
            themask,
            nprocs=optiondict["nprocs"],
            showprogressbar=True,
            chunksize=optiondict["mp_chunksize"],
        )
        # unpack the data
        volumetotal = 0
        for voxel in data_out:
            meanvalue[voxel[0]] = voxel[1]
            rvalue[voxel[0]] = voxel[2]
            r2value[voxel[0]] = voxel[3]
            fitcoff[voxel[0]] = voxel[4]
            fitNorm[voxel[0]] = voxel[5]
            datatoremove[voxel[0], :] = voxel[6]
            filtereddata[voxel[0], :] = voxel[7]
            volumetotal += 1
        data_out = []
    else:
        volumetotal = 0
        for vox in tqdm(
            range(0, numspatiallocs),
            desc="Voxel",
            unit="voxels",
            disable=(not optiondict["showprogressbar"]),
        ):
            inittc = fmri_data[vox, :].copy()
            if np.mean(inittc) >= threshval:
                (
                    dummy,
                    meanvalue[vox],
                    rvalue[vox],
                    r2value[vox],
                    fitcoff[vox],
                    fitNorm[vox],
                    datatoremove[vox],
                    filtereddata[vox],
                ) = _procOneVoxelWiener(
                    vox,
                    lagtc[vox, :],
                    inittc,
                    rt_floatset=rt_floatset,
                    t_floattype=rt_floattype,
                )
            volumetotal += 1

    return volumetotal
