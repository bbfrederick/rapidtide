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
from tqdm import tqdm

import rapidtide.fit as tide_fit
import rapidtide.multiproc as tide_multiproc
import rapidtide.resample as tide_resample

warnings.simplefilter(action="ignore", category=FutureWarning)


def _procOneVoxelPeaks(
    vox,
    thetc,
    theMutualInformationator,
    fmri_x,
    fmritc,
    os_fmri_x,
    xcorr_x,
    thexcorr,
    bipolar=False,
    oversampfactor=1,
    sort=True,
    interptype="univariate",
):
    if oversampfactor >= 1:
        thetc[:] = tide_resample.doresample(fmri_x, fmritc, os_fmri_x, method=interptype)
    else:
        thetc[:] = fmritc
    thepeaks = tide_fit.getpeaks(xcorr_x, thexcorr, bipolar=bipolar, displayplots=False)
    peaklocs = []
    for thepeak in thepeaks:
        peaklocs.append(int(round(thepeak[2], 0)))
    theMI_list = theMutualInformationator.run(thetc, locs=peaklocs)
    hybridpeaks = []
    for i in range(len(thepeaks)):
        hybridpeaks.append([thepeaks[i][0], thepeaks[i][1], theMI_list[i]])
    if sort:
        hybridpeaks.sort(key=lambda x: x[2], reverse=True)
    return vox, hybridpeaks


def peakevalpass(
    fmridata,
    referencetc,
    fmri_x,
    os_fmri_x,
    theMutualInformationator,
    xcorr_x,
    corrdata,
    nprocs=1,
    alwaysmultiproc=False,
    bipolar=False,
    oversampfactor=1,
    interptype="univariate",
    showprogressbar=True,
    chunksize=1000,
    rt_floatset=np.float64,
    rt_floattype="float64",
):
    """

    Parameters
    ----------
    fmridata
    referencetc
    theCorrelator
    fmri_x
    os_fmri_x
    tr
    corrout
    meanval
    nprocs
    alwaysmultiproc
    oversampfactor
    interptype
    showprogressbar
    chunksize
    rt_floatset
    rt_floattype

    Returns
    -------

    """
    peakdict = {}
    theMutualInformationator.setreftc(referencetc)

    inputshape = np.shape(fmridata)
    volumetotal = 0
    thetc = np.zeros(np.shape(os_fmri_x), dtype=rt_floattype)
    if nprocs > 1 or alwaysmultiproc:
        # define the consumer function here so it inherits most of the arguments
        def correlation_consumer(inQ, outQ):
            while True:
                try:
                    # get a new message
                    val = inQ.get()

                    # this is the 'TERM' signal
                    if val is None:
                        break

                    # process and send the data
                    outQ.put(
                        _procOneVoxelPeaks(
                            val,
                            thetc,
                            theMutualInformationator,
                            fmri_x,
                            fmridata[val, :],
                            os_fmri_x,
                            xcorr_x,
                            corrdata[val, :],
                            bipolar=bipolar,
                            oversampfactor=oversampfactor,
                            interptype=interptype,
                        )
                    )

                except Exception as e:
                    print("error!", e)
                    break

        data_out = tide_multiproc.run_multiproc(
            correlation_consumer,
            inputshape,
            None,
            nprocs=nprocs,
            showprogressbar=showprogressbar,
            chunksize=chunksize,
        )

        # unpack the data
        volumetotal = 0
        for voxel in data_out:
            peakdict[str(voxel[0])] = voxel[1]
            volumetotal += 1
        del data_out
    else:
        for vox in tqdm(
            range(0, inputshape[0]),
            desc="Voxel",
            unit="voxels",
            disable=(not showprogressbar),
        ):
            dummy, peakdict[str(vox)] = _procOneVoxelPeaks(
                vox,
                thetc,
                theMutualInformationator,
                fmri_x,
                fmridata[vox, :],
                os_fmri_x,
                xcorr_x,
                corrdata[vox, :],
                bipolar=bipolar,
                oversampfactor=oversampfactor,
                interptype=interptype,
            )
            volumetotal += 1
    print("\nPeak evaluation performed on " + str(volumetotal) + " voxels")

    # garbage collect
    collected = gc.collect()
    print("Garbage collector: collected %d objects." % collected)

    return volumetotal, peakdict
