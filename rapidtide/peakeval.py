#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2016-2026 Blaise Frederick
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
from typing import Any

import numpy as np
from numpy.typing import DTypeLike, NDArray
from tqdm import tqdm

import rapidtide.fit as tide_fit
import rapidtide.multiproc as tide_multiproc
import rapidtide.resample as tide_resample

warnings.simplefilter(action="ignore", category=FutureWarning)
LGR = logging.getLogger("GENERAL")


def _procOneVoxelPeaks(
    vox: int,
    thetc: NDArray,
    theMutualInformationator: Any,
    fmri_x: NDArray,
    fmritc: NDArray,
    os_fmri_x: NDArray,
    xcorr_x: NDArray,
    thexcorr: NDArray,
    bipolar: bool = False,
    oversampfactor: int = 1,
    sort: bool = True,
    interptype: str = "univariate",
) -> tuple[int, list]:
    """
    Evaluate candidate similarity peaks for one voxel.

    Parameters
    ----------
    vox : int
        Index of the voxel being processed.
    thetc : NDArray
        Time course to process.
    theMutualInformationator : object
        Mutual-information calculator used to score candidate peaks.
    fmri_x : NDArray
        Time coordinates of the original fMRI samples.
    fmritc : NDArray
        Input fMRI time course.
    os_fmri_x : NDArray
        Time coordinates on the oversampled fMRI grid.
    xcorr_x : NDArray
        Lag coordinates of the cross-correlation values.
    thexcorr : NDArray
        Cross-correlation values indexed by lag.
    bipolar : bool, optional
        Consider negative as well as positive correlation peaks.
    oversampfactor : int, optional
        Ratio of oversampled to original sampling frequency.
    sort : bool, optional
        Sort the candidate peaks before returning them.
    interptype : str, optional
        Interpolation method for time-series resampling.

    Returns
    -------
    tuple
        Voxel index and candidate peaks ranked by similarity.
    """
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
    fmridata: NDArray,
    referencetc: NDArray,
    fmri_x: NDArray,
    os_fmri_x: NDArray,
    theMutualInformationator: Any,
    xcorr_x: NDArray,
    corrdata: NDArray,
    nprocs: int = 1,
    alwaysmultiproc: bool = False,
    bipolar: bool = False,
    oversampfactor: int = 1,
    interptype: str = "univariate",
    showprogressbar: bool = True,
    chunksize: int = 1000,
    rt_floattype: DTypeLike = np.float64,
) -> tuple[int, dict]:
    """
    Evaluate candidate similarity peaks across voxels.

    Parameters
    ----------
    fmridata : NDArray
        Voxel time series to correlate with the reference.
    referencetc : NDArray
        Reference time series used for comparison.
    fmri_x : NDArray
        Time coordinates of the original fMRI samples.
    os_fmri_x : NDArray
        Time coordinates on the oversampled fMRI grid.
    theMutualInformationator : object
        Mutual-information calculator used to score candidate peaks.
    xcorr_x : NDArray
        Lag coordinates of the cross-correlation values.
    corrdata : NDArray
        Voxelwise correlation functions on the lag axis.
    nprocs : int, optional
        Number of worker processes.
    alwaysmultiproc : bool, optional
        Use multiprocessing even when only one worker is requested.
    bipolar : bool, optional
        Consider negative as well as positive correlation peaks.
    oversampfactor : int, optional
        Ratio of oversampled to original sampling frequency.
    interptype : str, optional
        Interpolation method for time-series resampling.
    showprogressbar : bool, optional
        Display processing progress.
    chunksize : int, optional
        Number of work items to submit to each worker at a time.
    rt_floattype : DTypeLike, optional
        Floating-point data type used for intermediate computations.

    Returns
    -------
    tuple
        Number of voxels processed and their peak lists.
    """
    peakdict = {}
    theMutualInformationator.setreftc(referencetc)

    inputshape = np.shape(fmridata)
    volumetotal = 0
    thetc = np.zeros(np.shape(os_fmri_x), dtype=rt_floattype)
    if nprocs > 1 or alwaysmultiproc:
        # define the consumer function here so it inherits most of the arguments
        def correlation_consumer(inQ, outQ):
            """
            Calculate correlations for queued voxel work items.

            Parameters
            ----------
            inQ : queue-like
                Queue supplying work items.
            outQ : queue-like
                Queue receiving processed results.
            """
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
    LGR.info(f"\nPeak evaluation performed on {volumetotal} voxels")

    # garbage collect
    uncollected = gc.collect()
    if uncollected != 0:
        LGR.info(f"garbage collected - unable to collect {uncollected} objects")
    else:
        LGR.info("garbage collected")

    return volumetotal, peakdict
