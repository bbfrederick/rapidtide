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
import gc
import logging
import warnings

import numpy as np
from tqdm import tqdm

import rapidtide.correlate as tide_corr
import rapidtide.fit as tide_fit
import rapidtide.multiproc as tide_multiproc

warnings.simplefilter(action="ignore", category=FutureWarning)
LGR = logging.getLogger("GENERAL")


def _procOneVoxelCorrelation(
    thedata,
    index,
    neighborindex,
    Fs,
    dofit=False,
    lagmin=-12.5,
    lagmax=12.5,
    widthmax=100.0,
    negsearch=15.0,
    possearch=15.0,
    padding=0,
    debug=False,
):
    tc1 = thedata[index, :]
    tc2 = thedata[neighborindex, :]
    if np.any(tc1) != 0.0 and np.any(tc2) != 0.0:
        thesimfunc = tide_corr.fastcorrelate(
            tc1,
            tc2,
            zeropadding=padding,
            usefft=True,
            debug=debug,
        )
        similarityfunclen = len(thesimfunc)
        similarityfuncorigin = similarityfunclen // 2 + 1

        negpoints = int(negsearch * Fs)
        pospoints = int(possearch * Fs)
        trimsimfunc = thesimfunc[
            similarityfuncorigin - negpoints : similarityfuncorigin + pospoints
        ]
        offset = 0.0
        trimtimeaxis = (
            (
                np.arange(0.0, similarityfunclen) * (1.0 / Fs)
                - ((similarityfunclen - 1) * (1.0 / Fs)) / 2.0
            )
            - offset
        )[similarityfuncorigin - negpoints : similarityfuncorigin + pospoints]
        if dofit:
            (
                maxindex,
                maxtime,
                maxcorr,
                maxsigma,
                maskval,
                failreason,
                peakstart,
                peakend,
            ) = tide_fit.simfuncpeakfit(
                trimsimfunc,
                trimtimeaxis,
                useguess=False,
                maxguess=0.0,
                displayplots=False,
                functype="correlation",
                peakfittype="gauss",
                searchfrac=0.5,
                lagmod=1000.0,
                enforcethresh=True,
                allowhighfitamps=False,
                lagmin=lagmin,
                lagmax=lagmax,
                absmaxsigma=1000.0,
                absminsigma=0.25,
                hardlimit=True,
                bipolar=False,
                lthreshval=0.0,
                uthreshval=1.0,
                zerooutbadfit=True,
                debug=False,
            )
        else:
            maxtime = trimtimeaxis[np.argmax(trimsimfunc)]
            maxcorr = np.max(trimsimfunc)
            maskval = 1
            failreason = 0
        if debug:
            print(f"{maxtime=}")
            print(f"{maxcorr=}")
            print(f"{maskval=}")
            print(f"{negsearch=}")
            print(f"{possearch=}")
            print(f"{Fs=}")
            print(f"{len(trimtimeaxis)=}")
            print(trimsimfunc, trimtimeaxis)
        return index, neighborindex, maxcorr, maxtime, maskval, failreason
    else:
        return index, neighborindex, 0.0, 0.0, 0, 0


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
    inputshape = np.shape(fmridata)
    volumetotal = 0
    thetc = np.zeros(np.shape(os_fmri_x), dtype=rt_floattype)
    theglobalmaxlist = []
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
                        _procOneVoxelCorrelation(
                            val,
                            idx1,
                            idx2,
                            Fs,
                            dofit=False,
                            lagmin=-12.5,
                            lagmax=12.5,
                            widthmax=100.0,
                            negsearch=15.0,
                            possearch=15.0,
                            padding=0,
                            debug=False,
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
            maxcorr[voxel[0], voxel[1]] = voxel[2]
            maxtime[voxel[0], voxel[1]] = voxel[3]
            maskval[voxel[0], voxel[1]] = voxel[4]
            failreason[voxel[0], voxel[1]] = voxel[5]
            volumetotal += 1
        del data_out
    else:
        for vox in tqdm(
            range(0, inputshape[0]),
            desc="Voxel",
            disable=(not showprogressbar),
        ):
            (
                dummy,
                meanval[vox],
                corrout[vox, :],
                thecorrscale,
                theglobalmax,
            ) = _procOneVoxelCorrelation(
                vox,
                thetc,
                theCorrelator,
                fmri_x,
                fmridata[vox, :],
                os_fmri_x,
                oversampfactor=oversampfactor,
                interptype=interptype,
                rt_floatset=rt_floatset,
                rt_floattype=rt_floattype,
            )
            theglobalmaxlist.append(theglobalmax + 0)
            volumetotal += 1
    LGR.info(f"\nSimilarity function calculated on {volumetotal} voxels")

    # garbage collect
    uncollected = gc.collect()
    if uncollected != 0:
        LGR.info(f"garbage collected - unable to collect {uncollected} objects")
    else:
        LGR.info("garbage collected")

    return volumetotal, theglobalmaxlist, thecorrscale
