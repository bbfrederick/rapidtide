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
import bisect
import gc
import logging

import numpy as np
from tqdm import tqdm

import rapidtide.fit as tide_fit
import rapidtide.multiproc as tide_multiproc

LGR = logging.getLogger("GENERAL")


def onesimfuncfit(
    correlationfunc,
    thefitter,
    disablethresholds=False,
    initiallag=None,
    despeckle_thresh=5.0,
    lthreshval=0.0,
    fixdelay=False,
    initialdelayvalue=0.0,
    rt_floatset=np.float64,
    rt_floattype="float64",
):
    if initiallag is not None:
        thefitter.setguess(True, maxguess=initiallag)
        thefitter.setrange(-despeckle_thresh / 2.0, despeckle_thresh / 2.0)
    else:
        thefitter.setguess(False)

    if disablethresholds:
        thefitter.setlthresh(0.0)
    else:
        thefitter.setlthresh(lthreshval)

    if not fixdelay:
        (
            maxindex,
            maxlag,
            maxval,
            maxsigma,
            maskval,
            failreason,
            peakstart,
            peakend,
        ) = thefitter.fit(correlationfunc)
    else:
        # do something different
        failreason = np.uint32(0)
        maxlag = rt_floatset(initialdelayvalue)
        maxindex = np.int16(bisect.bisect_left(thefitter.corrtimeaxis, initialdelayvalue))
        maxval = rt_floatset(correlationfunc[maxindex])
        maxsigma = rt_floatset(1.0)
        maskval = np.uint16(1)
        peakstart = maxindex
        peakend = maxindex

    return maxindex, maxlag, maxval, maxsigma, maskval, peakstart, peakend, failreason


def _procOneVoxelFitcorr(
    vox,
    corr_y,
    thefitter,
    disablethresholds=False,
    despeckle_thresh=5.0,
    initiallag=None,
    fixdelay=False,
    initialdelayvalue=0.0,
    rt_floatset=np.float64,
    rt_floattype="float64",
):
    (
        maxindex,
        maxlag,
        maxval,
        maxsigma,
        maskval,
        peakstart,
        peakend,
        failreason,
    ) = onesimfuncfit(
        corr_y,
        thefitter,
        disablethresholds=disablethresholds,
        despeckle_thresh=despeckle_thresh,
        fixdelay=fixdelay,
        initialdelayvalue=initialdelayvalue,
        initiallag=initiallag,
        rt_floatset=rt_floatset,
        rt_floattype=rt_floattype,
    )

    if maxval > 0.3:
        displayplots = False

    # now tuck everything away in the appropriate output array
    volumetotalinc = 0
    thewindowout = rt_floatset(0.0 * corr_y)
    thewindowout[peakstart : peakend + 1] = 1.0
    if (maskval == 0) and thefitter.zerooutbadfit:
        thetime = rt_floatset(0.0)
        thestrength = rt_floatset(0.0)
        thesigma = rt_floatset(0.0)
        thegaussout = 0.0 * corr_y
        theR2 = rt_floatset(0.0)
    else:
        volumetotalinc = 1
        thetime = rt_floatset(np.fmod(maxlag, thefitter.lagmod))
        thestrength = rt_floatset(maxval)
        thesigma = rt_floatset(maxsigma)
        thegaussout = rt_floatset(0.0 * corr_y)
        thewindowout = rt_floatset(0.0 * corr_y)
        if (not fixdelay) and (maxsigma != 0.0):
            thegaussout = rt_floatset(
                tide_fit.gauss_eval(thefitter.corrtimeaxis, [maxval, maxlag, maxsigma])
            )
        else:
            thegaussout = rt_floatset(0.0)
            thewindowout = rt_floatset(0.0)
        theR2 = rt_floatset(thestrength * thestrength)

    return (
        vox,
        volumetotalinc,
        thetime,
        thestrength,
        thesigma,
        thegaussout,
        thewindowout,
        theR2,
        maskval,
        failreason,
    )


def fitcorr(
    corrtimescale,
    thefitter,
    corrout,
    lagmask,
    failimage,
    lagtimes,
    lagstrengths,
    lagsigma,
    gaussout,
    windowout,
    R2,
    despeckling=False,
    peakdict=None,
    nprocs=1,
    alwaysmultiproc=False,
    fixdelay=False,
    initialdelayvalue=0.0,
    showprogressbar=True,
    chunksize=1000,
    despeckle_thresh=5.0,
    initiallags=None,
    rt_floatset=np.float64,
    rt_floattype="float64",
):
    thefitter.setcorrtimeaxis(corrtimescale)
    inputshape = np.shape(corrout)
    if initiallags is None:
        themask = None
    else:
        themask = np.where(initiallags > -100000.0, 1, 0)
    (
        volumetotal,
        ampfails,
        lowlagfails,
        highlagfails,
        lowwidthfails,
        highwidthfails,
        initfails,
        fitfails,
    ) = (0, 0, 0, 0, 0, 0, 0, 0)

    if nprocs > 1 or alwaysmultiproc:
        # define the consumer function here so it inherits most of the arguments
        def fitcorr_consumer(inQ, outQ):
            while True:
                try:
                    # get a new message
                    val = inQ.get()

                    # this is the 'TERM' signal
                    if val is None:
                        break

                    # process and send the data
                    if (themask is None) or (initiallags is None):
                        thislag = None
                    else:
                        if themask[val] > 0:
                            thislag = initiallags[val]
                        else:
                            thislag = None
                    if isinstance(initialdelayvalue, np.ndarray):
                        thisinitialdelayvalue = initialdelayvalue[val]
                    else:
                        thisinitialdelayvalue = initialdelayvalue
                    outQ.put(
                        _procOneVoxelFitcorr(
                            val,
                            corrout[val, :],
                            thefitter,
                            disablethresholds=False,
                            despeckle_thresh=despeckle_thresh,
                            initiallag=thislag,
                            fixdelay=fixdelay,
                            initialdelayvalue=thisinitialdelayvalue,
                            rt_floatset=rt_floatset,
                            rt_floattype=rt_floattype,
                        )
                    )
                except Exception as e:
                    print("error!", e)
                    break

        data_out = tide_multiproc.run_multiproc(
            fitcorr_consumer,
            inputshape,
            themask,
            nprocs=nprocs,
            showprogressbar=showprogressbar,
            chunksize=chunksize,
        )

        # unpack the data
        volumetotal = 0
        for voxel in data_out:
            if (
                thefitter.FML_INITAMPLOW
                | thefitter.FML_INITAMPHIGH
                | thefitter.FML_FITAMPLOW
                | thefitter.FML_FITAMPHIGH
            ) & voxel[9]:
                ampfails += 1
            if (thefitter.FML_INITWIDTHLOW | thefitter.FML_FITWIDTHLOW) & voxel[9]:
                lowwidthfails += 1
            if (thefitter.FML_INITWIDTHHIGH | thefitter.FML_FITWIDTHHIGH) & voxel[9]:
                highwidthfails += 1
            if (thefitter.FML_INITLAGLOW | thefitter.FML_FITLAGLOW) & voxel[9]:
                lowlagfails += 1
            if (thefitter.FML_INITLAGHIGH | thefitter.FML_FITLAGHIGH) & voxel[9]:
                highlagfails += 1
            if thefitter.FML_INITFAIL & voxel[9]:
                initfails += 1
            if thefitter.FML_FITFAIL & voxel[9]:
                fitfails += 1

            # if this is a despeckle pass, only accept the new values if the fit did not fail
            if (voxel[9] == 0) or not despeckling:
                volumetotal += voxel[1]
                lagtimes[voxel[0]] = voxel[2]
                lagstrengths[voxel[0]] = voxel[3]
                lagsigma[voxel[0]] = voxel[4]
                gaussout[voxel[0], :] = voxel[5]
                windowout[voxel[0], :] = voxel[6]
                R2[voxel[0]] = voxel[7]
                lagmask[voxel[0]] = voxel[8]
                failimage[voxel[0]] = voxel[9] & 0xFFFF

        del data_out
    else:
        for vox in tqdm(
            range(0, inputshape[0]),
            desc="Voxel",
            unit="voxels",
            disable=(not showprogressbar),
        ):
            # process and send the data
            if (themask is None) or (initiallags is None):
                thislag = None
                dothisone = True
            else:
                if themask[vox] > 0:
                    thislag = initiallags[vox]
                    dothisone = True
                else:
                    thislag = None
                    dothisone = False
            if isinstance(initialdelayvalue, np.ndarray):
                thisinitialdelayvalue = initialdelayvalue[vox]
            else:
                thisinitialdelayvalue = initialdelayvalue
            if dothisone:
                voxel = _procOneVoxelFitcorr(
                    vox,
                    corrout[vox, :],
                    thefitter,
                    disablethresholds=False,
                    despeckle_thresh=despeckle_thresh,
                    initiallag=thislag,
                    fixdelay=fixdelay,
                    initialdelayvalue=thisinitialdelayvalue,
                    rt_floatset=rt_floatset,
                    rt_floattype=rt_floattype,
                )
                if (
                    thefitter.FML_INITAMPLOW
                    | thefitter.FML_INITAMPHIGH
                    | thefitter.FML_FITAMPLOW
                    | thefitter.FML_FITAMPHIGH
                ) & voxel[9]:
                    ampfails += 1
                if (thefitter.FML_INITWIDTHLOW | thefitter.FML_FITWIDTHLOW) & voxel[9]:
                    lowwidthfails += 1
                if (thefitter.FML_INITWIDTHHIGH | thefitter.FML_FITWIDTHHIGH) & voxel[9]:
                    highwidthfails += 1
                if (thefitter.FML_INITLAGLOW | thefitter.FML_FITLAGLOW) & voxel[9]:
                    lowlagfails += 1
                if (thefitter.FML_INITLAGHIGH | thefitter.FML_FITLAGHIGH) & voxel[9]:
                    highlagfails += 1
                if thefitter.FML_INITFAIL & voxel[9]:
                    initfails += 1
                if thefitter.FML_FITFAIL & voxel[9]:
                    fitfails += 1

                # if this is a despeckle pass, only accept the new values if the fit did not fail
                if (voxel[9] == 0) or not despeckling:
                    volumetotal += voxel[1]
                    lagtimes[vox] = voxel[2]
                    lagstrengths[vox] = voxel[3]
                    lagsigma[vox] = voxel[4]
                    gaussout[vox, :] = voxel[5]
                    windowout[vox, :] = voxel[6]
                    R2[vox] = voxel[7]
                    lagmask[vox] = voxel[8]
                    failimage[vox] = voxel[9] & 0xFFFF

    LGR.info(f"\nSimilarity function fitted in {volumetotal} voxels")
    LGR.info(
        f"\tampfails: {ampfails}"
        + f"\n\tlowlagfails: {lowlagfails}"
        + f"\n\thighlagfails: {highlagfails}"
        + f"\n\tlowwidthfails: {lowwidthfails}"
        + f"\n\thighwidthfail: {highwidthfails}"
        + f"\n\ttotal initfails: {initfails}"
        + f"\n\ttotal fitfails: {fitfails}"
    )

    # garbage collect
    uncollected = gc.collect()
    if uncollected != 0:
        LGR.info(f"garbage collected - unable to collect {uncollected} objects")
    else:
        LGR.info("garbage collected")

    return volumetotal
