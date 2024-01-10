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
        thelagtc,
    )


def makelaggedtcs(
    lagtcgenerator,
    timeaxis,
    lagmask,
    lagtimes,
    lagstrengths,
    lagsigma,
    nprocs=1,
    alwaysmultiproc=False,
    showprogressbar=True,
    chunksize=1000,
    rt_floatset=np.float64,
    rt_floattype="float64",
):
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
            themask,
            nprocs=nprocs,
            showprogressbar=showprogressbar,
            chunksize=chunksize,
        )

        # unpack the data
        volumetotal = 0
        for voxel in data_out:
            # if this is a despeckle pass, only accept the new values if the fit did not fail
            if (voxel[10] == 0) or not despeckling:
                volumetotal += voxel[1]
                lagtc[voxel[0], :] = voxel[2]
        del data_out
    else:
        for vox in tqdm(
            range(0, inputshape[0]),
            desc="Voxel",
            unit="voxels",
            disable=(not showprogressbar),
        ):
            if themask is None:
                dothisone = True
                thislag = None
            elif themask[vox] > 0:
                dothisone = True
                thislag = initiallags[vox]
            else:
                dothisone = False
                thislag = None
            if dothisone:
                (
                    dummy,
                    volumetotalinc,
                    lagtc[vox, :],
                    lagtimes[vox],
                    lagstrengths[vox],
                    lagsigma[vox],
                    gaussout[vox, :],
                    windowout[vox, :],
                    R2[vox],
                    lagmask[vox],
                    failreason,
                ) = _procOneVoxelFitcorr(
                    vox,
                    corrout[vox, :],
                    lagtcgenerator,
                    timeaxis,
                    thefitter,
                    disablethresholds=False,
                    despeckle_thresh=despeckle_thresh,
                    initiallag=thislag,
                    fixdelay=fixdelay,
                    rt_floatset=rt_floatset,
                    rt_floattype=rt_floattype,
                )
                volumetotal += volumetotalinc
                if (
                    thefitter.FML_INITAMPLOW
                    | thefitter.FML_INITAMPHIGH
                    | thefitter.FML_FITAMPLOW
                    | thefitter.FML_FITAMPHIGH
                ) & failreason:
                    ampfails += 1
                if (thefitter.FML_INITWIDTHLOW | thefitter.FML_FITWIDTHLOW) & failreason:
                    lowwidthfails += 1
                if (thefitter.FML_INITWIDTHHIGH | thefitter.FML_FITWIDTHHIGH) & failreason:
                    highwidthfails += 1
                if (thefitter.FML_INITLAGLOW | thefitter.FML_INITLAGHIGH) & failreason:
                    lowlagfails += 1
                if (thefitter.FML_INITLAGLOW | thefitter.FML_FITLAGLOW) & failreason:
                    lowlagfails += 1
                if (thefitter.FML_INITLAGHIGH | thefitter.FML_FITLAGHIGH) & failreason:
                    highlagfails += 1
                if thefitter.FML_INITFAIL & failreason:
                    initfails += 1
                if thefitter.FML_FITFAIL & failreason:
                    fitfails += 1

    print("\nCorrelation fitted in " + str(volumetotal) + " voxels")
    print(
        "\tampfails:",
        ampfails,
        "\n\tlowlagfails:",
        lowlagfails,
        "\n\thighlagfails:",
        highlagfails,
        "\n\tlowwidthfails:",
        lowwidthfails,
        "\n\thighwidthfail:",
        highwidthfails,
        "\n\ttotal initfails:",
        initfails,
        "\n\ttotal fitfails:",
        fitfails,
    )

    # garbage collect
    collected = gc.collect()
    print("Garbage collector: collected %d objects." % collected)

    return volumetotal
