#!/usr/bin/env python
# -*- coding: latin-1 -*-
#
#   Copyright 2016-2019 Blaise Frederick
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

from __future__ import print_function, division

import bisect
import gc

import numpy as np

import rapidtide.fit as tide_fit
import rapidtide.multiproc as tide_multiproc
import rapidtide.util as tide_util


def onecorrfit(corr_y,
               corr_x,
               optiondict,
               zerooutbadfit=True,
               displayplots=False,
               initiallag=None,
               rt_floatset=np.float64,
               rt_floattype='float64'
               ):
    if initiallag is not None:
        maxguess = initiallag
        useguess = True
        widthlimit = optiondict['despeckle_thresh']
    else:
        maxguess = 0.0
        useguess = False
        widthlimit = optiondict['widthlimit']

    if optiondict['bipolar']:
        if max(corr_y) < -1.0 * min(corr_y):
            flipfac = rt_floatset(-1.0)
        else:
            flipfac = rt_floatset(1.0)
    else:
        flipfac = rt_floatset(1.0)

    if not optiondict['fixdelay']:
        if optiondict['findmaxtype'] == 'gauss':
            maxindex, maxlag, maxval, maxsigma, maskval, failreason, peakstart, peakend = tide_fit.findmaxlag_gauss(
                corr_x,
                flipfac * corr_y,
                optiondict['lagmin'], optiondict['lagmax'], widthlimit,
                edgebufferfrac=optiondict['edgebufferfrac'],
                threshval=optiondict['lthreshval'],
                uthreshval=optiondict['uthreshval'],
                debug=optiondict['debug'],
                refine=optiondict['gaussrefine'],
                maxguess=maxguess,
                useguess=useguess,
                fastgauss=optiondict['fastgauss'],
                enforcethresh=optiondict['enforcethresh'],
                zerooutbadfit=zerooutbadfit,
                lagmod=optiondict['lagmod'],
                displayplots=displayplots)
        else:
            maxindex, maxlag, maxval, maxsigma, maskval, failreason, peakstart, peakend = tide_fit.findmaxlag_quad(
                corr_x,
                flipfac * corr_y,
                optiondict['lagmin'], optiondict['lagmax'], widthlimit,
                edgebufferfrac=optiondict['edgebufferfrac'],
                threshval=optiondict['lthreshval'],
                uthreshval=optiondict['uthreshval'],
                debug=optiondict['debug'],
                refine=optiondict['gaussrefine'],
                maxguess=maxguess,
                useguess=useguess,
                fastgauss=optiondict['fastgauss'],
                enforcethresh=optiondict['enforcethresh'],
                zerooutbadfit=zerooutbadfit,
                lagmod=optiondict['lagmod'],
                displayplots=displayplots)
        maxval *= flipfac
    else:
        # do something different
        failreason = np.int16(0)
        maxlag = rt_floatset(optiondict['fixeddelayvalue'])
        maxindex = np.int16(bisect.bisect_left(corr_x, optiondict['fixeddelayvalue']))
        maxval = rt_floatset(flipfac * corr_y[maxindex])
        maxsigma = rt_floatset(1.0)
        maskval = np.uint16(1)

    return maxindex, maxlag, maxval, maxsigma, maskval, failreason


def _procOneVoxelFitcorr(vox,
                         corr_y,
                         corr_x,
                         lagtcgenerator,
                         timeaxis,
                         optiondict,
                         zerooutbadfit=True,
                         displayplots=False,
                         initiallag=None,
                         rt_floatset=np.float64,
                         rt_floattype='float64'
                         ):
    maxindex, maxlag, maxval, maxsigma, maskval, failreason = onecorrfit(corr_y,
                                                                         corr_x,
                                                                         optiondict,
                                                                         zerooutbadfit=zerooutbadfit,
                                                                         displayplots=displayplots,
                                                                         initiallag=initiallag,
                                                                         rt_floatset=rt_floatset,
                                                                         rt_floattype=rt_floattype)

    if maxval > 0.3:
        displayplots = False

    # question - should maxlag be added or subtracted?  As of 10/18, it is subtracted
    #  potential answer - tried adding, results are terrible.
    thelagtc = rt_floatset(lagtcgenerator.yfromx(timeaxis - maxlag))

    # now tuck everything away in the appropriate output array
    volumetotalinc = 0
    if (maskval == 0) and optiondict['zerooutbadfit']:
        thetime = rt_floatset(0.0)
        thestrength = rt_floatset(0.0)
        thesigma = rt_floatset(0.0)
        thegaussout = 0.0 * corr_y
        theR2 = rt_floatset(0.0)
    else:
        volumetotalinc = 1
        thetime = rt_floatset(np.fmod(maxlag, optiondict['lagmod']))
        thestrength = rt_floatset(maxval)
        thesigma = rt_floatset(maxsigma)
        if (not optiondict['fixdelay']) and (maxsigma != 0.0):
            thegaussout = rt_floatset(tide_fit.gauss_eval(corr_x, [maxval, maxlag, maxsigma]))
        else:
            thegaussout = rt_floatset(0.0)
        theR2 = rt_floatset(thestrength * thestrength)

    return vox, volumetotalinc, thelagtc, thetime, thestrength, thesigma, thegaussout, theR2, maskval, failreason


def fitcorr(lagtcgenerator,
            timeaxis,
            lagtc,
            slicesize,
            corr_x,
            lagmask,
            lagtimes,
            lagstrengths,
            lagsigma,
            corrout,
            meanval,
            gaussout,
            R2,
            optiondict,
            zerooutbadfit=True,
            initiallags=None,
            rt_floatset=np.float64,
            rt_floattype='float64'
            ):
    displayplots = False
    inputshape = np.shape(corrout)
    if initiallags is None:
        themask = None
    else:
        themask = np.where(initiallags > -1000000.0, 1, 0)
    volumetotal, ampfails, lagfails, widthfails, edgefails, fitfails = 0, 0, 0, 0, 0, 0
    reportstep = 1000
    zerolagtc = rt_floatset(lagtcgenerator.yfromx(timeaxis))

    if optiondict['nprocs'] > 1:
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
                    if initiallags is None:
                        thislag = None
                    else:
                        thislag = initiallags[val]
                    outQ.put(
                        _procOneVoxelFitcorr(val,
                                             corrout[val, :],
                                             corr_x, lagtcgenerator,
                                             timeaxis,
                                             optiondict,
                                             zerooutbadfit=zerooutbadfit,
                                             displayplots=False,
                                             initiallag=thislag,
                                             rt_floatset=rt_floatset,
                                             rt_floattype=rt_floattype))

                except Exception as e:
                    print("error!", e)
                    break

        data_out = tide_multiproc.run_multiproc(fitcorr_consumer,
                                                inputshape, themask,
                                                nprocs=optiondict['nprocs'],
                                                showprogressbar=True,
                                                chunksize=optiondict['mp_chunksize'])


        # unpack the data
        volumetotal = 0
        for voxel in data_out:
            volumetotal += voxel[1]
            lagtc[voxel[0], :] = voxel[2]
            lagtimes[voxel[0]] = voxel[3]
            lagstrengths[voxel[0]] = voxel[4]
            lagsigma[voxel[0]] = voxel[5]
            gaussout[voxel[0], :] = voxel[6]
            R2[voxel[0]] = voxel[7]
            lagmask[voxel[0]] = voxel[8]
        data_out = []
    else:
        for vox in range(0, inputshape[0]):
            if (vox % reportstep == 0 or vox == inputshape[0] - 1) and optiondict['showprogressbar']:
                tide_util.progressbar(vox + 1, inputshape[0], label='Percent complete')
            if themask is None:
                dothisone = True
                thislag = None
            elif themask[vox] > 0:
                dothisone = True
                thislag = initiallags[vox]
            else:
                dothisone = False
            if dothisone:
                dummy, \
                volumetotalinc, \
                lagtc[vox, :], \
                lagtimes[vox], \
                lagstrengths[vox], \
                lagsigma[vox], \
                gaussout[vox, :], \
                R2[vox], \
                lagmask[vox], \
                failreason = \
                    _procOneVoxelFitcorr(vox,
                                         corrout[vox, :],
                                         corr_x,
                                         lagtcgenerator,
                                         timeaxis,
                                         optiondict,
                                         zerooutbadfit=zerooutbadfit,
                                         displayplots=False,
                                         initiallag=thislag,
                                         rt_floatset=rt_floatset,
                                         rt_floattype=rt_floattype)

                volumetotal += volumetotalinc
    print('\nCorrelation fitted in ' + str(volumetotal) + ' voxels')
    print('\tampfails=', ampfails, '\n\tlagfails=', lagfails, '\n\twidthfail=', widthfails, '\n\tedgefail=', edgefails,
          '\n\tfitfail=', fitfails)

    # garbage collect
    collected = gc.collect()
    print("Garbage collector: collected %d objects." % collected)

    return volumetotal
