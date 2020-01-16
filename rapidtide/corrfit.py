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


def onecorrfitx(correlationfunc,
                thefitter,
                disablethresholds=False,
                initiallag=None,
                despeckle_thresh=5.0,
                lthreshval=0.0,
                fixdelay=False,
                fixeddelayvalue=0.0,
                rt_floatset=np.float64,
                rt_floattype='float64'):

    if initiallag is not None:
        thefitter.setguess(True, maxguess=initiallag)
        widthlimit = despeckle_thresh
    else:
        thefitter.setguess(False)

    if disablethresholds:
        thefitter.setlthresh(0.0)
    else:
        thefitter.setlthresh(lthreshval)

    flipfac = None
    if thefitter.bipolar:
        if max(correlationfunc) < -1.0 * min(correlationfunc):
            flipfac = rt_floatset(-1.0)
        else:
            flipfac = rt_floatset(1.0)
    else:
        flipfac = rt_floatset(1.0)

    if not fixdelay:
        if thefitter.findmaxtype == 'gauss':
            maxindex, maxlag, maxval, maxsigma, maskval, failreason, peakstart, peakend = thefitter.fit(correlationfunc)
        else:
            '''maxindex, maxlag, maxval, maxsigma, maskval, failreason, peakstart, peakend = tide_fit.findmaxlag_quad(
                corr_x,
                flipfac * thetc,
                lagmin, lagmax, widthlimit,
                edgebufferfrac=edgebufferfrac,
                threshval=lthreshval,
                uthreshval=uthreshval,
                debug=debug,
                refine=refine,
                maxguess=maxguess,
                useguess=useguess,
                fastgauss=fastgauss,
                enforcethresh=enforcethresh,
                zerooutbadfit=zerooutbadfit,
                lagmod=lagmod,
                displayplots=displayplots)'''
            maxval *= flipfac
    else:
        # do something different
        failreason = np.int16(0)
        maxlag = rt_floatset(fixeddelayvalue)
        maxindex = np.int16(bisect.bisect_left(thefitter.corrtimeaxis, fixeddelayvalue))
        maxval = rt_floatset(flipfac * correlationfunc[maxindex])
        maxsigma = rt_floatset(1.0)
        maskval = np.uint16(1)
        peakstart = maxindex
        peakend = maxindex

    return maxindex, maxlag, maxval, maxsigma, maskval, peakstart, peakend, failreason


def _procOneVoxelFitcorrx(vox,
                          corr_y,
                          lagtcgenerator,
                          timeaxis,
                          thefitter,
                          disablethresholds=False,
                          despeckle_thresh=5.0,
                          initiallag=None,
                          fixdelay=False,
                          fixeddelayvalue=0.0,
                          rt_floatset=np.float64,
                          rt_floattype='float64'):
    maxindex, maxlag, maxval, maxsigma, maskval, peakstart, peakend, failreason = onecorrfitx(corr_y,
                                                                                              thefitter,
                                                                                              disablethresholds=disablethresholds,
                                                                                              despeckle_thresh=despeckle_thresh,
                                                                                              fixdelay=fixdelay,
                                                                                              fixeddelayvalue=fixeddelayvalue,
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
    thewindowout = rt_floatset(0.0 * corr_y)
    thewindowout[peakstart:peakend + 1] = 1.0
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
            thegaussout = rt_floatset(tide_fit.gauss_eval(thefitter.corrtimeaxis, [maxval, maxlag, maxsigma]))
        else:
            thegaussout = rt_floatset(0.0)
            thewindowout = rt_floatset(0.0)
        theR2 = rt_floatset(thestrength * thestrength)

    return vox, volumetotalinc, thelagtc, thetime, thestrength, thesigma, thegaussout, \
           thewindowout, theR2, maskval, failreason


def fitcorrx(lagtcgenerator,
            timeaxis,
            lagtc,
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
            nprocs=1,
            fixdelay=False,
            showprogressbar=True,
            chunksize=1000,
            despeckle_thresh=5.0,
            initiallags=None,
            rt_floatset=np.float64,
            rt_floattype='float64'):

    thefitter.setcorrtimeaxis(corrtimescale)
    inputshape = np.shape(corrout)
    if initiallags is None:
        themask = None
    else:
        themask = np.where(initiallags > -1000000.0, 1, 0)
    reportstep = 1000
    volumetotal, ampfails, lagfails, windowfails, widthfails, edgefails, fitfails = 0, 0, 0, 0, 0, 0, 0
    FML_BADAMPLOW = np.uint16(0x01)
    FML_BADAMPHIGH = np.uint16(0x02)
    FML_BADSEARCHWINDOW = np.uint16(0x04)
    FML_BADWIDTH = np.uint16(0x08)
    FML_BADLAG = np.uint16(0x10)
    FML_HITEDGE = np.uint16(0x20)
    FML_FITFAIL = np.uint16(0x40)
    FML_INITFAIL = np.uint16(0x80)
    zerolagtc = rt_floatset(lagtcgenerator.yfromx(timeaxis))
    sliceoffsettime = 0.0

    if nprocs > 1:
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
                    outQ.put(_procOneVoxelFitcorrx(val,
                                                   corrout[val, :],
                                                   lagtcgenerator,
                                                   timeaxis,
                                                   thefitter,
                                                   disablethresholds=False,
                                                   despeckle_thresh=despeckle_thresh,
                                                   initiallag=thislag,
                                                   fixdelay=fixdelay,
                                                   fixeddelayvalue=0.0,
                                                   rt_floatset=rt_floatset,
                                                   rt_floattype=rt_floattype))
                except Exception as e:
                    print("error!", e)
                    break

        data_out = tide_multiproc.run_multiproc(fitcorr_consumer,
                                                inputshape, themask,
                                                nprocs=nprocs,
                                                showprogressbar=showprogressbar,
                                                chunksize=chunksize)

        # unpack the data
        volumetotal = 0
        for voxel in data_out:
            volumetotal += voxel[1]
            lagtc[voxel[0], :] = voxel[2]
            lagtimes[voxel[0]] = voxel[3]
            lagstrengths[voxel[0]] = voxel[4]
            lagsigma[voxel[0]] = voxel[5]
            gaussout[voxel[0], :] = voxel[6]
            windowout[voxel[0], :] = voxel[7]
            R2[voxel[0]] = voxel[8]
            lagmask[voxel[0]] = voxel[9]
            failimage[voxel[0]] = voxel[10] & 0x3f
            if (FML_BADAMPLOW | FML_BADAMPHIGH) & voxel[10]:
                ampfails += 1
            if FML_BADSEARCHWINDOW & voxel[10]:
                windowfails += 1
            if FML_BADWIDTH & voxel[10]:
                widthfails += 1
            if FML_BADLAG & voxel[10]:
                lagfails += 1
            if FML_HITEDGE & voxel[10]:
                edgefails += 1
            if (FML_FITFAIL | FML_INITFAIL) & voxel[10]:
                fitfails += 1
        del data_out
    else:
        for vox in range(0, inputshape[0]):
            if (vox % reportstep == 0 or vox == inputshape[0] - 1) and showprogressbar:
                tide_util.progressbar(vox + 1, inputshape[0], label='Percent complete')
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
                dummy, \
                volumetotalinc, \
                lagtc[vox, :], \
                lagtimes[vox], \
                lagstrengths[vox], \
                lagsigma[vox], \
                gaussout[vox, :], \
                windowout[vox, :], \
                R2[vox], \
                lagmask[vox], \
                failreason = \
                    _procOneVoxelFitcorrx(vox,
                                          corrout[vox, :],
                                          lagtcgenerator,
                                          timeaxis,
                                          thefitter,
                                          disablethresholds=False,
                                          despeckle_thresh=despeckle_thresh,
                                          initiallag=thislag,
                                          fixdelay=fixdelay,
                                          rt_floatset=rt_floatset,
                                          rt_floattype=rt_floattype)
                volumetotal += volumetotalinc
                if (FML_BADAMPLOW | FML_BADAMPHIGH) & failreason:
                    ampfails += 1
                if FML_BADSEARCHWINDOW & failreason:
                    windowfails += 1
                if FML_BADWIDTH & failreason:
                    widthfails += 1
                if FML_BADLAG & failreason:
                    lagfails += 1
                if FML_HITEDGE & failreason:
                    edgefails += 1
                if (FML_FITFAIL | FML_INITFAIL) & failreason:
                    fitfails += 1
    print('\nCorrelation fitted in ' + str(volumetotal) + ' voxels')
    print('\tampfails=', ampfails,
          '\n\tlagfails=', lagfails,
          '\n\twindowfails=', windowfails,
          '\n\twidthfail=', widthfails,
          '\n\tedgefail=', edgefails,
          '\n\tfitfail=', fitfails)

    # garbage collect
    collected = gc.collect()
    print("Garbage collector: collected %d objects." % collected)

    return volumetotal
