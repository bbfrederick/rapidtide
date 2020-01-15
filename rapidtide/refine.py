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

import gc
import sys

import rapidtide.miscmath as tide_math
import rapidtide.util as tide_util
import rapidtide.io as tide_io
import rapidtide.filter as tide_filt
import rapidtide.fit as tide_fit
import rapidtide.multiproc as tide_multiproc
import rapidtide.resample as tide_resample
import rapidtide.stats as tide_stats

from sklearn.decomposition import FastICA, PCA
import numpy as np
from scipy.stats.stats import pearsonr
from scipy.signal import welch


def _procOneVoxelTimeShift(vox,
                           fmritc,
                           lagstrength,
                           R2val,
                           lagtime,
                           padtrs,
                           fmritr,
                           theprefilter,
                           fmrifreq,
                           refineprenorm='mean',
                           lagmaxthresh=5.0,
                           refineweighting='R',
                           detrendorder=1,
                           offsettime=0.0,
                           filterbeforePCA=False,
                           psdfilter=False,
                           rt_floatset=np.float64,
                           rt_floattype='float64'
                           ):
    if refineprenorm == 'mean':
        thedivisor = np.mean(fmritc)
    elif refineprenorm == 'var':
        thedivisor = np.var(fmritc)
    elif refineprenorm == 'std':
        thedivisor = np.std(fmritc)
    elif refineprenorm == 'invlag':
        if lagtime < lagmaxthresh:
            thedivisor = lagmaxthresh - lagtime
        else:
            thedivisor = 0.0
    else:
        thedivisor = 1.0
    if thedivisor != 0.0:
        normfac = 1.0 / thedivisor
    else:
        normfac = 0.0

    if refineweighting == 'R':
        thisweight = lagstrength
    elif refineweighting == 'R2':
        thisweight = R2val
    else:
        thisweight = 1.0
    if detrendorder > 0:
        normtc = tide_fit.detrend(fmritc * normfac * thisweight, order=detrendorder, demean=True)
    else:
        normtc = fmritc * normfac * thisweight
    shifttr = -(-offsettime + lagtime) / fmritr  # lagtime is in seconds
    [shiftedtc, weights, paddedshiftedtc, paddedweights] = tide_resample.timeshift(normtc, shifttr, padtrs)
    if filterbeforePCA:
        outtc = theprefilter.apply(fmrifreq, shiftedtc)
        outweights = theprefilter.apply(fmrifreq, weights)
    else:
        outtc = 1.0 * shiftedtc
        outweights = 1.0 * weights
    if psdfilter:
        freqs, psd = welch(tide_math.corrnormalize(shiftedtc, True, True), fmritr, scaling='spectrum', window='hamming',
                           return_onesided=False, nperseg=len(shiftedtc))
        return vox, outtc, outweights, np.sqrt(psd)
    else:
        return vox, outtc, outweights, None


def refineregressor(fmridata,
                    fmritr,
                    shiftedtcs,
                    weights,
                    passnum,
                    lagstrengths,
                    lagtimes,
                    lagsigma,
                    R2,
                    theprefilter,
                    optiondict,
                    padtrs=60,
                    includemask=None,
                    excludemask=None,
                    rt_floatset=np.float64,
                    rt_floattype='float64'):
    """

    Parameters
    ----------
    fmridata : 4D numpy float array
       fMRI data
    fmritr : float
        Data repetition rate, in seconds
    shiftedtcs : 4D numpy float array
        Time aligned voxel timecourses
    weights :  unknown
        unknown
    passnum : int
        Number of the pass (for labelling output)
    lagstrengths : 3D numpy float array
        Maximum correlation coefficient in every voxel
    lagtimes : 3D numpy float array
        Time delay of maximum crosscorrelation in seconds
    lagsigma : 3D numpy float array
        Gaussian width of the crosscorrelation peak, in seconds.
    R2 : 3D numpy float array
        Square of the maximum correlation coefficient in every voxel
    theprefilter : function
        The filter function to use
    optiondict : dict
        Dictionary of all internal rapidtide configuration variables.
    padtrs : int, optional
        Number of timepoints to pad onto each end
    includemask : 3D array
        Mask of voxels to include in refinement.  Default is None (all voxels).
    excludemask : 3D array
        Mask of voxels to exclude from refinement.  Default is None (no voxels).
    rt_floatset : function
        Function to coerce variable types
    rt_floattype : {'float32', 'float64'}
        Data type for internal variables

    Returns
    -------
    volumetotal : int
        Number of voxels processed
    outputdata : float array
        New regressor
    maskarray : 3D array
        Mask of voxels used for refinement
    """
    inputshape = np.shape(fmridata)
    if optiondict['ampthresh'] < 0.0:
        theampthresh = tide_stats.getfracval(lagstrengths, -optiondict['ampthresh'], nozero=True)
        print('setting ampthresh to the', -100.0 * optiondict['ampthresh'], 'th percentile (', theampthresh, ')')
        ampmask = np.where(lagstrengths >= theampthresh, np.int16(1), np.int16(0))
    else:
        ampmask = np.where(lagstrengths >= optiondict['ampthresh'], np.int16(1), np.int16(0))
    if optiondict['lagmaskside'] == 'upper':
        delaymask = \
            np.where(lagtimes > optiondict['lagminthresh'], np.int16(1), np.int16(0)) * \
            np.where(lagtimes < optiondict['lagmaxthresh'], np.int16(1), np.int16(0))
    elif optiondict['lagmaskside'] == 'lower':
        delaymask = \
            np.where(lagtimes < -optiondict['lagminthresh'], np.int16(1), np.int16(0)) * \
            np.where(lagtimes > -optiondict['lagmaxthresh'], np.int16(1), np.int16(0))
    else:
        abslag = abs(lagtimes)
        delaymask = \
            np.where(abslag > optiondict['lagminthresh'], np.int16(1), np.int16(0)) * \
            np.where(abslag < optiondict['lagmaxthresh'], np.int16(1), np.int16(0))
    sigmamask = np.where(lagsigma < optiondict['sigmathresh'], np.int16(1), np.int16(0))
    locationmask = 0 * ampmask + 1
    if includemask is not None:
        locationmask = locationmask * includemask
    if excludemask is not None:
        locationmask = locationmask * (1 - excludemask)
    print('location mask created')

    # first generate the refine mask
    locationfails = np.sum(1 - locationmask)
    ampfails = np.sum(1 - ampmask)
    lagfails = np.sum(1 - delaymask)
    sigmafails = np.sum(1 - sigmamask)
    refinemask = locationmask * ampmask * delaymask * sigmamask
    if tide_stats.getmasksize(refinemask) == 0:
        if (includemask is None) and (excludemask is None):
            print('ERROR: no voxels in the refine mask - relax ampthresh, delaythresh, or sigmathresh - exiting')
            sys.exit()
        else:
            print('ERROR: no voxels in the refine mask - change include/exclude masks or relax ampthresh, delaythresh, or sigmathresh - exiting')
            sys.exit()

    if optiondict['cleanrefined']:
        shiftmask = locationmask
    else:
        shiftmask = refinemask
    volumetotal = np.sum(shiftmask)
    reportstep = 1000

    # timeshift the valid voxels
    if optiondict['nprocs'] > 1:
        # define the consumer function here so it inherits most of the arguments
        def timeshift_consumer(inQ, outQ):
            while True:
                try:
                    # get a new message
                    val = inQ.get()

                    # this is the 'TERM' signal
                    if val is None:
                        break

                    # process and send the data
                    outQ.put(_procOneVoxelTimeShift(val,
                                                    fmridata[val, :],
                                                    lagstrengths[val],
                                                    R2[val],
                                                    lagtimes[val],
                                                    padtrs,
                                                    fmritr,
                                                    theprefilter,
                                                    optiondict['fmrifreq'],
                                                    refineprenorm=optiondict['refineprenorm'],
                                                    lagmaxthresh=optiondict['lagmaxthresh'],
                                                    refineweighting=optiondict['refineweighting'],
                                                    detrendorder=optiondict['detrendorder'],
                                                    offsettime=optiondict['offsettime'],
                                                    filterbeforePCA=optiondict['filterbeforePCA'],
                                                    psdfilter=optiondict['psdfilter'],
                                                    rt_floatset=rt_floatset,
                                                    rt_floattype=rt_floattype
                                                    ))

                except Exception as e:
                    print("error!", e)
                    break

        data_out = tide_multiproc.run_multiproc(timeshift_consumer,
                                                inputshape, shiftmask,
                                                nprocs=optiondict['nprocs'],
                                                showprogressbar=True,
                                                chunksize=optiondict['mp_chunksize'])

        # unpack the data
        psdlist = []
        for voxel in data_out:
            shiftedtcs[voxel[0], :] = voxel[1]
            weights[voxel[0], :] = voxel[2]
            if optiondict['psdfilter']:
                psdlist.append(voxel[3])
        del data_out

    else:
        psdlist = []
        for vox in range(0, inputshape[0]):
            if (vox % reportstep == 0 or vox == inputshape[0] - 1) and optiondict['showprogressbar']:
                tide_util.progressbar(vox + 1, inputshape[0], label='Percent complete (timeshifting)')
            if shiftmask[vox] > 0:
                retvals = _procOneVoxelTimeShift(vox,
                                                 fmridata[vox, :],
                                                 lagstrengths[vox],
                                                 R2[vox],
                                                 lagtimes[vox],
                                                 padtrs,
                                                 fmritr,
                                                 theprefilter,
                                                 optiondict['fmrifreq'],
                                                 refineprenorm=optiondict['refineprenorm'],
                                                 lagmaxthresh=optiondict['lagmaxthresh'],
                                                 refineweighting=optiondict['refineweighting'],
                                                 detrendorder=optiondict['detrendorder'],
                                                 offsettime=optiondict['offsettime'],
                                                 filterbeforePCA=optiondict['filterbeforePCA'],
                                                 psdfilter=optiondict['psdfilter'],
                                                 rt_floatset=rt_floatset,
                                                 rt_floattype=rt_floattype)
                shiftedtcs[retvals[0], :] = retvals[1]
                weights[retvals[0], :] = retvals[2]
                if optiondict['psdfilter']:
                    psdlist.append(retvals[3])
        print()

    if optiondict['psdfilter']:
        print(len(psdlist))
        print(psdlist[0])
        print(np.shape(np.asarray(psdlist, dtype=rt_floattype)))
        averagepsd = np.mean(np.asarray(psdlist, dtype=rt_floattype), axis=0)
        stdpsd = np.std(np.asarray(psdlist, dtype=rt_floattype), axis=0)
        snr = np.nan_to_num(averagepsd / stdpsd)

    # now generate the refined timecourse(s)
    validlist = np.where(refinemask > 0)[0]
    refinevoxels = shiftedtcs[validlist]
    refineweights = weights[validlist]
    weightsum = np.sum(refineweights, axis=0) / volumetotal
    averagedata = np.sum(refinevoxels, axis=0) / volumetotal
    if optiondict['cleanrefined']:
        invalidlist = np.where((1 - ampmask) > 0)[0]
        discardvoxels = shiftedtcs[invalidlist]
        discardweights = weights[invalidlist]
        discardweightsum = np.sum(discardweights, axis=0) / volumetotal
        averagediscard = np.sum(discardvoxels, axis=0) / volumetotal
    if optiondict['dodispersioncalc']:
        print('splitting regressors by time lag for phase delay estimation')
        laglist = np.arange(optiondict['dispersioncalc_lower'], optiondict['dispersioncalc_upper'],
                            optiondict['dispersioncalc_step'])
        dispersioncalcout = np.zeros((np.shape(laglist)[0], inputshape[1]), dtype=rt_floattype)
        fftlen = int(inputshape[1] // 2)
        fftlen -= fftlen % 2
        dispersioncalcspecmag = np.zeros((np.shape(laglist)[0], fftlen), dtype=rt_floattype)
        dispersioncalcspecphase = np.zeros((np.shape(laglist)[0], fftlen), dtype=rt_floattype)
        for lagnum in range(0, np.shape(laglist)[0]):
            lower = laglist[lagnum] - optiondict['dispersioncalc_step'] / 2.0
            upper = laglist[lagnum] + optiondict['dispersioncalc_step'] / 2.0
            inlagrange = np.where(
                locationmask * ampmask * np.where(lower < lagtimes, np.int16(1), np.int16(0))
                * np.where(lagtimes < upper, np.int16(1), np.int16(0)))[0]
            print('    summing', np.shape(inlagrange)[0], 'regressors with lags from', lower, 'to', upper)
            if np.shape(inlagrange)[0] > 0:
                dispersioncalcout[lagnum, :] = tide_math.corrnormalize(np.mean(shiftedtcs[inlagrange], axis=0),
                                                                       prewindow=False,
                                                                       detrendorder=optiondict['detrendorder'],
                                                                       windowfunc=optiondict['windowfunc'])
                freqs, dispersioncalcspecmag[lagnum, :], dispersioncalcspecphase[lagnum, :] = tide_math.polarfft(
                    dispersioncalcout[lagnum, :],
                    1.0 / fmritr)
            inlagrange = None
        tide_io.writenpvecs(dispersioncalcout,
                            optiondict['outputname'] + '_dispersioncalcvecs_pass' + str(passnum) + '.txt')
        tide_io.writenpvecs(dispersioncalcspecmag,
                            optiondict['outputname'] + '_dispersioncalcspecmag_pass' + str(passnum) + '.txt')
        tide_io.writenpvecs(dispersioncalcspecphase,
                            optiondict['outputname'] + '_dispersioncalcspecphase_pass' + str(passnum) + '.txt')
        tide_io.writenpvecs(freqs, optiondict['outputname'] + '_dispersioncalcfreqs_pass' + str(passnum) + '.txt')

    if optiondict['estimatePCAdims']:
        pcacomponents = 'mle'
    else:
        pcacomponents = 1
    icacomponents = 1

    if optiondict['refinetype'] == 'ica':
        print('performing ica refinement')
        thefit = FastICA(n_components=icacomponents).fit(refinevoxels)  # Reconstruct signals
        print('Using first of ', len(thefit.components_), ' components')
        icadata = thefit.components_[0]
        filteredavg = tide_math.corrnormalize(theprefilter.apply(optiondict['fmrifreq'], averagedata), prewindow=True, detrendorder=optiondict['detrendorder'])
        filteredica = tide_math.corrnormalize(theprefilter.apply(optiondict['fmrifreq'], icadata), prewindow=True, detrendorder=optiondict['detrendorder'])
        thepxcorr = pearsonr(filteredavg, filteredica)[0]
        print('ica/avg correlation = ', thepxcorr)
        if thepxcorr > 0.0:
            outputdata = 1.0 * icadata
        else:
            outputdata = -1.0 * icadata
    elif optiondict['refinetype'] == 'pca':
        print('performing pca refinement')
        thefit = PCA(n_components=pcacomponents).fit(refinevoxels)
        print('Using first of ', len(thefit.components_), ' components')
        pcadata = thefit.components_[0]
        filteredavg = tide_math.corrnormalize(theprefilter.apply(optiondict['fmrifreq'], averagedata), prewindow=True, detrendorder=optiondict['detrendorder'])
        filteredpca = tide_math.corrnormalize(theprefilter.apply(optiondict['fmrifreq'], pcadata), prewindow=True, detrendorder=optiondict['detrendorder'])
        thepxcorr = pearsonr(filteredavg, filteredpca)[0]
        print('pca/avg correlation = ', thepxcorr)
        if thepxcorr > 0.0:
            outputdata = 1.0 * pcadata
        else:
            outputdata = -1.0 * pcadata
    elif optiondict['refinetype'] == 'weighted_average':
        print('performing weighted averaging refinement')
        outputdata = np.nan_to_num(averagedata / weightsum)
    else:
        print('performing unweighted averaging refinement')
        outputdata = averagedata

    if optiondict['cleanrefined']:
        thefit, R = tide_fit.mlregress(averagediscard, averagedata)
        fitcoff = rt_floatset(thefit[0, 1])
        datatoremove = rt_floatset(fitcoff * averagediscard)
        outputdata -= datatoremove
    print()
    print('Timeshift applied to ' + str(int(volumetotal)) + ' voxels, ' + str(len(validlist)) + ' used for refinement:',
          '\n	', locationfails, ' locationfails',
          '\n	', ampfails, ' ampfails',
          '\n	', lagfails, ' lagfails',
          '\n	', sigmafails, ' sigmafails')

    if optiondict['psdfilter']:
        outputdata = tide_filt.transferfuncfilt(outputdata, snr)

    # garbage collect
    collected = gc.collect()
    print("Garbage collector: collected %d objects." % collected)

    return volumetotal, outputdata, refinemask
