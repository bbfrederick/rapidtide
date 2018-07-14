#!/usr/bin/env python
#
#   Copyright 2016 Blaise Frederick
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

import time
import getopt
import platform
import bisect
import warnings
import os
import sys
import gc
import multiprocessing as mp
import resource

import rapidtide.miscmath as tide_math
import rapidtide.stats as tide_stats
import rapidtide.util as tide_util
import rapidtide.io as tide_io
import rapidtide.filter as tide_filt
import rapidtide.fit as tide_fit
import rapidtide.resample as tide_resample
import rapidtide.correlate as tide_corr

from sklearn.decomposition import FastICA, PCA
from matplotlib.pyplot import figure, plot, show
import numpy as np
from statsmodels.tsa.stattools import pacf_yw
from scipy.stats.stats import pearsonr
from scipy.signal import welch
from scipy import ndimage

try:
    from memory_profiler import profile

    memprofilerexists = True
except ImportError:
    memprofilerexists = False


def conditionalprofile():
    def resdec(f):
        if memprofilerexists:
            return profile(f)
        return f

    return resdec


global rt_floatset, rt_floattype


@conditionalprofile()
def memcheckpoint(message):
    print(message)


def startendcheck(timepoints, startpoint, endpoint):
    if startpoint > timepoints - 1:
        print('startpoint is too large (maximum is ', timepoints - 1, ')')
        sys.exit()
    if startpoint < 0:
        realstart = 0
        print('startpoint set to minimum, (0)')
    else:
        realstart = startpoint
        print('startpoint set to ', startpoint)
    if endpoint > timepoints - 1:
        realend = timepoints - 1
        print('endppoint set to maximum, (', timepoints - 1, ')')
    else:
        realend = endpoint
        print('endpoint set to ', endpoint)
    if realstart >= realend:
        print('endpoint (', realend, ') must be greater than startpoint (', realstart, ')')
        sys.exit()
    return realstart, realend


def procOneNullCorrelation(iteration, indata, ncprefilter, oversampfreq, corrscale, corrorigin, lagmininpts,
                           lagmaxinpts, optiondict):
    # make a shuffled copy of the regressors
    shuffleddata = np.random.permutation(indata)

    # crosscorrelate with original
    thexcorr, dummy = onecorrelation(shuffleddata, oversampfreq, corrorigin, lagmininpts, lagmaxinpts, ncprefilter,
                                     indata,
                                     optiondict)

    # fit the correlation
    maxindex, maxlag, maxval, maxsigma, maskval, failreason = \
        onecorrfit(thexcorr, corrscale[corrorigin - lagmininpts:corrorigin + lagmaxinpts],
                   optiondict)

    return maxval


def getNullDistributionData(indata, corrscale, ncprefilter, oversampfreq, corrorigin, lagmininpts, lagmaxinpts,
                            optiondict):
    if optiondict['multiproc']:
        # define the consumer function here so it inherits most of the arguments
        def nullCorrelation_consumer(inQ, outQ):
            while True:
                try:
                    # get a new message
                    val = inQ.get()

                    # this is the 'TERM' signal
                    if val is None:
                        break

                    # process and send the data
                    outQ.put(procOneNullCorrelation(val, indata, ncprefilter, oversampfreq, corrscale, corrorigin,
                                                    lagmininpts, lagmaxinpts, optiondict))

                except Exception as e:
                    print("error!", e)
                    break

        # initialize the workers and the queues
        n_workers = optiondict['nprocs']
        inQ = mp.Queue()
        outQ = mp.Queue()
        workers = [mp.Process(target=nullCorrelation_consumer, args=(inQ, outQ)) for i in range(n_workers)]
        for i, w in enumerate(workers):
            w.start()

        # pack the data and send to workers
        data_in = []
        for d in range(optiondict['numestreps']):
            data_in.append(d)
        print('processing', len(data_in), 'correlations with', n_workers, 'processes')
        data_out = process_data(data_in, inQ, outQ, showprogressbar=optiondict['showprogressbar'],
                                chunksize=optiondict['mp_chunksize'])

        # shut down workers
        for i in range(n_workers):
            inQ.put(None)
        for w in workers:
            w.terminate()
            w.join()

        # unpack the data
        volumetotal = 0
        corrlist = np.asarray(data_out, dtype=rt_floattype)
    else:
        corrlist = np.zeros((optiondict['numestreps']), dtype=rt_floattype)

        for i in range(0, optiondict['numestreps']):
            # make a shuffled copy of the regressors
            shuffleddata = np.random.permutation(indata)

            # crosscorrelate with original
            thexcorr, dummy = onecorrelation(shuffleddata, oversampfreq, corrorigin, lagmininpts, lagmaxinpts,
                                             ncprefilter,
                                             indata,
                                             optiondict)

            # fit the correlation
            maxindex, maxlag, maxval, maxsigma, maskval, failreason = \
                onecorrfit(thexcorr, corrscale[corrorigin - lagmininpts:corrorigin + lagmaxinpts],
                           optiondict)

            # find and tabulate correlation coefficient at optimal lag
            corrlist[i] = maxval

            # progress
            if optiondict['showprogressbar']:
                tide_util.progressbar(i + 1, optiondict['numestreps'], label='Percent complete')

        # jump to line after progress bar
        print()

    # return the distribution data
    numnonzero = len(np.where(corrlist != 0.0)[0])
    print(numnonzero, 'non-zero correlations out of', len(corrlist), '(', 100.0 * numnonzero / len(corrlist), '%)')
    return corrlist


def onecorrelation(thetc, oversampfreq, corrorigin, lagmininpts, lagmaxinpts, ncprefilter, referencetc, optiondict):
    thetc_classfilter = ncprefilter.apply(oversampfreq, thetc)
    thetc = thetc_classfilter

    # prepare timecourse by normalizing, detrending, and applying a window function
    preppedtc = tide_math.corrnormalize(thetc, optiondict['usewindowfunc'], optiondict['dodetrend'],
                                        windowfunc=optiondict['windowfunc'])

    # now actually do the correlation
    thexcorr = tide_corr.fastcorrelate(preppedtc, referencetc, usefft=True, weighting=optiondict['corrweighting'])

    # find the global maximum value
    theglobalmax = np.argmax(thexcorr)

    return thexcorr[corrorigin - lagmininpts:corrorigin + lagmaxinpts], theglobalmax


def procOneVoxelCorrelation(vox, thetc, optiondict, fmri_x, fmritc, os_fmri_x, oversampfreq,
                            corrorigin, lagmininpts, lagmaxinpts, ncprefilter, referencetc):
    global rt_floattype
    if optiondict['oversampfactor'] >= 1:
        thetc[:] = tide_resample.doresample(fmri_x, fmritc, os_fmri_x, method=optiondict['interptype'])
    else:
        thetc[:] = fmritc
    thexcorr, theglobalmax = onecorrelation(thetc, oversampfreq, corrorigin, lagmininpts, lagmaxinpts, ncprefilter,
                                            referencetc, optiondict)
    return vox, np.mean(thetc), thexcorr, theglobalmax


def process_data(data_in, inQ, outQ, showprogressbar=True, reportstep=1000, chunksize=10000):
    # send pos/data to workers
    data_out = []
    totalnum = len(data_in)
    numchunks = int(totalnum // chunksize)
    remainder = totalnum - numchunks * chunksize
    if showprogressbar:
        tide_util.progressbar(0, totalnum, label="Percent complete")

    # process all of the complete chunks
    for thechunk in range(numchunks):
        # queue the chunk
        for i, dat in enumerate(data_in[thechunk * chunksize:(thechunk + 1) * chunksize]):
            inQ.put(dat)
        offset = thechunk * chunksize

        # retrieve the chunk
        numreturned = 0
        while True:
            ret = outQ.get()
            if ret is not None:
                data_out.append(ret)
            numreturned += 1
            if (((numreturned + offset + 1) % reportstep) == 0) and showprogressbar:
                tide_util.progressbar(numreturned + offset + 1, totalnum, label="Percent complete")
            if numreturned > chunksize - 1:
                break

    # queue the remainder
    for i, dat in enumerate(data_in[numchunks * chunksize:numchunks * chunksize + remainder]):
        inQ.put(dat)
    numreturned = 0
    offset = numchunks * chunksize

    # retrieve the remainder
    while True:
        ret = outQ.get()
        if ret is not None:
            data_out.append(ret)
        numreturned += 1
        if (((numreturned + offset + 1) % reportstep) == 0) and showprogressbar:
            tide_util.progressbar(numreturned + offset + 1, totalnum, label="Percent complete")
        if numreturned > remainder - 1:
            break
    if showprogressbar:
        tide_util.progressbar(totalnum, totalnum, label="Percent complete")
    print()

    return data_out


def correlationpass(fmridata, fmrifftdata, referencetc,
                    fmri_x, os_fmri_x,
                    tr,
                    corrorigin, lagmininpts, lagmaxinpts,
                    corrmask, corrout, meanval,
                    ncprefilter,
                    optiondict):
    oversampfreq = optiondict['oversampfactor'] / tr
    inputshape = np.shape(fmridata)
    volumetotal = 0
    reportstep = 1000
    thetc = np.zeros(np.shape(os_fmri_x), dtype=rt_floattype)
    theglobalmaxlist = []
    if optiondict['multiproc']:
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
                    outQ.put(procOneVoxelCorrelation(val, thetc, optiondict, fmri_x, fmridata[val, :], os_fmri_x,
                                                     oversampfreq,
                                                     corrorigin, lagmininpts, lagmaxinpts, ncprefilter, referencetc))

                except Exception as e:
                    print("error!", e)
                    break

        # initialize the workers and the queues
        n_workers = optiondict['nprocs']
        inQ = mp.Queue()
        outQ = mp.Queue()
        workers = [mp.Process(target=correlation_consumer, args=(inQ, outQ)) for i in range(n_workers)]
        for i, w in enumerate(workers):
            w.start()

        # pack the data and send to workers
        data_in = []
        for d in range(inputshape[0]):
            data_in.append(d)
        print('processing', len(data_in), 'voxels with', n_workers, 'processes')
        data_out = process_data(data_in, inQ, outQ, showprogressbar=optiondict['showprogressbar'],
                                chunksize=optiondict['mp_chunksize'])

        # shut down workers
        for i in range(n_workers):
            inQ.put(None)
        for w in workers:
            w.terminate()
            w.join()

        # unpack the data
        volumetotal = 0
        for voxel in data_out:
            # corrmask[voxel[0]] = 1
            meanval[voxel[0]] = voxel[1]
            corrout[voxel[0], :] = voxel[2]
            theglobalmaxlist.append(voxel[3] + 0)
            volumetotal += 1
        data_out = []
    else:
        for vox in range(0, inputshape[0]):
            if (vox % reportstep == 0 or vox == inputshape[0] - 1) and optiondict['showprogressbar']:
                tide_util.progressbar(vox + 1, inputshape[0], label='Percent complete')
            dummy, meanval[vox], corrout[vox, :], theglobalmax = procOneVoxelCorrelation(vox, thetc, optiondict, fmri_x,
                                                                                         fmridata[vox, :], os_fmri_x,
                                                                                         oversampfreq,
                                                                                         corrorigin, lagmininpts,
                                                                                         lagmaxinpts,
                                                                                         ncprefilter, referencetc)
            theglobalmaxlist.append(theglobalmax + 0)
            volumetotal += 1
    print('\nCorrelation performed on ' + str(volumetotal) + ' voxels')

    # garbage collect
    collected = gc.collect()
    print("Garbage collector: collected %d objects." % collected)

    return volumetotal, theglobalmaxlist


def onecorrfit(thetc, corrscale, optiondict, displayplots=False, initiallag=None):
    if initiallag is not None:
        maxguess = initiallag
        useguess = True
        widthlimit = optiondict['despeckle_thresh']
    else:
        maxguess = 0.0
        useguess = False
        widthlimit = optiondict['widthlimit']

    if optiondict['bipolar']:
        if max(thetc) < -1.0 * min(thetc):
            flipfac = rt_floatset(-1.0)
        else:
            flipfac = rt_floatset(1.0)
    else:
        flipfac = rt_floatset(1.0)
    if not optiondict['fixdelay']:
        if optiondict['findmaxtype'] == 'gauss':
            maxindex, maxlag, maxval, maxsigma, maskval, failreason, peakstart, peakend = tide_fit.findmaxlag_gauss(
                corrscale,
                flipfac * thetc,
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
                zerooutbadfit=optiondict['zerooutbadfit'],
                lagmod=optiondict['lagmod'],
                displayplots=displayplots)
        else:
            maxindex, maxlag, maxval, maxsigma, maskval, failreason, peakstart, peakend = tide_fit.findmaxlag_quad(
                corrscale,
                flipfac * thetc,
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
                zerooutbadfit=optiondict['zerooutbadfit'],
                lagmod=optiondict['lagmod'],
                displayplots=displayplots)
        maxval *= flipfac
    else:
        # do something different
        failreason = np.int16(0)
        maxlag = rt_floatset(optiondict['fixeddelayvalue'])
        maxindex = np.int16(bisect.bisect_left(corrscale, optiondict['fixeddelayvalue']))
        maxval = rt_floatset(flipfac * thetc[maxindex])
        maxsigma = rt_floatset(1.0)
        maskval = np.uint16(1)

    return maxindex, maxlag, maxval, maxsigma, maskval, failreason


def procOneVoxelFitcorr(vox, corrtc, corrscale, genlagtc, initial_fmri_x, optiondict, displayplots, initiallag=None):
    if optiondict['slicetimes'] is not None:
        sliceoffsettime = optiondict['slicetimes'][vox % slicesize]
    else:
        sliceoffsettime = 0.0
    maxindex, maxlag, maxval, maxsigma, maskval, failreason = onecorrfit(corrtc, corrscale,
                                                                         optiondict, displayplots=displayplots,
                                                                         initiallag=initiallag)

    if maxval > 0.3:
        displayplots = False

    # question - should maxlag be added or subtracted?  As of 10/18, it is subtracted
    #  potential answer - tried adding, results are terrible.
    thelagtc = rt_floatset(genlagtc.yfromx(initial_fmri_x - maxlag))

    # now tuck everything away in the appropriate output array
    volumetotalinc = 0
    if (maskval == 0) and optiondict['zerooutbadfit']:
        thetime = rt_floatset(0.0)
        thestrength = rt_floatset(0.0)
        thesigma = rt_floatset(0.0)
        thegaussout = 0.0 * corrtc
        theR2 = rt_floatset(0.0)
    else:
        volumetotalinc = 1
        thetime = rt_floatset(np.fmod(maxlag, optiondict['lagmod']))
        thestrength = rt_floatset(maxval)
        thesigma = rt_floatset(maxsigma)
        if (not optiondict['fixdelay']) and (maxsigma != 0.0):
            thegaussout = rt_floatset(tide_fit.gauss_eval(corrscale, [maxval, maxlag, maxsigma]))
        else:
            thegaussout = rt_floatset(0.0)
        theR2 = rt_floatset(thestrength * thestrength)

    return vox, volumetotalinc, thelagtc, thetime, thestrength, thesigma, thegaussout, theR2, maskval, failreason


def fitcorr(genlagtc, initial_fmri_x, lagtc, slicesize,
            corrscale, lagmask, lagtimes, lagstrengths, lagsigma, corrout, meanval, gaussout,
            R2, optiondict, initiallags=None):
    displayplots = False
    inputshape = np.shape(corrout)
    volumetotal, ampfails, lagfails, widthfails, edgefails, fitfails = 0, 0, 0, 0, 0, 0
    reportstep = 1000
    zerolagtc = rt_floatset(genlagtc.yfromx(initial_fmri_x))
    sliceoffsettime = 0.0

    if optiondict['multiproc']:
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
                        outQ.put(
                            procOneVoxelFitcorr(val, corrout[val, :], corrscale, genlagtc, initial_fmri_x, optiondict,
                                                displayplots))
                    else:
                        outQ.put(
                            procOneVoxelFitcorr(val, corrout[val, :], corrscale, genlagtc, initial_fmri_x, optiondict,
                                                displayplots, initiallag=initiallags[val]))

                except Exception as e:
                    print("error!", e)
                    break

        # initialize the workers and the queues
        n_workers = optiondict['nprocs']
        inQ = mp.Queue()
        outQ = mp.Queue()
        workers = [mp.Process(target=fitcorr_consumer, args=(inQ, outQ)) for i in range(n_workers)]
        for i, w in enumerate(workers):
            w.start()

        # pack the data and send to workers
        data_in = []
        for d in range(inputshape[0]):
            if initiallags is None:
                data_in.append(d)
            else:
                if initiallags[d] > -1000000.0:
                    data_in.append(d)
        print('processing', len(data_in), 'voxels with', n_workers, 'processes')
        data_out = process_data(data_in, inQ, outQ, showprogressbar=optiondict['showprogressbar'],
                                chunksize=optiondict['mp_chunksize'])

        # shut down workers
        for i in range(n_workers):
            inQ.put(None)
        for w in workers:
            w.terminate()
            w.join()

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
            if initiallags is None:
                dummy, volumetotalinc, lagtc[vox, :], lagtimes[vox], lagstrengths[vox], lagsigma[vox], gaussout[vox, :], \
                R2[
                    vox], lagmask[vox], failreason = \
                    procOneVoxelFitcorr(vox, corrout[vox, :], corrscale, genlagtc, initial_fmri_x, optiondict,
                                        displayplots)
                volumetotal += volumetotalinc
            else:
                if initiallags[vox] != 0.0:
                    dummy, volumetotalinc, lagtc[vox, :], lagtimes[vox], lagstrengths[vox], lagsigma[vox], gaussout[vox,
                                                                                                           :], R2[
                        vox], lagmask[vox], failreason = \
                        procOneVoxelFitcorr(vox, corrout[vox, :], corrscale, genlagtc, initial_fmri_x, optiondict,
                                            displayplots, initiallags[vox])
                    volumetotal += volumetotalinc
    print('\nCorrelation fitted in ' + str(volumetotal) + ' voxels')
    print('\tampfails=', ampfails, ' lagfails=', lagfails, ' widthfail=', widthfails, ' edgefail=', edgefails,
          ' fitfail=', fitfails)

    # garbage collect
    collected = gc.collect()
    print("Garbage collector: collected %d objects." % collected)

    return volumetotal


def procOneVoxelTimeShift(vox, fmritc, optiondict, lagstrength, R2val, lagtime, padtrs, fmritr, theprefilter):
    if optiondict['refineprenorm'] == 'mean':
        thedivisor = np.mean(fmritc)
    elif optiondict['refineprenorm'] == 'var':
        thedivisor = np.var(fmritc)
    elif optiondict['refineprenorm'] == 'std':
        thedivisor = np.std(fmritc)
    elif optiondict['refineprenorm'] == 'invlag':
        if lagtime < optiondict['lagmaxthresh']:
            thedivisor = optiondict['lagmaxthresh'] - lagtime
        else:
            thedivisor = 0.0
    else:
        thedivisor = 1.0
    if thedivisor != 0.0:
        normfac = 1.0 / thedivisor
    else:
        normfac = 0.0

    if optiondict['refineweighting'] == 'R':
        thisweight = lagstrength
    elif optiondict['refineweighting'] == 'R2':
        thisweight = R2val
    else:
        thisweight = 1.0
    if optiondict['dodetrend']:
        normtc = tide_fit.detrend(fmritc * normfac * thisweight, demean=True)
    else:
        normtc = fmritc * normfac * thisweight
    shifttr = -(-optiondict['offsettime'] + lagtime) / fmritr  # lagtime is in seconds
    [shiftedtc, weights, paddedshiftedtc, paddedweights] = tide_resample.timeshift(normtc, shifttr, padtrs)
    if optiondict['filterbeforePCA']:
        outtc = theprefilter.apply(optiondict['fmrifreq'], shiftedtc)
        outweights = theprefilter.apply(optiondict['fmrifreq'], weights)
    else:
        outtc = 1.0 * shiftedtc
        outweights = 1.0 * weights
    if optiondict['psdfilter']:
        freqs, psd = welch(tide_math.corrnormalize(shiftedtc, True, True), fmritr, scaling='spectrum', window='hamming',
                           return_onesided=False, nperseg=len(shiftedtc))
        return vox, outtc, outweights, np.sqrt(psd)
    else:
        return vox, outtc, outweights, None


def refineregressor(reference, fmridata, fmritr, shiftedtcs, weights, passnum, lagstrengths, lagtimes,
                    lagsigma, R2,
                    theprefilter, optiondict, padtrs=60, includemask=None, excludemask=None):
    # print('entering refineregressor with padtrs=', padtrs)
    inputshape = np.shape(fmridata)
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
        locationmask = locationmask * excludemask
    print('location mask created')

    # first generate the refine mask
    locationfails = np.sum(1 - locationmask)
    ampfails = np.sum(1 - ampmask)
    lagfails = np.sum(1 - delaymask)
    sigmafails = np.sum(1 - sigmamask)
    maskarray = locationmask * ampmask * delaymask * sigmamask
    volumetotal = np.sum(maskarray)
    reportstep = 1000

    # timeshift the valid voxels
    if optiondict['multiproc']:
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
                    outQ.put(procOneVoxelTimeShift(val, fmridata[val, :], optiondict,
                                                   lagstrengths[val], R2[val], lagtimes[val], padtrs, fmritr,
                                                   theprefilter))

                except Exception as e:
                    print("error!", e)
                    break

        # initialize the workers and the queues
        n_workers = optiondict['nprocs']
        inQ = mp.Queue()
        outQ = mp.Queue()
        workers = [mp.Process(target=timeshift_consumer, args=(inQ, outQ)) for i in range(n_workers)]
        for i, w in enumerate(workers):
            w.start()

        # pack the data and send to workers
        data_in = []
        for d in range(inputshape[0]):
            if (maskarray[d] > 0) or optiondict['shiftall']:
                data_in.append(d)
        print('processing', len(data_in), 'voxels with', n_workers, 'processes')
        data_out = process_data(data_in, inQ, outQ, showprogressbar=optiondict['showprogressbar'],
                                chunksize=optiondict['mp_chunksize'])

        # shut down workers
        for i in range(n_workers):
            inQ.put(None)
        for w in workers:
            w.terminate()
            w.join()

        # unpack the data
        psdlist = []
        for voxel in data_out:
            shiftedtcs[voxel[0], :] = voxel[1]
            weights[voxel[0], :] = voxel[2]
            if optiondict['psdfilter']:
                psdlist.append(voxel[3])
        data_out = []
    else:
        psdlist = []
        for vox in range(0, inputshape[0]):
            if (vox % reportstep == 0 or vox == inputshape[0] - 1) and optiondict['showprogressbar']:
                tide_util.progressbar(vox + 1, inputshape[0], label='Percent complete (timeshifting)')
            if (maskarray[vox] > 0) or optiondict['shiftall']:
                retvals = procOneVoxelTimeShift(vox, fmridata[vox, :], optiondict, lagstrengths[vox], R2[vox],
                                                lagtimes[vox], padtrs, fmritr, theprefilter)
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
        # fig = figure()
        # ax = fig.add_subplot(111)
        # ax.set_title('Average and stedev of PSD')
        # plot(averagepsd)
        # plot(stdpsd)
        # show()
        # fig = figure()
        # ax = fig.add_subplot(111)
        # ax.set_title('SNR')
        # plot(snr)
        # show()

    # now generate the refined timecourse(s)
    validlist = np.where(maskarray > 0)[0]
    refinevoxels = shiftedtcs[validlist]
    refineweights = weights[validlist]
    weightsum = np.sum(refineweights, axis=0) / volumetotal
    averagedata = np.sum(refinevoxels, axis=0) / volumetotal
    if optiondict['shiftall']:
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
                dispersioncalcout[lagnum, :] = tide_math.corrnormalize(np.mean(shiftedtcs[inlagrange], axis=0), False,
                                                                       True,
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
        filteredavg = tide_math.corrnormalize(theprefilter.apply(optiondict['fmrifreq'], averagedata), True, True)
        filteredica = tide_math.corrnormalize(theprefilter.apply(optiondict['fmrifreq'], icadata), True, True)
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
        filteredavg = tide_math.corrnormalize(theprefilter.apply(optiondict['fmrifreq'], averagedata), True, True)
        filteredpca = tide_math.corrnormalize(theprefilter.apply(optiondict['fmrifreq'], pcadata), True, True)
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
    print(str(
        volumetotal) + ' voxels used for refinement:',
          '\n	', locationfails, ' locationfails',
          '\n	', ampfails, ' ampfails',
          '\n	', lagfails, ' lagfails',
          '\n	', sigmafails, ' sigmafails')

    if optiondict['psdfilter']:
        outputdata = tide_filt.xfuncfilt(outputdata, snr)

    # garbage collect
    collected = gc.collect()
    print("Garbage collector: collected %d objects." % collected)

    return volumetotal, outputdata, maskarray


def procOneVoxelWiener(vox, lagtc, inittc):
    thefit, R = tide_fit.mlregress(lagtc, inittc)
    fitcoff = rt_floatset(thefit[0, 1])
    datatoremove = rt_floatset(fitcoff * lagtc)
    return vox, rt_floatset(thefit[0, 0]), rt_floatset(R), rt_floatset(R * R), fitcoff, \
           rt_floatset(thefit[0, 1] / thefit[0, 0]), datatoremove, rt_floatset(inittc - datatoremove)


def wienerpass(numspatiallocs, reportstep, fmri_data, threshval, lagtc, optiondict, meanvalue, rvalue, r2value, fitcoff,
               fitNorm, datatoremove, filtereddata):
    if optiondict['multiproc']:
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
                    outQ.put(procOneVoxelWiener(val, lagtc[val, :], fmri_data[val, optiondict['addedskip']:]))

                except Exception as e:
                    print("error!", e)
                    break

        # initialize the workers and the queues
        n_workers = optiondict['nprocs']
        inQ = mp.Queue()
        outQ = mp.Queue()
        workers = [mp.Process(target=Wiener_consumer, args=(inQ, outQ)) for i in range(n_workers)]
        for i, w in enumerate(workers):
            w.start()

        # pack the data and send to workers
        data_in = []
        for d in range(numspatiallocs):
            if np.mean(fmri_data[d, optiondict['addedskip']:]) >= threshval:
                data_in.append(d)
        print('processing', len(data_in), 'voxels with', n_workers, 'processes')
        data_out = process_data(data_in, inQ, outQ, showprogressbar=optiondict['showprogressbar'],
                                chunksize=optiondict['mp_chunksize'])

        # shut down workers
        for i in range(n_workers):
            inQ.put(None)
        for w in workers:
            w.terminate()
            w.join()

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
        for vox in range(0, numspatiallocs):
            if (vox % reportstep == 0 or vox == numspatiallocs - 1) and optiondict['showprogressbar']:
                tide_util.progressbar(vox + 1, numspatiallocs, label='Percent complete')
            inittc = fmri_data[vox, optiondict['addedskip']:].copy()
            if np.mean(inittc) >= threshval:
                dummy, meanvalue[vox], rvalue[vox], r2value[vox], fitcoff[vox], fitNorm[vox], datatoremove[vox], \
                filtereddata[vox] = procOneVoxelWiener(vox, lagtc[vox, :], inittc)
                volumetotal += 1

    return volumetotal


def procOneVoxelGLM(vox, lagtc, inittc):
    thefit, R = tide_fit.mlregress(lagtc, inittc)
    fitcoff = rt_floatset(thefit[0, 1])
    datatoremove = rt_floatset(fitcoff * lagtc)
    return vox, rt_floatset(thefit[0, 0]), rt_floatset(R), rt_floatset(R * R), fitcoff, \
           rt_floatset(thefit[0, 1] / thefit[0, 0]), datatoremove, rt_floatset(inittc - datatoremove)


def glmpass(numspatiallocs, reportstep, fmri_data, threshval, lagtc, optiondict, meanvalue, rvalue, r2value, fitcoff,
            fitNorm, datatoremove, filtereddata):
    if optiondict['multiproc']:
        # define the consumer function here so it inherits most of the arguments
        def GLM_consumer(inQ, outQ):
            while True:
                try:
                    # get a new message
                    val = inQ.get()

                    # this is the 'TERM' signal
                    if val is None:
                        break

                    # process and send the data
                    outQ.put(procOneVoxelGLM(val, lagtc[val, :], fmri_data[val, optiondict['addedskip']:]))

                except Exception as e:
                    print("error!", e)
                    break

        # initialize the workers and the queues
        n_workers = optiondict['nprocs']
        inQ = mp.Queue()
        outQ = mp.Queue()
        workers = [mp.Process(target=GLM_consumer, args=(inQ, outQ)) for i in range(n_workers)]
        for i, w in enumerate(workers):
            w.start()

        # pack the data and send to workers
        data_in = []
        for d in range(numspatiallocs):
            if (np.mean(fmri_data[d, optiondict['addedskip']:]) >= threshval) or optiondict['nothresh']:
                data_in.append(d)
        print('processing', len(data_in), 'voxels with', n_workers, 'processes')
        data_out = process_data(data_in, inQ, outQ, showprogressbar=optiondict['showprogressbar'],
                                chunksize=optiondict['mp_chunksize'])

        # shut down workers
        for i in range(n_workers):
            inQ.put(None)
        for w in workers:
            w.terminate()
            w.join()

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
        for vox in range(0, numspatiallocs):
            if (vox % reportstep == 0 or vox == numspatiallocs - 1) and optiondict['showprogressbar']:
                tide_util.progressbar(vox + 1, numspatiallocs, label='Percent complete')
            inittc = fmri_data[vox, optiondict['addedskip']:].copy()
            if np.mean(inittc) >= threshval:
                dummy, meanvalue[vox], rvalue[vox], r2value[vox], fitcoff[vox], fitNorm[vox], datatoremove[vox], \
                filtereddata[vox] = \
                    procOneVoxelGLM(vox, lagtc[vox, :], inittc)
                volumetotal += 1
                # if optiondict['doprewhiten']:
                #    arcoffs[vox, :] = pacf_yw(thefilttc, nlags=optiondict['armodelorder'])[1:]
                #    prewhiteneddata[vox, :] = rt_floatset(prewhiten(inittc, arcoffs[vox, :]))

    return volumetotal


def maketmask(filename, timeaxis, maskvector, debug=False):
    inputdata = tide_io.readvecs(filename)
    theshape = np.shape(inputdata)
    if theshape[0] == 1:
        # this is simply a vector, one per TR.  If the value is nonzero, include the point, otherwise don't
        if theshape[1] == len(timeaxis):
            maskvector = np.where(inputdata[0, :] > 0.0, 1.0, 0.0)
        else:
            print('tmask length does not match fmri data')
            sys.exit(1)
    else:
        maskvector *= 0.0
        for idx in range(0, theshape[1]):
            starttime = inputdata[0, idx]
            endtime = starttime + inputdata[1, idx]
            startindex = np.max((bisect.bisect_left(timeaxis, starttime), 0))
            endindex = np.min((bisect.bisect_right(timeaxis, endtime), len(maskvector) - 1))
            maskvector[startindex:endindex] = 1.0
            print(starttime, startindex, endtime, endindex)
    if debug:
        fig = figure()
        ax = fig.add_subplot(111)
        ax.set_title('temporal mask vector')
        plot(timeaxis, maskvector)
        show()
    return maskvector


def prewhiten(indata, arcoffs):
    pwdata = 1.0 * indata
    for i in range(0, len(arcoffs)):
        pwdata[(i + 1):] = pwdata[(i + 1):] + arcoffs[i] * indata[:(-1 - i)]
    return pwdata


def numpy2shared(inarray, thetype):
    thesize = inarray.size
    theshape = inarray.shape
    if thetype == np.float64:
        inarray_shared = mp.RawArray('d', inarray.reshape((thesize)))
    else:
        inarray_shared = mp.RawArray('f', inarray.reshape((thesize)))
    inarray = np.frombuffer(inarray_shared, dtype=thetype, count=thesize)
    inarray.shape = theshape
    return inarray, inarray_shared, theshape


def allocshared(theshape, thetype):
    thesize = int(1)
    for element in theshape:
        thesize *= int(element)
    if thetype == np.float64:
        outarray_shared = mp.RawArray('d', thesize)
    else:
        outarray_shared = mp.RawArray('f', thesize)
    outarray = np.frombuffer(outarray_shared, dtype=thetype, count=thesize)
    outarray.shape = theshape
    return outarray, outarray_shared, theshape


def logmem(msg, file=None):
    if msg is None:
        logline = ','.join([
            '',
            'Self Max RSS',
            'Self Shared Mem',
            'Self Unshared Mem',
            'Self Unshared Stack',
            'Self Non IO Page Fault'
            'Self IO Page Fault'
            'Self Swap Out',
            'Children Max RSS',
            'Children Shared Mem',
            'Children Unshared Mem',
            'Children Unshared Stack',
            'Children Non IO Page Fault'
            'Children IO Page Fault'
            'Children Swap Out'])
    else:
        rcusage = resource.getrusage(resource.RUSAGE_SELF)
        outvals = [msg]
        outvals.append(str(rcusage.ru_maxrss))
        outvals.append(str(rcusage.ru_ixrss))
        outvals.append(str(rcusage.ru_idrss))
        outvals.append(str(rcusage.ru_isrss))
        outvals.append(str(rcusage.ru_minflt))
        outvals.append(str(rcusage.ru_majflt))
        outvals.append(str(rcusage.ru_nswap))
        rcusage = resource.getrusage(resource.RUSAGE_CHILDREN)
        outvals.append(str(rcusage.ru_maxrss))
        outvals.append(str(rcusage.ru_ixrss))
        outvals.append(str(rcusage.ru_idrss))
        outvals.append(str(rcusage.ru_isrss))
        outvals.append(str(rcusage.ru_minflt))
        outvals.append(str(rcusage.ru_majflt))
        outvals.append(str(rcusage.ru_nswap))
        logline = ','.join(outvals)
    if file is None:
        print(logline)
    else:
        file.writelines(logline + "\n")


def getglobalsignal(indata, optiondict, includemask=None, excludemask=None):
    # mask to interesting voxels
    if optiondict['globalmaskmethod'] == 'mean':
        themask = tide_stats.makemask(np.mean(indata, axis=1), optiondict['corrmaskthreshpct'])
    elif optiondict['globalmaskmethod'] == 'variance':
        themask = tide_stats.makemask(np.var(indata, axis=1), optiondict['corrmaskthreshpct'])
    if optiondict['nothresh']:
        themask *= 0
        themask += 1
    if includemask is not None:
        themask = themask * includemask
    if excludemask is not None:
        themask = themask * excludemask

    # add up all the voxels
    globalmean = rt_floatset(indata[0, :])
    thesize = np.shape(themask)
    numvoxelsused = 0
    for vox in range(0, thesize[0]):
        if themask[vox] > 0.0:
            numvoxelsused += 1
            if optiondict['meanscaleglobal']:
                themean = np.mean(indata[vox, :])
                if themean != 0.0:
                    globalmean = globalmean + indata[vox, :] / themean - 1.0
            else:
                globalmean = globalmean + indata[vox, :]
    print()
    print('used ', numvoxelsused, ' voxels to calculate global mean signal')
    return tide_math.stdnormalize(globalmean)


def rapidtide2(in_file, prefix, venousrefine=False, nirs=False,
               realtr='auto', antialias=True, invertregressor=False,
               interptype='univariate', offsettime=None,
               butterorder=None, arbvec=None, filtertype='arb',
               numestreps=10000, dosighistfit=True,
               windowfunc='hamming', gausssigma=0.,
               useglobalref=False, meanscaleglobal=False,
               slicetimes=None, preprocskip=0, nothresh=True,
               oversampfactor=2, regressorfile=None, inputfreq=1.,
               inputstarttime=0., corrweighting='none',
               dodetrend=True, corrmaskthreshpct=1.,
               corrmaskname=None, fixeddelayvalue=None,
               lag_extrema=(-30.0, 30.0), widthlimit=100.,
               bipolar=False, zerooutbadfit=True, findmaxtype='gauss',
               despeckle_passes=0, despeckle_thresh=5,
               refineprenorm='mean', refineweighting='R2', passes=1,
               includemaskname=None, excludemaskname=None,
               lagminthresh=0.5, lagmaxthresh=5., ampthresh=0.3,
               sigmathresh=100., refineoffset=False, psdfilter=False,
               lagmaskside='both', refinetype='avg',
               savelagregressors=True, savecorrtimes=False,
               histlen=100, timerange=(-1, 10000000),
               glmsourcefile=None, doglmfilt=True,
               preservefiltering=False, showprogressbar=True,
               dodeconv=False, internalprecision='double',
               isgrayordinate=False, fakerun=False, displayplots=False,
               nonumba=False, sharedmem=True, memprofile=False,
               nprocs=1, debug=False, cleanrefined=False,
               dodispersioncalc=False, fix_autocorrelation=False,
               tmaskname=None, doprewhiten=False, saveprewhiten=False,
               armodelorder=1, offsettime_total=None,
               ampthreshfromsig=False, nohistzero=False,
               fixdelay=False, usebutterworthfilter=False):
    pass
