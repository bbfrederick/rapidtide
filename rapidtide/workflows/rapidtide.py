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
import multiprocessing as mp
import platform
import sys
import time
import warnings

import numpy as np
from matplotlib.pyplot import figure, plot, show
from scipy import ndimage

import rapidtide.correlate as tide_corr
import rapidtide.filter as tide_filt
import rapidtide.fit as tide_fit
import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math
import rapidtide.multiproc as tide_multiproc
import rapidtide.resample as tide_resample
import rapidtide.stats as tide_stats
import rapidtide.util as tide_util

import rapidtide.nullcorrpass as tide_nullcorr
import rapidtide.corrpass as tide_corrpass
import rapidtide.corrfit as tide_corrfit
import rapidtide.refine as tide_refine
import rapidtide.glmpass as tide_glmpass
import rapidtide.helper_classes as tide_classes
import rapidtide.wiener as tide_wiener

import copy

try:
    import mkl

    mklexists = True
except ImportError:
    mklexists = False

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


@conditionalprofile()
def memcheckpoint(message):
    print(message)


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


def numpy2shared(inarray, thetype):
    thesize = inarray.size
    theshape = inarray.shape
    if thetype == np.float64:
        inarray_shared = mp.RawArray('d', inarray.reshape(thesize))
    else:
        inarray_shared = mp.RawArray('f', inarray.reshape(thesize))
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


def readamask(maskfilename, nim_hdr, xsize, istext=False, valslist=None, maskname='the', verbose=False):
    if verbose:
        print('readamask called with filename:', maskfilename, 'vals:', valslist)
    if istext:
        maskarray = tide_io.readvecs(maskfilename).astype('int16')
        theshape = np.shape(maskarray)
        theincludexsize = theshape[0]
        if not theincludexsize == xsize:
            print('Dimensions of ' + maskname + ' mask do not match the fmri data - exiting')
            sys.exit()
    else:
        themask, maskarray, mask_hdr, maskdims, masksizes = tide_io.readfromnifti(maskfilename)
        if not tide_io.checkspacematch(mask_hdr, nim_hdr):
            print('Dimensions of ' + maskname + ' mask do not match the fmri data - exiting')
            sys.exit()
    if valslist is not None:
        tempmask = (0 * maskarray).astype('int16')
        for theval in valslist:
            if verbose:
                print('looking for voxels matching', theval)
            tempmask[np.where(np.fabs(maskarray - theval) < 0.1)] += 1
        maskarray = np.where(tempmask > 0, 1, 0)
    return maskarray


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
        themask = themask * (1 - excludemask)

    # add up all the voxels
    global rt_floatset, rt_floattype
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
    return tide_math.stdnormalize(globalmean), themask


def addmemprofiling(thefunc, memprofile, memfile, themessage):
    if memprofile:
        return profile(thefunc, precision=2)
    else:
        tide_util.logmem(themessage, file=memfile)
        return thefunc


def rapidtide_main(argparsingfunc):
    timings = [['Start', time.time(), None, None]]
    optiondict, theprefilter = argparsingfunc()

    fmrifilename = optiondict['in_file']
    outputname = optiondict['outputname']
    filename = optiondict['regressorfile']

    timings.append(['Argument parsing done', time.time(), None, None])

    # don't use shared memory if there is only one process
    if optiondict['nprocs'] == 1:
        optiondict['sharedmem'] = False
        print('running single process - disabled shared memory use')

    # disable numba now if we're going to do it (before any jits)
    if optiondict['nonumba']:
        tide_util.disablenumba()

    # set the internal precision
    global rt_floatset, rt_floattype
    if optiondict['internalprecision'] == 'double':
        print('setting internal precision to double')
        rt_floattype = 'float64'
        rt_floatset = np.float64
    else:
        print('setting internal precision to single')
        rt_floattype = 'float32'
        rt_floatset = np.float32

    # set the output precision
    if optiondict['outputprecision'] == 'double':
        print('setting output precision to double')
        rt_outfloattype = 'float64'
        rt_outfloatset = np.float64
    else:
        print('setting output precision to single')
        rt_outfloattype = 'float32'
        rt_outfloatset = np.float32

    # set set the number of worker processes if multiprocessing
    if optiondict['nprocs'] < 1:
        optiondict['nprocs'] = tide_multiproc.maxcpus()

    # set the number of MKL threads to use
    if mklexists:
        mkl.set_num_threads(optiondict['mklthreads'])

    # open up the memory usage file
    if not optiondict['memprofile']:
        memfile = open(outputname + '_memusage.csv', 'w')
        tide_util.logmem(None, file=memfile)

    # open the fmri datafile
    tide_util.logmem('before reading in fmri data', file=memfile)
    if tide_io.checkiftext(fmrifilename):
        print('input file is text - all I/O will be to text files')
        optiondict['textio'] = True
        if optiondict['gausssigma'] > 0.0:
            optiondict['gausssigma'] = 0.0
            print('gaussian spatial filter disabled for text input files')
    else:
        optiondict['textio'] = False

    if optiondict['textio']:
        nim_data = tide_io.readvecs(fmrifilename)
        theshape = np.shape(nim_data)
        xsize = theshape[0]
        ysize = 1
        numslices = 1
        fileiscifti = False
        timepoints = theshape[1]
        thesizes = [0, int(xsize), 1, 1, int(timepoints)]
        numspatiallocs = int(xsize)
        slicesize = numspatiallocs
    else:
        nim, nim_data, nim_hdr, thedims, thesizes = tide_io.readfromnifti(fmrifilename)
        if nim_hdr['intent_code'] == 3002:
            print('input file is CIFTI')
            optiondict['isgrayordinate'] = True
            fileiscifti = True
            timepoints = nim_data.shape[4]
            numspatiallocs = nim_data.shape[5]
            slicesize = numspatiallocs
            outsuffix3d = '.dscalar'
            outsuffix4d = '.dtseries'
        else:
            print('input file is NIFTI')
            fileiscifti = False
            xsize, ysize, numslices, timepoints = tide_io.parseniftidims(thedims)
            numspatiallocs = int(xsize) * int(ysize) * int(numslices)
            slicesize = numspatiallocs / int(numslices)
            outsuffix3d = ''
            outsuffix4d = ''
        xdim, ydim, slicethickness, tr = tide_io.parseniftisizes(thesizes)
    tide_util.logmem('after reading in fmri data', file=memfile)

    # correct some fields if necessary
    if optiondict['isgrayordinate']:
        fmritr = 0.72  # this is wrong and is a hack until I can parse CIFTI XML
    else:
        if optiondict['textio']:
            if optiondict['realtr'] <= 0.0:
                print('for text file data input, you must use the -t option to set the timestep')
                sys.exit()
        else:
            if nim_hdr.get_xyzt_units()[1] == 'msec':
                fmritr = thesizes[4] / 1000.0
            else:
                fmritr = thesizes[4]
    if optiondict['realtr'] > 0.0:
        fmritr = optiondict['realtr']

    # check to see if we need to adjust the oversample factor
    if optiondict['oversampfactor'] < 0:
        optiondict['oversampfactor'] = int(np.max([np.ceil(fmritr // 0.5), 1]))
        print('oversample factor set to', optiondict['oversampfactor'])

    oversamptr = fmritr / optiondict['oversampfactor']
    if optiondict['verbose']:
        print('fmri data: ', timepoints, ' timepoints, tr = ', fmritr, ', oversamptr =', oversamptr)
    print(numspatiallocs, ' spatial locations, ', timepoints, ' timepoints')
    timings.append(['Finish reading fmrifile', time.time(), None, None])

    # if the user has specified start and stop points, limit check, then use these numbers
    validstart, validend = tide_util.startendcheck(timepoints, optiondict['startpoint'], optiondict['endpoint'])
    if abs(optiondict['lagmin']) > (validend - validstart + 1) * fmritr / 2.0:
        print('magnitude of lagmin exceeds', (validend - validstart + 1) * fmritr / 2.0, ' - invalid')
        sys.exit()
    if abs(optiondict['lagmax']) > (validend - validstart + 1) * fmritr / 2.0:
        print('magnitude of lagmax exceeds', (validend - validstart + 1) * fmritr / 2.0, ' - invalid')
        sys.exit()
    if optiondict['gausssigma'] > 0.0:
        print('applying gaussian spatial filter to timepoints ', validstart, ' to ', validend)
        reportstep = 10
        for i in range(validstart, validend + 1):
            if (i % reportstep == 0 or i == validend) and optiondict['showprogressbar']:
                tide_util.progressbar(i - validstart + 1, validend - validstart + 1, label='Percent complete')
            nim_data[:, :, :, i] = tide_filt.ssmooth(xdim, ydim, slicethickness, optiondict['gausssigma'],
                                                     nim_data[:, :, :, i])
        timings.append(['End 3D smoothing', time.time(), None, None])
        print()

    # reshape the data and trim to a time range, if specified.  Check for special case of no trimming to save RAM
    fmri_data = nim_data.reshape((numspatiallocs, timepoints))[:, validstart:validend + 1]
    validtimepoints = validend - validstart + 1

    # read in the optional masks
    tide_util.logmem('before setting masks', file=memfile)
    internalglobalmeanincludemask = None
    internalglobalmeanexcludemask = None
    internalrefineincludemask = None
    internalrefineexcludemask = None

    if optiondict['globalmeanincludename'] is not None:
        print('constructing global mean include mask')
        theglobalmeanincludemask = readamask(optiondict['globalmeanincludename'], nim_hdr, xsize,
                                             istext=optiondict['textio'],
                                             valslist=optiondict['globalmeanincludevals'],
                                             maskname='global mean include')
        internalglobalmeanincludemask = theglobalmeanincludemask.reshape(numspatiallocs)
        if tide_stats.getmasksize(internalglobalmeanincludemask) == 0:
            print('ERROR: there are no voxels in the global mean include mask - exiting')
            sys.exit()

    if optiondict['globalmeanexcludename'] is not None:
        print('constructing global mean exclude mask')
        theglobalmeanexcludemask = readamask(optiondict['globalmeanexcludename'], nim_hdr, xsize,
                                             istext=optiondict['textio'],
                                             valslist=optiondict['globalmeanexcludevals'],
                                             maskname='global mean exclude')
        internalglobalmeanexcludemask = theglobalmeanexcludemask.reshape(numspatiallocs)
        if tide_stats.getmasksize(internalglobalmeanexcludemask) == numspatiallocs:
            print('ERROR: the global mean exclude mask does not leave any voxels - exiting')
            sys.exit()

    if (internalglobalmeanincludemask is not None) and (internalglobalmeanexcludemask is not None):
        if tide_stats.getmasksize(internalglobalmeanincludemask * (1 - internalglobalmeanexcludemask)) == 0:
            print('ERROR: the global mean include and exclude masks not leave any voxels between them - exiting')
            sys.exit()

    if optiondict['refineincludename'] is not None:
        print('constructing refine include mask')
        therefineincludemask = readamask(optiondict['refineincludename'], nim_hdr, xsize,
                                             istext=optiondict['textio'],
                                             valslist=optiondict['refineincludevals'],
                                             maskname='refine include')
        internalrefineincludemask = therefineincludemask.reshape(numspatiallocs)
        if tide_stats.getmasksize(internalrefineincludemask) == 0:
            print('ERROR: there are no voxels in the refine include mask - exiting')
            sys.exit()

    if optiondict['refineexcludename'] is not None:
        print('constructing refine exclude mask')
        therefineexcludemask = readamask(optiondict['refineexcludename'], nim_hdr, xsize,
                                             istext=optiondict['textio'],
                                             valslist=optiondict['refineexcludevals'],
                                             maskname='refine exclude')
        internalrefineexcludemask = therefineexcludemask.reshape(numspatiallocs)
        if tide_stats.getmasksize(internalrefineexcludemask) == numspatiallocs:
            print('ERROR: the refine exclude mask does not leave any voxels - exiting')
            sys.exit()

    tide_util.logmem('after setting masks', file=memfile)

    # read or make a mask of where to calculate the correlations
    tide_util.logmem('before selecting valid voxels', file=memfile)
    threshval = tide_stats.getfracvals(fmri_data[:, :], [0.98])[0] / 25.0
    print('constructing correlation mask')
    if optiondict['corrmaskincludename'] is not None:
        thecorrmask = readamask(optiondict['corrmaskincludename'], nim_hdr, xsize,
                                             istext=optiondict['textio'],
                                             valslist=optiondict['corrmaskincludevals'],
                                             maskname='correlation')

        corrmask = np.uint16(np.where(thecorrmask > 0, 1, 0).reshape(numspatiallocs))
    else:
        # check to see if the data has been demeaned
        meanim = np.mean(fmri_data, axis=1)
        stdim = np.std(fmri_data, axis=1)
        if np.mean(stdim) < np.mean(meanim):
            print('generating correlation mask from mean image')
            corrmask = np.uint16(tide_stats.makemask(meanim, threshpct=optiondict['corrmaskthreshpct']))
        else:
            print('generating correlation mask from std image')
            corrmask = np.uint16(tide_stats.makemask(stdim, threshpct=optiondict['corrmaskthreshpct']))
    if tide_stats.getmasksize(corrmask) == 0:
        print('ERROR: there are no voxels in the correlation mask - exiting')
        sys.exit()
    if internalrefineincludemask is not None:
        if internalrefineexcludemask is not None:
            if tide_stats.getmasksize(corrmask * internalrefineincludemask * (1 - internalrefineexcludemask)) == 0:
                print('ERROR: the refine include and exclude masks not leave any voxels in the corrmask - exiting')
                sys.exit()
        else:
            if tide_stats.getmasksize(corrmask * internalrefineincludemask) == 0:
                print('ERROR: the refine include mask does not leave any voxels in the corrmask - exiting')
                sys.exit()
    else:
        if internalrefineexcludemask is not None:
            if tide_stats.getmasksize(corrmask * (1 - internalrefineexcludemask)) == 0:
                print('ERROR: the refine exclude mask does not leave any voxels in the corrmask - exiting')
                sys.exit()

    if optiondict['nothresh']:
        corrmask *= 0
        corrmask += 1
        threshval = -10000000.0
    if optiondict['savecorrmask']:
        theheader = copy.deepcopy(nim_hdr)
        if fileiscifti:
            theheader['intent_code'] = 3006
        else:
            theheader['dim'][0] = 3
            theheader['dim'][4] = 1
        tide_io.savetonifti(corrmask.reshape(xsize, ysize, numslices), theheader, outputname + '_corrmask')

    if optiondict['verbose']:
        print('image threshval =', threshval)
    validvoxels = np.where(corrmask > 0)[0]
    numvalidspatiallocs = np.shape(validvoxels)[0]
    print('validvoxels shape =', numvalidspatiallocs)
    fmri_data_valid = fmri_data[validvoxels, :] + 0.0
    print('original size =', np.shape(fmri_data), ', trimmed size =', np.shape(fmri_data_valid))
    if internalglobalmeanincludemask is not None:
        internalglobalmeanincludemask_valid = 1.0 * internalglobalmeanincludemask[validvoxels]
        del internalglobalmeanincludemask
        print('internalglobalmeanincludemask_valid has size:', internalglobalmeanincludemask_valid.size)
    else:
        internalglobalmeanincludemask_valid = None
    if internalglobalmeanexcludemask is not None:
        internalglobalmeanexcludemask_valid = 1.0 * internalglobalmeanexcludemask[validvoxels]
        del internalglobalmeanexcludemask
        print('internalglobalmeanexcludemask_valid has size:', internalglobalmeanexcludemask_valid.size)
    else:
        internalglobalmeanexcludemask_valid = None
    if internalrefineincludemask is not None:
        internalrefineincludemask_valid = 1.0 * internalrefineincludemask[validvoxels]
        del internalrefineincludemask
        print('internalrefineincludemask_valid has size:', internalrefineincludemask_valid.size)
    else:
        internalrefineincludemask_valid = None
    if internalrefineexcludemask is not None:
        internalrefineexcludemask_valid = 1.0 * internalrefineexcludemask[validvoxels]
        del internalrefineexcludemask
        print('internalrefineexcludemask_valid has size:', internalrefineexcludemask_valid.size)
    else:
        internalrefineexcludemask_valid = None
    tide_util.logmem('after selecting valid voxels', file=memfile)

    # move fmri_data_valid into shared memory
    if optiondict['sharedmem']:
        print('moving fmri data to shared memory')
        timings.append(['Start moving fmri_data to shared memory', time.time(), None, None])
        numpy2shared_func = addmemprofiling(numpy2shared,
                                            optiondict['memprofile'],
                                            memfile,
                                            'before fmri data move')
        fmri_data_valid, fmri_data_valid_shared, fmri_data_valid_shared_shape = numpy2shared_func(fmri_data_valid,
                                                                                                  rt_floatset)
        timings.append(['End moving fmri_data to shared memory', time.time(), None, None])

    # get rid of memory we aren't using
    tide_util.logmem('before purging full sized fmri data', file=memfile)
    del fmri_data
    del nim_data
    tide_util.logmem('after purging full sized fmri data', file=memfile)

    # filter out motion regressors here
    if optiondict['motionfilename'] is not None:
        print('regressing out motion')

        timings.append(['Motion filtering start', time.time(), None, None])
        motionregressors, fmri_data_valid = tide_glmpass.motionregress(optiondict['motionfilename'],
                                                                    fmri_data_valid,
                                                                    tr,
                                                                    motstart=validstart,
                                                                    motend=validend + 1,
                                                                    position=optiondict['mot_pos'],
                                                                    deriv=optiondict['mot_deriv'],
                                                                    derivdelayed=optiondict['mot_delayderiv'])

        timings.append(['Motion filtering end', time.time(), fmri_data_valid.shape[0], 'voxels'])
        tide_io.writenpvecs(motionregressors, outputname + '_orthogonalizedmotion.txt')
        if optiondict['memprofile']:
            memcheckpoint('...done')
        else:
            tide_util.logmem('after motion glm filter', file=memfile)

        if optiondict['savemotionfiltered']:
            outfmriarray = np.zeros((numspatiallocs, validtimepoints), dtype=rt_floattype)
            outfmriarray[validvoxels, :] = fmri_data_valid[:, :]
            if optiondict['textio']:
                tide_io.writenpvecs(outfmriarray.reshape((numspatiallocs, validtimepoints)),
                                outputname + '_motionfiltered' + '' + '.txt')
            else:
                tide_io.savetonifti(outfmriarray.reshape((xsize, ysize, numslices, validtimepoints)), nim_hdr,
                                outputname + '_motionfiltered' + '')


    # read in the timecourse to resample
    timings.append(['Start of reference prep', time.time(), None, None])
    if filename is None:
        print('no regressor file specified - will use the global mean regressor')
        optiondict['useglobalref'] = True

    if optiondict['useglobalref']:
        inputfreq = 1.0 / fmritr
        inputperiod = 1.0 * fmritr
        inputstarttime = 0.0
        inputvec, meanmask = getglobalsignal(fmri_data_valid, optiondict,
                                             includemask=internalglobalmeanincludemask_valid,
                                             excludemask=internalglobalmeanexcludemask_valid)
        fullmeanmask = np.zeros((numspatiallocs), dtype=rt_floattype)
        fullmeanmask[validvoxels] = meanmask[:]
        theheader = copy.deepcopy(nim_hdr)
        if fileiscifti:
            theheader['intent_code'] = 3006
        else:
            theheader['dim'][0] = 3
            theheader['dim'][4] = 1
        tide_io.savetonifti(fullmeanmask.reshape((xsize, ysize, numslices)), theheader,
                            outputname + '_meanmask' + '')
        optiondict['preprocskip'] = 0
    else:
        inputfreq = optiondict['inputfreq']
        inputstarttime = optiondict['inputstarttime']
        if inputfreq is None:
            print('no regressor frequency specified - defaulting to 1/tr')
            inputfreq = 1.0 / fmritr
        if inputstarttime is None:
            print('no regressor start time specified - defaulting to 0.0')
            inputstarttime = 0.0
        inputperiod = 1.0 / inputfreq
        inputvec = tide_io.readvec(filename)
    numreference = len(inputvec)
    optiondict['inputfreq'] = inputfreq
    optiondict['inputstarttime'] = inputstarttime
    print('regressor start time, end time, and step', inputstarttime, inputstarttime + numreference * inputperiod,
          inputperiod)
    if optiondict['verbose']:
        print('input vector length', len(inputvec), 'input freq', inputfreq, 'input start time', inputstarttime)

    reference_x = np.arange(0.0, numreference) * inputperiod - (inputstarttime + optiondict['offsettime'])

    # Print out initial information
    if optiondict['verbose']:
        print('there are ', numreference, ' points in the original regressor')
        print('the timepoint spacing is ', 1.0 / inputfreq)
        print('the input timecourse start time is ', inputstarttime)

    # generate the time axes
    fmrifreq = 1.0 / fmritr
    optiondict['fmrifreq'] = fmrifreq
    skiptime = fmritr * (optiondict['preprocskip'])
    print('first fMRI point is at ', skiptime, ' seconds relative to time origin')
    initial_fmri_x = np.arange(0.0, validtimepoints) * fmritr + skiptime
    os_fmri_x = np.arange(0.0, validtimepoints * optiondict['oversampfactor'] - (
            optiondict['oversampfactor'] - 1)) * oversamptr + skiptime

    if optiondict['verbose']:
        print(np.shape(os_fmri_x)[0])
        print(np.shape(initial_fmri_x)[0])

    # generate the comparison regressor from the input timecourse
    # correct the output time points
    # check for extrapolation
    if os_fmri_x[0] < reference_x[0]:
        print('WARNING: extrapolating ', os_fmri_x[0] - reference_x[0], ' seconds of data at beginning of timecourse')
    if os_fmri_x[-1] > reference_x[-1]:
        print('WARNING: extrapolating ', os_fmri_x[-1] - reference_x[-1], ' seconds of data at end of timecourse')

    # invert the regressor if necessary
    if optiondict['invertregressor']:
        invertfac = -1.0
    else:
        invertfac = 1.0

    # detrend the regressor if necessary
    if optiondict['detrendorder'] > 0:
        reference_y = invertfac * tide_fit.detrend(inputvec[0:numreference],
                                                   order=optiondict['detrendorder'],
                                                   demean=optiondict['dodemean'])
    else:
        reference_y = invertfac * (inputvec[0:numreference] - np.mean(inputvec[0:numreference]))

    # write out the reference regressor prior to filtering
    tide_io.writenpvecs(reference_y, outputname + '_reference_origres_prefilt.txt')

    # band limit the regressor if that is needed
    print('filtering to ', theprefilter.gettype(), ' band')
    optiondict['lowerstop'], optiondict['lowerpass'], optiondict['upperpass'], optiondict['upperstop'] = theprefilter.getfreqs()
    reference_y_classfilter = theprefilter.apply(inputfreq, reference_y)
    reference_y = reference_y_classfilter

    # write out the reference regressor used
    tide_io.writenpvecs(tide_math.stdnormalize(reference_y), outputname + '_reference_origres.txt')

    # filter the input data for antialiasing
    if optiondict['antialias']:
        print('applying trapezoidal antialiasing filter')
        reference_y_filt = tide_filt.dolptrapfftfilt(inputfreq, 0.25 * fmrifreq, 0.5 * fmrifreq, reference_y,
                                                     padlen=int(inputfreq * optiondict['padseconds']),
                                                     debug=optiondict['debug'])
        reference_y = rt_floatset(reference_y_filt.real)

    warnings.filterwarnings('ignore', 'Casting*')

    if optiondict['fakerun']:
        return

    # generate the resampled reference regressors
    oversampfreq = optiondict['oversampfactor'] / fmritr
    if optiondict['detrendorder'] > 0:
        resampnonosref_y = tide_fit.detrend(
            tide_resample.doresample(reference_x, reference_y, initial_fmri_x,
                                     padlen=int(inputfreq * optiondict['padseconds']),
                                     method=optiondict['interptype']),
            order=optiondict['detrendorder'],
            demean=optiondict['dodemean'])
        # int(inputfreq * optiondict['padseconds'])
        resampref_y = tide_fit.detrend(
            tide_resample.doresample(reference_x, reference_y, os_fmri_x,
                                     padlen=int(oversampfreq * optiondict['padseconds']),
                                     method=optiondict['interptype']),
            order=optiondict['detrendorder'],
            demean=optiondict['dodemean'])
        # int(oversampfreq * optiondict['padseconds'])
    else:
        resampnonosref_y = tide_resample.doresample(reference_x, reference_y, initial_fmri_x,
                                                    padlen=int(inputfreq * optiondict['padseconds']),
                                                    method=optiondict['interptype'])
        resampref_y = tide_resample.doresample(reference_x, reference_y, os_fmri_x,
                                               padlen=int(oversampfreq * optiondict['padseconds']),
                                               method=optiondict['interptype'])
    print(len(os_fmri_x,), len(resampref_y), len(initial_fmri_x,), len(resampnonosref_y))

    # prepare the temporal mask
    if optiondict['tmaskname'] is not None:
        tmask_y = maketmask(optiondict['tmaskname'], reference_x, rt_floatset(reference_y))
        tmaskos_y = tide_resample.doresample(reference_x, tmask_y, os_fmri_x, method=optiondict['interptype'])
        tide_io.writenpvecs(tmask_y, outputname + '_temporalmask.txt')
        resampnonosref_y *= tmask_y
        thefit, R = tide_fit.mlregress(tmask_y, resampnonosref_y)
        resampnonosref_y -= thefit[0, 1] * tmask_y
        resampref_y *= tmaskos_y
        thefit, R = tide_fit.mlregress(tmaskos_y, resampref_y)
        resampref_y -= thefit[0, 1] * tmaskos_y

    nonosrefname = '_reference_fmrires_pass1.txt'
    osrefname = '_reference_resampres_pass1.txt'

    optiondict['kurtosis_reference_pass1'], \
        optiondict['kurtosisz_reference_pass1'], \
        optiondict['kurtosisp_reference_pass1'] = tide_stats.kurtosisstats(resampref_y)
    tide_io.writenpvecs(tide_math.stdnormalize(resampnonosref_y), outputname + nonosrefname)
    tide_io.writenpvecs(tide_math.stdnormalize(resampref_y), outputname + osrefname)
    timings.append(['End of reference prep', time.time(), None, None])

    corrtr = oversamptr
    if optiondict['verbose']:
        print('corrtr=', corrtr)

    # initialize the correlator
    thecorrelator = tide_classes.correlator(Fs=oversampfreq,
                                            ncprefilter=theprefilter,
                                            detrendorder=optiondict['detrendorder'],
                                            windowfunc=optiondict['windowfunc'],
                                            corrweighting=optiondict['corrweighting'],
                                            hpfreq=optiondict['correlator_hpfreq'])
    thecorrelator.setreftc(np.zeros((optiondict['oversampfactor'] * validtimepoints),
                                    dtype=np.float))
    numccorrlags = thecorrelator.corrlen
    corrorigin = thecorrelator.corrorigin
    dummy, corrscale, dummy = thecorrelator.getcorrelation(trim=False)

    lagmininpts = int((-optiondict['lagmin'] / corrtr) - 0.5)
    lagmaxinpts = int((optiondict['lagmax'] / corrtr) + 0.5)

    if (lagmaxinpts + lagmininpts) < 3:
        print('correlation search range is too narrow - decrease lagmin, increase lagmax, or increase oversample factor')
        sys.exit(1)

    thecorrelator.setlimits(lagmininpts, lagmaxinpts)
    dummy, trimmedcorrscale, dummy = thecorrelator.getcorrelation()

    if optiondict['verbose']:
        print('corrorigin at point ', corrorigin, corrscale[corrorigin])
        print('corr range from ', corrorigin - lagmininpts, '(', corrscale[
            corrorigin - lagmininpts], ') to ', corrorigin + lagmaxinpts, '(', corrscale[corrorigin + lagmaxinpts], ')')

    if optiondict['savecorrtimes']:
        tide_io.writenpvecs(trimmedcorrscale, outputname + '_corrtimes.txt')

    # allocate all of the data arrays
    tide_util.logmem('before main array allocation', file=memfile)
    if optiondict['textio']:
        nativespaceshape = xsize
    else:
        if fileiscifti:
            nativespaceshape = (1, 1, 1, 1, numspatiallocs)
        else:
            nativespaceshape = (xsize, ysize, numslices)
    internalspaceshape = numspatiallocs
    internalvalidspaceshape = numvalidspatiallocs
    meanval = np.zeros(internalvalidspaceshape, dtype=rt_floattype)
    lagtimes = np.zeros(internalvalidspaceshape, dtype=rt_floattype)
    lagstrengths = np.zeros(internalvalidspaceshape, dtype=rt_floattype)
    lagsigma = np.zeros(internalvalidspaceshape, dtype=rt_floattype)
    lagmask = np.zeros(internalvalidspaceshape, dtype='uint16')
    failimage = np.zeros(internalvalidspaceshape, dtype='uint16')
    R2 = np.zeros(internalvalidspaceshape, dtype=rt_floattype)
    outmaparray = np.zeros(internalspaceshape, dtype=rt_floattype)
    tide_util.logmem('after main array allocation', file=memfile)

    corroutlen = np.shape(trimmedcorrscale)[0]
    if optiondict['textio']:
        nativecorrshape = (xsize, corroutlen)
    else:
        if fileiscifti:
            nativecorrshape = (1, 1, 1, corroutlen, numspatiallocs)
        else:
            nativecorrshape = (xsize, ysize, numslices, corroutlen)
    internalcorrshape = (numspatiallocs, corroutlen)
    internalvalidcorrshape = (numvalidspatiallocs, corroutlen)
    print('allocating memory for correlation arrays', internalcorrshape, internalvalidcorrshape)
    if optiondict['sharedmem']:
        corrout, dummy, dummy = allocshared(internalvalidcorrshape, rt_floatset)
        gaussout, dummy, dummy = allocshared(internalvalidcorrshape, rt_floatset)
        windowout, dummy, dummy = allocshared(internalvalidcorrshape, rt_floatset)
        outcorrarray, dummy, dummy = allocshared(internalcorrshape, rt_floatset)
    else:
        corrout = np.zeros(internalvalidcorrshape, dtype=rt_floattype)
        gaussout = np.zeros(internalvalidcorrshape, dtype=rt_floattype)
        windowout = np.zeros(internalvalidcorrshape, dtype=rt_floattype)
        outcorrarray = np.zeros(internalcorrshape, dtype=rt_floattype)
    tide_util.logmem('after correlation array allocation', file=memfile)

    if optiondict['textio']:
        nativefmrishape = (xsize, np.shape(initial_fmri_x)[0])
    else:
        if fileiscifti:
            nativefmrishape = (1, 1, 1, np.shape(initial_fmri_x)[0], numspatiallocs)
        else:
            nativefmrishape = (xsize, ysize, numslices, np.shape(initial_fmri_x)[0])
    internalfmrishape = (numspatiallocs, np.shape(initial_fmri_x)[0])
    internalvalidfmrishape = (numvalidspatiallocs, np.shape(initial_fmri_x)[0])
    lagtc = np.zeros(internalvalidfmrishape, dtype=rt_floattype)
    tide_util.logmem('after lagtc array allocation', file=memfile)

    if optiondict['passes'] > 1:
        if optiondict['sharedmem']:
            shiftedtcs, dummy, dummy = allocshared(internalvalidfmrishape, rt_floatset)
            weights, dummy, dummy = allocshared(internalvalidfmrishape, rt_floatset)
        else:
            shiftedtcs = np.zeros(internalvalidfmrishape, dtype=rt_floattype)
            weights = np.zeros(internalvalidfmrishape, dtype=rt_floattype)
        tide_util.logmem('after refinement array allocation', file=memfile)
    if optiondict['sharedmem']:
        outfmriarray, dummy, dummy = allocshared(internalfmrishape, rt_floatset)
    else:
        outfmriarray = np.zeros(internalfmrishape, dtype=rt_floattype)

    # prepare for fast resampling
    padvalue = max((-optiondict['lagmin'], optiondict['lagmax'])) + 30.0 + np.abs(optiondict['offsettime'])
    # print('setting up fast resampling with padvalue =',padvalue)
    numpadtrs = int(padvalue // fmritr)
    padvalue = fmritr * numpadtrs
    genlagtc = tide_resample.fastresampler(reference_x, reference_y, padvalue=padvalue)

    # cycle over all voxels
    refine = True
    if optiondict['verbose']:
        print('refine is set to ', refine)
    optiondict['edgebufferfrac'] = max([optiondict['edgebufferfrac'], 2.0 / np.shape(corrscale)[0]])
    if optiondict['verbose']:
        print('edgebufferfrac set to ', optiondict['edgebufferfrac'])

    # intitialize the correlation fitter
    thefitter = tide_classes.correlation_fitter(lagmod=optiondict['lagmod'],
                                             lthreshval=optiondict['lthreshval'],
                                             uthreshval=optiondict['uthreshval'],
                                             bipolar=optiondict['bipolar'],
                                             lagmin=optiondict['lagmin'],
                                             lagmax=optiondict['lagmax'],
                                             absmaxsigma=optiondict['absmaxsigma'],
                                             absminsigma=optiondict['absminsigma'],
                                             debug=optiondict['debug'],
                                             findmaxtype=optiondict['findmaxtype'],
                                             refine=optiondict['gaussrefine'],
                                             searchfrac=optiondict['searchfrac'],
                                             fastgauss=optiondict['fastgauss'],
                                             enforcethresh=optiondict['enforcethresh'],
                                             hardlimit=optiondict['hardlimit'])

    for thepass in range(1, optiondict['passes'] + 1):
        # initialize the pass
        if optiondict['passes'] > 1:
            print('\n\n*********************')
            print('Pass number ', thepass)

        referencetc = tide_math.corrnormalize(resampref_y,
                                              prewindow=optiondict['usewindowfunc'],
                                              detrendorder=optiondict['detrendorder'],
                                              windowfunc=optiondict['windowfunc'])

        # Step -1 - check the regressor for periodic components in the passband
        dolagmod = True
        doreferencenotch = True
        if optiondict['respdelete']:
            resptracker = tide_classes.freqtrack(nperseg=64)
            thetimes, thefreqs = resptracker.track(resampref_y, 1.0 / oversamptr)
            tide_io.writevec(thefreqs, outputname + '_peakfreaks_pass' + str(thepass) + '.txt')
            resampref_y = resptracker.clean(resampref_y, 1.0 / oversamptr, thetimes, thefreqs)
            tide_io.writevec(resampref_y, outputname + '_respfilt_pass' + str(thepass) + '.txt')
            referencetc = tide_math.corrnormalize(resampref_y,
                                                  prewindow=optiondict['usewindowfunc'],
                                                  detrendorder=optiondict['detrendorder'],
                                                  windowfunc=optiondict['windowfunc'])
        if optiondict['check_autocorrelation']:
            print('checking reference regressor autocorrelation properties')
            optiondict['lagmod'] = 1000.0
            lagindpad = corrorigin - 2 * np.max((lagmininpts, lagmaxinpts))
            acmininpts = lagmininpts + lagindpad
            acmaxinpts = lagmaxinpts + lagindpad
            thecorrelator.setreftc(referencetc)
            thecorrelator.setlimits(acmininpts, acmaxinpts)
            thexcorr, accheckcorrscale, dummy = thecorrelator.run(resampref_y)
            thefitter.setcorrtimeaxis(accheckcorrscale)
            maxindex, maxlag, maxval, acwidth, maskval, peakstart, peakend, failreason = \
                tide_corrfit.onecorrfitx(thexcorr,
                                         thefitter,
                                         despeckle_thresh=optiondict['despeckle_thresh'],
                                         lthreshval=optiondict['lthreshval'],
                                         fixdelay=optiondict['fixdelay'],
                                         rt_floatset=rt_floatset,
                                         rt_floattype=rt_floattype
                                         )
            outputarray = np.asarray([accheckcorrscale, thexcorr])
            tide_io.writenpvecs(outputarray, outputname + '_referenceautocorr_pass' + str(thepass) + '.txt')
            thelagthresh = np.max((abs(optiondict['lagmin']), abs(optiondict['lagmax'])))
            theampthresh = 0.1
            print('searching for sidelobes with amplitude >', theampthresh, 'with abs(lag) <', thelagthresh, 's')
            sidelobetime, sidelobeamp = tide_corr.autocorrcheck(
                accheckcorrscale,
                thexcorr,
                acampthresh=theampthresh,
                aclagthresh=thelagthresh,
                prewindow=optiondict['usewindowfunc'],
                detrendorder=optiondict['detrendorder'])
            optiondict['acwidth'] = acwidth + 0.0
            optiondict['absmaxsigma'] = acwidth * 10.0
            if sidelobetime is not None:
                passsuffix = '_pass' + str(thepass + 1)
                optiondict['acsidelobelag' + passsuffix] = sidelobetime
                optiondict['despeckle_thresh'] = np.max([optiondict['despeckle_thresh'], sidelobetime / 2.0])
                optiondict['acsidelobeamp' + passsuffix] = sidelobeamp
                print('\n\nWARNING: autocorrcheck found bad sidelobe at', sidelobetime, 'seconds (', 1.0 / sidelobetime,
                      'Hz)...')
                tide_io.writenpvecs(np.array([sidelobetime]),
                                    outputname + '_autocorr_sidelobetime' + passsuffix + '.txt')
                if optiondict['fix_autocorrelation']:
                    print('Removing sidelobe')
                    if dolagmod:
                        print('subjecting lag times to modulus')
                        optiondict['lagmod'] = sidelobetime / 2.0
                    if doreferencenotch:
                        print('removing spectral component at sidelobe frequency')
                        acstopfreq = 1.0 / sidelobetime
                        acfixfilter = tide_filt.noncausalfilter(debug=optiondict['debug'])
                        acfixfilter.settype('arb_stop')
                        acfixfilter.setfreqs(acstopfreq * 0.9, acstopfreq * 0.95, acstopfreq * 1.05, acstopfreq * 1.1)
                        cleaned_resampref_y = tide_math.corrnormalize(acfixfilter.apply(fmrifreq, resampref_y),
                                                                      prewindow=False,
                                                                      detrendorder=optiondict['detrendorder'])
                        cleaned_referencetc = tide_math.corrnormalize(cleaned_resampref_y,
                                                                      prewindow=optiondict['usewindowfunc'],
                                                                      detrendorder=optiondict['detrendorder'],
                                                                      windowfunc=optiondict['windowfunc'])
                        cleaned_nonosreferencetc = tide_math.stdnormalize(acfixfilter.apply(fmrifreq, resampnonosref_y))
                        tide_io.writenpvecs(cleaned_nonosreferencetc,
                                            outputname + '_cleanedreference_fmrires_pass' + str(thepass) + '.txt')
                        tide_io.writenpvecs(cleaned_referencetc,
                                            outputname + '_cleanedreference_pass' + str(thepass) + '.txt')
                        tide_io.writenpvecs(cleaned_resampref_y,
                                            outputname + '_cleanedresampref_y_pass' + str(thepass) + '.txt')
                else:
                    cleaned_resampref_y = 1.0 * tide_math.corrnormalize(resampref_y,
                                                                        prewindow=False,
                                                                        detrendorder=optiondict['detrendorder'])
                    cleaned_referencetc = 1.0 * referencetc
            else:
                print('no sidelobes found in range')
                cleaned_resampref_y = 1.0 * tide_math.corrnormalize(resampref_y,
                                                                    prewindow=False,
                                                                    detrendorder=optiondict['detrendorder'])
                cleaned_referencetc = 1.0 * referencetc
        else:
            cleaned_resampref_y = 1.0 * tide_math.corrnormalize(resampref_y,
                                                                prewindow=False,
                                                                detrendorder=optiondict['detrendorder'])
            cleaned_referencetc = 1.0 * referencetc

        # Step 0 - estimate significance
        if optiondict['numestreps'] > 0:
            timings.append(['Significance estimation start, pass ' + str(thepass), time.time(), None, None])
            print('\n\nSignificance estimation, pass ' + str(thepass))
            if optiondict['verbose']:
                print('calling getNullDistributionData with args:', oversampfreq, fmritr, corrorigin, lagmininpts,
                      lagmaxinpts)
            getNullDistributionData_func = addmemprofiling(tide_nullcorr.getNullDistributionDatax,
                                                           optiondict['memprofile'],
                                                           memfile,
                                                           'before getnulldistristributiondata')
            if optiondict['checkpoint']:
                tide_io.writenpvecs(cleaned_referencetc,
                                    outputname + '_cleanedreference_pass' + str(thepass) + '.txt')
                tide_io.writenpvecs(cleaned_resampref_y,
                                    outputname + '_cleanedresampref_y_pass' + str(thepass) + '.txt')

                plot(cleaned_resampref_y)
                plot(cleaned_referencetc)
                show()
                if optiondict['saveoptionsasjson']:
                    tide_io.writedicttojson(optiondict, outputname + '_options_pregetnull_pass' + str(thepass) + '.json')
                else:
                    tide_io.writedict(optiondict, outputname + '_options_pregetnull_pass' + str(thepass) + '.txt')
            thecorrelator.setlimits(lagmininpts, lagmaxinpts)
            thecorrelator.setreftc(cleaned_resampref_y)
            dummy, trimmedcorrscale, dummy = thecorrelator.getcorrelation()
            thefitter.setcorrtimeaxis(trimmedcorrscale)
            corrdistdata = getNullDistributionData_func(cleaned_resampref_y,
                                                         oversampfreq,
                                                         thecorrelator,
                                                         thefitter,
                                                         numestreps=optiondict['numestreps'],
                                                         nprocs=optiondict['nprocs'],
                                                         showprogressbar=optiondict['showprogressbar'],
                                                         chunksize=optiondict['mp_chunksize'],
                                                         permutationmethod=optiondict['permutationmethod'],
                                                         fixdelay=optiondict['fixdelay'],
                                                         fixeddelayvalue=optiondict['fixeddelayvalue'],
                                                         rt_floatset=np.float64,
                                                         rt_floattype='float64')
            tide_io.writenpvecs(corrdistdata, outputname + '_corrdistdata_pass' + str(thepass) + '.txt')

            # calculate percentiles for the crosscorrelation from the distribution data
            thepercentiles = np.array([0.95, 0.99, 0.995, 0.999])
            thepvalnames = []
            for thispercentile in thepercentiles:
                thepvalnames.append("{:.3f}".format(1.0 - thispercentile).replace('.', 'p'))

            pcts, pcts_fit, sigfit = tide_stats.sigFromDistributionData(corrdistdata, optiondict['sighistlen'],
                                                                        thepercentiles, twotail=optiondict['bipolar'],
                                                                        displayplots=optiondict['displayplots'],
                                                                        nozero=optiondict['nohistzero'],
                                                                        dosighistfit=optiondict['dosighistfit'])
            if optiondict['ampthreshfromsig']:
                if pcts is not None:
                    print('setting ampthresh to the p<', "{:.3f}".format(1.0 - thepercentiles[0]), ' threshhold')
                    optiondict['ampthresh'] = pcts[2]
                    tide_stats.printthresholds(pcts, thepercentiles, 'Crosscorrelation significance thresholds from data:')
                    if optiondict['dosighistfit']:
                        tide_stats.printthresholds(pcts_fit, thepercentiles,
                                                   'Crosscorrelation significance thresholds from fit:')
                        tide_stats.makeandsavehistogram(corrdistdata, optiondict['sighistlen'], 0,
                                                        outputname + '_nullcorrelationhist_pass' + str(thepass),
                                                        displaytitle='Null correlation histogram, pass' + str(thepass),
                                                        displayplots=optiondict['displayplots'], refine=False)
                else:
                    print('leaving ampthresh unchanged')

            del corrdistdata
            timings.append(['Significance estimation end, pass ' + str(thepass), time.time(), optiondict['numestreps'],
                            'repetitions'])

        # Step 1 - Correlation step
        print('\n\nCorrelation calculation, pass ' + str(thepass))
        timings.append(['Correlation calculation start, pass ' + str(thepass), time.time(), None, None])
        correlationpass_func = addmemprofiling(tide_corrpass.correlationpass,
                                               optiondict['memprofile'],
                                               memfile,
                                               'before correlationpass')

        thecorrelator.setlimits(lagmininpts, lagmaxinpts)
        voxelsprocessed_cp, theglobalmaxlist, trimmedcorrscale = correlationpass_func(fmri_data_valid[:, :],
                                                               cleaned_referencetc,
                                                               thecorrelator,
                                                               initial_fmri_x,
                                                               os_fmri_x,
                                                               corrorigin,
                                                               lagmininpts,
                                                               lagmaxinpts,
                                                               corrout,
                                                               meanval,
                                                               nprocs=optiondict['nprocs'],
                                                               oversampfactor=optiondict['oversampfactor'],
                                                               interptype=optiondict['interptype'],
                                                               showprogressbar=optiondict['showprogressbar'],
                                                               chunksize=optiondict['mp_chunksize'],
                                                               rt_floatset=rt_floatset,
                                                               rt_floattype=rt_floattype)

        for i in range(len(theglobalmaxlist)):
            theglobalmaxlist[i] = corrscale[theglobalmaxlist[i]]
        tide_stats.makeandsavehistogram(np.asarray(theglobalmaxlist), len(corrscale), 0,
                                        outputname + '_globallaghist_pass' + str(thepass),
                                        displaytitle='lagtime histogram', displayplots=optiondict['displayplots'],
                                        therange=(corrscale[0], corrscale[-1]), refine=False)

        if optiondict['checkpoint']:
            outcorrarray[:, :] = 0.0
            outcorrarray[validvoxels, :] = corrout[:, :]
            if optiondict['textio']:
                tide_io.writenpvecs(outcorrarray.reshape(nativecorrshape),
                                    outputname + '_corrout_prefit_pass' + str(thepass) + outsuffix4d + '.txt')
            else:
                tide_io.savetonifti(outcorrarray.reshape(nativecorrshape), theheader,
                                    outputname + '_corrout_prefit_pass' + str(thepass)+ outsuffix4d)

        timings.append(['Correlation calculation end, pass ' + str(thepass), time.time(), voxelsprocessed_cp, 'voxels'])

        # Step 2 - correlation fitting and time lag estimation
        print('\n\nTime lag estimation pass ' + str(thepass))
        timings.append(['Time lag estimation start, pass ' + str(thepass), time.time(), None, None])
        fitcorr_func = addmemprofiling(tide_corrfit.fitcorrx,
                                       optiondict['memprofile'],
                                       memfile,
                                       'before fitcorr')
        thefitter.setcorrtimeaxis(trimmedcorrscale)
        voxelsprocessed_fc = fitcorr_func(genlagtc,
                                          initial_fmri_x,
                                          lagtc,
                                          trimmedcorrscale,
                                          thefitter,
                                          corrout,
                                          lagmask, failimage, lagtimes, lagstrengths, lagsigma,
                                          gaussout, windowout, R2,
                                          nprocs=optiondict['nprocs'],
                                          fixdelay=optiondict['fixdelay'],
                                          showprogressbar=optiondict['showprogressbar'],
                                          chunksize=optiondict['mp_chunksize'],
                                          despeckle_thresh=optiondict['despeckle_thresh'],
                                          rt_floatset=rt_floatset,
                                          rt_floattype=rt_floattype
                                          )

        timings.append(['Time lag estimation end, pass ' + str(thepass), time.time(), voxelsprocessed_fc, 'voxels'])

        # Step 2b - Correlation time despeckle
        if optiondict['despeckle_passes'] > 0:
            print('\n\nCorrelation despeckling pass ' + str(thepass))
            print('\tUsing despeckle_thresh =' + str(optiondict['despeckle_thresh']))
            timings.append(['Correlation despeckle start, pass ' + str(thepass), time.time(), None, None])

            # find lags that are very different from their neighbors, and refit starting at the median lag for the point
            voxelsprocessed_fc_ds = 0
            despecklingdone = False
            for despecklepass in range(optiondict['despeckle_passes']):
                print('\n\nCorrelation despeckling subpass ' + str(despecklepass + 1))
                outmaparray *= 0.0
                outmaparray[validvoxels] = eval('lagtimes')[:]
                medianlags = ndimage.median_filter(outmaparray.reshape(nativespaceshape), 3).reshape(numspatiallocs)
                initlags = \
                    np.where(np.abs(outmaparray - medianlags) > optiondict['despeckle_thresh'], medianlags, -1000000.0)[
                        validvoxels]
                if len(initlags) > 0:
                    if len(np.where(initlags != -1000000.0)[0]) > 0:
                        voxelsprocessed_fc_ds += fitcorr_func(genlagtc,
                                                              initial_fmri_x,
                                                              lagtc,
                                                              trimmedcorrscale,
                                                              thefitter,
                                                              corrout,
                                                              lagmask, failimage, lagtimes, lagstrengths, lagsigma,
                                                              gaussout, windowout, R2,
                                                              nprocs=optiondict['nprocs'],
                                                              fixdelay=optiondict['fixdelay'],
                                                              showprogressbar=optiondict['showprogressbar'],
                                                              chunksize=optiondict['mp_chunksize'],
                                                              despeckle_thresh=optiondict['despeckle_thresh'],
                                                              initiallags=initlags,
                                                              rt_floatset=rt_floatset,
                                                              rt_floattype=rt_floattype
                                                              )
                    else:
                        despecklingdone = True
                else:
                    despecklingdone = True
                if despecklingdone:
                    print('Nothing left to do! Terminating despeckling')
                    break

            if optiondict['savedespecklemasks']:
                theheader = copy.deepcopy(nim_hdr)
                if fileiscifti:
                    theheader['intent_code'] = 3006
                else:
                    theheader['dim'][0] = 3
                    theheader['dim'][4] = 1
                tide_io.savetonifti((np.where(np.abs(outmaparray - medianlags) > optiondict['despeckle_thresh'], medianlags, 0.0)).reshape(nativespaceshape), theheader,
                                 outputname + '_despecklemask_pass' + str(thepass))
            print('\n\n', voxelsprocessed_fc_ds, 'voxels despeckled in', optiondict['despeckle_passes'], 'passes')
            timings.append(
                ['Correlation despeckle end, pass ' + str(thepass), time.time(), voxelsprocessed_fc_ds, 'voxels'])

        # Step 3 - regressor refinement for next pass
        if thepass < optiondict['passes']:
            print('\n\nRegressor refinement, pass' + str(thepass))
            timings.append(['Regressor refinement start, pass ' + str(thepass), time.time(), None, None])
            if optiondict['refineoffset']:
                peaklag, peakheight, peakwidth = tide_stats.gethistprops(lagtimes[np.where(lagmask > 0)],
                                                                         optiondict['histlen'],
                                                                         pickleft=optiondict['pickleft'],
                                                                         peakthresh=optiondict['pickleftthresh'])
                optiondict['offsettime'] = peaklag
                optiondict['offsettime_total'] += peaklag
                print('offset time set to ', optiondict['offsettime'], ', total is ', optiondict['offsettime_total'])

            # regenerate regressor for next pass
            refineregressor_func = addmemprofiling(tide_refine.refineregressor,
                                                   optiondict['memprofile'],
                                                   memfile,
                                                   'before refineregressor')
            voxelsprocessed_rr, outputdata, refinemask = refineregressor_func(
                fmri_data_valid,
                fmritr,
                shiftedtcs,
                weights,
                thepass,
                lagstrengths,
                lagtimes,
                lagsigma,
                R2,
                theprefilter,
                optiondict,
                padtrs=numpadtrs,
                includemask=internalrefineincludemask_valid,
                excludemask=internalrefineexcludemask_valid,
                rt_floatset=rt_floatset,
                rt_floattype=rt_floattype)
            normoutputdata = tide_math.stdnormalize(theprefilter.apply(fmrifreq, outputdata))
            tide_io.writenpvecs(normoutputdata, outputname + '_refinedregressor_pass' + str(thepass) + '.txt')

            if optiondict['detrendorder'] > 0:
                resampnonosref_y = tide_fit.detrend(
                    tide_resample.doresample(initial_fmri_x,
                                             normoutputdata,
                                             initial_fmri_x,
                                             method=optiondict['interptype']),
                    order=optiondict['detrendorder'],
                    demean=optiondict['dodemean'])
                resampref_y = tide_fit.detrend(
                    tide_resample.doresample(initial_fmri_x,
                                             normoutputdata,
                                             os_fmri_x,
                                             method=optiondict['interptype']),
                    order=optiondict['detrendorder'],
                    demean=optiondict['dodemean'])
            else:
                resampnonosref_y = tide_resample.doresample(initial_fmri_x,
                                                            normoutputdata,
                                                            initial_fmri_x,
                                                            method=optiondict['interptype'])
                resampref_y = tide_resample.doresample(initial_fmri_x,
                                                       normoutputdata,
                                                       os_fmri_x,
                                                       method=optiondict['interptype'])
            if optiondict['tmaskname'] is not None:
                resampnonosref_y *= tmask_y
                thefit, R = tide_fit.mlregress(tmask_y, resampnonosref_y)
                resampnonosref_y -= thefit[0, 1] * tmask_y
                resampref_y *= tmaskos_y
                thefit, R = tide_fit.mlregress(tmaskos_y, resampref_y)
                resampref_y -= thefit[0, 1] * tmaskos_y

            # reinitialize lagtc for resampling
            genlagtc = tide_resample.fastresampler(initial_fmri_x, normoutputdata, padvalue=padvalue)
            nonosrefname = '_reference_fmrires_pass' + str(thepass + 1) + '.txt'
            osrefname = '_reference_resampres_pass' + str(thepass + 1) + '.txt'
            optiondict['kurtosis_reference_pass' + str(thepass + 1)], \
                optiondict['kurtosisz_reference_pass' + str(thepass + 1)], \
                optiondict['kurtosisp_reference_pass' + str(thepass + 1)] = tide_stats.kurtosisstats(resampref_y)
            tide_io.writenpvecs(tide_math.stdnormalize(resampnonosref_y), outputname + nonosrefname)
            tide_io.writenpvecs(tide_math.stdnormalize(resampref_y), outputname + osrefname)
            timings.append(
                ['Regressor refinement end, pass ' + str(thepass), time.time(), voxelsprocessed_rr, 'voxels'])

    # Post refinement step 0 - Wiener deconvolution
    if optiondict['dodeconv']:
        timings.append(['Wiener deconvolution start', time.time(), None, None])
        print('\n\nWiener deconvolution')
        reportstep = 1000

        # now allocate the arrays needed for Wiener deconvolution
        wienerdeconv = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)
        wpeak = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)

        wienerpass_func = addmemprofiling(tide_wiener.wienerpass,
                                          optiondict['memprofile'],
                                          memfile,
                                          'before wienerpass')
        voxelsprocessed_wiener = wienerpass_func(numspatiallocs,
                                                 reportstep,
                                                 fmri_data_valid,
                                                 threshval,
                                                 optiondict,
                                                 wienerdeconv,
                                                 wpeak,
                                                 resampref_y,
                                                 rt_floatset=rt_floatset,
                                                 rt_floattype=rt_floattype
                                                 )
        timings.append(['Wiener deconvolution end', time.time(), voxelsprocessed_wiener, 'voxels'])

    # Post refinement step 1 - GLM fitting to remove moving signal
    if optiondict['doglmfilt']:
        timings.append(['GLM filtering start', time.time(), None, None])
        print('\n\nGLM filtering')
        reportstep = 1000
        if (optiondict['gausssigma'] > 0.0) or (optiondict['glmsourcefile'] is not None):
            if optiondict['glmsourcefile'] is not None:
                print('reading in ', optiondict['glmsourcefile'], 'for GLM filter, please wait')
                if optiondict['textio']:
                    nim_data = tide_io.readvecs(optiondict['glmsourcefile'])
                else:
                    nim, nim_data, nim_hdr, thedims, thesizes = tide_io.readfromnifti(optiondict['glmsourcefile'])
            else:
                print('rereading', fmrifilename, ' for GLM filter, please wait')
                if optiondict['textio']:
                    nim_data = tide_io.readvecs(fmrifilename)
                else:
                    nim, nim_data, nim_hdr, thedims, thesizes = tide_io.readfromnifti(fmrifilename)
            fmri_data_valid = (nim_data.reshape((numspatiallocs, timepoints))[:, validstart:validend + 1])[validvoxels,
                              :] + 0.0

            # move fmri_data_valid into shared memory
            if optiondict['sharedmem']:
                print('moving fmri data to shared memory')
                timings.append(['Start moving fmri_data to shared memory', time.time(), None, None])
                numpy2shared_func = addmemprofiling(numpy2shared,
                                                    optiondict['memprofile'],
                                                    memfile,
                                                    'before movetoshared (glm)')
                fmri_data_valid, fmri_data_valid_shared, fmri_data_valid_shared_shape = numpy2shared_func(
                    fmri_data_valid, rt_floatset)
                timings.append(['End moving fmri_data to shared memory', time.time(), None, None])
            del nim_data

        # now allocate the arrays needed for GLM filtering
        meanvalue = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)
        rvalue = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)
        r2value = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)
        fitNorm = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)
        fitcoff = np.zeros(internalvalidspaceshape, dtype=rt_outfloattype)
        if optiondict['sharedmem']:
            datatoremove, dummy, dummy = allocshared(internalvalidfmrishape, rt_outfloatset)
            filtereddata, dummy, dummy = allocshared(internalvalidfmrishape, rt_outfloatset)
        else:
            datatoremove = np.zeros(internalvalidfmrishape, dtype=rt_outfloattype)
            filtereddata = np.zeros(internalvalidfmrishape, dtype=rt_outfloattype)

        if optiondict['memprofile']:
            memcheckpoint('about to start glm noise removal...')
        else:
            tide_util.logmem('before glm', file=memfile)

        if optiondict['preservefiltering']:
            for i in range(len(validvoxels)):
                fmri_data_valid[i] = theprefilter.apply(optiondict['fmrifreq'], fmri_data_valid[i])
        glmpass_func = addmemprofiling(tide_glmpass.glmpass,
                                       optiondict['memprofile'],
                                       memfile,
                                       'before glmpass')
        voxelsprocessed_glm = glmpass_func(numvalidspatiallocs,
                                           fmri_data_valid,
                                           threshval,
                                           lagtc,
                                           meanvalue,
                                           rvalue,
                                           r2value,
                                           fitcoff,
                                           fitNorm,
                                           datatoremove,
                                           filtereddata,
                                           reportstep=reportstep,
                                           nprocs=optiondict['nprocs'],
                                           showprogressbar=optiondict['showprogressbar'],
                                           mp_chunksize=optiondict['mp_chunksize'],
                                           rt_floatset=rt_floatset,
                                           rt_floattype=rt_floattype
                                           )
        del fmri_data_valid

        timings.append(['GLM filtering end, pass ' + str(thepass), time.time(), voxelsprocessed_glm, 'voxels'])
        if optiondict['memprofile']:
            memcheckpoint('...done')
        else:
            tide_util.logmem('after glm filter', file=memfile)
        print('')
    else:
        # get the original data to calculate the mean
        print('rereading', fmrifilename, ' to calculate mean value, please wait')
        if optiondict['textio']:
            nim_data = tide_io.readvecs(fmrifilename)
        else:
            nim, nim_data, nim_hdr, thedims, thesizes = tide_io.readfromnifti(fmrifilename)
        fmri_data = nim_data.reshape((numspatiallocs, timepoints))[:, validstart:validend + 1]
        meanvalue = np.mean(fmri_data, axis=1)


    # Post refinement step 2 - make and save interesting histograms
    timings.append(['Start saving histograms', time.time(), None, None])
    tide_stats.makeandsavehistogram(lagtimes[np.where(lagmask > 0)], optiondict['histlen'], 0, outputname + '_laghist',
                                    displaytitle='lagtime histogram', displayplots=optiondict['displayplots'],
                                    refine=False)
    tide_stats.makeandsavehistogram(lagstrengths[np.where(lagmask > 0)], optiondict['histlen'], 0,
                                    outputname + '_strengthhist',
                                    displaytitle='lagstrength histogram', displayplots=optiondict['displayplots'],
                                    therange=(0.0, 1.0))
    tide_stats.makeandsavehistogram(lagsigma[np.where(lagmask > 0)], optiondict['histlen'], 1,
                                    outputname + '_widthhist',
                                    displaytitle='lagsigma histogram', displayplots=optiondict['displayplots'])
    if optiondict['doglmfilt']:
        tide_stats.makeandsavehistogram(r2value[np.where(lagmask > 0)], optiondict['histlen'], 1, outputname + '_Rhist',
                                        displaytitle='correlation R2 histogram',
                                        displayplots=optiondict['displayplots'])
    timings.append(['Finished saving histograms', time.time(), None, None])

    # Post refinement step 3 - save out all of the important arrays to nifti files
    # write out the options used
    if optiondict['saveoptionsasjson']:
        tide_io.writedicttojson(optiondict, outputname + '_options.json')
    else:
        tide_io.writedict(optiondict, outputname + '_options.txt')

    # do ones with one time point first
    timings.append(['Start saving maps', time.time(), None, None])
    if not optiondict['textio']:
        theheader = copy.deepcopy(nim_hdr)
        if fileiscifti:
            theheader['intent_code'] = 3006
        else:
            theheader['dim'][0] = 3
            theheader['dim'][4] = 1

    # first generate the MTT map
    MTT = np.square(lagsigma) - (optiondict['acwidth'] * optiondict['acwidth'])
    MTT = np.where(MTT > 0.0, MTT, 0.0)
    MTT = np.sqrt(MTT)

    for mapname in ['lagtimes', 'lagstrengths', 'R2', 'lagsigma', 'lagmask', 'failimage', 'MTT']:
        if optiondict['memprofile']:
            memcheckpoint('about to write ' + mapname)
        else:
            tide_util.logmem('about to write ' + mapname, file=memfile)
        outmaparray[:] = 0.0
        outmaparray[validvoxels] = eval(mapname)[:]
        if optiondict['textio']:
            tide_io.writenpvecs(outmaparray.reshape(nativespaceshape, 1),
                                outputname + '_' + mapname + outsuffix3d + '.txt')
        else:
            tide_io.savetonifti(outmaparray.reshape(nativespaceshape), theheader,
                                outputname + '_' + mapname + outsuffix3d)

    if optiondict['doglmfilt']:
        for mapname, mapsuffix in [('rvalue', 'fitR'), ('r2value', 'fitR2'), ('meanvalue', 'mean'),
                                   ('fitcoff', 'fitcoff'), ('fitNorm', 'fitNorm')]:
            if optiondict['memprofile']:
                memcheckpoint('about to write ' + mapname)
            else:
                tide_util.logmem('about to write ' + mapname, file=memfile)
            outmaparray[:] = 0.0
            outmaparray[validvoxels] = eval(mapname)[:]
            if optiondict['textio']:
                tide_io.writenpvecs(outmaparray.reshape(nativespaceshape),
                                    outputname + '_' + mapsuffix + outsuffix3d + '.txt')
            else:
                tide_io.savetonifti(outmaparray.reshape(nativespaceshape), theheader,
                                    outputname + '_' + mapsuffix + outsuffix3d)
        del rvalue
        del r2value
        del meanvalue
        del fitcoff
        del fitNorm
    else:
        for mapname, mapsuffix in [('meanvalue', 'mean')]:
            if optiondict['memprofile']:
                memcheckpoint('about to write ' + mapname)
            else:
                tide_util.logmem('about to write ' + mapname, file=memfile)
            outmaparray[:] = 0.0
            outmaparray = eval(mapname)[:]
            if optiondict['textio']:
                tide_io.writenpvecs(outmaparray.reshape(nativespaceshape),
                                    outputname + '_' + mapsuffix + outsuffix3d + '.txt')
            else:
                tide_io.savetonifti(outmaparray.reshape(nativespaceshape), theheader,
                                    outputname + '_' + mapsuffix + outsuffix3d)
        del meanvalue

    if optiondict['numestreps'] > 0:
        for i in range(0, len(thepercentiles)):
            pmask = np.where(np.abs(lagstrengths) > pcts[i], lagmask, 0 * lagmask)
            if optiondict['dosighistfit']:
                tide_io.writenpvecs(sigfit, outputname + '_sigfit' + '.txt')
            tide_io.writenpvecs(np.array([pcts[i]]), outputname + '_p_lt_' + thepvalnames[i] + '_thresh.txt')
            outmaparray[:] = 0.0
            outmaparray[validvoxels] = pmask[:]
            if optiondict['textio']:
                tide_io.writenpvecs(outmaparray.reshape(nativespaceshape),
                                    outputname + '_p_lt_' + thepvalnames[i] + '_mask' + outsuffix3d + '.txt')
            else:
                tide_io.savetonifti(outmaparray.reshape(nativespaceshape), theheader,
                                    outputname + '_p_lt_' + thepvalnames[i] + '_mask' + outsuffix3d)

    if optiondict['passes'] > 1:
        outmaparray[:] = 0.0
        outmaparray[validvoxels] = refinemask[:]
        if optiondict['textio']:
            tide_io.writenpvecs(outfmriarray.reshape(nativefmrishape),
                                outputname + '_lagregressor' + outsuffix4d + '.txt')
        else:
            tide_io.savetonifti(outmaparray.reshape(nativespaceshape), theheader,
                                outputname + '_refinemask' + outsuffix3d)
        del refinemask

    # clean up arrays that will no longer be needed
    del lagtimes
    del lagstrengths
    del lagsigma
    del R2
    del lagmask

    # now do the ones with other numbers of time points
    if not optiondict['textio']:
        theheader = copy.deepcopy(nim_hdr)
        if fileiscifti:
            theheader['intent_code'] = 3002
        else:
            theheader['dim'][4] = np.shape(corrscale)[0]
        theheader['toffset'] = corrscale[corrorigin - lagmininpts]
        theheader['pixdim'][4] = corrtr
    outcorrarray[:, :] = 0.0
    outcorrarray[validvoxels, :] = gaussout[:, :]
    if optiondict['textio']:
        tide_io.writenpvecs(outcorrarray.reshape(nativecorrshape),
                            outputname + '_gaussout' + outsuffix4d + '.txt')
    else:
        tide_io.savetonifti(outcorrarray.reshape(nativecorrshape), theheader,
                            outputname + '_gaussout' + outsuffix4d)
    del gaussout
    outcorrarray[:, :] = 0.0
    outcorrarray[validvoxels, :] = windowout[:, :]
    if optiondict['textio']:
        tide_io.writenpvecs(outcorrarray.reshape(nativecorrshape),
                            outputname + '_windowout' + outsuffix4d + '.txt')
    else:
        tide_io.savetonifti(outcorrarray.reshape(nativecorrshape), theheader,
                            outputname + '_windowout' + outsuffix4d)
    del windowout
    outcorrarray[:, :] = 0.0
    outcorrarray[validvoxels, :] = corrout[:, :]
    if optiondict['textio']:
        tide_io.writenpvecs(outcorrarray.reshape(nativecorrshape),
                            outputname + '_corrout' + outsuffix4d + '.txt')
    else:
        tide_io.savetonifti(outcorrarray.reshape(nativecorrshape), theheader,
                            outputname + '_corrout' + outsuffix4d)
    del corrout

    if not optiondict['textio']:
        theheader = copy.deepcopy(nim_hdr)
        theheader['pixdim'][4] = fmritr
        theheader['toffset'] = 0.0
        if fileiscifti:
            theheader['intent_code'] = 3002
        else:
            theheader['dim'][4] = np.shape(initial_fmri_x)[0]

    if optiondict['savelagregressors']:
        outfmriarray[validvoxels, :] = lagtc[:, :]
        if optiondict['textio']:
            tide_io.writenpvecs(outfmriarray.reshape(nativefmrishape),
                                outputname + '_lagregressor' + outsuffix4d + '.txt')
        else:
            tide_io.savetonifti(outfmriarray.reshape(nativefmrishape), theheader,
                                outputname + '_lagregressor' + outsuffix4d)
        del lagtc

    if optiondict['passes'] > 1:
        if optiondict['savelagregressors']:
            outfmriarray[validvoxels, :] = shiftedtcs[:, :]
            if optiondict['textio']:
                tide_io.writenpvecs(outfmriarray.reshape(nativefmrishape),
                                    outputname + '_shiftedtcs' + outsuffix4d + '.txt')
            else:
                tide_io.savetonifti(outfmriarray.reshape(nativefmrishape), theheader,
                                    outputname + '_shiftedtcs' + outsuffix4d)
        del shiftedtcs

    if optiondict['doglmfilt'] and optiondict['saveglmfiltered']:
        if optiondict['savedatatoremove']:
            outfmriarray[validvoxels, :] = datatoremove[:, :]
            if optiondict['textio']:
                tide_io.writenpvecs(outfmriarray.reshape(nativefmrishape),
                                outputname + '_datatoremove' + outsuffix4d + '.txt')
            else:
                tide_io.savetonifti(outfmriarray.reshape(nativefmrishape), theheader,
                                outputname + '_datatoremove' + outsuffix4d)
        del datatoremove
        outfmriarray[validvoxels, :] = filtereddata[:, :]
        if optiondict['textio']:
            tide_io.writenpvecs(outfmriarray.reshape(nativefmrishape),
                                outputname + '_filtereddata' + outsuffix4d + '.txt')
        else:
            tide_io.savetonifti(outfmriarray.reshape(nativefmrishape), theheader,
                                outputname + '_filtereddata' + outsuffix4d)
        del filtereddata

    timings.append(['Finished saving maps', time.time(), None, None])
    memfile.close()
    print('done')

    if optiondict['displayplots']:
        show()
    timings.append(['Done', time.time(), None, None])

    # Post refinement step 5 - process and save timing information
    nodeline = 'Processed on ' + platform.node()
    tide_util.proctiminginfo(timings, outputfile=outputname + '_runtimings.txt', extraheader=nodeline)


if __name__ == '__main__':
    from rapidtide.workflows.rapidtide_parser import process_args

    rapidtide_main(process_args)
