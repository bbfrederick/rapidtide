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

import numpy as np

import rapidtide.correlate as tide_corr
import rapidtide.miscmath as tide_math
import rapidtide.multiproc as tide_multiproc
import rapidtide.resample as tide_resample
import rapidtide.util as tide_util

# this is here until numpy deals with their fft issue
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def onecorrelation(thetc,
                   oversampfreq,
                   corrorigin,
                   lagmininpts,
                   lagmaxinpts,
                   ncprefilter,
                   referencetc,
                   usewindowfunc=True,
                   detrendorder=1,
                   windowfunc='hamming',
                   corrweighting='none'):
    thetc_classfilter = ncprefilter.apply(oversampfreq, thetc)
    thetc = thetc_classfilter

    # prepare timecourse by normalizing, detrending, and applying a window function 
    preppedtc = tide_math.corrnormalize(thetc,
                                        prewindow=usewindowfunc,
                                        detrendorder=detrendorder,
                                        windowfunc=windowfunc)

    # now actually do the correlation
    thexcorr = tide_corr.fastcorrelate(preppedtc, referencetc, usefft=True, weighting=corrweighting)

    # find the global maximum value
    theglobalmax = np.argmax(thexcorr)

    return thexcorr[corrorigin - lagmininpts:corrorigin + lagmaxinpts], theglobalmax


def _procOneVoxelCorrelation(vox,
                             thetc,
                             optiondict,
                             fmri_x,
                             fmritc,
                             os_fmri_x,
                             oversampfreq,
                             corrorigin,
                             lagmininpts,
                             lagmaxinpts,
                             ncprefilter,
                             referencetc,
                             rt_floatset=np.float64,
                             rt_floattype='float64'
                             ):
    if optiondict['oversampfactor'] >= 1:
        thetc[:] = tide_resample.doresample(fmri_x, fmritc, os_fmri_x, method=optiondict['interptype'])
    else:
        thetc[:] = fmritc
    thexcorr, theglobalmax = onecorrelation(thetc,
                                            oversampfreq,
                                            corrorigin,
                                            lagmininpts,
                                            lagmaxinpts,
                                            ncprefilter,
                                            referencetc,
                                            usewindowfunc=optiondict['usewindowfunc'],
                                            detrendorder=optiondict['detrendorder'],
                                            windowfunc=optiondict['windowfunc'],
                                            corrweighting=optiondict['corrweighting'])

    return vox, np.mean(thetc), thexcorr, theglobalmax


def correlationpass(fmridata,
                    fmrifftdata,
                    referencetc,
                    fmri_x,
                    os_fmri_x,
                    tr,
                    corrorigin,
                    lagmininpts,
                    lagmaxinpts,
                    corrout,
                    meanval,
                    ncprefilter,
                    optiondict,
                    rt_floatset=np.float64,
                    rt_floattype='float64'):
    """

    Parameters
    ----------
    fmridata
    fmrifftdata
    referencetc
    fmri_x
    os_fmri_x
    tr
    corrorigin
    lagmininpts
    lagmaxinpts
    corrout
    meanval
    ncprefilter
    optiondict
    rt_floatset
    rt_floattype

    Returns
    -------

    """
    oversampfreq = optiondict['oversampfactor'] / tr
    inputshape = np.shape(fmridata)
    volumetotal = 0
    reportstep = 1000
    thetc = np.zeros(np.shape(os_fmri_x), dtype=rt_floattype)
    theglobalmaxlist = []
    if optiondict['nprocs'] > 1:
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
                    outQ.put(_procOneVoxelCorrelation(val,
                                                      thetc,
                                                      optiondict,
                                                      fmri_x,
                                                      fmridata[val, :],
                                                      os_fmri_x,
                                                      oversampfreq,
                                                      corrorigin,
                                                      lagmininpts,
                                                      lagmaxinpts,
                                                      ncprefilter,
                                                      referencetc,
                                                      rt_floatset=rt_floatset,
                                                      rt_floattype=rt_floattype))

                except Exception as e:
                    print("error!", e)
                    break

        data_out = tide_multiproc.run_multiproc(correlation_consumer,
                                                inputshape, None,
                                                nprocs=optiondict['nprocs'],
                                                showprogressbar=True,
                                                chunksize=optiondict['mp_chunksize'])

        # unpack the data
        volumetotal = 0
        for voxel in data_out:
            # corrmask[voxel[0]] = 1
            meanval[voxel[0]] = voxel[1]
            corrout[voxel[0], :] = voxel[2]
            theglobalmaxlist.append(voxel[3] + 0)
            volumetotal += 1
        del data_out
    else:
        for vox in range(0, inputshape[0]):
            if (vox % reportstep == 0 or vox == inputshape[0] - 1) and optiondict['showprogressbar']:
                tide_util.progressbar(vox + 1, inputshape[0], label='Percent complete')
            dummy, meanval[vox], corrout[vox, :], theglobalmax = _procOneVoxelCorrelation(vox,
                                                                                          thetc,
                                                                                          optiondict,
                                                                                          fmri_x,
                                                                                          fmridata[vox, :],
                                                                                          os_fmri_x,
                                                                                          oversampfreq,
                                                                                          corrorigin,
                                                                                          lagmininpts,
                                                                                          lagmaxinpts,
                                                                                          ncprefilter,
                                                                                          referencetc,
                                                                                          rt_floatset=rt_floatset,
                                                                                          rt_floattype=rt_floattype
                                                                                          )
            theglobalmaxlist.append(theglobalmax + 0)
            volumetotal += 1
    print('\nCorrelation performed on ' + str(volumetotal) + ' voxels')

    # garbage collect
    collected = gc.collect()
    print("Garbage collector: collected %d objects." % collected)

    return volumetotal, theglobalmaxlist
