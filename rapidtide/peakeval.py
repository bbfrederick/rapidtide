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

import rapidtide.multiproc as tide_multiproc
import rapidtide.resample as tide_resample
import rapidtide.util as tide_util

# this is here until numpy deals with their fft issue
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def _procOneVoxelPeaks(vox,
                       thetc,
                       themutualinformationator,
                       fmri_x,
                       fmritc,
                       os_fmri_x,
                       xcorr_x,
                       thexcorr,
                       lagmin,
                       lagmax,
                       oversampfactor=1,
                       interptype='univariate'
                       ):

    if oversampfactor >= 1:
        thetc[:] = tide_resample.doresample(fmri_x, fmritc, os_fmri_x, method=interptype)
    else:
        thetc[:] = fmritc
    thepeaks = tide_fit.getpeaks(xcorr_x, thexcorr, xrange=(lagmin, lagmax), display=True)
    peaklocs = []
    for thepeak in thepeaks:
        peaklocs.append(int(round(thepeak[2], 0)))
    theMI_list = themutualinformationator.run(thetc, locs=peaklocs)
    hybridpeaks = []
    for i in range(len(thepeaks)):
        hybridpeaks.append([thepeaks[i][0], thepeaks[i][1], theMI_list[i]])
    return vox, hybridpeaks


def peakevalpass(
        fmridata,
        referencetc,
        themutualinformationator,
        fmri_x,
        os_fmri_x,
        lagmininpts,
        lagmaxinpts,
        corrout,
        meanval,
        nprocs=1,
        oversampfactor=1,
        interptype='univariate',
        showprogressbar=True,
        chunksize=1000):
    """

    Parameters
    ----------
    fmridata
    referencetc
    thecorrelator
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
    themutualinformationator.setreftc(referencetc)
    themutualinformationator.setlimits(lagmininpts, lagmaxinpts)

    inputshape = np.shape(fmridata)
    volumetotal = 0
    reportstep = 1000
    thetc = np.zeros(np.shape(os_fmri_x), dtype=rt_floattype)
    theglobalmaxlist = []
    if nprocs > 1:
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
                    outQ.put(_procOneVoxelPeaks(
                        val,
                        thetc,
                        themutualinformationator,
                        fmri_x,
                        fmridata[val, :],
                        os_fmri_x,
                        xcorr_x,
                        thexcorr,
                        lagmin,
                        lagmax,
                        oversampfactor=oversampfactor,
                        interptype=interptype)
                    )

                except Exception as e:
                    print("error!", e)
                    break

        data_out = tide_multiproc.run_multiproc(
            correlation_consumer,
            inputshape, None,
            nprocs=nprocs,
            showprogressbar=showprogressbar,
            chunksize=chunksize)

        # unpack the data
        volumetotal = 0
        for voxel in data_out:
            meanval[voxel[0]] = voxel[1]
            corrout[voxel[0], :] = voxel[2]
            thecorrscale = voxel[3]
            theglobalmaxlist.append(voxel[4] + 0)
            volumetotal += 1
        del data_out
    else:
        for vox in range(0, inputshape[0]):
            if (vox % reportstep == 0 or vox == inputshape[0] - 1) and showprogressbar:
                tide_util.progressbar(vox + 1, inputshape[0], label='Percent complete')
            dummy, thepeaks = \
                _procOneVoxelPeaks(
                    vox,
                    thetc,
                    themutualinformationator,
                    fmri_x,
                    fmridata[val, :],
                    os_fmri_x,
                    xcorr_x,
                    thexcorr,
                    lagmin,
                    lagmax,
                    oversampfactor=oversampfactor,
                    interptype=interptype)
            theglobalmaxlist.append(theglobalmax + 0)
            volumetotal += 1
    print('\nPeak evaluation performed on ' + str(volumetotal) + ' voxels')

    # garbage collect
    collected = gc.collect()
    print("Garbage collector: collected %d objects." % collected)

    return volumetotal, theglobalmaxlist, thecorrscale
