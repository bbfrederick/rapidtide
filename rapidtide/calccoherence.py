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
import rapidtide.util as tide_util

# this is here until numpy deals with their fft issue
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def _procOneVoxelCoherence(vox,
                             thetc,
                             thecoherer,
                             fmri_x,
                             fmritc,
                             rt_floatset=np.float64,
                             rt_floattype='float64'
                             ):
    thecoherence_y, thecoherence_x, theglobalmax = thecoherer.run(fmritc)

    return vox, np.mean(thetc), thecoherence_y, thecoherence_x, theglobalmax


def coherencepass(fmridata,
                    referencetc,
                    thecoherer,
                    maxfreq,
                    cohereout,
                    nprocs=1,
                    alwaysmultiproc=False,
                    showprogressbar=True,
                    chunksize=1000,
                    rt_floatset=np.float64,
                    rt_floattype='float64',
                    debug=False):
    """

    Parameters
    ----------
    fmridata
    referencetc - the reference regressor, already oversampled
    thecoherer
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
    thecoherer.setreftc(referencetc)
    thecoherer.setlimits(0.0, maxfreq)

    inputshape = np.shape(fmridata)
    volumetotal = 0
    reportstep = 1000
    thetc = np.zeros(np.shape(os_fmri_x), dtype=rt_floattype)
    theglobalmaxlist = []
    if nprocs > 1 or alwaysmultiproc:
        # define the consumer function here so it inherits most of the arguments
        def coherence_consumer(inQ, outQ):
            while True:
                try:
                    # get a new message
                    val = inQ.get()

                    # this is the 'TERM' signal
                    if val is None:
                        break

                    # process and send the data
                    outQ.put(_procOneVoxelCoherence(val,
                                                      thetc,
                                                      thecoherer,
                                                      fmri_x,
                                                      fmridata[val, :],
                                                      os_fmri_x,
                                                      oversampfactor=oversampfactor,
                                                      interptype=interptype,
                                                      rt_floatset=rt_floatset,
                                                      rt_floattype=rt_floattype))

                except Exception as e:
                    print("error!", e)
                    break

        data_out = tide_multiproc.run_multiproc(coherence_consumer,
                                                inputshape, None,
                                                nprocs=nprocs,
                                                showprogressbar=showprogressbar,
                                                chunksize=chunksize)

        # unpack the data
        volumetotal = 0
        for voxel in data_out:
            # corrmask[voxel[0]] = 1
            meanval[voxel[0]] = voxel[1]
            cohereout[voxel[0], :] = voxel[2]
            thecoherescale = voxel[3]
            theglobalmaxlist.append(voxel[4] + 0)
            volumetotal += 1
        del data_out
    else:
        for vox in range(0, inputshape[0]):
            if (vox % reportstep == 0 or vox == inputshape[0] - 1) and showprogressbar:
                tide_util.progressbar(vox + 1, inputshape[0], label='Percent complete')
            dummy, meanval[vox], corrout[vox, :], thecoherescale, theglobalmax = _procOneVoxelCorrelation(vox,
                                                                                          thetc,
                                                                                          thecorrelator,
                                                                                          fmri_x,
                                                                                          fmridata[vox, :],
                                                                                          os_fmri_x,
                                                                                          oversampfactor=oversampfactor,
                                                                                          interptype=interptype,
                                                                                          rt_floatset=rt_floatset,
                                                                                          rt_floattype=rt_floattype
                                                                                          )
            theglobalmaxlist.append(theglobalmax + 0)
            volumetotal += 1
    print('\nCorrelation performed on ' + str(volumetotal) + ' voxels')

    # garbage collect
    collected = gc.collect()
    print("Garbage collector: collected %d objects." % collected)

    return volumetotal, theglobalmaxlist, thecoherescale
