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

import numpy as np

import rapidtide.fit as tide_fit
import rapidtide.io as tide_io
import rapidtide.multiproc as tide_multiproc
import rapidtide.util as tide_util


def _procOneItemGLM(vox,
                     lagtc,
                     inittc,
                     rt_floatset=np.float64,
                     rt_floattype='float64'):
    thefit, R = tide_fit.mlregress(lagtc, inittc)
    fitcoff = rt_floatset(thefit[0, 1])
    datatoremove = rt_floatset(fitcoff * lagtc)
    return vox, rt_floatset(thefit[0, 0]), rt_floatset(R), rt_floatset(R * R), fitcoff, \
           rt_floatset(thefit[0, 1] / thefit[0, 0]), datatoremove, rt_floatset(inittc - datatoremove)


def glmpass(numprocitems,
            fmri_data,
            threshval,
            lagtc,
            meanvalue,
            rvalue,
            r2value,
            fitcoff,
            fitNorm,
            datatoremove,
            filtereddata,
            reportstep=1000,
            nprocs=1,
            procbyvoxel=True,
            showprogressbar=True,
            addedskip=0,
            mp_chunksize=1000,
            rt_floatset=np.float64,
            rt_floattype='float64'):
    inputshape = np.shape(fmri_data)
    if threshval is None:
        themask = None
    else:
        themask = np.where(np.mean(fmri_data, axis=1) > threshval, 1, 0)
    if nprocs > 1:
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
                    if procbyvoxel:
                        outQ.put(_procOneItemGLM(val,
                                                  lagtc[val, :],
                                                  fmri_data[val, addedskip:],
                                                  rt_floatset=rt_floatset,
                                                  rt_floattype=rt_floattype))
                    else:
                        outQ.put(_procOneItemGLM(val,
                                                  lagtc[:, val],
                                                  fmri_data[:, addedskip + val],
                                                  rt_floatset=rt_floatset,
                                                  rt_floattype=rt_floattype))


                except Exception as e:
                    print("error!", e)
                    break

        data_out = tide_multiproc.run_multiproc(GLM_consumer,
                                                inputshape, themask,
                                                nprocs=nprocs,
                                                procbyvoxel=procbyvoxel,
                                                showprogressbar=True,
                                                chunksize=mp_chunksize)

        # unpack the data
        itemstotal = 0
        if procbyvoxel:
            for voxel in data_out:
                meanvalue[voxel[0]] = voxel[1]
                rvalue[voxel[0]] = voxel[2]
                r2value[voxel[0]] = voxel[3]
                fitcoff[voxel[0]] = voxel[4]
                fitNorm[voxel[0]] = voxel[5]
                datatoremove[voxel[0], :] = voxel[6]
                filtereddata[voxel[0], :] = voxel[7]
                itemstotal += 1
        else:
            for timepoint in data_out:
                meanvalue[timepoint[0]] = timepoint[1]
                rvalue[timepoint[0]] = timepoint[2]
                r2value[timepoint[0]] = timepoint[3]
                fitcoff[timepoint[0]] = timepoint[4]
                fitNorm[timepoint[0]] = timepoint[5]
                datatoremove[:, timepoint[0]] = timepoint[6]
                filtereddata[:, timepoint[0]] = timepoint[7]
                itemstotal += 1

        del data_out
    else:
        itemstotal = 0
        if procbyvoxel:
            for vox in range(0, numprocitems):
                if (vox % reportstep == 0 or vox == numprocitems - 1) and showprogressbar:
                    tide_util.progressbar(vox + 1, numprocitems, label='Percent complete')
                inittc = fmri_data[vox, addedskip:].copy()
                if themask[vox] > 0:
                    dummy, \
                    meanvalue[vox],\
                    rvalue[vox], \
                    r2value[vox], \
                    fitcoff[vox], \
                    fitNorm[vox], \
                    datatoremove[vox, :], \
                    filtereddata[vox, :] = \
                        _procOneItemGLM(vox,
                                         lagtc[vox, :],
                                         inittc,
                                         rt_floatset=rt_floatset,
                                         rt_floattype=rt_floattype)
                    itemstotal += 1
        else:
            for timepoint in range(0, numprocitems):
                if (timepoint % reportstep == 0 or timepoint == numprocitems - 1) and showprogressbar:
                    tide_util.progressbar(timepoint + 1, numprocitems, label='Percent complete')
                inittc = fmri_data[:, addedskip + timepoint].copy()
                if themask[timepoint] > 0:
                    dummy, \
                    meanvalue[timepoint], \
                    rvalue[timepoint], \
                    r2value[timepoint], \
                    fitcoff[timepoint], \
                    fitNorm[timepoint], \
                    datatoremove[:, timepoint], \
                    filtereddata[:, timepoint] = \
                        _procOneItemGLM(timepoint,
                                        lagtc[:, timepoint],
                                        inittc,
                                        rt_floatset=rt_floatset,
                                        rt_floattype=rt_floattype)
                    itemstotal += 1
    return itemstotal


def confoundglm(data,
                     regressors,
                     debug=False,
                     showprogressbar=True,
                     reportstep=1000,
                     rt_floatset=np.float64,
                     rt_floattype='float64'):
    r"""Filters multiple regressors out of an array of data in place

    Parameters
    ----------
    data : 2d numpy array
        A data array.  First index is the spatial dimension, second is the time (filtering) dimension.

    regressors: 2d numpy array
        The set of regressors to filter out of each timecourse.  The first dimension is the regressor number, second is the time (filtering) dimension:

    debug : boolean
        Print additional diagnostic information if True

    Returns
    -------
    """
    if debug:
        print('data shape:', data.shape)
        print('regressors shape:', regressors.shape)
    datatoremove = np.zeros(data.shape[1], dtype=rt_floattype)
    for i in range(data.shape[0]):
        if showprogressbar and (i % reportstep == 0):
            tide_util.progressbar(i + 1, data.shape[0], label='Percent complete')
        if showprogressbar:
            tide_util.progressbar(data.shape[0], data.shape[0], label='Percent complete')
        datatoremove *= 0.0
        thefit, R = tide_fit.mlregress(regressors, data[i, :])
        for j in range(regressors.shape[0]):
            datatoremove += rt_floatset(rt_floatset(thefit[0, 1 + j]) * regressors[j, :])
        data[i, :] -= datatoremove

