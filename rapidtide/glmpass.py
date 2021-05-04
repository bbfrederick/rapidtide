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
import numpy as np

import rapidtide.filter as tide_filt
import rapidtide.fit as tide_fit
import rapidtide.io as tide_io
import rapidtide.multiproc as tide_multiproc
import rapidtide.util as tide_util


def _procOneItemGLM(vox, theevs, thedata, rt_floatset=np.float64, rt_floattype="float64"):
    thefit, R = tide_fit.mlregress(theevs, thedata)
    fitcoeff = rt_floatset(thefit[0, 1])
    datatoremove = rt_floatset(fitcoeff * theevs)
    return (
        vox,
        rt_floatset(thefit[0, 0]),
        rt_floatset(R),
        rt_floatset(R * R),
        fitcoeff,
        rt_floatset(thefit[0, 1] / thefit[0, 0]),
        datatoremove,
        rt_floatset(thedata - datatoremove),
    )


def glmpass(
    numprocitems,
    fmri_data,
    threshval,
    theevs,
    meanvalue,
    rvalue,
    r2value,
    fitcoeff,
    fitNorm,
    datatoremove,
    filtereddata,
    reportstep=1000,
    nprocs=1,
    alwaysmultiproc=False,
    procbyvoxel=True,
    showprogressbar=True,
    mp_chunksize=1000,
    rt_floatset=np.float64,
    rt_floattype="float64",
):
    inputshape = np.shape(fmri_data)
    if threshval is None:
        themask = None
    else:
        if procbyvoxel:
            meanim = np.mean(fmri_data, axis=1)
            stdim = np.std(fmri_data, axis=1)
        else:
            meanim = np.mean(fmri_data, axis=0)
            stdim = np.std(fmri_data, axis=0)
        if np.mean(stdim) < np.mean(meanim):
            themask = np.where(meanim > threshval, 1, 0)
        else:
            themask = np.where(stdim > threshval, 1, 0)
    if (
        nprocs > 1 or alwaysmultiproc
    ):  # temporary workaround until I figure out why nprocs > 1 is failing
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
                        outQ.put(
                            _procOneItemGLM(
                                val,
                                theevs[val, :],
                                fmri_data[val, :],
                                rt_floatset=rt_floatset,
                                rt_floattype=rt_floattype,
                            )
                        )
                    else:
                        outQ.put(
                            _procOneItemGLM(
                                val,
                                theevs[:, val],
                                fmri_data[:, val],
                                rt_floatset=rt_floatset,
                                rt_floattype=rt_floattype,
                            )
                        )

                except Exception as e:
                    print("error!", e)
                    break

        data_out = tide_multiproc.run_multiproc(
            GLM_consumer,
            inputshape,
            themask,
            nprocs=nprocs,
            procbyvoxel=procbyvoxel,
            showprogressbar=showprogressbar,
            chunksize=mp_chunksize,
        )

        # unpack the data
        itemstotal = 0
        if procbyvoxel:
            for voxel in data_out:
                meanvalue[voxel[0]] = voxel[1]
                rvalue[voxel[0]] = voxel[2]
                r2value[voxel[0]] = voxel[3]
                fitcoeff[voxel[0]] = voxel[4]
                fitNorm[voxel[0]] = voxel[5]
                datatoremove[voxel[0], :] = voxel[6]
                filtereddata[voxel[0], :] = voxel[7]
                itemstotal += 1
        else:
            for timepoint in data_out:
                meanvalue[timepoint[0]] = timepoint[1]
                rvalue[timepoint[0]] = timepoint[2]
                r2value[timepoint[0]] = timepoint[3]
                fitcoeff[timepoint[0]] = timepoint[4]
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
                    tide_util.progressbar(vox + 1, numprocitems, label="Percent complete")
                thedata = fmri_data[vox, :].copy()
                if (themask is None) or (themask[vox] > 0):
                    (
                        dummy,
                        meanvalue[vox],
                        rvalue[vox],
                        r2value[vox],
                        fitcoeff[vox],
                        fitNorm[vox],
                        datatoremove[vox, :],
                        filtereddata[vox, :],
                    ) = _procOneItemGLM(
                        vox,
                        theevs[vox, :],
                        thedata,
                        rt_floatset=rt_floatset,
                        rt_floattype=rt_floattype,
                    )
                    itemstotal += 1
        else:
            for timepoint in range(0, numprocitems):
                if (
                    timepoint % reportstep == 0 or timepoint == numprocitems - 1
                ) and showprogressbar:
                    tide_util.progressbar(timepoint + 1, numprocitems, label="Percent complete")
                thedata = fmri_data[:, timepoint].copy()
                if (themask is None) or (themask[timepoint] > 0):
                    (
                        dummy,
                        meanvalue[timepoint],
                        rvalue[timepoint],
                        r2value[timepoint],
                        fitcoeff[timepoint],
                        fitNorm[timepoint],
                        datatoremove[:, timepoint],
                        filtereddata[:, timepoint],
                    ) = _procOneItemGLM(
                        timepoint,
                        theevs[:, timepoint],
                        thedata,
                        rt_floatset=rt_floatset,
                        rt_floattype=rt_floattype,
                    )
                    itemstotal += 1
        if showprogressbar:
            print()
    return itemstotal


def motionregress(
    themotionfilename,
    thedataarray,
    tr,
    orthogonalize=True,
    motstart=0,
    motend=-1,
    motionhp=None,
    motionlp=None,
    position=True,
    deriv=True,
    derivdelayed=False,
    debug=False,
):
    print("regressing out motion")
    motionregressors, motionregressorlabels = tide_io.calcmotregressors(
        tide_io.readmotion(themotionfilename),
        position=position,
        deriv=deriv,
        derivdelayed=derivdelayed,
    )
    if motend == -1:
        motionregressors = motionregressors[:, motstart:]
    else:
        motionregressors = motionregressors[:, motstart:motend]
    if (motionlp is not None) or (motionhp is not None):
        mothpfilt = tide_filt.NoncausalFilter(filtertype="arb", transferfunc="trapezoidal")
        if motionlp is None:
            motionlp = 0.5 / tr
        else:
            motionlp = np.min([0.5 / tr, motionlp])
        if motionhp is None:
            motionhp = 0.0
        mothpfilt.setfreqs(0.9 * motionhp, motionhp, motionlp, np.min([0.5 / tr, motionlp * 1.1]))
        for i in range(motionregressors.shape[0]):
            motionregressors[i, :] = mothpfilt.apply(1.0 / tr, motionregressors[i, :])
    if orthogonalize:
        motionregressors = tide_fit.gram_schmidt(motionregressors)
        initregressors = len(motionregressorlabels)
        motionregressorlabels = []
        for theregressor in range(motionregressors.shape[0]):
            motionregressorlabels.append("orthogmotion_{:02d}".format(theregressor))
        print(
            "After orthogonalization, {0} of {1} regressors remain.".format(
                len(motionregressorlabels), initregressors
            )
        )

    print("start motion filtering")
    filtereddata = confoundglm(thedataarray, motionregressors, debug=debug)
    print()
    print("motion filtering complete")
    return motionregressors, motionregressorlabels, filtereddata


def confoundglm(
    data,
    regressors,
    debug=False,
    showprogressbar=True,
    reportstep=1000,
    rt_floatset=np.float64,
    rt_floattype="float64",
):
    r"""Filters multiple regressors out of an array of data

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
        print("data shape:", data.shape)
        print("regressors shape:", regressors.shape)
    datatoremove = np.zeros(data.shape[1], dtype=rt_floattype)
    filtereddata = data * 0.0
    for i in range(data.shape[0]):
        if showprogressbar and (i > 0) and (i % reportstep == 0 or i == data.shape[0] - 1):
            tide_util.progressbar(i + 1, data.shape[0], label="Percent complete")
        datatoremove *= 0.0
        thefit, R = tide_fit.mlregress(regressors, data[i, :])
        if i == 0 and debug:
            print("fit shape:", thefit.shape)
        for j in range(regressors.shape[0]):
            datatoremove += rt_floatset(rt_floatset(thefit[0, 1 + j]) * regressors[j, :])
        filtereddata[i, :] = data[i, :] - datatoremove
    return filtereddata
