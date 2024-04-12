#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2016-2024 Blaise Frederick
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
import numpy as np
from tqdm import tqdm

import rapidtide.filter as tide_filt
import rapidtide.fit as tide_fit
import rapidtide.io as tide_io
import rapidtide.multiproc as tide_multiproc


def _procOneGLMItem(vox, theevs, thedata, rt_floatset=np.float64, rt_floattype="float64"):
    # NOTE: if theevs is 2D, dimension 0 is number of points, dimension 1 is number of evs
    thefit, R = tide_fit.mlregress(theevs, thedata)
    if theevs.ndim > 1:
        if thefit is None:
            thefit = np.matrix(np.zeros((1, theevs.shape[1] + 1), dtype=rt_floattype))
        fitcoeffs = rt_floatset(thefit[0, 1:])
        datatoremove = theevs[:, 0] * 0.0
        for j in range(theevs.shape[1]):
            datatoremove += rt_floatset(rt_floatset(thefit[0, 1 + j]) * theevs[:, j])
        if np.any(fitcoeffs) != 0.0:
            pass
        else:
            R = 0.0
        return (
            vox,
            rt_floatset(thefit[0, 0]),
            rt_floatset(R),
            rt_floatset(R * R),
            fitcoeffs,
            rt_floatset(thefit[0, 1:] / thefit[0, 0]),
            datatoremove,
            rt_floatset(thedata - datatoremove),
        )
    else:
        fitcoeff = rt_floatset(thefit[0, 1])
        datatoremove = rt_floatset(fitcoeff * theevs)
        if fitcoeff == 0.0:
            R = 0.0
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
    nprocs=1,
    alwaysmultiproc=False,
    confoundglm=False,
    procbyvoxel=True,
    showprogressbar=True,
    mp_chunksize=1000,
    rt_floatset=np.float64,
    rt_floattype="float64",
    debug=False,
):
    inputshape = np.shape(fmri_data)
    if debug:
        print(f"{numprocitems=}")
        print(f"{fmri_data.shape=}")
        print(f"{threshval=}")
        print(f"{theevs.shape=}")
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
                        if confoundglm:
                            outQ.put(
                                _procOneGLMItem(
                                    val,
                                    theevs,
                                    fmri_data[val, :],
                                    rt_floatset=rt_floatset,
                                    rt_floattype=rt_floattype,
                                )
                            )
                        else:
                            outQ.put(
                                _procOneGLMItem(
                                    val,
                                    theevs[val, :],
                                    fmri_data[val, :],
                                    rt_floatset=rt_floatset,
                                    rt_floattype=rt_floattype,
                                )
                            )
                    else:
                        if confoundglm:
                            outQ.put(
                                _procOneGLMItem(
                                    val,
                                    theevs,
                                    fmri_data[:, val],
                                    rt_floatset=rt_floatset,
                                    rt_floattype=rt_floattype,
                                )
                            )
                        else:
                            outQ.put(
                                _procOneGLMItem(
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
            if confoundglm:
                for voxel in data_out:
                    r2value[voxel[0]] = voxel[3]
                    filtereddata[voxel[0], :] = voxel[7]
                    itemstotal += 1
            else:
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
            if confoundglm:
                for timepoint in data_out:
                    r2value[timepoint[0]] = timepoint[3]
                    filtereddata[:, timepoint[0]] = timepoint[7]
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
            for vox in tqdm(
                range(0, numprocitems),
                desc="Voxel",
                unit="voxels",
                disable=(not showprogressbar),
            ):
                thedata = fmri_data[vox, :].copy()
                if (themask is None) or (themask[vox] > 0):
                    if confoundglm:
                        (
                            dummy,
                            dummy,
                            dummy,
                            r2value[vox],
                            dummy,
                            dummy,
                            dummy,
                            filtereddata[vox, :],
                        ) = _procOneGLMItem(
                            vox,
                            theevs,
                            thedata,
                            rt_floatset=rt_floatset,
                            rt_floattype=rt_floattype,
                        )
                    else:
                        (
                            dummy,
                            meanvalue[vox],
                            rvalue[vox],
                            r2value[vox],
                            fitcoeff[vox],
                            fitNorm[vox],
                            datatoremove[vox, :],
                            filtereddata[vox, :],
                        ) = _procOneGLMItem(
                            vox,
                            theevs[vox, :],
                            thedata,
                            rt_floatset=rt_floatset,
                            rt_floattype=rt_floattype,
                        )
                    itemstotal += 1
        else:
            for timepoint in tqdm(
                range(0, numprocitems),
                desc="Timepoint",
                unit="timepoints",
                disable=(not showprogressbar),
            ):
                thedata = fmri_data[:, timepoint].copy()
                if (themask is None) or (themask[timepoint] > 0):
                    if confoundglm:
                        (
                            dummy,
                            dummy,
                            dummy,
                            r2value[timepoint],
                            dummy,
                            dummy,
                            dummy,
                            filtereddata[:, timepoint],
                        ) = _procOneGLMItem(
                            timepoint,
                            theevs,
                            thedata,
                            rt_floatset=rt_floatset,
                            rt_floattype=rt_floattype,
                        )
                    else:
                        (
                            dummy,
                            meanvalue[timepoint],
                            rvalue[timepoint],
                            r2value[timepoint],
                            fitcoeff[timepoint],
                            fitNorm[timepoint],
                            datatoremove[:, timepoint],
                            filtereddata[:, timepoint],
                        ) = _procOneGLMItem(
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


def makevoxelspecificderivs(theevs, nderivs=1, debug=False):
    r"""Perform multicomponent expansion on theevs (each ev replaced by itself,
    its square, its cube, etc.).

    Parameters
    ----------
    theevs : 2D numpy array
        NxP array of voxel specific explanatory variables (one timecourse per voxel)
        :param theevs:

    nderivs : integer
        Number of components to use for each ev.  Each successive component is a
        higher power of the initial ev (initial, square, cube, etc.)
        :param nderivs:

    debug: bool
        Flag to toggle debugging output
        :param debug:
    """
    if debug:
        print(f"{theevs.shape=}")
    if nderivs == 0:
        thenewevs = theevs
    else:
        thenewevs = np.zeros((theevs.shape[0], theevs.shape[1], nderivs + 1), dtype=float)
        for thevoxel in range(0, theevs.shape[0]):
            thenewevs[thevoxel, :, 0] = theevs[thevoxel, :] * 1.0
            for i in range(1, nderivs + 1):
                thenewevs[thevoxel, :, i] = np.gradient(thenewevs[thevoxel, :, i - 1])
    if debug:
        print(f"{nderivs=}")
        print(f"{thenewevs.shape=}")

    return thenewevs


def confoundregress(
    theregressors,
    theregressorlabels,
    thedataarray,
    tr,
    nprocs=1,
    orthogonalize=True,
    tcstart=0,
    tcend=-1,
    tchp=None,
    tclp=None,
    showprogressbar=True,
    debug=False,
):
    if tcend == -1:
        theregressors = theregressors[:, tcstart:]
    else:
        theregressors = theregressors[:, tcstart:tcend]
    if (tclp is not None) or (tchp is not None):
        mothpfilt = tide_filt.NoncausalFilter(filtertype="arb", transferfunc="trapezoidal")
        if tclp is None:
            tclp = 0.5 / tr
        else:
            tclp = np.min([0.5 / tr, tclp])
        if tchp is None:
            tchp = 0.0
        mothpfilt.setfreqs(0.9 * tchp, tchp, tclp, np.min([0.5 / tr, tclp * 1.1]))
        for i in range(theregressors.shape[0]):
            theregressors[i, :] = mothpfilt.apply(1.0 / tr, theregressors[i, :])
    if orthogonalize:
        theregressors = tide_fit.gram_schmidt(theregressors)
        initregressors = len(theregressorlabels)
        theregressorlabels = []
        for theregressor in range(theregressors.shape[0]):
            theregressorlabels.append("orthogconfound_{:02d}".format(theregressor))
        print(
            "After orthogonalization, {0} of {1} regressors remain.".format(
                len(theregressorlabels), initregressors
            )
        )

    print("start confound filtering")

    numprocitems = thedataarray.shape[0]
    filtereddata = thedataarray * 0.0
    r2value = np.zeros(numprocitems)
    numfiltered = glmpass(
        numprocitems,
        thedataarray,
        None,
        np.transpose(theregressors),
        None,
        None,
        r2value,
        None,
        None,
        None,
        filtereddata,
        confoundglm=True,
        nprocs=nprocs,
        showprogressbar=showprogressbar,
        procbyvoxel=True,
        debug=debug,
    )

    print()
    print(f"confound filtering on {numfiltered} voxels complete")
    return theregressors, theregressorlabels, filtereddata, r2value
