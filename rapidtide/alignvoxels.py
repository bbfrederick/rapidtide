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
import gc

import numpy as np
from tqdm import tqdm

import rapidtide.fit as tide_fit
import rapidtide.multiproc as tide_multiproc
import rapidtide.resample as tide_resample


def _procOneVoxelTimeShift(
    vox,
    fmritc,
    lagtime,
    padtrs,
    fmritr,
    detrendorder=1,
    offsettime=0.0,
    rt_floatset=np.float64,
    rt_floattype="float64",
):
    if detrendorder > 0:
        normtc = tide_fit.detrend(fmritc, order=detrendorder, demean=True)
    else:
        normtc = fmritc
    shifttr = -(-offsettime + lagtime) / fmritr  # lagtime is in seconds
    [shiftedtc, weights, paddedshiftedtc, paddedweights] = tide_resample.timeshift(
        normtc, shifttr, padtrs
    )
    return vox, shiftedtc, weights, paddedshiftedtc, paddedweights


def alignvoxels(
    fmridata,
    fmritr,
    shiftedtcs,
    weights,
    paddedshiftedtcs,
    paddedweights,
    lagtimes,
    lagmask,
    detrendorder=1,
    offsettime=0.0,
    nprocs=1,
    alwaysmultiproc=False,
    showprogressbar=True,
    chunksize=1000,
    padtrs=60,
    debug=False,
    rt_floatset=np.float64,
    rt_floattype="float64",
):
    """
    This routine applies a timeshift to every voxel in the image.
    Inputs are:
        fmridata - the fmri data, filtered to the passband
        fmritr - the timestep of the data
        shiftedtcs,
        weights,
        paddedshiftedtcs,
        paddedweights,
        lagtimes, lagmask - the results of the correlation fit.
        detrendorder - the order of the polynomial to use to detrend the data
        offsettime - the global timeshift to apply to all timecourses
        nprocs - the number of processes to use if multiprocessing is enabled

    Explicit outputs are:
        volumetotal - the number of voxels processed

    Implicit outputs:
        shiftedtcs - voxelwise fmri data timeshifted to zero lag
        weights - the weights of every timepoint in the final regressor
        paddedshiftedtcs - voxelwise fmri data timeshifted to zero lag, with a bufffer of padtrs on each end
        paddedweights - the weights of every timepoint in the final regressor, with a bufffer of padtrs on each end


    Parameters
    ----------
    fmridata : 4D numpy float array
       fMRI data
    fmritr : float
        Data repetition rate, in seconds
    shiftedtcs : 4D numpy float array
        Destination array for time aligned voxel timecourses
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
    lagmask : 3D numpy float array
        Mask of voxels with successful correlation fits.
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
    debug : bool
        Enable additional debugging output.  Default is False
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
    volumetotal = np.sum(lagmask)

    # timeshift the valid voxels
    if nprocs > 1 or alwaysmultiproc:
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
                    outQ.put(
                        _procOneVoxelTimeShift(
                            val,
                            fmridata[val, :],
                            lagtimes[val],
                            padtrs,
                            fmritr,
                            detrendorder=detrendorder,
                            offsettime=offsettime,
                            rt_floatset=rt_floatset,
                            rt_floattype=rt_floattype,
                        )
                    )

                except Exception as e:
                    print("error!", e)
                    break

        data_out = tide_multiproc.run_multiproc(
            timeshift_consumer,
            inputshape,
            lagmask,
            nprocs=nprocs,
            showprogressbar=showprogressbar,
            chunksize=chunksize,
        )

        # unpack the data
        for voxel in data_out:
            shiftedtcs[voxel[0], :] = voxel[1]
            weights[voxel[0], :] = voxel[2]
            paddedshiftedtcs[voxel[0], :] = voxel[3]
            paddedweights[voxel[0], :] = voxel[4]
        del data_out

    else:
        for vox in tqdm(
            range(0, inputshape[0]),
            desc="Voxel timeshifts",
            unit="voxels",
            disable=(not showprogressbar),
        ):
            if lagmask[vox] > 0.5:
                retvals = _procOneVoxelTimeShift(
                    vox,
                    fmridata[vox, :],
                    lagtimes[vox],
                    padtrs,
                    fmritr,
                    detrendorder=detrendorder,
                    offsettime=offsettime,
                    rt_floatset=rt_floatset,
                    rt_floattype=rt_floattype,
                )
                shiftedtcs[retvals[0], :] = retvals[1]
                weights[retvals[0], :] = retvals[2]
                paddedshiftedtcs[retvals[0], :] = retvals[3]
                paddedweights[retvals[0], :] = retvals[4]
        print()
    print(
        "Timeshift applied to " + str(int(volumetotal)) + " voxels",
    )

    # garbage collect
    collected = gc.collect()
    print("Garbage collector: collected %d objects." % collected)

    return volumetotal
