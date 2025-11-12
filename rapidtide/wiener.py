#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2016-2025 Blaise Frederick
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
from numpy.typing import NDArray
from tqdm import tqdm

import rapidtide.fit as tide_fit
import rapidtide.multiproc as tide_multiproc


def _procOneVoxelWiener(
    vox: int,
    lagtc: NDArray,
    inittc: NDArray,
    rt_floattype: np.dtype = np.float64,
) -> tuple[int, float, float, float, float, NDArray, NDArray, NDArray]:
    """
    Perform Wiener filter processing on a single voxel time series.

    This function applies a Wiener filter to remove the lagged component from
    the initial time course, returning both the filtered and unfiltered results
    along with fitting statistics.

    Parameters
    ----------
    vox : int
        Voxel index identifier
    lagtc : NDArray
        Lagged time course data (input signal)
    inittc : NDArray
        Initial time course data (target signal)
    rt_floattype : np.dtype, optional
        Rapidtide float type for output arrays, default is np.float64

    Returns
    -------
    tuple[int, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]
        A tuple containing:
        - vox (int): Input voxel index
        - intercept (NDArray): Regression intercept term
        - sqrt_R2 (NDArray): Square root of coefficient of determination
        - R2 (NDArray): Coefficient of determination
        - fitcoff (NDArray): Fitting coefficient
        - ratio (NDArray): Ratio of slope to intercept
        - datatoremove (NDArray): Data to be removed (filtered signal)
        - residual (NDArray): Residual signal (unfiltered data)

    Notes
    -----
    This function uses maximum likelihood regression to estimate the relationship
    between lagged and initial time courses, then applies the Wiener filter
    to remove the lagged component from the initial signal.

    Examples
    --------
    >>> import numpy as np
    >>> lagtc = np.array([1.0, 2.0, 3.0, 4.0])
    >>> inittc = np.array([2.0, 4.0, 6.0, 8.0])
    >>> result = _procOneVoxelWiener(0, lagtc, inittc)
    >>> print(result[0])  # voxel index
    0
    >>> print(result[4])  # fitting coefficient
    2.0
    """
    thefit, R2 = tide_fit.mlregress(lagtc, inittc)
    fitcoff = thefit[0, 1]
    datatoremove = (fitcoff * lagtc).astype(rt_floattype)
    return (
        vox,
        thefit[0, 0],
        np.sqrt(R2),
        R2,
        fitcoff,
        thefit[0, 1] / thefit[0, 0],
        datatoremove,
        (inittc - datatoremove).astype(rt_floattype),
    )


def wienerpass(
    numspatiallocs: int,
    fmri_data: NDArray,
    threshval: float,
    lagtc: NDArray,
    optiondict: dict,
    wienerdeconv: NDArray,
    wpeak: NDArray,
    resampref_y: NDArray,
    rt_floattype: np.dtype = np.float64,
) -> int:
    """
    Perform Wiener deconvolution on fMRI data voxels.

    This function applies Wiener deconvolution to each voxel in the fMRI data
    based on the provided lagged time course and threshold. It supports both
    single-threaded and multi-threaded processing depending on the configuration
    in `optiondict`.

    Parameters
    ----------
    numspatiallocs : int
        Number of spatial locations (voxels) in the fMRI data.
    fmri_data : NDArray
        2D array of fMRI data with shape (numspatiallocs, timepoints).
    threshval : float
        Threshold value for masking voxels based on mean signal intensity.
    lagtc : NDArray
        2D array of lagged time courses with shape (numspatiallocs, timepoints).
    optiondict : dict
        Dictionary containing processing options including:
        - 'nprocs': number of processors to use (default: 1)
        - 'showprogressbar': whether to show progress bar (default: True)
        - 'mp_chunksize': chunk size for multiprocessing (default: 10)
    wienerdeconv : NDArray
        Wiener deconvolution kernel or filter.
    wpeak : NDArray
        Peak values associated with the Wiener deconvolution.
    resampref_y : NDArray
        Resampled reference signal for filtering.
    rt_floattype : np.dtype, optional
        Data type for floating-point numbers, default is `np.float64`.

    Returns
    -------
    int
        Total number of voxels processed.

    Notes
    -----
    - Voxels are masked based on their mean signal intensity exceeding `threshval`.
    - If `nprocs` > 1, multiprocessing is used to process voxels in parallel.
    - The function modifies global variables such as `meanvalue`, `rvalue`, etc.,
      which are assumed to be defined in the outer scope.

    Examples
    --------
    >>> import numpy as np
    >>> fmri_data = np.random.rand(100, 50)
    >>> lagtc = np.random.rand(100, 50)
    >>> optiondict = {'nprocs': 4, 'showprogressbar': True, 'mp_chunksize': 5}
    >>> result = wienerpass(
    ...     numspatiallocs=100,
    ...     fmri_data=fmri_data,
    ...     threshval=0.1,
    ...     lagtc=lagtc,
    ...     optiondict=optiondict,
    ...     wienerdeconv=np.array([1, 2, 1]),
    ...     wpeak=np.array([0.5]),
    ...     resampref_y=np.array([1, 1, 1])
    ... )
    >>> print(result)
    100
    """
    rt_floattype = rt_floattype
    inputshape = np.shape(fmri_data)
    themask = np.where(np.mean(fmri_data, axis=1) > threshval, 1, 0)
    if optiondict["nprocs"] > 1:
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
                    outQ.put(
                        _procOneVoxelWiener(
                            val,
                            lagtc[val, :],
                            fmri_data[val, :],
                            rt_floattype=rt_floattype,
                        )
                    )

                except Exception as e:
                    print("error!", e)
                    break

        data_out = tide_multiproc.run_multiproc(
            Wiener_consumer,
            inputshape,
            themask,
            nprocs=optiondict["nprocs"],
            showprogressbar=True,
            chunksize=optiondict["mp_chunksize"],
        )
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
        for vox in tqdm(
            range(0, numspatiallocs),
            desc="Voxel",
            unit="voxels",
            disable=(not optiondict["showprogressbar"]),
        ):
            inittc = fmri_data[vox, :].copy()
            if np.mean(inittc) >= threshval:
                (
                    dummy,
                    meanvalue[vox],
                    rvalue[vox],
                    r2value[vox],
                    fitcoff[vox],
                    fitNorm[vox],
                    datatoremove[vox],
                    filtereddata[vox],
                ) = _procOneVoxelWiener(
                    vox,
                    lagtc[vox, :],
                    inittc,
                    rt_floattype=rt_floattype,
                )
            volumetotal += 1

    return volumetotal
