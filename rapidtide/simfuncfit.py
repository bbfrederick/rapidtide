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
import bisect
import gc
import logging
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from tqdm import tqdm

import rapidtide.fit as tide_fit
import rapidtide.multiproc as tide_multiproc

LGR = logging.getLogger("GENERAL")


def onesimfuncfit(
    correlationfunc: ArrayLike,
    thefitter: Any,
    disablethresholds: bool = False,
    initiallag: Optional[float] = None,
    despeckle_thresh: float = 5.0,
    lthreshval: float = 0.0,
    fixdelay: bool = False,
    initialdelayvalue: float = 0.0,
    rt_floattype: np.dtype = np.float64,
) -> Tuple[int, float, float, float, int, int, int, int]:
    """
    Perform a single fit on a correlation function using the provided fitter.

    This function sets up the fitter with initial parameters and thresholds,
    then performs a fit on the provided correlation function. If `fixdelay` is
    True, the fit is skipped and a fixed delay value is used instead.

    Parameters
    ----------
    correlationfunc : ArrayLike
        The correlation function data to be fitted.
    thefitter : Any
        An object with methods `setguess`, `setrange`, `setlthresh`, and `fit`.
    disablethresholds : bool, optional
        If True, disables the threshold setting in the fitter. Default is False.
    initiallag : float, optional
        Initial guess for the lag value. If None, no initial guess is set.
        Default is None.
    despeckle_thresh : float, optional
        Threshold for despeckling. Default is 5.0.
    lthreshval : float, optional
        Low threshold value for the fitter. Default is 0.0.
    fixdelay : bool, optional
        If True, uses a fixed delay value instead of performing a fit.
        Default is False.
    initialdelayvalue : float, optional
        The fixed delay value to use when `fixdelay=True`. Default is 0.0.
    rt_floattype : np.dtype, optional
        The data type to use for floating-point values. Default is `np.float64`.

    Returns
    -------
    tuple of (int, float, float, float, int, int, int, int)
        A tuple containing:
        - maxindex (int): Index of the maximum value in the correlation function.
        - maxlag (float): The lag value at the maximum.
        - maxval (float): The maximum value in the correlation function.
        - maxsigma (float): The sigma (standard deviation) of the fit.
        - maskval (int): A mask value indicating fit quality.
        - peakstart (int): Start index of the fitted peak.
        - peakend (int): End index of the fitted peak.
        - failreason (int): Reason for fit failure (0 if successful).

    Notes
    -----
    When `fixdelay=True`, the function bypasses the fitting process and returns
    precomputed values based on `initialdelayvalue`.

    Examples
    --------
    >>> import numpy as np
    >>> corr_func = np.random.rand(100)
    >>> fitter = some_fitter_class()
    >>> result = onesimfuncfit(corr_func, fitter)
    >>> print(result)
    (50, 0.5, 0.95, 0.02, 1, 45, 55, 0)
    """
    if initiallag is not None:
        thefitter.setguess(True, maxguess=initiallag)
        thefitter.setrange(-despeckle_thresh / 2.0, despeckle_thresh / 2.0)
    else:
        thefitter.setguess(False)

    if disablethresholds:
        thefitter.setlthresh(0.0)
    else:
        thefitter.setlthresh(lthreshval)

    if not fixdelay:
        (
            maxindex,
            maxlag,
            maxval,
            maxsigma,
            maskval,
            failreason,
            peakstart,
            peakend,
        ) = thefitter.fit(correlationfunc)
    else:
        # do something different
        failreason = np.uint32(0)
        maxlag = initialdelayvalue
        maxindex = np.int16(bisect.bisect_left(thefitter.corrtimeaxis, initialdelayvalue))
        maxval = correlationfunc[maxindex]
        maxsigma = 1.0
        maskval = np.uint16(1)
        peakstart = maxindex
        peakend = maxindex

    return maxindex, maxlag, maxval, maxsigma, maskval, peakstart, peakend, failreason


def _procOneVoxelFitcorr(
    vox: int,
    corr_y: ArrayLike,
    thefitter: Any,
    disablethresholds: bool = False,
    despeckle_thresh: float = 5.0,
    initiallag: Optional[float] = None,
    fixdelay: bool = False,
    initialdelayvalue: float = 0.0,
    rt_floattype: np.dtype = np.float64,
) -> Tuple[int, int, float, float, float, NDArray, NDArray, float, int, int]:
    """
    Process a single voxel for correlation fitting.

    This function performs correlation fitting on a single voxel's data using the provided fitter.
    It returns fitting results including time, strength, sigma, Gaussian fit, window mask, R² value,
    and metadata such as mask value and failure reason.

    Parameters
    ----------
    vox : int
        Voxel index.
    corr_y : ArrayLike
        Correlation data for the voxel.
    thefitter : Any
        Fitter object containing fitting parameters and methods.
    disablethresholds : bool, optional
        If True, disables thresholding during fitting. Default is False.
    despeckle_thresh : float, optional
        Threshold for despeckling. Default is 5.0.
    initiallag : float, optional
        Initial lag value for fitting. Default is None.
    fixdelay : bool, optional
        If True, fixes the delay during fitting. Default is False.
    initialdelayvalue : float, optional
        Initial delay value if `fixdelay` is True. Default is 0.0.
    rt_floattype : np.dtype, optional
        Type to use for real-valued floating-point arrays. Default is `np.float64`.

    Returns
    -------
    tuple of (int, int, float, float, float, ndarray, ndarray, float, int, int)
        A tuple containing:
        - `vox`: Voxel index.
        - `volumetotalinc`: 1 if fit was successful, 0 otherwise.
        - `thetime`: Fitted time value.
        - `thestrength`: Fitted strength value.
        - `thesigma`: Fitted sigma value.
        - `thegaussout`: Gaussian fit evaluated over the time axis.
        - `thewindowout`: Binary window mask indicating the peak region.
        - `theR2`: R-squared value of the fit.
        - `maskval`: Mask value from the fitting process.
        - `failreason`: Reason for failure, if any.

    Notes
    -----
    - If `maxval > 0.3`, plotting is disabled.
    - The function uses `onesimfuncfit` to perform the actual fitting.
    - The `thefitter` object must have attributes like `zerooutbadfit`, `lagmod`, and `corrtimeaxis`.

    Examples
    --------
    >>> result = _procOneVoxelFitcorr(
    ...     vox=10,
    ...     corr_y=corr_data,
    ...     thefitter=fitter_obj,
    ...     disablethresholds=False,
    ...     despeckle_thresh=5.0,
    ...     fixdelay=False,
    ...     initialdelayvalue=0.0,
    ...     rt_floattype=np.float64,
    ... )
    >>> print(result)
    (10, 1, 1.23, 0.95, 0.12, array([...]), array([...]), 0.90, 1, 0)
    """
    (
        maxindex,
        maxlag,
        maxval,
        maxsigma,
        maskval,
        peakstart,
        peakend,
        failreason,
    ) = onesimfuncfit(
        corr_y,
        thefitter,
        disablethresholds=disablethresholds,
        despeckle_thresh=despeckle_thresh,
        fixdelay=fixdelay,
        initialdelayvalue=initialdelayvalue,
        initiallag=initiallag,
        rt_floattype=rt_floattype,
    )

    if maxval > 0.3:
        displayplots = False

    # now tuck everything away in the appropriate output array
    volumetotalinc = 0
    thewindowout = np.zeros_like(corr_y, rt_floattype)
    thewindowout[peakstart : peakend + 1] = 1.0
    if (maskval == 0) and thefitter.zerooutbadfit:
        thetime = 0.0
        thestrength = 0.0
        thesigma = 0.0
        thegaussout = np.zeros_like(corr_y, rt_floattype)
        theR2 = 0.0
    else:
        volumetotalinc = 1
        thetime = np.fmod(maxlag, thefitter.lagmod)
        thestrength = maxval
        thesigma = maxsigma
        thegaussout = np.zeros_like(corr_y, rt_floattype)
        thewindowout = np.zeros_like(corr_y, rt_floattype)
        if (not fixdelay) and (maxsigma != 0.0):
            thegaussout = tide_fit.gauss_eval(
                thefitter.corrtimeaxis, [maxval, maxlag, maxsigma]
            ).astype(rt_floattype)
        else:
            thegaussout = 0.0
            thewindowout = 0.0
        theR2 = thestrength * thestrength

    return (
        vox,
        volumetotalinc,
        thetime,
        thestrength,
        thesigma,
        thegaussout,
        thewindowout,
        theR2,
        maskval,
        failreason,
    )


def fitcorr(
    corrtimescale: ArrayLike,
    thefitter: Any,
    corrout: NDArray,
    lagmask: NDArray,
    failimage: NDArray,
    lagtimes: NDArray,
    lagstrengths: NDArray,
    lagsigma: NDArray,
    gaussout: NDArray,
    windowout: NDArray,
    R2: NDArray,
    despeckling: bool = False,
    peakdict: Optional[dict] = None,
    nprocs: int = 1,
    alwaysmultiproc: bool = False,
    fixdelay: bool = False,
    initialdelayvalue: Union[float, NDArray] = 0.0,
    showprogressbar: bool = True,
    chunksize: int = 1000,
    despeckle_thresh: float = 5.0,
    initiallags: Optional[NDArray] = None,
    rt_floattype: np.dtype = np.float64,
) -> int:
    """
    Fit correlation data to extract lag parameters and related statistics for each voxel.

    This function performs a fitting procedure on correlation data for each voxel,
    extracting lag times, strengths, sigma values, Gaussian fits, window functions,
    and R² values. It supports both single-threaded and multi-threaded processing.

    Parameters
    ----------
    corrtimescale : ArrayLike
        Time scale of the correlation data.
    thefitter : Any
        Fitter object used to perform the fitting. Must have methods like `setcorrtimeaxis`.
    corrout : NDArray
        Correlation data for all voxels, shape (n_voxels, n_timepoints).
    lagmask : NDArray
        Mask indicating valid lags for each voxel, shape (n_voxels,).
    failimage : NDArray
        Image to store failure flags for each voxel, shape (n_voxels,).
    lagtimes : NDArray
        Output array for lag times, shape (n_voxels,).
    lagstrengths : NDArray
        Output array for lag strengths, shape (n_voxels,).
    lagsigma : NDArray
        Output array for lag sigma values, shape (n_voxels,).
    gaussout : NDArray
        Output array for Gaussian fit parameters, shape (n_voxels, n_timepoints).
    windowout : NDArray
        Output array for window function values, shape (n_voxels, n_timepoints).
    R2 : NDArray
        Output array for R² values, shape (n_voxels,).
    despeckling : bool, optional
        If True, performs despeckling pass, only accepting successful fits, by default False.
    peakdict : dict, optional
        Dictionary of peak information, by default None.
    nprocs : int, optional
        Number of processes to use for multiprocessing, by default 1.
    alwaysmultiproc : bool, optional
        If True, always use multiprocessing even for single process, by default False.
    fixdelay : bool, optional
        If True, fix the delay value, by default False.
    initialdelayvalue : Union[float, NDArray], optional
        Initial delay value(s), by default 0.0.
    showprogressbar : bool, optional
        If True, show progress bar, by default True.
    chunksize : int, optional
        Size of chunks for multiprocessing, by default 1000.
    despeckle_thresh : float, optional
        Threshold for despeckling, by default 5.0.
    initiallags : NDArray, optional
        Initial lag values for each voxel, by default None.
    rt_floattype : np.dtype, optional
        Floating-point type for runtime, by default np.float64.

    Returns
    -------
    int
        Total number of voxels successfully processed.

    Notes
    -----
    The function modifies the input arrays (`lagtimes`, `lagstrengths`, `lagsigma`,
    `gaussout`, `windowout`, `R2`, `lagmask`, `failimage`) in-place.

    Examples
    --------
    >>> fitcorr(
    ...     corrtimescale=timescale,
    ...     thefitter=fitter,
    ...     corrout=correlation_data,
    ...     lagmask=lag_mask,
    ...     failimage=fail_image,
    ...     lagtimes=lag_times,
    ...     lagstrengths=lag_strengths,
    ...     lagsigma=lag_sigma,
    ...     gaussout=gaussian_out,
    ...     windowout=window_out,
    ...     R2=r2_values,
    ...     nprocs=4,
    ...     despeckling=True,
    ... )
    12345
    """
    thefitter.setcorrtimeaxis(corrtimescale)
    inputshape = np.shape(corrout)
    if initiallags is None:
        themask = None
    else:
        themask = np.where(initiallags > -100000.0, 1, 0)
    (
        volumetotal,
        ampfails,
        lowlagfails,
        highlagfails,
        lowwidthfails,
        highwidthfails,
        initfails,
        fitfails,
    ) = (0, 0, 0, 0, 0, 0, 0, 0)

    if nprocs > 1 or alwaysmultiproc:
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
                    if (themask is None) or (initiallags is None):
                        thislag = None
                    else:
                        if themask[val] > 0:
                            thislag = initiallags[val]
                        else:
                            thislag = None
                    if isinstance(initialdelayvalue, np.ndarray):
                        thisinitialdelayvalue = initialdelayvalue[val]
                    else:
                        thisinitialdelayvalue = initialdelayvalue
                    outQ.put(
                        _procOneVoxelFitcorr(
                            val,
                            corrout[val, :],
                            thefitter,
                            disablethresholds=False,
                            despeckle_thresh=despeckle_thresh,
                            initiallag=thislag,
                            fixdelay=fixdelay,
                            initialdelayvalue=thisinitialdelayvalue,
                            rt_floattype=rt_floattype,
                        )
                    )
                except Exception as e:
                    print("error!", e)
                    break

        data_out = tide_multiproc.run_multiproc(
            fitcorr_consumer,
            inputshape,
            themask,
            nprocs=nprocs,
            showprogressbar=showprogressbar,
            chunksize=chunksize,
        )

        # unpack the data
        volumetotal = 0
        for voxel in data_out:
            if (
                thefitter.FML_INITAMPLOW
                | thefitter.FML_INITAMPHIGH
                | thefitter.FML_FITAMPLOW
                | thefitter.FML_FITAMPHIGH
            ) & voxel[9]:
                ampfails += 1
            if (thefitter.FML_INITWIDTHLOW | thefitter.FML_FITWIDTHLOW) & voxel[9]:
                lowwidthfails += 1
            if (thefitter.FML_INITWIDTHHIGH | thefitter.FML_FITWIDTHHIGH) & voxel[9]:
                highwidthfails += 1
            if (thefitter.FML_INITLAGLOW | thefitter.FML_FITLAGLOW) & voxel[9]:
                lowlagfails += 1
            if (thefitter.FML_INITLAGHIGH | thefitter.FML_FITLAGHIGH) & voxel[9]:
                highlagfails += 1
            if thefitter.FML_INITFAIL & voxel[9]:
                initfails += 1
            if thefitter.FML_FITFAIL & voxel[9]:
                fitfails += 1

            # if this is a despeckle pass, only accept the new values if the fit did not fail
            if (voxel[9] == 0) or not despeckling:
                volumetotal += voxel[1]
                lagtimes[voxel[0]] = voxel[2]
                lagstrengths[voxel[0]] = voxel[3]
                lagsigma[voxel[0]] = voxel[4]
                gaussout[voxel[0], :] = voxel[5]
                windowout[voxel[0], :] = voxel[6]
                R2[voxel[0]] = voxel[7]
                lagmask[voxel[0]] = voxel[8]
                failimage[voxel[0]] = voxel[9] & 0xFFFF

        del data_out
    else:
        for vox in tqdm(
            range(0, inputshape[0]),
            desc="Voxel",
            unit="voxels",
            disable=(not showprogressbar),
        ):
            # process and send the data
            if (themask is None) or (initiallags is None):
                thislag = None
                dothisone = True
            else:
                if themask[vox] > 0:
                    thislag = initiallags[vox]
                    dothisone = True
                else:
                    thislag = None
                    dothisone = False
            if isinstance(initialdelayvalue, np.ndarray):
                thisinitialdelayvalue = initialdelayvalue[vox]
            else:
                thisinitialdelayvalue = initialdelayvalue
            if dothisone:
                voxel = _procOneVoxelFitcorr(
                    vox,
                    corrout[vox, :],
                    thefitter,
                    disablethresholds=False,
                    despeckle_thresh=despeckle_thresh,
                    initiallag=thislag,
                    fixdelay=fixdelay,
                    initialdelayvalue=thisinitialdelayvalue,
                    rt_floattype=rt_floattype,
                )
                if (
                    thefitter.FML_INITAMPLOW
                    | thefitter.FML_INITAMPHIGH
                    | thefitter.FML_FITAMPLOW
                    | thefitter.FML_FITAMPHIGH
                ) & voxel[9]:
                    ampfails += 1
                if (thefitter.FML_INITWIDTHLOW | thefitter.FML_FITWIDTHLOW) & voxel[9]:
                    lowwidthfails += 1
                if (thefitter.FML_INITWIDTHHIGH | thefitter.FML_FITWIDTHHIGH) & voxel[9]:
                    highwidthfails += 1
                if (thefitter.FML_INITLAGLOW | thefitter.FML_FITLAGLOW) & voxel[9]:
                    lowlagfails += 1
                if (thefitter.FML_INITLAGHIGH | thefitter.FML_FITLAGHIGH) & voxel[9]:
                    highlagfails += 1
                if thefitter.FML_INITFAIL & voxel[9]:
                    initfails += 1
                if thefitter.FML_FITFAIL & voxel[9]:
                    fitfails += 1

                # if this is a despeckle pass, only accept the new values if the fit did not fail
                if (voxel[9] == 0) or not despeckling:
                    volumetotal += voxel[1]
                    lagtimes[vox] = voxel[2]
                    lagstrengths[vox] = voxel[3]
                    lagsigma[vox] = voxel[4]
                    gaussout[vox, :] = voxel[5]
                    windowout[vox, :] = voxel[6]
                    R2[vox] = voxel[7]
                    lagmask[vox] = voxel[8]
                    failimage[vox] = voxel[9] & 0xFFFF

    LGR.info(f"\nSimilarity function fitted in {volumetotal} voxels")
    LGR.info(
        f"\tampfails: {ampfails}"
        + f"\n\tlowlagfails: {lowlagfails}"
        + f"\n\thighlagfails: {highlagfails}"
        + f"\n\tlowwidthfails: {lowwidthfails}"
        + f"\n\thighwidthfail: {highwidthfails}"
        + f"\n\ttotal initfails: {initfails}"
        + f"\n\ttotal fitfails: {fitfails}"
    )

    # garbage collect
    uncollected = gc.collect()
    if uncollected != 0:
        LGR.info(f"garbage collected - unable to collect {uncollected} objects")
    else:
        LGR.info("garbage collected")

    return volumetotal
