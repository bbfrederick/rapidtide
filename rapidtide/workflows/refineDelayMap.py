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
from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray

import rapidtide.refinedelay as tide_refinedelay
import rapidtide.stats as tide_stats


def refineDelay(
    fmri_data_valid: NDArray,
    initial_fmri_x: NDArray,
    xdim: float,
    ydim: float,
    slicethickness: float,
    sLFOfiltmask: NDArray,
    genlagtc: Any,
    oversamptr: float,
    sLFOfitmean: NDArray,
    rvalue: NDArray,
    r2value: NDArray,
    fitNorm: Any,
    fitcoeff: NDArray,
    lagtc: NDArray,
    outputname: str,
    validvoxels: Any,
    nativespaceshape: Any,
    theinputdata: Any,
    lagtimes: Any,
    optiondict: Any,
    LGR: Any,
    TimingLGR: Any,
    outputlevel: str = "normal",
    gausssigma: int = -1,
    patchthresh: float = 3.0,
    timemask: NDArray | None = None,
    mindelay: float = -5.0,
    maxdelay: float = 5.0,
    numpoints: int = 501,
    histlen: int = 101,
    rt_floattype: np.dtype = np.dtype(np.float64),
    debug: bool = False,
) -> Tuple[
    NDArray,
    NDArray,
    NDArray,
    NDArray,
    float
    ]:
    """
    Refine delay estimates using regression derivative ratios and histogram-based calibration.

    This function performs calibration of delay estimates by computing regression derivative
    ratios, filtering them, training a mapping from ratios to delays, and finally calculating
    delay offsets for each voxel. It also generates a histogram of the computed delay offsets.

    Parameters
    ----------
    fmri_data_valid : Any
        Valid fMRI data used for delay refinement.
    initial_fmri_x : Any
        Initial fMRI design matrix.
    xdim : Any
        X dimension of the data.
    ydim : Any
        Y dimension of the data.
    slicethickness : Any
        Thickness of the slices in the data.
    sLFOfiltmask : NDArray
        Mask for filtering based on SLOF (slice timing) effects.
    genlagtc : Any
        Generated lag time course.
    oversamptr : Any
        Oversampling time resolution.
    sLFOfitmean : Any
        Mean SLOF fit values.
    rvalue : Any
        R-values from regression.
    r2value : Any
        R-squared values from regression.
    fitNorm : Any
        Normalized fit coefficients.
    fitcoeff : Any
        Fit coefficients.
    lagtc : Any
        Lag time course.
    outputname : Any
        Base name for output files.
    validvoxels : Any
        Indices of valid voxels.
    nativespaceshape : Any
        Shape of the native space.
    theinputdata : Any
        Input data object.
    lagtimes : Any
        Time lags used for analysis.
    optiondict : Any
        Dictionary of options for processing.
    LGR : Any
        Logger for general messages.
    TimingLGR : Any
        Logger for timing-related messages.
    outputlevel : str, optional
        Level of output verbosity, default is "normal".
    gausssigma : int, optional
        Sigma for Gaussian filtering, default is -1 (no filtering).
    patchthresh : float, optional
        Threshold for patch-based filtering, default is 3.0.
    timemask : NDArray, optional
       Mask of timepoints to include in regression filtering. Default is None.
    mindelay : float, optional
        Minimum delay value, default is -5.0.
    maxdelay : float, optional
        Maximum delay value, default is 5.0.
    numpoints : int, optional
        Number of points for delay interpolation, default is 501.
    histlen : int, optional
        Length of histogram bins, default is 101.
    rt_floattype : np.dtype, optional
        Data type for rapidtide float operations, default is np.float64.
    debug : bool, optional
        Enable debug mode, default is False.

    Returns
    -------
    tuple
        A tuple containing:
        - delayoffset : ndarray
            Calculated delay offsets for each voxel.
        - regressderivratios : ndarray
            Raw regression derivative ratios.
        - medfiltregressderivratios : ndarray
            Median-filtered regression derivative ratios.
        - filteredregressderivratios : ndarray
            Final filtered regression derivative ratios.
        - delayoffsetMAD : ndarray
            Median absolute deviation of delay offsets.

    Notes
    -----
    The function uses the `tide_refinedelay` module to perform various steps:
    1. Computes regression derivative ratios using `getderivratios`.
    2. Filters these ratios using `filterderivratios`.
    3. Trains a ratio-to-delay mapping using `trainratiotooffset`.
    4. Converts ratios to delay offsets using `ratiotodelay`.
    5. Saves a histogram of delay offsets to disk.

    Examples
    --------
    >>> delayoffset, regressderivratios, medfiltregressderivratios, filteredregressderivratios, delayoffsetMAD = refineDelay(
    ...     fmri_data_valid, initial_fmri_x, xdim, ydim, slicethickness,
    ...     sLFOfiltmask, genlagtc, oversamptr, sLFOfitmean, rvalue, r2value,
    ...     fitNorm, fitcoeff, lagtc, outputname, validvoxels, nativespaceshape,
    ...     theinputdata, lagtimes, optiondict, LGR, TimingLGR
    ... )
    """
    # do the calibration
    TimingLGR.info("Refinement calibration start")
    regressderivratios, regressrvalues = tide_refinedelay.getderivratios(
        fmri_data_valid,
        validvoxels,
        initial_fmri_x,
        lagtimes,
        sLFOfiltmask,
        genlagtc,
        "glm",
        outputname,
        oversamptr,
        sLFOfitmean,
        rvalue,
        r2value,
        fitNorm[:, :2],
        fitcoeff[:, :2],
        None,
        lagtc,
        None,
        LGR,
        TimingLGR,
        optiondict,
        regressderivs=1,
        timemask=timemask,
        debug=debug,
    )

    medfiltregressderivratios, filteredregressderivratios, delayoffsetMAD = (
        tide_refinedelay.filterderivratios(
            regressderivratios,
            nativespaceshape,
            validvoxels,
            (xdim, ydim, slicethickness),
            gausssigma=gausssigma,
            patchthresh=patchthresh,
            filetype=theinputdata.filetype,
            rt_floattype=rt_floattype,
            debug=debug,
        )
    )

    # find the mapping of derivative ratios to delays
    tide_refinedelay.trainratiotooffset(
        genlagtc,
        initial_fmri_x,
        outputname,
        outputlevel,
        mindelay=mindelay,
        maxdelay=maxdelay,
        numpoints=numpoints,
        debug=debug,
    )

    # now calculate the delay offsets
    delayoffset = np.zeros_like(filteredregressderivratios)
    if debug:
        print(f"calculating delayoffsets for {filteredregressderivratios.shape[0]} voxels")
    for i in range(filteredregressderivratios.shape[0]):
        delayoffset[i], closestoffset = tide_refinedelay.ratiotodelay(
            filteredregressderivratios[i]
        )

    namesuffix = "_desc-delayoffset_hist"
    tide_stats.makeandsavehistogram(
        delayoffset[np.where(sLFOfiltmask > 0)],
        histlen,
        1,
        outputname + namesuffix,
        displaytitle="Histogram of delay offsets calculated from regression coefficients",
        dictvarname="delayoffsethist",
        thedict=optiondict,
    )

    return (
        delayoffset,
        regressderivratios,
        medfiltregressderivratios,
        filteredregressderivratios,
        delayoffsetMAD,
    )
