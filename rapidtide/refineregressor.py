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
import gc
import logging
import sys
from typing import Any

import numpy as np
import statsmodels as sm
from numpy.typing import NDArray
from scipy.stats import pearsonr
from sklearn.decomposition import PCA, FastICA

import rapidtide.fit as tide_fit
import rapidtide.genericmultiproc as tide_genericmultiproc
import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math
import rapidtide.resample as tide_resample
import rapidtide.stats as tide_stats

LGR = logging.getLogger("GENERAL")


def _procOneVoxelTimeShift(
    vox: int,
    voxelargs: tuple,
    **kwargs: Any,
) -> tuple[int, NDArray, NDArray, NDArray, NDArray]:
    options = {
        "detrendorder": 1,
        "offsettime": 0.0,
        "debug": False,
    }
    options.update(kwargs)
    detrendorder = int(options["detrendorder"])
    offsettime = options["offsettime"]
    debug = options["debug"]
    if debug:
        print(f"{detrendorder=} {offsettime=}")
    (
        fmritc,
        lagtime,
        padtrs,
        fmritr,
    ) = voxelargs
    if detrendorder > 0:
        normtc = tide_fit.detrend(fmritc, order=detrendorder, demean=True)
    else:
        normtc = fmritc + 0.0
    shifttr = -(-offsettime + lagtime) / fmritr  # lagtime is in seconds
    [shiftedtc, weights, paddedshiftedtc, paddedweights] = tide_resample.timeshift(
        normtc, shifttr, padtrs
    )
    return vox, shiftedtc, weights, paddedshiftedtc, paddedweights


def _packvoxeldata(voxnum: int, voxelargs: tuple) -> list:
    """
    Pack voxel data into a list structure.

    Parameters
    ----------
    voxnum : int
        The voxel index to extract data from.
    voxelargs : tuple
        A tuple containing voxel data with the following structure:
        - voxelargs[0]: 2D array of shape (n_voxels, n_features) containing voxel features
        - voxelargs[1]: 1D array of shape (n_voxels,) containing voxel labels or values
        - voxelargs[2]: Additional voxel parameter (type depends on context)
        - voxelargs[3]: Additional voxel parameter (type depends on context)

    Returns
    -------
    list
        A list containing:
        - [0]: 1D array of shape (n_features,) representing voxel features at voxnum
        - [1]: scalar value representing voxel label or value at voxnum
        - [2]: third element from voxelargs tuple
        - [3]: fourth element from voxelargs tuple

    Notes
    -----
    This function is typically used for extracting and packaging voxel data for further processing
    in neuroimaging or 3D data analysis workflows.

    Examples
    --------
    >>> voxel_features = np.array([[1, 2, 3], [4, 5, 6]])
    >>> voxel_labels = np.array([10, 20])
    >>> extra_param1 = "param1"
    >>> extra_param2 = "param2"
    >>> result = _packvoxeldata(0, (voxel_features, voxel_labels, extra_param1, extra_param2))
    >>> print(result)
    [[1, 2, 3], 10, 'param1', 'param2']
    """
    return [(voxelargs[0])[voxnum, :], (voxelargs[1])[voxnum], voxelargs[2], voxelargs[3]]


def _unpackvoxeldata(retvals: tuple, voxelproducts: list) -> None:
    """
    Unpack voxel data from retvals into voxelproducts arrays.

    Parameters
    ----------
    retvals : tuple
        Tuple containing voxel data to be unpacked. Expected to contain at least 5 elements
        where retvals[0] is the index for assignment and retvals[1:5] are the data arrays.
    voxelproducts : list
        List of arrays where voxel data will be unpacked. Expected to contain exactly 4 arrays
        that will be modified in-place.

    Returns
    -------
    None
        This function modifies the voxelproducts arrays in-place and does not return anything.

    Notes
    -----
    This function performs in-place assignment of voxel data. The first element of retvals
    is used as an index for row-wise assignment into each of the four arrays in voxelproducts.
    All arrays in voxelproducts must have sufficient dimensions to accommodate the assignment.

    Examples
    --------
    >>> import numpy as np
    >>> voxel_data = [np.zeros((10, 5)), np.zeros((10, 5)), np.zeros((10, 5)), np.zeros((10, 5))]
    >>> retvals = (2, np.array([1, 2, 3, 4, 5]), np.array([6, 7, 8, 9, 10]),
    ...            np.array([11, 12, 13, 14, 15]), np.array([16, 17, 18, 19, 20]))
    >>> _unpackvoxeldata(retvals, voxel_data)
    >>> print(voxel_data[0][2, :])  # Should print [1 2 3 4 5]
    """
    (voxelproducts[0])[retvals[0], :] = retvals[1]
    (voxelproducts[1])[retvals[0], :] = retvals[2]
    (voxelproducts[2])[retvals[0], :] = retvals[3]
    (voxelproducts[3])[retvals[0], :] = retvals[4]


def findecho(
    nlags: int,
    shiftedtcs: NDArray,
    sigmav: NDArray,
    arcoefs: NDArray,
    pacf: NDArray,
    sigma: NDArray,
    phi: NDArray,
) -> None:
    """
    Compute autoregressive parameters and related statistics for each voxel using Levinson-Durbin recursion.

    This function applies the Levinson-Durbin algorithm to estimate autoregressive coefficients
    and associated statistics for time series data from multiple voxels. The algorithm computes
    the variance, autoregressive coefficients, partial autocorrelations, and other related
    parameters for each voxel's time series.

    Parameters
    ----------
    nlags : int
        Number of lags to compute for the autoregressive model.
    shiftedtcs : NDArray
        Input time series data with shape (n_voxels, n_timepoints), where each row represents
        a voxel's time series.
    sigmav : NDArray
        Output array for variance estimates, shape (n_voxels,).
    arcoefs : NDArray
        Output array for autoregressive coefficients, shape (n_voxels, nlags).
    pacf : NDArray
        Output array for partial autocorrelations, shape (n_voxels, nlags).
    sigma : NDArray
        Output array for sigma values, shape (n_voxels, nlags).
    phi : NDArray
        Output array for phi values, shape (n_voxels, nlags).

    Returns
    -------
    None
        This function modifies the input arrays in-place and does not return any value.

    Notes
    -----
    The function uses `statsmodels.tsa.stattools.levinson_durbin` to compute the autoregressive
    parameters. This algorithm is efficient for computing autoregressive parameters and is
    commonly used in time series analysis for estimating model parameters.

    Examples
    --------
    >>> import numpy as np
    >>> from statsmodels.tsa import stattools
    >>> nlags = 5
    >>> shiftedtcs = np.random.randn(100, 1000)
    >>> sigmav = np.zeros(100)
    >>> arcoefs = np.zeros((100, 5))
    >>> pacf = np.zeros((100, 5))
    >>> sigma = np.zeros((100, 5))
    >>> phi = np.zeros((100, 5))
    >>> findecho(nlags, shiftedtcs, sigmav, arcoefs, pacf, sigma, phi)
    """
    inputshape = np.shape(shiftedtcs)
    for voxel in range(inputshape[0]):
        sigmav[voxel], arcoefs[voxel, :], pacf[voxel, :], sigma[voxel, :], phi[voxel, :] = (
            sm.tsa.stattools.levinson_durbin(shiftedtcs[voxel, :], nlags=nlags, isacov=False)
        )


def alignvoxels(
    fmridata: NDArray,
    fmritr: float,
    shiftedtcs: NDArray,
    weights: NDArray,
    paddedshiftedtcs: NDArray,
    paddedweights: NDArray,
    lagtimes: NDArray,
    lagmask: NDArray,
    detrendorder: int = 1,
    offsettime: float = 0.0,
    nprocs: int = 1,
    alwaysmultiproc: bool = False,
    showprogressbar: bool = True,
    chunksize: int = 1000,
    padtrs: int = 60,
    debug: bool = False,
    rt_floattype: np.dtype = np.float64,
) -> int:
    """
    Apply temporal alignment (timeshift) to all voxels in fMRI data based on correlation peaks.

    This routine applies a time shift to every voxel in the fMRI data based on the lag times
    determined from cross-correlation with a reference signal. The function modifies the input
    arrays in-place to store the aligned timecourses and associated weights.

    Parameters
    ----------
    fmridata : 4D NDArray
        fMRI data, filtered to the passband, with shape (nx, ny, nz, nt)
    fmritr : float
        Data repetition time (TR), in seconds
    shiftedtcs : 4D NDArray
        Destination array for time-aligned voxel timecourses, shape (nx, ny, nz, nt)
    weights : 4D NDArray
        Weights for each timepoint in the final regressor, shape (nx, ny, nz, nt)
    paddedshiftedtcs : 4D NDArray
        Time-aligned voxel timecourses with padding, shape (nx, ny, nz, nt + 2*padtrs)
    paddedweights : 4D NDArray
        Weights for each timepoint in the padded regressor, shape (nx, ny, nz, nt + 2*padtrs)
    lagtimes : 3D NDArray
        Time delay of maximum crosscorrelation in seconds, shape (nx, ny, nz)
    lagmask : 3D NDArray
        Mask of voxels with successful correlation fits, shape (nx, ny, nz)
    detrendorder : int, optional
        Order of polynomial used to detrend the data (default is 1)
    offsettime : float, optional
        Global time shift to apply to all timecourses in seconds (default is 0.0)
    nprocs : int, optional
        Number of processes to use for multiprocessing (default is 1)
    alwaysmultiproc : bool, optional
        If True, always use multiprocessing even for small datasets (default is False)
    showprogressbar : bool, optional
        If True, show a progress bar during processing (default is True)
    chunksize : int, optional
        Number of voxels to process per chunk in multiprocessing (default is 1000)
    padtrs : int, optional
        Number of timepoints to pad on each end of the timecourses (default is 60)
    debug : bool, optional
        If True, enable additional debugging output (default is False)
    rt_floattype : np.dtype, optional
        Function to coerce variable types (default is np.float64)

    Returns
    -------
    volumetotal : int
        Total number of voxels processed

    Notes
    -----
    This function modifies the input arrays `shiftedtcs`, `weights`, `paddedshiftedtcs`, and
    `paddedweights` in-place. The `lagtimes` and `lagmask` arrays are used to determine the
    appropriate time shifts for each voxel.

    Examples
    --------
    >>> import numpy as np
    >>> from rapidtide import alignvoxels
    >>> fmridata = np.random.rand(64, 64, 32, 100)
    >>> fmritr = 2.0
    >>> shiftedtcs = np.zeros_like(fmridata)
    >>> weights = np.ones_like(fmridata)
    >>> paddedshiftedtcs = np.zeros((64, 64, 32, 100 + 2*60))
    >>> paddedweights = np.ones((64, 64, 32, 100 + 2*60))
    >>> lagtimes = np.random.rand(64, 64, 32)
    >>> lagmask = np.ones((64, 64, 32))
    >>> volumetotal = alignvoxels(
    ...     fmridata, fmritr, shiftedtcs, weights, paddedshiftedtcs, paddedweights,
    ...     lagtimes, lagmask, nprocs=4
    ... )
    >>> print(f"Processed {volumetotal} voxels")
    """
    inputshape = np.shape(fmridata)
    voxelargs = [fmridata, lagtimes, padtrs, fmritr]
    voxelfunc = _procOneVoxelTimeShift
    packfunc = _packvoxeldata
    unpackfunc = _unpackvoxeldata
    voxeltargets = [
        shiftedtcs,
        weights,
        paddedshiftedtcs,
        paddedweights,
    ]
    if debug:
        print("alignvoxels: {inputshape}")
        print("volumetotal: {volumetotal}")

    # timeshift the valid voxels
    # NOTE need to figure out how to use kwargs to pass extra arguments
    volumetotal = tide_genericmultiproc.run_multiproc(
        voxelfunc,
        packfunc,
        unpackfunc,
        voxelargs,
        voxeltargets,
        inputshape,
        lagmask,
        LGR,
        nprocs,
        alwaysmultiproc,
        showprogressbar,
        chunksize,
        detrendorder=detrendorder,
        offsettime=offsettime,
        debug=debug,
    )

    LGR.info(
        "Timeshift applied to " + str(int(volumetotal)) + " voxels",
    )

    # garbage collect
    uncollected = gc.collect()
    if uncollected != 0:
        LGR.info(f"garbage collected - unable to collect {uncollected} objects")
    else:
        LGR.info("garbage collected")

    return volumetotal


def makerefinemask(
    lagstrengths: NDArray,
    lagtimes: NDArray,
    lagsigma: NDArray,
    lagmask: NDArray,
    offsettime: float = 0.0,
    ampthresh: float = 0.3,
    lagmaskside: str = "both",
    lagminthresh: float = 0.5,
    lagmaxthresh: float = 5.0,
    sigmathresh: float = 100,
    cleanrefined: bool = False,
    bipolar: bool = False,
    includemask: NDArray | None = None,
    excludemask: NDArray | None = None,
    fixdelay: bool = False,
    debug: bool = False,
    rt_floattype: np.dtype = np.float64,
) -> tuple[int, NDArray | None, int, int, int, int, int]:
    """
    Determine which voxels should be used for regressor refinement based on correlation strength,
    time delay, and peak width criteria.

    This routine evaluates a set of voxels defined by their correlation properties and applies
    various thresholds to determine which ones are suitable for refinement. It supports optional
    masking, bipolar correlation handling, and debugging output.

    Parameters
    ----------
    lagstrengths : ndarray
        3D numpy float array of maximum correlation coefficients in every voxel.
    lagtimes : ndarray
        3D numpy float array of time delays (in seconds) of maximum crosscorrelation.
    lagsigma : ndarray
        3D numpy float array of Gaussian widths (in seconds) of the crosscorrelation peaks.
    lagmask : ndarray
        3D numpy float array masking voxels with successful correlation fits.
    offsettime : float, optional
        Offset time in seconds to apply to all regressors. Default is 0.0.
    ampthresh : float, optional
        Lower limit of correlation values to consider for refine mask inclusion.
        If negative, treated as percentile. Default is 0.3.
    lagmaskside : str, optional
        Which side of the lag values to consider: 'upper', 'lower', or 'both'.
        Default is 'both'.
    lagminthresh : float, optional
        Lower limit of absolute lag values to consider for inclusion. Default is 0.5.
    lagmaxthresh : float, optional
        Upper limit of absolute lag values to consider for inclusion. Default is 5.0.
    sigmathresh : float, optional
        Upper limit of lag peak width (in seconds) for inclusion. Default is 100.
    cleanrefined : bool, optional
        If True, uses the full location mask for refinement; otherwise, uses the refined mask.
        Default is False.
    bipolar : bool, optional
        If True, considers both positive and negative correlation peaks. Default is False.
    includemask : ndarray, optional
        3D array masking voxels to include in refinement. Default is None (all voxels).
    excludemask : ndarray, optional
        3D array masking voxels to exclude from refinement. Default is None (no voxels).
    fixdelay : bool, optional
        If True, uses the raw `lagmask` without applying delay thresholds. Default is False.
    debug : bool, optional
        Enable additional debugging output. Default is False.
    rt_floattype : np.dtype, optional
        Data type for internal arrays. Default is `np.float64`.

    Returns
    -------
    volumetotal : int
        Number of voxels processed for refinement.
    maskarray : ndarray or None
        3D mask of voxels used for refinement. Returns None if no voxels remain after filtering.
    locationfails : int
        Number of voxels eliminated due to include/exclude mask constraints.
    ampfails : int
        Number of voxels eliminated due to low correlation amplitude.
    lagfails : int
        Number of voxels eliminated due to lag value out of range.
    sigmafails : int
        Number of voxels eliminated due to wide correlation peak.
    numinmask : int
        Total number of voxels in the original `lagmask`.

    Notes
    -----
    - The function applies multiple filtering steps: amplitude, lag time, and sigma (peak width).
    - If `ampthresh` is negative, it is interpreted as a percentile threshold.
    - The `lagmaskside` parameter controls which direction of the lag values to consider:
      'upper' for positive lags, 'lower' for negative lags, 'both' for both.
    - If no voxels remain after filtering, an error is printed and the function returns early.

    Examples
    --------
    >>> import numpy as np
    >>> lagstrengths = np.random.rand(10, 10, 10)
    >>> lagtimes = np.random.rand(10, 10, 10) * 10
    >>> lagsigma = np.random.rand(10, 10, 10) * 50
    >>> lagmask = np.ones((10, 10, 10))
    >>> volumetotal, maskarray, locfails, ampfails, lagfails, sigfails, numinmask = makerefinemask(
    ...     lagstrengths, lagtimes, lagsigma, lagmask, ampthresh=0.4, lagminthresh=1.0
    ... )
    """
    if ampthresh < 0.0:
        if bipolar:
            theampthresh = tide_stats.getfracval(np.fabs(lagstrengths), -ampthresh, nozero=True)
        else:
            theampthresh = tide_stats.getfracval(lagstrengths, -ampthresh, nozero=True)
        LGR.info(f"setting ampthresh to the {-100.0 * ampthresh}th percentile ({theampthresh})")
    else:
        theampthresh = ampthresh
    if debug:
        print(f"makerefinemask: {theampthresh=}")
    if bipolar:
        ampmask = np.where(np.fabs(lagstrengths) >= theampthresh, np.int16(1), np.int16(0))
    else:
        ampmask = np.where(lagstrengths >= theampthresh, np.int16(1), np.int16(0))
    if fixdelay:
        delaymask = lagmask + 0
    else:
        if lagmaskside == "upper":
            delaymask = np.where(
                (lagtimes - offsettime) > lagminthresh,
                np.int16(1),
                np.int16(0),
            ) * np.where(
                (lagtimes - offsettime) < lagmaxthresh,
                np.int16(1),
                np.int16(0),
            )
        elif lagmaskside == "lower":
            delaymask = np.where(
                (lagtimes - offsettime) < -lagminthresh,
                np.int16(1),
                np.int16(0),
            ) * np.where(
                (lagtimes - offsettime) > -lagmaxthresh,
                np.int16(1),
                np.int16(0),
            )
        else:
            abslag = abs(lagtimes - offsettime)
            delaymask = np.where(abslag > lagminthresh, np.int16(1), np.int16(0)) * np.where(
                abslag < lagmaxthresh, np.int16(1), np.int16(0)
            )
        if debug:
            print(f"makerefinemask: {lagmaskside=}")
            print(f"makerefinemask: {lagminthresh=}")
            print(f"makerefinemask: {lagmaxthresh=}")
            print(f"makerefinemask: {offsettime=}")
    sigmamask = np.where(lagsigma < sigmathresh, np.int16(1), np.int16(0))
    locationmask = lagmask + 0
    if includemask is not None:
        locationmask = locationmask * includemask
    if excludemask is not None:
        locationmask = locationmask * (1 - excludemask)
    locationmask = locationmask.astype(np.int16)
    LGR.info("location mask created")

    # first generate the refine mask
    locationfails = np.sum(1 - locationmask)
    ampfails = np.sum(1 - ampmask * locationmask)
    lagfails = np.sum(1 - delaymask * locationmask)
    sigmafails = np.sum(1 - sigmamask * locationmask)
    refinemask = locationmask * ampmask * delaymask * sigmamask
    if tide_stats.getmasksize(refinemask) == 0:
        print("ERROR: no voxels in the refine mask:")
        print(
            "\n	",
            locationfails,
            " locationfails",
            "\n	",
            ampfails,
            " ampfails",
            "\n	",
            lagfails,
            " lagfails",
            "\n	",
            sigmafails,
            " sigmafails",
        )
        if (includemask is None) and (excludemask is None):
            print("\nRelax ampthresh, delaythresh, or sigmathresh - exiting")
        else:
            print(
                "\nChange include/exclude masks or relax ampthresh, delaythresh, or sigmathresh - exiting"
            )
        return 0, None, locationfails, ampfails, lagfails, sigmafails, 0

    if cleanrefined:
        shiftmask = locationmask
    else:
        shiftmask = refinemask
    volumetotal = np.sum(shiftmask)
    LGR.info(
        f"{int(volumetotal)} voxels will be used for refinement:"
        + f"\n	{locationfails} locationfails"
        + f"\n	{ampfails} ampfails"
        + f"\n	{lagfails} lagfails"
        + f"\n	{sigmafails} sigmafails"
    )
    numinmask = np.sum(lagmask)
    if numinmask is None:
        numinmask = 0

    return volumetotal, shiftmask, locationfails, ampfails, lagfails, sigmafails, numinmask


def prenorm(
    shiftedtcs: NDArray,
    refinemask: NDArray,
    lagtimes: NDArray,
    lagmaxthresh: float,
    lagstrengths: NDArray,
    R2vals: NDArray,
    refineprenorm: str,
    refineweighting: str,
    debug: bool = False,
) -> None:
    """
    Apply pre-normalization and weighting to shifted time correlation data.

    This function performs normalization and weighting of time correlation data
    based on specified criteria. It modifies the input `shiftedtcs` array in-place.

    Parameters
    ----------
    shiftedtcs : NDArray
        Array of shifted time correlation data, shape (n_samples, n_timepoints).
    refinemask : NDArray
        Boolean mask for refining data, shape (n_samples,).
    lagtimes : NDArray
        Array of lag times, shape (n_samples,).
    lagmaxthresh : float
        Threshold for lag time normalization.
    lagstrengths : NDArray
        Array of lag strengths, shape (n_samples,).
    R2vals : NDArray
        Array of R-squared values, shape (n_samples,).
    refineprenorm : str
        Normalization method to use: 'mean', 'var', 'std', or 'invlag'.
        If any other value is provided, unit normalization is applied.
    refineweighting : str
        Weighting method to use: 'R', 'R2', or other (default weighting based on lagstrengths).
    debug : bool, optional
        If True, print debug information about input shapes and intermediate values.

    Returns
    -------
    None
        The function modifies `shiftedtcs` in-place.

    Notes
    -----
    The function applies normalization using a divisor computed according to the
    `refineprenorm` parameter and then applies weights based on `refineweighting`.
    The `shiftedtcs` array is updated in-place.

    Examples
    --------
    >>> import numpy as np
    >>> shiftedtcs = np.random.rand(10, 5)
    >>> refinemask = np.ones(10, dtype=bool)
    >>> lagtimes = np.arange(10)
    >>> lagmaxthresh = 2.0
    >>> lagstrengths = np.random.rand(10)
    >>> R2vals = np.random.rand(10)
    >>> prenorm(shiftedtcs, refinemask, lagtimes, lagmaxthresh, lagstrengths, R2vals, "mean", "R", debug=True)
    """
    if debug:
        print(f"{shiftedtcs.shape=}"),
        print(f"{refinemask.shape=}"),
        print(f"{lagtimes.shape=}"),
        print(f"{lagmaxthresh=}"),
        print(f"{lagstrengths.shape=}"),
        print(f"{R2vals.shape=}"),
        print(f"{refineprenorm=}"),
        print(f"{refineweighting=}"),
    if refineprenorm == "mean":
        thedivisor = np.mean(shiftedtcs, axis=1)
    elif refineprenorm == "var":
        thedivisor = np.var(shiftedtcs, axis=1)
    elif refineprenorm == "std":
        thedivisor = np.std(shiftedtcs, axis=1)
    elif refineprenorm == "invlag":
        thedivisor = np.where(np.fabs(lagtimes) < lagmaxthresh, lagmaxthresh - lagtimes, 0.0)
    else:
        thedivisor = np.ones_like(shiftedtcs[:, 0])

    normfac = np.where(thedivisor != 0.0, 1.0 / thedivisor, 0.0)

    if refineweighting == "R":
        thisweight = lagstrengths
    elif refineweighting == "R2":
        thisweight = R2vals
    else:
        thisweight = np.where(lagstrengths > 0.0, 1.0, -1.0)
    thisweight *= refinemask

    if debug:
        print(f"{thedivisor.shape=}")
        print(f"{normfac.shape=}")
        print(f"{thisweight.shape=}")

    shiftedtcs *= (normfac * thisweight)[:, None]


def dorefine(
    shiftedtcs: NDArray,
    refinemask: NDArray,
    weights: NDArray,
    theprefilter: Any,
    fmritr: float,
    passnum: int,
    lagstrengths: NDArray,
    lagtimes: NDArray,
    refinetype: str,
    fmrifreq: float,
    outputname: str,
    detrendorder: int = 1,
    pcacomponents: float | str = 0.8,
    dodispersioncalc: bool = False,
    dispersioncalc_lower: float = 0.0,
    dispersioncalc_upper: float = 0.0,
    dispersioncalc_step: float = 0.0,
    windowfunc: str = "hamming",
    cleanrefined: bool = False,
    bipolar: bool = False,
    debug: bool = False,
    rt_floattype: np.dtype = np.float64,
) -> tuple[int, NDArray]:
    """
    Refine timecourses using specified method (ICA, PCA, weighted average, or unweighted average).

    This function applies a refinement process to a set of timecourses based on a mask and
    weights. It supports multiple refinement techniques including ICA, PCA, and averaging,
    and can optionally perform dispersion calculation and cleaning of refined data.

    Parameters
    ----------
    shiftedtcs : ndarray
        Array of shape (n_voxels, n_timepoints) containing the shifted timecourses.
    refinemask : ndarray
        Boolean mask indicating which voxels to include in refinement.
    weights : ndarray
        Array of shape (n_voxels, n_timepoints) containing weights for each voxel.
    theprefilter : Any
        Pre-filter object with an `apply` method to filter the data.
    fmritr : float
        fMRI repetition time in seconds.
    passnum : int
        Pass number for output file naming.
    lagstrengths : ndarray
        Array of lag strengths for each voxel.
    lagtimes : ndarray
        Array of lag times for each voxel.
    refinetype : str
        Type of refinement to perform: 'ica', 'pca', 'weighted_average', or 'unweighted_average'.
    fmrifreq : float
        fMRI frequency in Hz.
    outputname : str
        Base name for output files.
    detrendorder : int, optional
        Order of detrending for correlation normalization (default is 1).
    pcacomponents : float or str, optional
        Number of PCA components to use. If < 1, treated as fraction of variance; if 'mle', uses MLE.
        Default is 0.8.
    dodispersioncalc : bool, optional
        If True, compute dispersion calculation across lag ranges (default is False).
    dispersioncalc_lower : float, optional
        Lower bound for dispersion calculation lag range (default is 0.0).
    dispersioncalc_upper : float, optional
        Upper bound for dispersion calculation lag range (default is 0.0).
    dispersioncalc_step : float, optional
        Step size for dispersion calculation lag range (default is 0.0).
    windowfunc : str, optional
        Window function for correlation normalization (default is "hamming").
    cleanrefined : bool, optional
        If True, remove linearly fitted discard data from refined output (default is False).
    bipolar : bool, optional
        If True, flip sign of negative lag strengths (default is False).
    debug : bool, optional
        If True, print debug information (default is False).
    rt_floattype : np.dtype, optional
        Data type for floating-point numbers (default is np.float64).

    Returns
    -------
    tuple[int, ndarray]
        A tuple containing:
        - `volumetotal`: int, total number of voxels included in refinement.
        - `outputdata`: ndarray, refined timecourse of shape (n_timepoints,).

    Notes
    -----
    - The function supports multiple refinement methods: ICA, PCA, weighted average, and
      unweighted average.
    - If `cleanrefined` is True, a linear regression is performed to remove discard data
      from the refined output.
    - If `dodispersioncalc` is True, dispersion calculation is performed across lag ranges
      and outputs are saved to files with the prefix `outputname`.

    Examples
    --------
    >>> import numpy as np
    >>> shiftedtcs = np.random.rand(100, 200)
    >>> refinemask = np.ones(100)
    >>> weights = np.ones((100, 200))
    >>> theprefilter = SomeFilter()
    >>> fmritr = 2.0
    >>> passnum = 1
    >>> lagstrengths = np.random.rand(100)
    >>> lagtimes = np.random.rand(100)
    >>> refinetype = "pca"
    >>> fmrifreq = 0.1
    >>> outputname = "test_output"
    >>> volumetotal, outputdata = dorefine(
    ...     shiftedtcs, refinemask, weights, theprefilter, fmritr, passnum,
    ...     lagstrengths, lagtimes, refinetype, fmrifreq, outputname
    ... )
    """
    # now generate the refined timecourse(s)
    inputshape = np.shape(shiftedtcs)
    validlist = np.where(refinemask > 0)[0]
    volumetotal = len(validlist)
    refinevoxels = shiftedtcs[validlist, :]
    if bipolar:
        for thevoxel in range(len(validlist)):
            if lagstrengths[validlist][thevoxel] < 0.0:
                refinevoxels[thevoxel, :] *= -1.0
    refineweights = weights[validlist]
    weightsum = np.sum(refineweights, axis=0) / volumetotal
    averagedata = np.sum(refinevoxels, axis=0) / volumetotal
    if cleanrefined:
        invalidlist = np.where((1 - refinemask) > 0)[0]
        discardvoxels = shiftedtcs[invalidlist]
        discardweights = weights[invalidlist]
        discardweightsum = np.sum(discardweights, axis=0) / volumetotal
        averagediscard = np.sum(discardvoxels, axis=0) / volumetotal
    if dodispersioncalc:
        LGR.info("splitting regressors by time lag for phase delay estimation")
        laglist = np.arange(
            dispersioncalc_lower,
            dispersioncalc_upper,
            dispersioncalc_step,
        )
        dispersioncalcout = np.zeros((np.shape(laglist)[0], inputshape[1]), dtype=rt_floattype)
        fftlen = int(inputshape[1] // 2)
        fftlen -= fftlen % 2
        dispersioncalcspecmag = np.zeros((np.shape(laglist)[0], fftlen), dtype=rt_floattype)
        dispersioncalcspecphase = np.zeros((np.shape(laglist)[0], fftlen), dtype=rt_floattype)
        ###### BBF dispersioncalc fails when the number of timepoints is odd (or even - not sure).  Works the other way.
        for lagnum in range(0, np.shape(laglist)[0]):
            lower = laglist[lagnum] - dispersioncalc_step / 2.0
            upper = laglist[lagnum] + dispersioncalc_step / 2.0
            inlagrange = np.where(
                refinemask
                * np.where(lower < lagtimes, np.int16(1), np.int16(0))
                * np.where(lagtimes < upper, np.int16(1), np.int16(0))
            )[0]
            LGR.info(
                f"\tsumming {np.shape(inlagrange)[0]} regressors with lags from {lower} to {upper}"
            )
            if np.shape(inlagrange)[0] > 0:
                dispersioncalcout[lagnum, :] = tide_math.corrnormalize(
                    np.mean(shiftedtcs[inlagrange], axis=0),
                    detrendorder=detrendorder,
                    windowfunc=windowfunc,
                )
                (
                    freqs,
                    dispersioncalcspecmag[lagnum, :],
                    dispersioncalcspecphase[lagnum, :],
                ) = tide_math.polarfft(dispersioncalcout[lagnum, :], 1.0 / fmritr)
            inlagrange = None
        tide_io.writenpvecs(
            dispersioncalcout,
            outputname + "_dispersioncalcvecs_pass" + str(passnum) + ".txt",
        )
        tide_io.writenpvecs(
            dispersioncalcspecmag,
            outputname + "_dispersioncalcspecmag_pass" + str(passnum) + ".txt",
        )
        tide_io.writenpvecs(
            dispersioncalcspecphase,
            outputname + "_dispersioncalcspecphase_pass" + str(passnum) + ".txt",
        )
        tide_io.writenpvecs(
            freqs,
            outputname + "_dispersioncalcfreqs_pass" + str(passnum) + ".txt",
        )

    if pcacomponents < 0.0:
        pcacomponents = "mle"
    elif pcacomponents >= 1.0:
        pcacomponents = int(np.round(pcacomponents))
    elif pcacomponents == 0.0:
        print("0.0 is not an allowed value for pcacomponents")
        sys.exit()
    else:
        pcacomponents = pcacomponents
    icacomponents = 1

    if refinetype == "ica":
        LGR.info("performing ica refinement")
        thefit = FastICA(n_components=icacomponents).fit(refinevoxels)  # Reconstruct signals
        LGR.info(f"Using first of {len(thefit.components_)} components")
        icadata = thefit.components_[0]
        filteredavg = tide_math.corrnormalize(
            theprefilter.apply(fmrifreq, averagedata),
            detrendorder=detrendorder,
        )
        filteredica = tide_math.corrnormalize(
            theprefilter.apply(fmrifreq, icadata),
            detrendorder=detrendorder,
        )
        thepxcorr = pearsonr(filteredavg, filteredica).statistic
        LGR.info(f"ica/avg correlation = {thepxcorr}")
        if thepxcorr > 0.0:
            outputdata = 1.0 * icadata
        else:
            outputdata = -1.0 * icadata
    elif refinetype == "pca":
        # use the method of "A novel perspective to calibrate temporal delays in cerebrovascular reactivity
        # using hypercapnic and hyperoxic respiratory challenges". NeuroImage 187, 154?165 (2019).
        LGR.info(f"performing pca refinement with pcacomponents set to {pcacomponents}")
        try:
            thefit = PCA(n_components=pcacomponents).fit(refinevoxels)
        except ValueError:
            if pcacomponents == "mle":
                LGR.info("mle estimation failed - falling back to pcacomponents=0.8")
                thefit = PCA(n_components=0.8).fit(refinevoxels)
            else:
                print("unhandled math exception in PCA refinement - exiting")
                sys.exit()
        LGR.info(
            f"Using {len(thefit.components_)} component(s), accounting for "
            + f"{100.0 * np.cumsum(thefit.explained_variance_ratio_)[len(thefit.components_) - 1]:.2f}% of the variance"
        )
        reduceddata = thefit.inverse_transform(thefit.transform(refinevoxels))
        if debug:
            print("complex processing: reduceddata.shape =", reduceddata.shape)
        pcadata = np.mean(reduceddata, axis=0)
        filteredavg = tide_math.corrnormalize(
            theprefilter.apply(fmrifreq, averagedata),
            detrendorder=detrendorder,
        )
        filteredpca = tide_math.corrnormalize(
            theprefilter.apply(fmrifreq, pcadata),
            detrendorder=detrendorder,
        )
        thepxcorr = pearsonr(filteredavg, filteredpca).statistic
        LGR.info(f"pca/avg correlation = {thepxcorr}")
        if thepxcorr > 0.0:
            outputdata = 1.0 * pcadata
        else:
            outputdata = -1.0 * pcadata
    elif refinetype == "weighted_average":
        LGR.info("performing weighted averaging refinement")
        outputdata = np.nan_to_num(averagedata / weightsum)
    else:
        LGR.info("performing unweighted averaging refinement")
        outputdata = averagedata

    if cleanrefined:
        thefit, R2 = tide_fit.mlregress(averagediscard, averagedata)

        fitcoff = thefit[0, 1]
        datatoremove = (fitcoff * averagediscard).astype(rt_floattype)
        outputdata -= datatoremove

    # garbage collect
    uncollected = gc.collect()
    if uncollected != 0:
        LGR.info(f"garbage collected - unable to collect {uncollected} objects")
    else:
        LGR.info("garbage collected")

    return volumetotal, outputdata
