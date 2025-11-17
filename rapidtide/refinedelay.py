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
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import numpy.polynomial.polynomial as poly
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.ndimage import median_filter
from statsmodels.robust import mad

import rapidtide.filter as tide_filt
import rapidtide.io as tide_io
import rapidtide.workflows.regressfrommaps as tide_regressfrommaps

global ratiotooffsetfunc, funcoffsets, maplimits


def smooth(y: NDArray, box_pts: int) -> NDArray:
    """Apply a simple moving average smooth to the input array.

    This function performs convolution with a uniform boxcar filter to smooth
    the input data. The smoothing is applied using the 'same' mode which
    returns an array of the same length as the input.

    Parameters
    ----------
    y : NDArray
        Input array to be smoothed.
    box_pts : int
        Number of points in the smoothing window. Must be a positive integer.

    Returns
    -------
    NDArray
        Smoothed array of the same shape as input `y`.

    Notes
    -----
    The smoothing is performed using numpy's convolve function with a boxcar
    filter of uniform weights. The 'same' mode ensures the output has the
    same length as the input, with edge effects handled by padding.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> smooth(data, 3)
    array([1.33333333, 2.33333333, 3.33333333, 4.33333333, 5.33333333])
    """
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth


def trainratiotooffset(
    lagtcgenerator: Any,
    timeaxis: NDArray,
    outputname: str,
    outputlevel: str,
    trainlagmin: float = 0.0,
    trainlagmax: float = 0.0,
    trainlagstep: float = 0.5,
    mindelay: float = -3.0,
    maxdelay: float = 3.0,
    numpoints: int = 501,
    smoothpts: int = 3,
    edgepad: int = 5,
    regressderivs: int = 1,
    LGR: Optional[Any] = None,
    TimingLGR: Optional[Any] = None,
    verbose: bool = False,
    debug: bool = False,
) -> None:
    """
    Train a mapping from derivative ratio to delay offset using lagged time courses.

    This function generates synthetic fMRI data by applying time shifts to the
    input lagged time course generator and computes derivative ratios to estimate
    the relationship between the ratio of derivatives and the corresponding delay.
    The resulting mapping is stored globally and optionally saved to BIDS-style
    TSV files.

    Parameters
    ----------
    lagtcgenerator : Any
        An object that provides the `yfromx` method for generating lagged time courses.
    timeaxis : NDArray
        The time axis (in seconds) for the fMRI data.
    outputname : str
        Base name for output files (e.g., BIDS entity).
    outputlevel : str
        Determines level of output; valid values are "min", "onlyregressors", or others.
    trainlagmin : float, optional
        Minimum lag value for training offsets (default is 0.0).
    trainlagmax : float, optional
        Maximum lag value for training offsets (default is 0.0).
    trainlagstep : float, optional
        Step size for generating training offsets (default is 0.5).
    mindelay : float, optional
        Minimum delay to consider in the delay map (default is -3.0).
    maxdelay : float, optional
        Maximum delay to consider in the delay map (default is 3.0).
    numpoints : int, optional
        Number of points in the delay grid (default is 501).
    smoothpts : int, optional
        Number of points for smoothing (default is 3).
    edgepad : int, optional
        Padding applied to edges during processing (default is 5).
    regressderivs : int, optional
        Number of derivatives to regress (default is 1).
    LGR : Optional[Any], optional
        Logging object for verbose output (default is None).
    TimingLGR : Optional[Any], optional
        Timing logging object (default is None).
    verbose : bool, optional
        Enable verbose logging if True (default is False).
    debug : bool, optional
        Enable debug output if True (default is False).

    Returns
    -------
    None
        This function does not return any value but updates global variables:
        - `ratiotooffsetfunc`: List of cubic spline interpolants mapping ratio to delay.
        - `funcoffsets`: List of offset values used in training.
        - `maplimits`: Tuple of minimum and maximum ratio values for the mapping.

    Notes
    -----
    - The function uses `getderivratios` to compute derivative ratios.
    - Output files are written in BIDS TSV format.
    - The mapping function is saved globally for future use.

    Examples
    --------
    >>> trainratiotooffset(
    ...     lagtcgenerator=generator,
    ...     timeaxis=time_axis,
    ...     outputname="sub-01",
    ...     outputlevel="full",
    ...     trainlagmin=-1.0,
    ...     trainlagmax=1.0,
    ...     trainlagstep=0.2,
    ...     mindelay=-2.0,
    ...     maxdelay=2.0,
    ...     numpoints=201,
    ...     smoothpts=5,
    ...     edgepad=3,
    ...     regressderivs=2,
    ...     verbose=True,
    ...     debug=False
    ... )
    """
    global ratiotooffsetfunc, funcoffsets, maplimits

    if debug:
        print("ratiotooffsetfunc:")
        lagtcgenerator.info(prefix="\t")
        print("\ttimeaxis:", timeaxis)
        print("\toutputname:", outputname)
        print("\ttrainlagmin:", trainlagmin)
        print("\ttrainlagmax:", trainlagmax)
        print("\ttrainlagstep:", trainlagstep)
        print("\tmindelay:", mindelay)
        print("\tmaxdelay:", maxdelay)
        print("\tsmoothpts:", smoothpts)
        print("\tedgepad:", edgepad)
        print("\tregressderivs:", regressderivs)
        print("\tlagtcgenerator:", lagtcgenerator)

    # make a delay map
    delaystep = (maxdelay - mindelay) / (numpoints - 1)
    lagtimes = np.linspace(
        mindelay - edgepad * delaystep,
        maxdelay + edgepad * delaystep,
        numpoints + 2 * edgepad,
        endpoint=True,
    )
    if debug:
        print(f"{mindelay=}")
        print(f"{maxdelay=}")
        print(f"{delaystep=}")
        print("lagtimes=", lagtimes)

    # set up for getratioderivs call
    rt_floattype = "float64"
    internalvalidfmrishape = (numpoints + 2 * edgepad, timeaxis.shape[0])
    fmridata = np.zeros(internalvalidfmrishape, dtype=float)
    fmrimask = np.ones(numpoints + 2 * edgepad, dtype=float)
    validvoxels = np.where(fmrimask > 0)[0]
    sLFOfitmean = np.zeros(numpoints + 2 * edgepad, dtype=rt_floattype)
    rvalue = np.zeros(numpoints + 2 * edgepad, dtype=rt_floattype)
    r2value = np.zeros(numpoints + 2 * edgepad, dtype=rt_floattype)
    fitNorm = np.zeros((numpoints + 2 * edgepad, 2), dtype=rt_floattype)
    fitcoeff = np.zeros((numpoints + 2 * edgepad, 2), dtype=rt_floattype)
    movingsignal = np.zeros(internalvalidfmrishape, dtype=rt_floattype)
    lagtc = np.zeros(internalvalidfmrishape, dtype=rt_floattype)
    filtereddata = np.zeros(internalvalidfmrishape, dtype=rt_floattype)
    sampletime = timeaxis[1] - timeaxis[0]
    optiondict = {
        "regressfiltthreshval": 0.0,
        "saveminimumsLFOfiltfiles": False,
        "nprocs_makelaggedtcs": 1,
        "nprocs_regressionfilt": 1,
        "mp_chunksize": 1000,
        "showprogressbar": False,
        "alwaysmultiproc": False,
        "focaldebug": debug,
        "fmrifreq": 1.0 / sampletime,
    }

    if trainlagmax - trainlagmin > 0.0:
        numnegoffsets = np.max((-int(np.round(trainlagmin / trainlagstep, 0)), 1))
        numposoffsets = np.max((int(np.round(trainlagmax / trainlagstep, 0)), 1))
        numoffsets = numnegoffsets + 1 + numposoffsets
        trainoffsets = (
            np.linspace(0, (numoffsets - 1) * trainlagstep, numoffsets, endpoint=True)
            - numnegoffsets * trainlagstep
        )
    else:
        trainoffsets = np.array([0.0], dtype=float)
        numoffsets = 1
    if debug:
        print("trainoffsets:", trainoffsets)
    allsmoothregressderivratios = np.zeros(
        (numpoints + 2 * edgepad, numoffsets), dtype=rt_floattype
    )
    theEVs = np.zeros((numoffsets, timeaxis.shape[0]), dtype=float)

    if verbose and (LGR is not None):
        thisLGR = LGR
        thisTimingLGR = TimingLGR
    else:
        thisLGR = None
        thisTimingLGR = None

    for whichoffset in range(numoffsets):
        thisoffset = trainoffsets[whichoffset]

        # now make synthetic fMRI data
        for i in range(numpoints + 2 * edgepad):
            fmridata[i, :] = lagtcgenerator.yfromx(timeaxis - lagtimes[i] + thisoffset)

        theEVs[whichoffset, :] = lagtcgenerator.yfromx(timeaxis + thisoffset)

        regressderivratios, regressrvalues = getderivratios(
            fmridata,
            validvoxels,
            timeaxis + thisoffset,
            0.0 * lagtimes,
            fmrimask,
            lagtcgenerator,
            "glm",
            "refinedelaytest",
            sampletime,
            sLFOfitmean,
            rvalue,
            r2value,
            fitNorm[:, :2],
            fitcoeff[:, :2],
            movingsignal,
            lagtc,
            filtereddata,
            thisLGR,
            thisTimingLGR,
            optiondict,
            regressderivs=regressderivs,
            debug=debug,
        )
        if debug:
            print("before trimming")
            print(f"{regressderivratios.shape=}")
            print(f"{lagtimes.shape=}")
        if regressderivs == 1:
            smoothregressderivratios = tide_filt.unpadvec(
                smooth(
                    tide_filt.padvec(regressderivratios, padlen=20, padtype="constant"), smoothpts
                ),
                padlen=20,
            )
            # regressderivratios = regressderivratios[edgepad:-edgepad]
            allsmoothregressderivratios[:, whichoffset] = smoothregressderivratios + 0.0
        else:
            smoothregressderivratios = np.zeros_like(regressderivratios)
            for i in range(regressderivs):
                allsmoothregressderivratios[i, :] = tide_filt.unpadvec(
                    smooth(
                        tide_filt.padvec(regressderivratios[i, :], padlen=20, padtype="constant"),
                        smoothpts,
                    ),
                    padlen=20,
                )
            # regressderivratios = regressderivratios[:, edgepad:-edgepad]
            allsmoothregressderivratios = smoothregressderivratios + 0.0

    allsmoothregressderivratios = allsmoothregressderivratios[edgepad:-edgepad, :]
    lagtimes = lagtimes[edgepad:-edgepad]
    if debug:
        print("after trimming")
        print(f"{regressderivratios.shape=}")
        print(f"{allsmoothregressderivratios.shape=}")
        print(f"{lagtimes.shape=}")

    # find the minimum legal limits of the mapping function
    highestlowerlim = 0
    lowestupperlim = numpoints
    for whichoffset in range(numoffsets):
        xaxis = allsmoothregressderivratios[::-1, whichoffset]
        midpoint = int(len(xaxis) // 2)
        lowerlim = midpoint + 0
        while (lowerlim > 1) and xaxis[lowerlim] > xaxis[lowerlim - 1]:
            lowerlim -= 1
        upperlim = midpoint + 0
        while (upperlim < len(xaxis) - 2) and xaxis[upperlim] < xaxis[upperlim + 1]:
            upperlim += 1
        if lowerlim > highestlowerlim:
            highestlowerlim = lowerlim
        if upperlim < lowestupperlim:
            lowestupperlim = upperlim

    ratiotooffsetfunc = []
    funcoffsets = []
    for whichoffset in range(numoffsets):
        xaxis = allsmoothregressderivratios[::-1, whichoffset]
        yaxis = lagtimes[::-1]
        xaxis = xaxis[highestlowerlim : lowestupperlim + 1]
        yaxis = yaxis[highestlowerlim : lowestupperlim + 1]
        ratiotooffsetfunc.append(CubicSpline(xaxis, yaxis))
        funcoffsets.append(trainoffsets[whichoffset] + 0.0)
    maplimits = (xaxis[0], xaxis[-1])

    if outputlevel != "min" and outputlevel != "onlyregressors":
        resampaxis = np.linspace(xaxis[0], xaxis[-1], num=len(xaxis), endpoint=True)
        outputfuncs = np.zeros((resampaxis.size, numoffsets), dtype=float)
        colnames = []
        for whichoffset in range(numoffsets):
            colnames.append(f"{funcoffsets[whichoffset]}")
            outputfuncs[:, whichoffset] = ratiotooffsetfunc[whichoffset](resampaxis)
        if debug:
            print(f"{colnames=}")
            print(f"{outputfuncs.shape=}")
        tide_io.writebidstsv(
            f"{outputname}_desc-ratiotodelayfunc_timeseries",
            np.transpose(outputfuncs),
            1.0 / (resampaxis[1] - resampaxis[0]),
            starttime=resampaxis[0],
            columns=colnames,
            extraheaderinfo={
                "Description": "The function mapping derivative ratio to delay",
                "minratio": f"{resampaxis[0]}",
                "maxratio": f"{resampaxis[-1]}",
            },
            xaxislabel="coefficientratio",
            yaxislabel="time",
            append=False,
        )
        if numoffsets > 1:
            print(f"{theEVs.shape=}, {numoffsets=}, {(numoffsets>1)=}")
            tide_io.writebidstsv(
                f"{outputname}_desc-trainratioEV_timeseries",
                theEVs,
                1.0 / (timeaxis[1] - timeaxis[0]),
                starttime=timeaxis[0],
                columns=colnames,
                extraheaderinfo={"Description": f"EVs used for each offset"},
                append=False,
            )


def ratiotodelay(theratio: float, offset: float = 0.0, debug: bool = False) -> Tuple[float, float]:
    """
    Convert a ratio to a delay value using lookup tables and offset compensation.

    This function maps a given ratio to a corresponding delay value by interpolating
    between pre-calculated offset values. It handles boundary conditions by clamping
    the ratio to predefined limits and applies offset compensation for accurate
    delay calculation.

    Parameters
    ----------
    theratio : float
        The input ratio value to be converted to delay. This value is used as input
        to the lookup function after being clamped to the valid range.
    offset : float, optional
        Offset value used for compensation and lookup table selection. Default is 0.0.
    debug : bool, optional
        Flag to enable debug output. Default is False.

    Returns
    -------
    Tuple[float, float]
        A tuple containing:
        - The calculated delay value based on the ratio and offset
        - The closest offset value used for the lookup

    Notes
    -----
    The function uses global variables `ratiotooffsetfunc`, `funcoffsets`, and `maplimits`:
    - `ratiotooffsetfunc`: List of lookup functions for different offset values
    - `funcoffsets`: List of pre-calculated offset values
    - `maplimits`: Tuple containing minimum and maximum valid ratio limits

    Examples
    --------
    >>> result = ratiotodelay(0.5, offset=0.1)
    >>> print(result)
    (0.48, 0.1)

    >>> result = ratiotodelay(1.5, offset=0.0)
    >>> print(result)
    (0.95, 0.0)
    """
    global ratiotooffsetfunc, funcoffsets, maplimits

    # find the closest calculated offset
    closestindex = 0
    for offsetindex in range(1, len(funcoffsets)):
        if np.fabs(funcoffsets[offsetindex] - offset) < np.fabs(
            funcoffsets[closestindex] - offset
        ):
            closestindex = offsetindex
    closestoffset = funcoffsets[closestindex]
    distance = np.fabs(funcoffsets[closestindex] - offset)

    if theratio < maplimits[0]:
        return (
            ratiotooffsetfunc[closestindex](maplimits[0]) + (offset - closestoffset),
            closestoffset,
        )
    elif theratio > maplimits[1]:
        return (
            ratiotooffsetfunc[closestindex](maplimits[1]) - (offset - closestoffset),
            closestoffset,
        )
    else:
        return (
            ratiotooffsetfunc[closestindex](theratio),
            closestoffset,
        )


def coffstodelay(
    thecoffs: NDArray, mindelay: float = -3.0, maxdelay: float = 3.0, debug: bool = False
) -> float:
    """
    Convert polynomial coefficients to delay value by finding roots within specified bounds.

    This function constructs a polynomial from the given coefficients and finds its roots
    within the specified delay range. It returns the root closest to zero that lies
    within the valid range, or 0.0 if no valid roots are found.

    Parameters
    ----------
    thecoffs : NDArray
        Array of polynomial coefficients (excluding the leading 1.0 term).
    mindelay : float, optional
        Minimum allowed delay value, default is -3.0.
    maxdelay : float, optional
        Maximum allowed delay value, default is 3.0.
    debug : bool, optional
        If True, prints debugging information about root selection process,
        default is False.

    Returns
    -------
    float
        The selected delay value (root closest to zero within bounds),
        or 0.0 if no valid roots are found.

    Notes
    -----
    The function constructs a polynomial with coefficients [1.0, *thecoffs] and
    finds all roots within the interval [mindelay, maxdelay]. Only real roots
    within the specified bounds are considered valid candidates.

    Examples
    --------
    >>> import numpy as np
    >>> coeffs = np.array([0.5, -1.0])
    >>> delay = coffstodelay(coeffs, mindelay=-2.0, maxdelay=2.0)
    >>> print(delay)
    0.5

    >>> coeffs = np.array([1.0, 0.0, -1.0])
    >>> delay = coffstodelay(coeffs, mindelay=-1.0, maxdelay=1.0, debug=True)
    keeping root 0 (1.0)
    keeping root 1 (-1.0)
    chosen = -1.0
    >>> print(delay)
    -1.0
    """
    justaone = np.array([1.0], dtype=thecoffs.dtype)
    allcoffs = np.concatenate((justaone, thecoffs))
    theroots = (poly.Polynomial(allcoffs, domain=(mindelay, maxdelay))).roots()
    if theroots is None:
        return 0.0
    elif len(theroots) == 1:
        return theroots[0].real
    else:
        candidates = []
        for i in range(len(theroots)):
            if np.isreal(theroots[i]) and (mindelay <= theroots[i] <= maxdelay):
                if debug:
                    print(f"keeping root {i} ({theroots[i]})")
                candidates.append(theroots[i].real)
            else:
                if debug:
                    print(f"discarding root {i} ({theroots[i]})")
                else:
                    pass
        if len(candidates) > 0:
            chosen = candidates[np.argmin(np.fabs(np.array(candidates)))].real
            if debug:
                print(f"{theroots=}, {candidates=}, {chosen=}")
            return chosen
        return 0.0


def getderivratios(
    fmri_data_valid: NDArray,
    validvoxels: NDArray,
    initial_fmri_x: NDArray,
    lagtimes: NDArray,
    fitmask: NDArray,
    genlagtc: Any,
    mode: str,
    outputname: str,
    oversamptr: float,
    sLFOfitmean: NDArray,
    rvalue: NDArray,
    r2value: NDArray,
    fitNorm: NDArray,
    fitcoeff: NDArray,
    movingsignal: Optional[NDArray],
    lagtc: NDArray,
    filtereddata: Optional[NDArray],
    LGR: Optional[Any],
    TimingLGR: Optional[Any],
    optiondict: dict,
    regressderivs: int = 1,
    timemask: Optional[NDArray] = None,
    starttr: Optional[int] = None,
    endtr: Optional[int] = None,
    debug: bool = False,
) -> Tuple[NDArray, NDArray]:
    """
    Compute the ratio of the first (or higher-order) derivative of regressors to the main regressor.

    This function performs regression analysis on fMRI data using lagged timecourses and
    calculates the ratio of each derivative regressor to the main (zeroth-order) regressor.
    It is typically used in the context of hemodynamic response function (HRF) modeling and
    temporal filtering.

    Parameters
    ----------
    fmri_data_valid : NDArray
        Valid fMRI data (voxels x timepoints).
    validvoxels : NDArray
        Boolean mask indicating valid voxels.
    initial_fmri_x : NDArray
        Initial fMRI design matrix (timepoints x regressors).
    lagtimes : NDArray
        Array of lag times for generating lagged regressors.
    fitmask : NDArray
        Mask for fitting regressors.
    genlagtc : Any
        Function or object for generating lagged timecourses.
    mode : str
        Regression mode (e.g., 'ols', 'wls').
    outputname : str
        Name of the output file or identifier.
    oversamptr : float
        Oversampling factor for temporal resolution.
    sLFOfitmean : NDArray
        Mean of the low-frequency fit.
    rvalue : NDArray
        R-values from regression.
    r2value : NDArray
        R-squared values from regression.
    fitNorm : NDArray
        Normalization factors from regression.
    fitcoeff : NDArray
        Regression coefficients (voxels x regressors).
    movingsignal : NDArray
        Moving signal data.
    lagtc : NDArray
        Lagged timecourses.
    filtereddata : NDArray
        Filtered fMRI data.
    LGR : Optional[Any]
        LGR object for temporal filtering.
    TimingLGR : Optional[Any]
        Timing LGR object for temporal filtering.
    optiondict : dict
        Dictionary of options for regression and processing.
    regressderivs : int, optional
        Number of derivative regressors to include (default is 1).
    timemask : NDArray, optional
       Mask of timepoints to include in regression filtering.
    starttr : Optional[int], optional
        Start timepoint for processing (default is 0).
    endtr : Optional[int], optional
        End timepoint for processing (default is number of timepoints).
    debug : bool, optional
        If True, print debug information (default is False).

    Returns
    -------
    Tuple[NDArray, NDArray]
        A tuple containing:
        - `regressderivratios`: Array of derivative-to-main regressor ratios (regressors x voxels).
        - `rvalue`: R-values from regression (same as input `rvalue`).

    Notes
    -----
    - The function uses `tide_regressfrommaps.regressfrommaps` internally for regression.
    - Derivative ratios are computed as `fitcoeff[:, i+1] / fitcoeff[:, 0]` for i in 0 to `regressderivs-1`.
    - NaN values are replaced with 0 using `np.nan_to_num`.

    Examples
    --------
    >>> ratios, rvals = getderivratios(
    ...     fmri_data_valid,
    ...     validvoxels,
    ...     initial_fmri_x,
    ...     lagtimes,
    ...     fitmask,
    ...     genlagtc,
    ...     'ols',
    ...     'output',
    ...     2.0,
    ...     sLFOfitmean,
    ...     rvalue,
    ...     r2value,
    ...     fitNorm,
    ...     fitcoeff,
    ...     movingsignal,
    ...     lagtc,
    ...     filtereddata,
    ...     None,
    ...     None,
    ...     optiondict,
    ...     regressderivs=2,
    ...     starttr=0,
    ...     endtr=100,
    ...     debug=False
    ... )
    """
    if starttr is None:
        starttr = 0
    if endtr is None:
        endtr = fmri_data_valid.shape[1]
    if debug:
        print("getderivratios: Starting")
        print(f"\t{fitNorm.shape=}")
        print(f"\t{fitcoeff.shape=}")
        print(f"\t{regressderivs=}")
        print(f"\t{starttr=}")
        print(f"\t{endtr=}")

    if timemask is not None:
        trimmedtimemask = timemask[starttr:endtr]
    else:
        trimmedtimemask = None

    voxelsprocessed_regressionfilt, regressorset, evset = tide_regressfrommaps.regressfrommaps(
        fmri_data_valid[:, starttr:endtr],
        validvoxels,
        initial_fmri_x[starttr:endtr],
        lagtimes,
        fitmask,
        genlagtc,
        mode,
        outputname,
        oversamptr,
        sLFOfitmean,
        rvalue,
        r2value,
        fitNorm,
        fitcoeff,
        movingsignal,
        lagtc,
        filtereddata,
        LGR,
        TimingLGR,
        optiondict["regressfiltthreshval"],
        False,
        nprocs_makelaggedtcs=optiondict["nprocs_makelaggedtcs"],
        nprocs_regressionfilt=optiondict["nprocs_regressionfilt"],
        regressderivs=regressderivs,
        chunksize=optiondict["mp_chunksize"],
        showprogressbar=optiondict["showprogressbar"],
        alwaysmultiproc=optiondict["alwaysmultiproc"],
        coefficientsonly=True,
        timemask=trimmedtimemask,
        debug=debug,
    )

    # calculate the ratio of the first derivative to the main regressor
    if regressderivs == 1:
        regressderivratios = np.nan_to_num(fitcoeff[:, 1] / fitcoeff[:, 0])
    else:
        numvoxels = fitcoeff.shape[0]
        regressderivratios = np.zeros((regressderivs, numvoxels), dtype=np.float64)
        for i in range(regressderivs):
            regressderivratios[i, :] = np.nan_to_num(fitcoeff[:, i + 1] / fitcoeff[:, 0])

    if debug:
        print("getderivratios: End\n\n")
    return regressderivratios, rvalue


def filterderivratios(
    regressderivratios: NDArray,
    nativespaceshape: Tuple[int, ...],
    validvoxels: NDArray,
    thedims: Tuple[float, ...],
    patchthresh: float = 3.0,
    gausssigma: float = 0,
    filetype: str = "nifti",
    rt_floattype: np.dtype = np.float64,
    verbose: bool = True,
    debug: bool = False,
) -> Tuple[NDArray, NDArray, float]:
    """
    Filter derivative ratios using median absolute deviation (MAD) and optional smoothing.

    This function applies a filtering procedure to regression derivative ratios to
    identify and correct outliers. It uses median filtering and compares deviations
    from the median using the median absolute deviation (MAD). Optionally, Gaussian
    smoothing is applied to the filtered data.

    Parameters
    ----------
    regressderivratios : ndarray
        Array of regression derivative ratios to be filtered.
    nativespaceshape : tuple of int
        Shape of the native space (e.g., (nx, ny, nz)).
    validvoxels : ndarray
        Boolean or integer array indicating valid voxels in the data.
    thedims : tuple of float
        Voxel dimensions (dx, dy, dz) used for Gaussian smoothing.
    patchthresh : float, optional
        Threshold for outlier detection in units of MAD. Default is 3.0.
    gausssigma : float, optional
        Standard deviation for Gaussian smoothing. If 0, no smoothing is applied.
        Default is 0.
    filetype : str, optional
        File type for output mapping. If not "nifti", no median filtering is applied.
        Default is "nifti".
    rt_floattype : str, optional
        Data type for the output arrays. Default is "float64".
    verbose : bool, optional
        If True, print diagnostic information. Default is True.
    debug : bool, optional
        If True, enable debug printing. Default is False.

    Returns
    -------
    medfilt : ndarray
        Median-filtered version of the input `regressderivratios`.
    filteredarray : ndarray
        Final filtered array with outliers replaced by median values.
    themad : float
        Median absolute deviation of the input `regressderivratios`.

    Notes
    -----
    - The function uses `median_filter` from `scipy.ndimage` for spatial filtering.
    - If `filetype` is "nifti", the input data is first mapped to a destination array
      using `tide_io.makedestarray` and `tide_io.populatemap`.
    - Gaussian smoothing is applied using `tide_filt.ssmooth` if `gausssigma > 0`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.ndimage import median_filter
    >>> # Assume inputs are prepared
    >>> medfilt, filtered, mad_val = filterderivratios(
    ...     regressderivratios, nativespaceshape, validvoxels, thedims,
    ...     patchthresh=3.0, gausssigma=1.0, verbose=True
    ... )
    """

    if debug:
        print("filterderivratios:")
        print(f"\t{patchthresh=}")
        print(f"\t{validvoxels.shape=}")
        print(f"\t{nativespaceshape=}")

    # filter the ratio to find weird values
    themad = mad(regressderivratios).astype(np.float64)
    if verbose:
        print(f"MAD of regression fit derivative ratios = {themad}")
    outmaparray, internalspaceshape = tide_io.makedestarray(
        nativespaceshape,
        filetype=filetype,
        rt_floattype=rt_floattype,
    )
    mappedregressderivratios = tide_io.populatemap(
        regressderivratios,
        internalspaceshape,
        validvoxels,
        outmaparray,
        debug=debug,
    )
    if filetype != "nifti":
        medfilt = regressderivratios
        filteredarray = regressderivratios
    else:
        if debug:
            print(f"{regressderivratios.shape=}, {mappedregressderivratios.shape=}")
        medfilt = median_filter(
            mappedregressderivratios.reshape(nativespaceshape), size=(3, 3, 3)
        ).reshape(internalspaceshape)[validvoxels]
        filteredarray = np.where(
            np.fabs(regressderivratios - medfilt) > patchthresh * themad,
            medfilt,
            regressderivratios,
        )
        if gausssigma > 0:
            mappedfilteredarray = tide_io.populatemap(
                filteredarray,
                internalspaceshape,
                validvoxels,
                outmaparray,
                debug=debug,
            )
            filteredarray = tide_filt.ssmooth(
                thedims[0],
                thedims[1],
                thedims[2],
                gausssigma,
                mappedfilteredarray.reshape(nativespaceshape),
            ).reshape(internalspaceshape)[validvoxels]

    return medfilt, filteredarray, themad
