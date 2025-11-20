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
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import factorial
from tqdm import tqdm

import rapidtide.filter as tide_filt
import rapidtide.fit as tide_fit
import rapidtide.miscmath as tide_math
import rapidtide.multiproc as tide_multiproc


def _procOneRegressionFitItem(
    vox: int,
    theevs: NDArray,
    thedata: NDArray,
    rt_floattype: np.dtype = np.float64,
) -> tuple[int, float, float, float, float | NDArray, Any, NDArray, NDArray]:
    """
    Perform single regression fit on voxel data and return fit results.

    This function fits a linear regression model to the provided evs and data,
    handling both univariate and multivariate cases. It computes fit coefficients,
    R-squared value, and residual data.

    Parameters
    ----------
    vox : int
        Voxel index.
    theevs : NDArray
        Experimental design matrix. If 2D, dimension 0 is number of points,
        dimension 1 is number of evs.
    thedata : NDArray
        Dependent variable data corresponding to the evs.
    rt_floattype : str, optional
        String representation of the floating-point type, default is ``np.float64``.

    Returns
    -------
    tuple[int, float, float, float, float, Any, NDArray, NDArray]
        A tuple containing:
        - voxel index (`int`)
        - intercept term (`float`)
        - signed square root of R-squared (`float`)
        - R-squared value (`float`)
        - fit coefficients (`float` or `NDArray`)
        - normalized fit coefficients (`Any`)
        - data removed by fitting (`NDArray`)
        - residuals (`NDArray`)

    Notes
    -----
    For multivariate regressions (2D `theevs`), the function computes the fit
    using `tide_fit.mlregress`. If the fit fails, a zero matrix is returned.
    For univariate regressions (1D `theevs`), the function directly computes
    the fit and handles edge cases such as zero coefficients.

    Examples
    --------
    >>> import numpy as np
    >>> theevs = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    >>> thedata = np.array([1, 2, 3], dtype=np.float64)
    >>> result = _procOneRegressionFitItem(0, theevs, thedata)
    >>> print(result[0])  # voxel index
    0
    """
    # NOTE: if theevs is 2D, dimension 0 is number of points, dimension 1 is number of evs
    thefit, R2 = tide_fit.mlregress(theevs, thedata)
    if theevs.ndim > 1:
        if thefit is None:
            thefit = np.matrix(np.zeros((1, theevs.shape[1] + 1), dtype=rt_floattype))
        fitcoeffs = (thefit[0, 1:]).astype(rt_floattype)
        if fitcoeffs[0, 0] < 0.0:
            coeffsign = -1.0
        else:
            coeffsign = 1.0
        datatoremove = np.zeros_like(theevs[:, 0])
        for j in range(theevs.shape[1]):
            datatoremove += (thefit[0, 1 + j] * theevs[:, j]).astype(rt_floattype)
        if np.any(fitcoeffs) != 0.0:
            pass
        else:
            R2 = 0.0
        return (
            vox,
            thefit[0, 0],
            coeffsign * np.sqrt(R2),
            R2,
            fitcoeffs,
            (thefit[0, 1:] / thefit[0, 0]).astype(rt_floattype),
            datatoremove,
            (thedata - datatoremove).astype(rt_floattype),
        )
    else:
        fitcoeff = (thefit[0, 1]).astype(rt_floattype)
        datatoremove = (fitcoeff * theevs).astype(rt_floattype)
        if fitcoeff < 0.0:
            coeffsign = -1.0
        else:
            coeffsign = 1.0
        if fitcoeff == 0.0:
            R2 = 0.0
        return (
            vox,
            thefit[0, 0],
            coeffsign * np.sqrt(R2),
            R2,
            fitcoeff,
            (thefit[0, 1] / thefit[0, 0]).astype(rt_floattype),
            datatoremove,
            (thedata - datatoremove).astype(rt_floattype),
        )


def linfitfiltpass(
    numprocitems: int,
    fmri_data: NDArray,
    threshval: float | None,
    theevs: NDArray,
    meanvalue: NDArray | None,
    rvalue: NDArray | None,
    r2value: NDArray,
    fitcoeff: NDArray | None,
    fitNorm: NDArray | None,
    datatoremove: NDArray | None,
    filtereddata: NDArray | None,
    nprocs: int = 1,
    alwaysmultiproc: bool = False,
    constantevs: bool = False,
    confoundregress: bool = False,
    coefficientsonly: bool = False,
    procbyvoxel: bool = True,
    showprogressbar: bool = True,
    chunksize: int = 1000,
    rt_floattype: np.dtype = np.dtype(np.float64),
    verbose: bool = True,
    debug: bool = False,
) -> int:
    """
    Perform linear regression fitting and filtering on fMRI data.

    This function fits a linear model to fMRI data using specified experimental variables
    and applies filtering to remove noise. It supports both voxel-wise and timepoint-wise
    processing, with optional multiprocessing for performance.

    Parameters
    ----------
    numprocitems : int
        Number of items to process (voxels or timepoints depending on ``procbyvoxel``).
    fmri_data : ndarray
        Input fMRI data array with shape ``(n_voxels, n_timepoints)`` or ``(n_timepoints, n_voxels)``.
    threshval : float, optional
        Threshold value for masking. If ``None``, no masking is applied.
    theevs : ndarray
        Experimental variables (design matrix) with shape ``(n_voxels, n_timepoints)`` or ``(n_timepoints, n_voxels)``.
    meanvalue : ndarray, optional
        Array to store mean values of the data. Shape depends on ``procbyvoxel``.
    rvalue : ndarray, optional
        Array to store correlation coefficients. Shape depends on ``procbyvoxel``.
    r2value : ndarray
        Array to store R-squared values. Shape depends on ``procbyvoxel``.
    fitcoeff : ndarray, optional
        Array to store fit coefficients. Shape depends on ``procbyvoxel`` and ``constantevs``.
    fitNorm : ndarray, optional
        Array to store normalized fit coefficients. Shape depends on ``procbyvoxel``.
    datatoremove : ndarray, optional
        Array to store data to be removed after fitting. Shape depends on ``procbyvoxel``.
    filtereddata : ndarray
        Array to store filtered data after regression. Shape depends on ``procbyvoxel``.
    nprocs : int, default: 1
        Number of processes to use for multiprocessing. If 1 and ``alwaysmultiproc`` is False, uses single-threaded processing.
    alwaysmultiproc : bool, default: False
        If True, always use multiprocessing even if ``nprocs`` is 1.
    constantevs : bool, default: False
        If True, treat experimental variables as constant across voxels/timepoints.
    confoundregress : bool, default: False
        If True, perform confound regression only (no output of coefficients or residuals).
    coefficientsonly : bool, default: False
        If True, store only regression coefficients and R-squared values.
    procbyvoxel : bool, default: True
        If True, process data voxel-wise; otherwise, process by timepoint.
    showprogressbar : bool, default: True
        If True, display a progress bar during processing.
    chunksize : int, default: 1000
        Size of chunks for multiprocessing.
    rt_floattype : str, default: np.float64
        Data type for internal floating-point calculations.
    verbose : bool, default: True
        If True, print verbose output.
    debug : bool, default: False
        If True, enable debug printing.

    Returns
    -------
    int
        Total number of items processed.

    Notes
    -----
    - The function modifies the output arrays in-place.
    - For ``confoundregress=True``, only ``r2value`` and ``filtereddata`` are populated.
    - When ``coefficientsonly=True``, only ``meanvalue``, ``rvalue``, ``r2value``, ``fitcoeff``, and ``fitNorm`` are populated.
    - If ``threshval`` is provided, a mask is generated based on mean or standard deviation of the data.

    Examples
    --------
    >>> import numpy as np
    >>> fmri_data = np.random.rand(100, 200)
    >>> theevs = np.random.rand(100, 200)
    >>> r2value = np.zeros(100)
    >>> filtereddata = np.zeros_like(fmri_data)
    >>> numprocitems = 100
    >>> items_processed = linfitfiltpass(
    ...     numprocitems=numprocitems,
    ...     fmri_data=fmri_data,
    ...     threshval=None,
    ...     theevs=theevs,
    ...     meanvalue=None,
    ...     rvalue=None,
    ...     r2value=r2value,
    ...     fitcoeff=None,
    ...     fitNorm=None,
    ...     datatoremove=None,
    ...     filtereddata=filtereddata,
    ...     nprocs=4,
    ...     procbyvoxel=True,
    ...     showprogressbar=True
    ... )
    >>> print(f"Processed {items_processed} items.")
    """
    inputshape = np.shape(fmri_data)
    if debug:
        print(f"{numprocitems=}")
        print(f"{fmri_data.shape=}")
        print(f"{threshval=}")
        print(f"{theevs.shape=}, {np.min(theevs)=}, {np.max(theevs)=}")
        print(f"{theevs=}")
    if procbyvoxel:
        indexaxis = 0
        procunit = "voxels"
    else:
        indexaxis = 1
        procunit = "timepoints"
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
                        if confoundregress or constantevs:
                            outQ.put(
                                _procOneRegressionFitItem(
                                    val,
                                    theevs,
                                    fmri_data[val, :],
                                    rt_floattype=rt_floattype,
                                )
                            )
                        else:
                            outQ.put(
                                _procOneRegressionFitItem(
                                    val,
                                    theevs[val, :],
                                    fmri_data[val, :],
                                    rt_floattype=rt_floattype,
                                )
                            )
                    else:
                        if confoundregress or constantevs:
                            outQ.put(
                                _procOneRegressionFitItem(
                                    val,
                                    theevs,
                                    fmri_data[:, val],
                                    rt_floattype=rt_floattype,
                                )
                            )
                        else:
                            outQ.put(
                                _procOneRegressionFitItem(
                                    val,
                                    theevs[:, val],
                                    fmri_data[:, val],
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
            verbose=verbose,
            nprocs=nprocs,
            indexaxis=indexaxis,
            procunit=procunit,
            showprogressbar=showprogressbar,
            chunksize=chunksize,
        )

        # unpack the data
        itemstotal = 0
        if procbyvoxel:
            if confoundregress:
                for voxel in data_out:
                    r2value[voxel[0]] = voxel[3]
                    filtereddata[voxel[0], :] = voxel[7]
                    itemstotal += 1
            elif coefficientsonly:
                for voxel in data_out:
                    meanvalue[voxel[0]] = voxel[1]
                    rvalue[voxel[0]] = voxel[2]
                    r2value[voxel[0]] = voxel[3]
                    if fitcoeff.ndim > 1:
                        fitcoeff[voxel[0], :] = voxel[4]
                        fitNorm[voxel[0], :] = voxel[5]
                    else:
                        fitcoeff[voxel[0]] = voxel[4]
                        fitNorm[voxel[0]] = voxel[5]
                    itemstotal += 1
            else:
                for voxel in data_out:
                    meanvalue[voxel[0]] = voxel[1]
                    rvalue[voxel[0]] = voxel[2]
                    r2value[voxel[0]] = voxel[3]
                    if fitcoeff.ndim > 1:
                        fitcoeff[voxel[0], :] = voxel[4]
                        fitNorm[voxel[0], :] = voxel[5]
                    else:
                        fitcoeff[voxel[0]] = voxel[4]
                        fitNorm[voxel[0]] = voxel[5]
                    datatoremove[voxel[0], :] = voxel[6]
                    filtereddata[voxel[0], :] = voxel[7]
                    itemstotal += 1
        else:
            if confoundregress:
                for timepoint in data_out:
                    r2value[timepoint[0]] = timepoint[3]
                    filtereddata[:, timepoint[0]] = timepoint[7]
                    itemstotal += 1
            elif coefficientsonly:
                for timepoint in data_out:
                    meanvalue[timepoint[0]] = timepoint[1]
                    rvalue[timepoint[0]] = timepoint[2]
                    r2value[timepoint[0]] = timepoint[3]
                    if fitcoeff.ndim > 1:
                        fitcoeff[:, timepoint[0]] = timepoint[4]
                        fitNorm[:, timepoint[0]] = timepoint[5]
                    else:
                        fitcoeff[timepoint[0]] = timepoint[4]
                        fitNorm[timepoint[0]] = timepoint[5]
                    itemstotal += 1
            else:
                for timepoint in data_out:
                    meanvalue[timepoint[0]] = timepoint[1]
                    rvalue[timepoint[0]] = timepoint[2]
                    r2value[timepoint[0]] = timepoint[3]
                    if fitcoeff.ndim > 1:
                        fitcoeff[:, timepoint[0]] = timepoint[4]
                        fitNorm[:, timepoint[0]] = timepoint[5]
                    else:
                        fitcoeff[timepoint[0]] = timepoint[4]
                        fitNorm[timepoint[0]] = timepoint[5]
                    datatoremove[:, timepoint[0]] = timepoint[6]
                    filtereddata[:, timepoint[0]] = timepoint[7]
                    itemstotal += 1

        del data_out
    else:
        # this is the single proc path
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
                    if confoundregress:
                        (
                            dummy,
                            dummy,
                            dummy,
                            r2value[vox],
                            dummy,
                            dummy,
                            dummy,
                            filtereddata[vox, :],
                        ) = _procOneRegressionFitItem(
                            vox,
                            theevs,
                            thedata,
                            rt_floattype=rt_floattype,
                        )
                    elif coefficientsonly:
                        if not constantevs:
                            (
                                dummy,
                                meanvalue[vox],
                                rvalue[vox],
                                r2value[vox],
                                fitcoeff[vox],
                                fitNorm[vox],
                                dummy,
                                dummy,
                            ) = _procOneRegressionFitItem(
                                vox,
                                theevs[vox, :],
                                thedata,
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
                                dummy,
                                dummy,
                            ) = _procOneRegressionFitItem(
                                vox,
                                theevs,
                                thedata,
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
                        ) = _procOneRegressionFitItem(
                            vox,
                            theevs[vox, :],
                            thedata,
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
                    if confoundregress:
                        (
                            dummy,
                            dummy,
                            dummy,
                            r2value[timepoint],
                            dummy,
                            dummy,
                            dummy,
                            filtereddata[:, timepoint],
                        ) = _procOneRegressionFitItem(
                            timepoint,
                            theevs,
                            thedata,
                            rt_floattype=rt_floattype,
                        )
                    elif coefficientsonly:
                        if not constantevs:
                            (
                                dummy,
                                meanvalue[timepoint],
                                rvalue[timepoint],
                                r2value[timepoint],
                                fitcoeff[timepoint],
                                fitNorm[timepoint],
                                dummy,
                                dummy,
                            ) = _procOneRegressionFitItem(
                                timepoint,
                                theevs[:, timepoint],
                                thedata,
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
                                dummy,
                                dummy,
                            ) = _procOneRegressionFitItem(
                                timepoint,
                                theevs,
                                thedata,
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
                        ) = _procOneRegressionFitItem(
                            timepoint,
                            theevs[:, timepoint],
                            thedata,
                            rt_floattype=rt_floattype,
                        )

                    itemstotal += 1
        if showprogressbar:
            print()
    return itemstotal


def makevoxelspecificderivs(theevs: NDArray, nderivs: int = 1, debug: bool = False) -> NDArray:
    """
    Perform multicomponent expansion on voxel-specific explanatory variables by computing
    derivatives up to a specified order.

    This function takes an array of voxel-specific timecourses and expands each one
    into a set of components representing the original signal and its derivatives.
    Each component corresponds to a higher-order derivative of the signal, scaled by
    the inverse factorial of the derivative order (Taylor series coefficients).

    Parameters
    ----------
    theevs : 2D numpy array
        NxP array of voxel-specific explanatory variables, where N is the number of voxels
        and P is the number of timepoints.
    nderivs : int, optional
        Number of derivative components to compute for each voxel. Default is 1.
        If 0, the original `theevs` are returned without modification.
    debug : bool, optional
        If True, print debugging information including input and output shapes.
        Default is False.

    Returns
    -------
    3D numpy array
        Array of shape (N, P, nderivs + 1) containing the original signal and its
        derivatives up to order `nderivs` for each voxel. The first component is the
        original signal, followed by the first, second, ..., up to `nderivs`-th derivative.

    Notes
    -----
    - The function uses `numpy.gradient` to compute numerical derivatives.
    - Each derivative component is scaled by the inverse factorial of the derivative order
      to align with Taylor series expansion coefficients.
    - If `nderivs=0`, the function returns a copy of the input `theevs`.

    Examples
    --------
    >>> import numpy as np
    >>> theevs = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    >>> result = makevoxelspecificderivs(theevs, nderivs=2)
    >>> print(result.shape)
    (2, 4, 3)
    """
    if debug:
        print(f"{theevs.shape=}")
    if nderivs == 0:
        thenewevs = theevs
    else:
        taylorcoffs = np.zeros((nderivs + 1), dtype=np.float64)
        taylorcoffs[0] = 1.0
        thenewevs = np.zeros((theevs.shape[0], theevs.shape[1], nderivs + 1), dtype=float)
        for i in range(1, nderivs + 1):
            taylorcoffs[i] = 1.0 / factorial(i)
        for thevoxel in range(0, theevs.shape[0]):
            thenewevs[thevoxel, :, 0] = theevs[thevoxel, :] * 1.0
            for i in range(1, nderivs + 1):
                thenewevs[thevoxel, :, i] = taylorcoffs[i] * np.gradient(
                    thenewevs[thevoxel, :, i - 1]
                )
    if debug:
        print(f"{nderivs=}")
        print(f"{thenewevs.shape=}")

    return thenewevs


def confoundregress(
    theregressors: NDArray,
    theregressorlabels: list[str],
    thedataarray: NDArray,
    tr: float,
    nprocs: int = 1,
    orthogonalize: bool = True,
    tcstart: int = 0,
    tcend: int = -1,
    tchp: float | None = None,
    tclp: float | None = None,
    showprogressbar: bool = True,
    debug: bool = False,
) -> tuple[NDArray, list[str], NDArray, NDArray]:
    """
    Perform confound regression on fMRI data using linear regression.

    This function applies confound regression to remove noise from fMRI time series
    by regressing out specified confounding variables (e.g., motion parameters,
    physiological signals). It supports optional filtering, orthogonalization of
    regressors, and parallel processing for performance.

    Parameters
    ----------
    theregressors : ndarray
        Array of confounding variables with shape (n_regressors, n_timepoints).
    theregressorlabels : list of str
        List of labels corresponding to each regressor.
    thedataarray : ndarray
        3D or 4D array of fMRI data with shape (n_voxels, n_timepoints) or
        (n_voxels, n_timepoints, n_volumes).
    tr : float
        Repetition time (TR) in seconds.
    nprocs : int, optional
        Number of processes to use for parallel processing. Default is 1.
    orthogonalize : bool, optional
        If True, orthogonalize the regressors to reduce multicollinearity.
        Default is True.
    tcstart : int, optional
        Start timepoint index for regressor data. Default is 0.
    tcend : int, optional
        End timepoint index for regressor data. If -1, use all timepoints
        from `tcstart`. Default is -1.
    tchp : float, optional
        High-pass cutoff frequency for filtering. If None, no high-pass filtering
        is applied.
    tclp : float, optional
        Low-pass cutoff frequency for filtering. If None, no low-pass filtering
        is applied.
    showprogressbar : bool, optional
        If True, display a progress bar during processing. Default is True.
    debug : bool, optional
        If True, enable debug output. Default is False.

    Returns
    -------
    tuple of (NDArray, list of str, NDArray, NDArray)
        - `theregressors`: Processed regressors (possibly orthogonalized).
        - `theregressorlabels`: Updated labels for the regressors.
        - `filtereddata`: Data with confounds removed.
        - `r2value`: R-squared values for each voxel (or None if not computed).

    Notes
    -----
    - The function applies standard deviation normalization to regressors for
      numerical stability.
    - If `orthogonalize` is True, Gram-Schmidt orthogonalization is applied to
      the regressors.
    - Filtering is applied using a trapezoidal filter if `tchp` or `tclp` are provided.
    - The function uses `linfitfiltpass` internally for the actual regression.

    Examples
    --------
    >>> regressors = np.random.rand(3, 100)
    >>> labels = ['motion_x', 'motion_y', 'motion_z']
    >>> data = np.random.rand(50, 100)
    >>> tr = 2.0
    >>> processed_regressors, labels, filtered_data, r2 = confoundregress(
    ...     regressors, labels, data, tr, nprocs=4
    ... )
    """
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

    # stddev normalize the regressors.  Not strictly necessary, but might help with stability.
    for i in range(theregressors.shape[0]):
        theregressors[i, :] = tide_math.normalize(theregressors[i, :], method="stddev")

    if orthogonalize:
        theregressors = tide_fit.gram_schmidt(theregressors)
        initregressors = len(theregressorlabels)
        theregressorlabels = []
        for theregressor in range(theregressors.shape[0]):
            theregressorlabels.append("orthogconfound_{:02d}".format(theregressor))
        if len(theregressorlabels) == 0:
            print("No regressors survived orthogonalization - skipping confound regression")
            return theregressors, theregressorlabels, thedataarray, None
        print(
            f"After orthogonalization, {len(theregressorlabels)} of {initregressors} regressors remain."
        )

    # stddev normalize the regressors.  Not strictly necessary, but might help with stability.
    for i in range(theregressors.shape[0]):
        theregressors[i, :] = tide_math.normalize(theregressors[i, :], method="stddev")

    print("start confound filtering")

    numprocitems = thedataarray.shape[0]
    filtereddata = np.zeros_like(thedataarray)
    r2value = np.zeros(numprocitems)
    numfiltered = linfitfiltpass(
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
        confoundregress=True,
        nprocs=nprocs,
        showprogressbar=showprogressbar,
        procbyvoxel=True,
        debug=debug,
    )

    print()
    print(f"confound filtering on {numfiltered} voxels complete")
    return theregressors, theregressorlabels, filtereddata, r2value
