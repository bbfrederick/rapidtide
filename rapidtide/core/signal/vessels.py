#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2026-2026 Blaise Frederick
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
import copy

import numpy as np
from numpy.typing import NDArray
from scipy.stats import pearsonr
from tqdm import tqdm

import rapidtide.filter as tide_filt
import rapidtide.fit as tide_fit
import rapidtide.genericmultiproc as tide_genericmultiproc
import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math
import rapidtide.resample as tide_resample
import rapidtide.stats as tide_stats
import rapidtide.util as tide_util


def circularderivs(timecourse: NDArray) -> tuple[NDArray, float, float]:
    """
    Compute circular first derivatives and their extremal values.

    This function calculates the circular first derivative of a time course,
    which is the difference between consecutive elements with the last element
    wrapped around to the first. It then returns the maximum and minimum values
    of these derivatives along with their indices.

    Parameters
    ----------
    timecourse : array-like
        Input time course data as a 1D array or sequence of numerical values.

    Returns
    -------
    tuple
        A tuple containing four elements:
        - max_derivative : float
            The maximum value of the circular first derivative
        - argmax_index : int
            The index of the maximum derivative value
        - min_derivative : float
            The minimum value of the circular first derivative
        - argmin_index : int
            The index of the minimum derivative value

    Notes
    -----
    The circular first derivative is computed as:
    ``first_deriv[i] = timecourse[i+1] - timecourse[i]`` for i < n-1,
    and ``first_deriv[n-1] = timecourse[0] - timecourse[n-1]``.

    Examples
    --------
    >>> import numpy as np
    >>> timecourse = [1, 2, 3, 2, 1]
    >>> max_val, max_idx, min_val, min_idx = circularderivs(timecourse)
    >>> print(f"Max derivative: {max_val} at index {max_idx}")
    >>> print(f"Min derivative: {min_val} at index {min_idx}")
    """
    firstderiv = np.diff(timecourse, append=[timecourse[0]])
    return (
        np.max(firstderiv),
        np.argmax(firstderiv),
        np.min(firstderiv),
        np.argmin(firstderiv),
    )


def _procOnePhaseProject(slice, sliceargs, **kwargs):
    """
    Process a single phase project for fMRI data resampling and averaging.

    This function performs temporal resampling of fMRI data along the phase dimension
    using a congrid-based interpolation scheme. It updates weight, raw application,
    and cine data arrays based on the resampled values.

    Parameters
    ----------
    slice : int
        The slice index to process.
    sliceargs : tuple
        A tuple containing the following elements:
        - validlocslist : list of arrays
          List of valid location indices for each slice.
        - proctrs : array-like
          Time indices to process.
        - demeandata_byslice : ndarray
          Demeaned fMRI data organized by slice and time.
        - fmri_data_byslice : ndarray
          Raw fMRI data organized by slice and time.
        - outphases : array-like
          Output phase values for resampling.
        - cardphasevals : ndarray
          Cardinality of phase values for each slice and time.
        - congridbins : int
          Number of bins for congrid interpolation.
        - gridkernel : str
          Interpolation kernel to use.
        - weights_byslice : ndarray
          Weight array to be updated.
        - cine_byslice : ndarray
          Cine data array to be updated.
        - destpoints : int
          Number of destination points.
        - rawapp_byslice : ndarray
          Raw application data array to be updated.
    **kwargs : dict
        Additional options to override default settings:
        - cache : bool, optional
          Whether to use caching in congrid (default: True).
        - debug : bool, optional
          Whether to enable debug mode (default: False).

    Returns
    -------
    tuple
        A tuple containing:
        - slice : int
          The input slice index.
        - rawapp_byslice : ndarray
          Updated raw application data for the slice.
        - cine_byslice : ndarray
          Updated cine data for the slice.
        - weights_byslice : ndarray
          Updated weights for the slice.
        - validlocs : array-like
          Valid location indices for the slice.

    Notes
    -----
    This function modifies the input arrays `weights_byslice`, `rawapp_byslice`,
    and `cine_byslice` in-place. The function assumes that the data has already
    been preprocessed and organized into slices and time points.

    Examples
    --------
    >>> slice_idx = 0
    >>> args = (validlocslist, proctrs, demeandata_byslice, fmri_data_byslice,
    ...         outphases, cardphasevals, congridbins, gridkernel,
    ...         weights_byslice, cine_byslice, destpoints, rawapp_byslice)
    >>> result = _procOnePhaseProject(slice_idx, args, cache=False)
    """
    options = {
        "cache": True,
        "debug": False,
    }
    options.update(kwargs)
    cache = options["cache"]
    debug = options["debug"]
    (
        validlocslist,
        proctrs,
        demeandata_byslice,
        fmri_data_byslice,
        outphases,
        cardphasevals,
        congridbins,
        gridkernel,
        weights_byslice,
        cine_byslice,
        destpoints,
        rawapp_byslice,
    ) = sliceargs
    # now smooth the projected data along the time dimension
    validlocs = validlocslist[slice]
    if len(validlocs) > 0:
        for t in proctrs:
            filteredmr = -demeandata_byslice[validlocs, slice, t]
            cinemr = fmri_data_byslice[validlocs, slice, t]
            thevals, theweights, theindices = tide_resample.congrid(
                outphases,
                cardphasevals[slice, t],
                1.0,
                congridbins,
                kernel=gridkernel,
                cache=cache,
                cyclic=True,
            )
            for i in range(len(theindices)):
                weights_byslice[validlocs, slice, theindices[i]] += theweights[i]
                rawapp_byslice[validlocs, slice, theindices[i]] += filteredmr
                cine_byslice[validlocs, slice, theindices[i]] += theweights[i] * cinemr
        for d in range(destpoints):
            if weights_byslice[validlocs[0], slice, d] == 0.0:
                weights_byslice[validlocs, slice, d] = 1.0
        rawapp_byslice[validlocs, slice, :] = np.nan_to_num(
            rawapp_byslice[validlocs, slice, :] / weights_byslice[validlocs, slice, :]
        )
        cine_byslice[validlocs, slice, :] = np.nan_to_num(
            cine_byslice[validlocs, slice, :] / weights_byslice[validlocs, slice, :]
        )
    else:
        rawapp_byslice[:, slice, :] = 0.0
        cine_byslice[:, slice, :] = 0.0

    return (
        slice,
        rawapp_byslice[:, slice, :],
        cine_byslice[:, slice, :],
        weights_byslice[:, slice, :],
        validlocs,
    )


def _packslicedataPhaseProject(slicenum, sliceargs):
    """
    Pack slice data for phase projection.

    This function takes a slice number and slice arguments, then returns a
    flattened list containing all the slice arguments in order.

    Parameters
    ----------
    slicenum : int
        The slice number identifier.
    sliceargs : list or tuple
        Collection of slice arguments to be packed into a flat list.

    Returns
    -------
    list
        A list containing all elements from sliceargs in the same order.

    Notes
    -----
    This function essentially performs a flattening operation on the slice
    arguments, converting them into a fixed-length list format.

    Examples
    --------
    >>> _packslicedataPhaseProject(0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    """
    return [
        sliceargs[0],
        sliceargs[1],
        sliceargs[2],
        sliceargs[3],
        sliceargs[4],
        sliceargs[5],
        sliceargs[6],
        sliceargs[7],
        sliceargs[8],
        sliceargs[9],
        sliceargs[10],
        sliceargs[11],
    ]


def _unpackslicedataPhaseProject(retvals, voxelproducts):
    """
    Unpack slice data for phase project operation.

    This function assigns sliced data from retvals to corresponding voxelproducts
    based on index mappings. It performs three simultaneous assignments using
    slicing operations on 3D arrays.

    Parameters
    ----------
    retvals : tuple of array-like
        A tuple containing 5 elements where:
        - retvals[0], retvals[1], retvals[2], retvals[3], retvals[4]
        - retvals[4] is used as row index for slicing
        - retvals[0] is used as column index for slicing
    voxelproducts : list of array-like
        A list of 3 arrays that will be modified in-place with the sliced data.
        Each array is expected to be 3D and will be indexed using retvals[4] and retvals[0].

    Returns
    -------
    None
        This function modifies voxelproducts in-place and does not return any value.

    Notes
    -----
    The function performs three assignments:
    1. voxelproducts[0][retvals[4], retvals[0], :] = retvals[1][retvals[4], :]
    2. voxelproducts[1][retvals[4], retvals[0], :] = retvals[2][retvals[4], :]
    3. voxelproducts[2][retvals[4], retvals[0], :] = retvals[3][retvals[4], :]

    All arrays must be compatible for the specified slicing operations.

    Examples
    --------
    >>> retvals = (np.array([0, 1]), np.array([[1, 2], [3, 4]]),
    ...            np.array([[5, 6], [7, 8]]), np.array([[9, 10], [11, 12]]),
    ...            np.array([0, 1]))
    >>> voxelproducts = [np.zeros((2, 2, 2)), np.zeros((2, 2, 2)), np.zeros((2, 2, 2))]
    >>> _unpackslicedataPhaseProject(retvals, voxelproducts)
    """
    (voxelproducts[0])[retvals[4], retvals[0], :] = (retvals[1])[retvals[4], :]
    (voxelproducts[1])[retvals[4], retvals[0], :] = (retvals[2])[retvals[4], :]
    (voxelproducts[2])[retvals[4], retvals[0], :] = (retvals[3])[retvals[4], :]


def preloadcongrid(
    outphases: NDArray,
    congridbins: int,
    gridkernel: str = "kaiser",
    cyclic: bool = True,
    debug: bool = False,
) -> None:
    """
    Preload congrid interpolation cache for efficient subsequent calls.

    This function preloads the congrid interpolation cache by performing a series
    of interpolation operations with different phase values. This avoids the
    computational overhead of cache initialization during subsequent calls to
    tide_resample.congrid with the same parameters.

    Parameters
    ----------
    outphases : array-like
        Output phase values for the interpolation grid.
    congridbins : array-like
        Binning parameters for the congrid interpolation.
    gridkernel : str, optional
        Interpolation kernel to use. Default is "kaiser".
    cyclic : bool, optional
        Whether to treat the data as cyclic. Default is True.
    debug : bool, optional
        Enable debug output. Default is False.

    Returns
    -------
    None
        This function does not return any value.

    Notes
    -----
    This function is designed to improve performance when calling tide_resample.congrid
    multiple times with the same parameters. By preloading the cache with various
    phase values, subsequent calls will be faster as the cache is already populated.

    Examples
    --------
    >>> import numpy as np
    >>> outphases = np.linspace(0, 2*np.pi, 100)
    >>> congridbins = [10, 20]
    >>> preloadcongrid(outphases, congridbins, gridkernel="kaiser", cyclic=True)
    """
    outphasestep = outphases[1] - outphases[0]
    outphasecenter = outphases[int(len(outphases) / 2)]
    fillargs = outphasestep * (
        np.linspace(-0.5, 0.5, 10001, endpoint=True, dtype=float) + outphasecenter
    )
    for thearg in fillargs:
        dummy, dummy, dummy = tide_resample.congrid(
            outphases,
            thearg,
            1.0,
            congridbins,
            kernel=gridkernel,
            cyclic=cyclic,
            cache=True,
            debug=debug,
        )


def phaseprojectpass(
    numslices,
    demeandata_byslice,
    fmri_data_byslice,
    validlocslist,
    proctrs,
    weights_byslice,
    cine_byslice,
    rawapp_byslice,
    outphases,
    cardphasevals,
    congridbins,
    gridkernel,
    destpoints,
    mpcode=False,
    nprocs=1,
    alwaysmultiproc=False,
    showprogressbar=True,
    cache=True,
    debug=False,
):
    """
    Perform phase-encoding projection for fMRI data across slices.

    This function projects fMRI data onto a set of phase values using congrid
    resampling, accumulating results in `rawapp_byslice` and `cine_byslice` arrays.
    It supports both single-threaded and multi-processed execution.

    Parameters
    ----------
    numslices : int
        Number of slices to process.
    demeandata_byslice : ndarray
        Demeaned fMRI data, shape (nvoxels, nslices, ntr).
    fmri_data_byslice : ndarray
        Raw fMRI data, shape (nvoxels, nslices, ntr).
    validlocslist : list of ndarray
        List of valid voxel indices for each slice.
    proctrs : ndarray
        Timepoints to process.
    weights_byslice : ndarray
        Weight array, shape (nvoxels, nslices, ndestpoints).
    cine_byslice : ndarray
        Cine data array, shape (nvoxels, nslices, ndestpoints).
    rawapp_byslice : ndarray
        Raw application data array, shape (nvoxels, nslices, ndestpoints).
    outphases : ndarray
        Output phase values.
    cardphasevals : ndarray
        Cardinal phase values for each slice and timepoint, shape (nslices, ntr).
    congridbins : int
        Number of bins for congrid resampling.
    gridkernel : str
        Kernel to use for congrid resampling.
    destpoints : int
        Number of destination points.
    mpcode : bool, optional
        If True, use multiprocessing. Default is False.
    nprocs : int, optional
        Number of processes to use if `mpcode` is True. Default is 1.
    alwaysmultiproc : bool, optional
        If True, always use multiprocessing even for small datasets. Default is False.
    showprogressbar : bool, optional
        If True, show progress bar. Default is True.
    cache : bool, optional
        If True, enable caching for congrid. Default is True.
    debug : bool, optional
        If True, enable debug output. Default is False.

    Returns
    -------
    None
        The function modifies `weights_byslice`, `cine_byslice`, and `rawapp_byslice` in-place.

    Notes
    -----
    This function is typically used in the context of phase-encoded fMRI analysis.
    It applies a congrid-based resampling technique to project data onto a specified
    phase grid, accumulating weighted contributions in the output arrays.

    Examples
    --------
    >>> phaseprojectpass(
    ...     numslices=10,
    ...     demeandata_byslice=demean_data,
    ...     fmri_data_byslice=fmri_data,
    ...     validlocslist=valid_locs_list,
    ...     proctrs=tr_list,
    ...     weights_byslice=weights,
    ...     cine_byslice=cine_data,
    ...     rawapp_byslice=rawapp_data,
    ...     outphases=phase_vals,
    ...     cardphasevals=card_phase_vals,
    ...     congridbins=100,
    ...     gridkernel='gaussian',
    ...     destpoints=50,
    ...     mpcode=False,
    ...     nprocs=4,
    ...     showprogressbar=True,
    ...     cache=True,
    ...     debug=False,
    ... )
    """
    if mpcode:
        inputshape = rawapp_byslice.shape
        sliceargs = [
            validlocslist,
            proctrs,
            demeandata_byslice,
            fmri_data_byslice,
            outphases,
            cardphasevals,
            congridbins,
            gridkernel,
            weights_byslice,
            cine_byslice,
            destpoints,
            rawapp_byslice,
        ]
        slicefunc = _procOnePhaseProject
        packfunc = _packslicedataPhaseProject
        unpackfunc = _unpackslicedataPhaseProject
        slicetargets = [rawapp_byslice, cine_byslice, weights_byslice]
        slicemask = np.ones_like(rawapp_byslice[0, :, 0])

        slicetotal = tide_genericmultiproc.run_multiproc(
            slicefunc,
            packfunc,
            unpackfunc,
            sliceargs,
            slicetargets,
            inputshape,
            slicemask,
            None,
            nprocs,
            alwaysmultiproc,
            showprogressbar,
            8,
            indexaxis=1,
            procunit="slices",
            cache=cache,
            debug=debug,
        )
    else:
        for theslice in tqdm(
            range(numslices),
            desc="Slice",
            unit="slices",
            disable=(not showprogressbar),
        ):
            validlocs = validlocslist[theslice]
            if len(validlocs) > 0:
                for t in proctrs:
                    filteredmr = -demeandata_byslice[validlocs, theslice, t]
                    cinemr = fmri_data_byslice[validlocs, theslice, t]
                    thevals, theweights, theindices = tide_resample.congrid(
                        outphases,
                        cardphasevals[theslice, t],
                        1.0,
                        congridbins,
                        kernel=gridkernel,
                        cyclic=True,
                        cache=cache,
                        debug=debug,
                    )
                    for i in range(len(theindices)):
                        weights_byslice[validlocs, theslice, theindices[i]] += theweights[i]
                        rawapp_byslice[validlocs, theslice, theindices[i]] += filteredmr
                        cine_byslice[validlocs, theslice, theindices[i]] += theweights[i] * cinemr
                for d in range(destpoints):
                    if weights_byslice[validlocs[0], theslice, d] == 0.0:
                        weights_byslice[validlocs, theslice, d] = 1.0
                rawapp_byslice[validlocs, theslice, :] = np.nan_to_num(
                    rawapp_byslice[validlocs, theslice, :]
                    / weights_byslice[validlocs, theslice, :]
                )
                cine_byslice[validlocs, theslice, :] = np.nan_to_num(
                    cine_byslice[validlocs, theslice, :] / weights_byslice[validlocs, theslice, :]
                )
            else:
                rawapp_byslice[:, theslice, :] = 0.0
                cine_byslice[:, theslice, :] = 0.0


def _procOneSliceSmoothing(slice, sliceargs, **kwargs):
    """
    Apply smoothing filter to a single slice of projected data along time dimension.

    This function processes a single slice of data by applying a smoothing filter
    to the raw application data and computing circular derivatives for the
    specified slice. The smoothing is applied only to valid locations within the slice.

    Parameters
    ----------
    slice : int
        The slice index to process.
    sliceargs : tuple
        A tuple containing the following elements:

        - validlocslist : list of arrays
          List of arrays containing valid location indices for each slice
        - rawapp_byslice : ndarray
          Array containing raw application data by slice [locations, slices, time_points]
        - appsmoothingfilter : object
          Smoothing filter object with an apply method
        - phaseFs : array-like
          Frequency values for smoothing filter application
        - derivatives_byslice : ndarray
          Array to store computed derivatives [locations, slices, time_points]
    **kwargs : dict
        Additional keyword arguments:
        - debug : bool, optional
          Enable debug mode (default: False)

    Returns
    -------
    tuple
        A tuple containing:

        - slice : int
          The input slice index
        - rawapp_byslice : ndarray
          Smoothed raw application data for the specified slice [locations, time_points]
        - derivatives_byslice : ndarray
          Computed circular derivatives for the specified slice [locations, time_points]

    Notes
    -----
    - The function only processes slices with valid locations (len(validlocs) > 0)
    - Smoothing is applied using the provided smoothing filter's apply method
    - Circular derivatives are computed using the `circularderivs` function
    - The function modifies the input arrays in-place

    Examples
    --------
    >>> slice_idx = 5
    >>> sliceargs = (validlocslist, rawapp_byslice, appsmoothingfilter, phaseFs, derivatives_byslice)
    >>> result = _procOneSliceSmoothing(slice_idx, sliceargs, debug=True)
    """
    options = {
        "debug": False,
    }
    options.update(kwargs)
    debug = options["debug"]
    validlocslist, rawapp_byslice, appsmoothingfilter, phaseFs, derivatives_byslice = sliceargs
    # now smooth the projected data along the time dimension
    validlocs = validlocslist[slice]
    if len(validlocs) > 0:
        for loc in validlocs:
            rawapp_byslice[loc, slice, :] = appsmoothingfilter.apply(
                phaseFs, rawapp_byslice[loc, slice, :]
            )
            derivatives_byslice[loc, slice, :] = circularderivs(rawapp_byslice[loc, slice, :])
    return slice, rawapp_byslice[:, slice, :], derivatives_byslice[:, slice, :]


def _packslicedataSliceSmoothing(slicenum, sliceargs):
    """Pack slice data for slice smoothing operation.

    Parameters
    ----------
    slicenum : int
        The slice number identifier.
    sliceargs : list
        List containing slice arguments with at least 5 elements.

    Returns
    -------
    list
        A list containing the first 5 elements from sliceargs in the same order.

    Notes
    -----
    This function extracts the first five elements from the sliceargs parameter
    and returns them as a new list. It's typically used as part of a slice
    smoothing pipeline where slice arguments need to be packed for further processing.

    Examples
    --------
    >>> _packslicedataSliceSmoothing(1, [10, 20, 30, 40, 50, 60])
    [10, 20, 30, 40, 50]
    """
    return [
        sliceargs[0],
        sliceargs[1],
        sliceargs[2],
        sliceargs[3],
        sliceargs[4],
    ]


def _unpackslicedataSliceSmoothing(retvals, voxelproducts):
    """
    Unpack slice data for smoothing operation.

    This function assigns smoothed slice data back to the voxel products array
    based on the provided retvals structure.

    Parameters
    ----------
    retvals : tuple of array-like
        A tuple containing:
        - retvals[0] : array-like
            Index array for slice selection
        - retvals[1] : array-like
            First set of smoothed data to assign
        - retvals[2] : array-like
            Second set of smoothed data to assign
    voxelproducts : list of array-like
        A list containing two array-like objects where:
        - voxelproducts[0] : array-like
            First voxel product array to be modified
        - voxelproducts[1] : array-like
            Second voxel product array to be modified

    Returns
    -------
    None
        This function modifies the voxelproducts arrays in-place and does not return anything.

    Notes
    -----
    The function performs in-place assignment operations on the voxelproducts arrays.
    The first dimension of voxelproducts arrays is modified using retvals[0] as indices,
    while the second and third dimensions are directly assigned from retvals[1] and retvals[2].

    Examples
    --------
    >>> import numpy as np
    >>> retvals = (np.array([0, 1, 2]), np.array([[1, 2], [3, 4], [5, 6]]), np.array([[7, 8], [9, 10], [11, 12]]))
    >>> voxelproducts = [np.zeros((3, 3, 2)), np.zeros((3, 3, 2))]
    >>> _unpackslicedataSliceSmoothing(retvals, voxelproducts)
    >>> print(voxelproducts[0])
    >>> print(voxelproducts[1])
    """
    (voxelproducts[0])[:, retvals[0], :] = retvals[1]
    (voxelproducts[1])[:, retvals[0], :] = retvals[2]


def tcsmoothingpass(
    numslices,
    validlocslist,
    rawapp_byslice,
    appsmoothingfilter,
    phaseFs,
    derivatives_byslice,
    nprocs=1,
    alwaysmultiproc=False,
    showprogressbar=True,
    debug=False,
):
    """
    Apply smoothing to time course data across slices using multiprocessing.

    This function performs smoothing operations on time course data organized by slices,
    utilizing multiprocessing for improved performance when processing large datasets.

    Parameters
    ----------
    numslices : int
        Number of slices in the dataset
    validlocslist : list
        List of valid locations for processing
    rawapp_byslice : NDArray
        Raw application data organized by slice
    appsmoothingfilter : NDArray
        Smoothing filter to be applied
    phaseFs : float
        Phase frequency parameter for smoothing operations
    derivatives_byslice : NDArray
        Derivative data organized by slice
    nprocs : int, optional
        Number of processors to use for multiprocessing (default is 1)
    alwaysmultiproc : bool, optional
        Whether to always use multiprocessing regardless of data size (default is False)
    showprogressbar : bool, optional
        Whether to display progress bar during processing (default is True)
    debug : bool, optional
        Enable debug mode for additional logging (default is False)

    Returns
    -------
    NDArray
        Processed data after smoothing operations have been applied

    Notes
    -----
    This function uses the `tide_genericmultiproc.run_multiproc` utility to distribute
    the smoothing workload across multiple processors. The function handles data organization
    and processing for each slice individually, then combines results.

    Examples
    --------
    >>> result = tcsmoothingpass(
    ...     numslices=10,
    ...     validlocslist=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    ...     rawapp_byslice=raw_data,
    ...     appsmoothingfilter=smoothing_filter,
    ...     phaseFs=100.0,
    ...     derivatives_byslice=derivatives,
    ...     nprocs=4
    ... )
    """
    inputshape = rawapp_byslice.shape
    sliceargs = [validlocslist, rawapp_byslice, appsmoothingfilter, phaseFs, derivatives_byslice]
    slicefunc = _procOneSliceSmoothing
    packfunc = _packslicedataSliceSmoothing
    unpackfunc = _unpackslicedataSliceSmoothing
    slicetargets = [rawapp_byslice, derivatives_byslice]
    slicemask = np.ones_like(rawapp_byslice[0, :, 0])

    slicetotal = tide_genericmultiproc.run_multiproc(
        slicefunc,
        packfunc,
        unpackfunc,
        sliceargs,
        slicetargets,
        inputshape,
        slicemask,
        None,
        nprocs,
        alwaysmultiproc,
        showprogressbar,
        16,
        indexaxis=1,
        procunit="slices",
        debug=debug,
    )


def phaseproject(
    input_data,
    demeandata_byslice,
    means_byslice,
    rawapp_byslice,
    app_byslice,
    normapp_byslice,
    weights_byslice,
    cine_byslice,
    projmask_byslice,
    derivatives_byslice,
    proctrs,
    thispass,
    args,
    sliceoffsets,
    cardphasevals,
    outphases,
    appsmoothingfilter,
    phaseFs,
    thecorrfunc_byslice,
    waveamp_byslice,
    wavedelay_byslice,
    wavedelayCOM_byslice,
    corrected_rawapp_byslice,
    corrstartloc,
    correndloc,
    thealiasedcorrx,
    theAliasedCorrelator,
):
    """
    Perform phase projection and related processing on fMRI data across slices.

    This function performs phase projection on fMRI data, optionally smoothing
    timecourses, and applying flips based on derivative information. It also
    computes wavelet-based correlation measures and updates relevant arrays
    in-place for further processing.

    Parameters
    ----------
    input_data : object
        Input fMRI data container with `getdims()` and `byslice()` methods.
    demeandata_byslice : array_like
        Demeaned fMRI data by slice.
    means_byslice : array_like
        Mean values by slice for normalization.
    rawapp_byslice : array_like
        Raw APP data by slice.
    app_byslice : array_like
        APP data after initial processing.
    normapp_byslice : array_like
        Normalized APP data.
    weights_byslice : array_like
        Weights by slice for processing.
    cine_byslice : array_like
        Cine data by slice.
    projmask_byslice : array_like
        Projection mask by slice.
    derivatives_byslice : array_like
        Derivative data by slice, used for determining flips.
    proctrs : array_like
        Processing timepoints or transformation parameters.
    thispass : int
        Current processing pass number.
    args : argparse.Namespace
        Command-line arguments controlling processing behavior.
    sliceoffsets : array_like
        Slice offset values.
    cardphasevals : array_like
        Cardiac phase values.
    outphases : array_like
        Output phases.
    appsmoothingfilter : array_like
        Smoothing filter for timecourses.
    phaseFs : float
        Sampling frequency for phase processing.
    thecorrfunc_byslice : array_like
        Correlation function by slice.
    waveamp_byslice : array_like
        Wave amplitude by slice.
    wavedelay_byslice : array_like
        Wave delay by slice.
    wavedelayCOM_byslice : array_like
        Center of mass of wave delay by slice.
    corrected_rawapp_byslice : array_like
        Corrected raw APP data by slice.
    corrstartloc : int
        Start location for correlation computation.
    correndloc : int
        End location for correlation computation.
    thealiasedcorrx : array_like
        Aliased correlation x-axis values.
    theAliasedCorrelator : object
        Correlator object for aliased correlation computation.

    Returns
    -------
    appflips_byslice : array_like
        Flip values applied to the APP data by slice.

    Notes
    -----
    - The function modifies several input arrays in-place.
    - If `args.smoothapp` is True, smoothing is applied to the raw APP data.
    - If `args.fliparteries` is True, flips are applied to correct arterial
      orientation.
    - If `args.doaliasedcorrelation` is True, aliased correlation is computed
      and stored in `thecorrfunc_byslice`.

    Examples
    --------
    >>> phaseproject(
    ...     input_data, demeandata_byslice, means_byslice, rawapp_byslice,
    ...     app_byslice, normapp_byslice, weights_byslice, cine_byslice,
    ...     projmask_byslice, derivatives_byslice, proctrs, thispass, args,
    ...     sliceoffsets, cardphasevals, outphases, appsmoothingfilter,
    ...     phaseFs, thecorrfunc_byslice, waveamp_byslice, wavedelay_byslice,
    ...     wavedelayCOM_byslice, corrected_rawapp_byslice, corrstartloc,
    ...     correndloc, thealiasedcorrx, theAliasedCorrelator
    ... )
    """
    xsize, ysize, numslices, timepoints = input_data.getdims()
    fmri_data_byslice = input_data.byslice()

    # first find the validlocs for each slice
    validlocslist = []
    if args.verbose:
        print("Finding validlocs")
    for theslice in range(numslices):
        validlocslist.append(np.where(projmask_byslice[:, theslice] > 0)[0])

    # phase project each slice
    print("Phase projecting")
    phaseprojectpass(
        numslices,
        demeandata_byslice,
        fmri_data_byslice,
        validlocslist,
        proctrs,
        weights_byslice,
        cine_byslice,
        rawapp_byslice,
        outphases,
        cardphasevals,
        args.congridbins,
        args.gridkernel,
        args.destpoints,
        cache=args.congridcache,
        mpcode=args.mpphaseproject,
        nprocs=args.nprocs,
        showprogressbar=args.showprogressbar,
    )

    # smooth the phase projection, if requested
    if args.smoothapp:
        print("Smoothing timecourses")
        tcsmoothingpass(
            numslices,
            validlocslist,
            rawapp_byslice,
            appsmoothingfilter,
            phaseFs,
            derivatives_byslice,
            nprocs=args.nprocs,
            showprogressbar=args.showprogressbar,
        )

    # now do the flips
    print("Doing flips")
    appflips_byslice = np.where(
        -derivatives_byslice[:, :, 2] > derivatives_byslice[:, :, 0], -1.0, 1.0
    )
    for theslice in tqdm(
        range(numslices),
        desc="Slice",
        unit="slices",
        disable=(not args.showprogressbar),
    ):
        # now do the flips
        validlocs = validlocslist[theslice]
        if len(validlocs) > 0:
            timecoursemean = np.mean(rawapp_byslice[validlocs, theslice, :], axis=1).reshape(
                (-1, 1)
            )
            if args.fliparteries:
                corrected_rawapp_byslice[validlocs, theslice, :] = (
                    rawapp_byslice[validlocs, theslice, :] - timecoursemean
                ) * appflips_byslice[validlocs, theslice, None] + timecoursemean
                if args.doaliasedcorrelation and (thispass > 0):
                    for theloc in validlocs:
                        thecorrfunc_byslice[theloc, theslice, :] = theAliasedCorrelator.apply(
                            -appflips_byslice[theloc, theslice]
                            * demeandata_byslice[theloc, theslice, :],
                            int(sliceoffsets[theslice]),
                        )[corrstartloc : correndloc + 1]
                        maxloc = np.argmax(thecorrfunc_byslice[theloc, theslice, :])
                        wavedelay_byslice[theloc, theslice] = (
                            thealiasedcorrx[corrstartloc : correndloc + 1]
                        )[maxloc]
                        waveamp_byslice[theloc, theslice] = np.fabs(
                            thecorrfunc_byslice[theloc, theslice, maxloc]
                        )
                        wavedelayCOM_byslice[theloc, theslice] = theCOM(
                            thealiasedcorrx[corrstartloc : correndloc + 1],
                            np.fabs(thecorrfunc_byslice[theloc, theslice, :]),
                        )
            else:
                corrected_rawapp_byslice[validlocs, theslice, :] = rawapp_byslice[
                    validlocs, theslice, :
                ]
                if args.doaliasedcorrelation and (thispass > 0):
                    for theloc in validlocs:
                        thecorrfunc_byslice[theloc, theslice, :] = theAliasedCorrelator.apply(
                            -demeandata_byslice[theloc, theslice, :],
                            int(sliceoffsets[theslice]),
                        )[corrstartloc : correndloc + 1]
                        maxloc = np.argmax(np.abs(thecorrfunc_byslice[theloc, theslice, :]))
                        wavedelay_byslice[theloc, theslice] = (
                            thealiasedcorrx[corrstartloc : correndloc + 1]
                        )[maxloc]
                        waveamp_byslice[theloc, theslice] = np.fabs(
                            thecorrfunc_byslice[theloc, theslice, maxloc]
                        )
            timecoursemin = np.min(
                corrected_rawapp_byslice[validlocs, theslice, :], axis=1
            ).reshape((-1, 1))
            app_byslice[validlocs, theslice, :] = (
                corrected_rawapp_byslice[validlocs, theslice, :] - timecoursemin
            )
            normapp_byslice[validlocs, theslice, :] = np.nan_to_num(
                app_byslice[validlocs, theslice, :] / means_byslice[validlocs, theslice, None]
            )
    return appflips_byslice


def findvessels(
    app,
    normapp,
    validlocs,
    numspatiallocs,
    outputroot,
    unnormvesselmap,
    destpoints,
    softvesselfrac,
    histlen,
    outputlevel,
    debug=False,
) -> tuple[float, float]:
    """
    Find vessel thresholds and generate vessel masks from app data.

    This function processes app data to identify vessel thresholds and optionally
    generates histograms for visualization. It handles both normalized and
    unnormalized vessel maps based on the input parameters.

    Parameters
    ----------
    app : NDArray
        Raw app data array
    normapp : NDArray
        Normalized app data array
    validlocs : NDArray
        Array of valid locations for processing
    numspatiallocs : int
        Number of spatial locations
    outputroot : str
        Root directory path for output files
    unnormvesselmap : bool
        Flag indicating whether to use unnormalized vessel map
    destpoints : int
        Number of destination points
    softvesselfrac : float
        Fractional multiplier for soft vessel threshold
    histlen : int
        Length of histogram bins
    outputlevel : int
        Level of output generation (0 = no histogram, 1 = histogram only)
    debug : bool, optional
        Debug flag for additional logging (default is False)

    Returns
    -------
    tuple[float, float]
        A tuple containing:
        - hardvesselthresh: float
          Hard threshold for vessel detection.
        - softvesselthresh: float
          Soft threshold for vessel detection.

    Notes
    -----
    The function performs the following steps:
    1. Reshapes app data based on unnormvesselmap flag
    2. Extracts valid locations from the reshaped data
    3. Generates histogram if outputlevel > 0
    4. Calculates hard and soft vessel thresholds based on 98th percentile
    5. Prints threshold values to console

    Examples
    --------
    >>> hard_thresh, soft_thresh = findvessels(
    ...     app=app_data,
    ...     normapp=norm_app_data,
    ...     validlocs=valid_indices,
    ...     numspatiallocs=100,
    ...     outputroot='/path/to/output',
    ...     unnormvesselmap=True,
    ...     destpoints=50,
    ...     softvesselfrac=0.5,
    ...     histlen=100,
    ...     outputlevel=1
    ... )
    """
    if unnormvesselmap:
        app2d = app.reshape((numspatiallocs, destpoints))
    else:
        app2d = normapp.reshape((numspatiallocs, destpoints))
    histinput = app2d[validlocs, :].reshape((len(validlocs), destpoints))
    if outputlevel > 0:
        namesuffix = "_desc-apppeaks_hist"
        tide_stats.makeandsavehistogram(
            histinput,
            histlen,
            0,
            outputroot + namesuffix,
            debug=debug,
        )

    # find vessel thresholds
    tide_util.logmem("before making vessel masks")
    hardvesselthresh = tide_stats.getfracvals(np.max(histinput, axis=1), [0.98])[0] / 2.0
    softvesselthresh = softvesselfrac * hardvesselthresh
    print(
        "hard, soft vessel thresholds set to",
        "{:.3f}".format(hardvesselthresh),
        "{:.3f}".format(softvesselthresh),
    )
    return hardvesselthresh, softvesselthresh


def upsampleimage(input_data, numsteps, sliceoffsets, slicesamplerate, outputroot):
    """
    Upsample fMRI data along the temporal and slice dimensions.

    This function takes fMRI data and upsamples it by a factor of `numsteps` along
    the temporal dimension, and interpolates across slices to align with specified
    slice offsets. The resulting upsampled data is saved as a NIfTI file.

    Parameters
    ----------
    input_data : object
        Input fMRI data object with attributes: `byvol()`, `timepoints`, `xsize`,
        `ysize`, `numslices`, and `copyheader()`.
    numsteps : int
        Upsampling factor along the temporal dimension.
    sliceoffsets : array-like of int
        Slice offset indices indicating where each slice's data should be placed
        in the upsampled volume.
    slicesamplerate : float
        Sampling rate of the slice acquisition (used to set the TR in the output header).
    outputroot : str
        Root name for the output NIfTI file (will be suffixed with "_upsampled").

    Returns
    -------
    None
        The function saves the upsampled data to a NIfTI file and does not return any value.

    Notes
    -----
    - The function demeanes the input data before upsampling.
    - Interpolation is performed along the slice direction using linear interpolation.
    - The output file is saved using `tide_io.savetonifti`.

    Examples
    --------
    >>> upsampleimage(fmri_data, numsteps=2, sliceoffsets=[0, 1], slicesamplerate=2.0, outputroot='output')
    Upsamples the fMRI data by a factor of 2 and saves to 'output_upsampled.nii'.
    """
    fmri_data = input_data.byvol()
    timepoints = input_data.timepoints
    xsize = input_data.xsize
    ysize = input_data.ysize
    numslices = input_data.numslices

    # allocate the image
    print(f"upsampling fmri data by a factor of {numsteps}")
    upsampleimage = np.zeros((xsize, ysize, numslices, numsteps * timepoints), dtype=float)

    # demean the raw data
    meanfmri = fmri_data.mean(axis=1)
    demeaned_data = fmri_data - meanfmri[:, None]

    # drop in the raw data
    for theslice in range(numslices):
        upsampleimage[
            :, :, theslice, sliceoffsets[theslice] : timepoints * numsteps : numsteps
        ] = demeaned_data.reshape((xsize, ysize, numslices, timepoints))[:, :, theslice, :]

    upsampleimage_byslice = upsampleimage.reshape(xsize * ysize, numslices, timepoints * numsteps)

    # interpolate along the slice direction
    thedstlocs = np.linspace(0, numslices, num=len(sliceoffsets), endpoint=False)
    print(f"len(destlocst), destlocs: {len(thedstlocs)}, {thedstlocs}")
    for thetimepoint in range(0, timepoints * numsteps):
        thestep = thetimepoint % numsteps
        print(f"interpolating step {thestep}")
        thesrclocs = np.where(sliceoffsets == thestep)[0]
        print(f"timepoint: {thetimepoint}, sourcelocs: {thesrclocs}")
        for thexyvoxel in range(xsize * ysize):
            theinterps = np.interp(
                thedstlocs,
                1.0 * thesrclocs,
                upsampleimage_byslice[thexyvoxel, thesrclocs, thetimepoint],
            )
            upsampleimage_byslice[thexyvoxel, :, thetimepoint] = 1.0 * theinterps

    theheader = input_data.copyheader(
        numtimepoints=(timepoints * numsteps), tr=(1.0 / slicesamplerate)
    )
    tide_io.savetonifti(upsampleimage, theheader, outputroot + "_upsampled")
    print("upsampling complete")


def wrightmap(
    input_data,
    demeandata_byslice,
    rawapp_byslice,
    projmask_byslice,
    outphases,
    cardphasevals,
    proctrs,
    congridbins,
    gridkernel,
    destpoints,
    iterations=100,
    nprocs=-1,
    verbose=False,
    debug=False,
):
    """
    Compute a vessel map using Wright's method by performing phase correlation
    analysis across randomized subsets of timecourses.

    This function implements Wright's method for estimating vessel maps by
    splitting the timecourse data into two random halves, projecting each half
    separately, and computing the Pearson correlation between the resulting
    projections for each voxel and slice. The final map is derived as the mean
    of these correlations across iterations.

    Parameters
    ----------
    input_data : object
        Input data container with attributes `xsize`, `ysize`, and `numslices`.
    demeandata_byslice : array_like
        Demeaned data organized by slice, shape ``(nvoxels, numslices)``.
    rawapp_byslice : array_like
        Raw application data by slice, shape ``(nvoxels, numslices)``.
    projmask_byslice : array_like
        Projection mask by slice, shape ``(nvoxels, numslices)``.
    outphases : array_like
        Output phases, shape ``(nphases,)``.
    cardphasevals : array_like
        Cardinal phase values, shape ``(nphases,)``.
    proctrs : array_like
        Timecourse indices to be processed, shape ``(ntimepoints,)``.
    congridbins : array_like
        Binning information for congrid interpolation.
    gridkernel : array_like
        Kernel for grid interpolation.
    destpoints : array_like
        Destination points for projection.
    iterations : int, optional
        Number of iterations for random splitting (default is 100).
    nprocs : int, optional
        Number of processes to use for parallel computation; -1 uses all
        available cores (default is -1).
    verbose : bool, optional
        If True, print progress messages (default is False).
    debug : bool, optional
        If True, print additional debug information (default is False).

    Returns
    -------
    wrightcorrs : ndarray
        Computed vessel map with shape ``(xsize, ysize, numslices)``.

    Notes
    -----
    This function performs a bootstrap-like procedure where the input timecourse
    is randomly split into two halves, and phase projections are computed for
    each half. Pearson correlation is computed between the two projections for
    each voxel and slice. The result is averaged over all iterations to produce
    the final vessel map.

    Examples
    --------
    >>> wrightcorrs = wrightmap(
    ...     input_data,
    ...     demeandata_byslice,
    ...     rawapp_byslice,
    ...     projmask_byslice,
    ...     outphases,
    ...     cardphasevals,
    ...     proctrs,
    ...     congridbins,
    ...     gridkernel,
    ...     destpoints,
    ...     iterations=50,
    ...     verbose=True
    ... )
    """
    xsize = input_data.xsize
    ysize = input_data.ysize
    numslices = input_data.numslices
    # make a vessel map using Wright's method
    wrightcorrs_byslice = np.zeros((xsize * ysize, numslices, iterations))
    # first find the validlocs for each slice
    validlocslist = []
    if verbose:
        print("Finding validlocs")
    for theslice in range(numslices):
        validlocslist.append(np.where(projmask_byslice[:, theslice] > 0)[0])
    for theiteration in range(iterations):
        print(f"wright iteration: {theiteration + 1} of {iterations}")
        # split timecourse into two sets
        scrambledprocs = np.random.permutation(proctrs)
        proctrs1 = scrambledprocs[: int(len(scrambledprocs) // 2)]
        proctrs2 = scrambledprocs[int(len(scrambledprocs) // 2) :]
        if debug:
            print(f"{proctrs1=}, {proctrs2=}")

        # phase project each slice
        rawapp_byslice1 = np.zeros_like(rawapp_byslice)
        cine_byslice1 = np.zeros_like(rawapp_byslice)
        weights_byslice1 = np.zeros_like(rawapp_byslice)
        phaseprojectpass(
            numslices,
            demeandata_byslice,
            input_data.byslice(),
            validlocslist,
            proctrs1,
            weights_byslice1,
            cine_byslice1,
            rawapp_byslice1,
            outphases,
            cardphasevals,
            congridbins,
            gridkernel,
            destpoints,
            nprocs=nprocs,
            showprogressbar=False,
        )
        rawapp_byslice2 = np.zeros_like(rawapp_byslice)
        cine_byslice2 = np.zeros_like(rawapp_byslice)
        weights_byslice2 = np.zeros_like(rawapp_byslice)
        phaseprojectpass(
            numslices,
            demeandata_byslice,
            input_data.byslice(),
            validlocslist,
            proctrs2,
            weights_byslice2,
            cine_byslice2,
            rawapp_byslice2,
            outphases,
            cardphasevals,
            congridbins,
            gridkernel,
            destpoints,
            nprocs=nprocs,
            showprogressbar=False,
        )
        for theslice in range(numslices):
            for thepoint in validlocslist[theslice]:
                theresult = pearsonr(
                    rawapp_byslice1[thepoint, theslice, :],
                    rawapp_byslice2[thepoint, theslice, :],
                )
                theRvalue = theresult.statistic
                if debug:
                    print("theRvalue = ", theRvalue)
                wrightcorrs_byslice[thepoint, theslice, theiteration] = theRvalue
    wrightcorrs = np.mean(wrightcorrs_byslice, axis=2).reshape(xsize, ysize, numslices)
    return wrightcorrs




__all__ = [
    "circularderivs",
    "preloadcongrid",
    "phaseprojectpass",
    "tcsmoothingpass",
    "phaseproject",
    "findvessels",
    "upsampleimage",
    "wrightmap",
]
