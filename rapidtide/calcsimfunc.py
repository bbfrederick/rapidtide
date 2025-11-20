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
import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray

import rapidtide.genericmultiproc as tide_genericmultiproc
import rapidtide.resample as tide_resample

warnings.simplefilter(action="ignore", category=FutureWarning)
LGR = logging.getLogger("GENERAL")


def _procOneVoxelCorrelation(
    vox: int,
    voxelargs: list[Any],
    **kwargs: Any,
) -> tuple[int, float, NDArray, NDArray, float, list[float]]:
    """
    Process correlation for a single voxel.

    This function performs correlation analysis on a single voxel using the provided
    fMRI data and correlation parameters. It handles resampling of fMRI data based
    on the oversampling factor and computes the correlation between the resampled
    data and the target time course.

    Parameters
    ----------
    vox : int
        The voxel index being processed.
    voxelargs : list[Any]
        List containing the following elements in order:
        - thetc : array-like
        - theCorrelator : object
        - fmri_x : array-like
        - fmritc : array-like
        - os_fmri_x : array-like
        - theglobalmaxlist : list
        - thexcorr_y : array-like
    **kwargs : Any
        Additional keyword arguments that override default options:
        - oversampfactor : int, optional
            Oversampling factor for resampling (default: 1)
        - interptype : str, optional
            Interpolation type for resampling (default: "univariate")
        - debug : bool, optional
            Enable debug printing (default: False)

    Returns
    -------
    tuple[int, float, NDArray, NDArray, float, list[float]]
        A tuple containing:
        - vox : int
            The input voxel index
        - np.mean(thetc) : float
            Mean of the processed time course
        - thexcorr_y : NDArray
            Correlation values
        - thexcorr_x : NDArray
            Correlation lags
        - theglobalmax : float
            Global maximum correlation value
        - theglobalmaxlist : list[float]
            List of global maximum correlation values

    Notes
    -----
    The function modifies the input `thetc` array in-place with the resampled data.
    If oversampfactor is less than 1, no resampling is performed and the original
    time course is used.

    Examples
    --------
    >>> result = _procOneVoxelCorrelation(
    ...     vox=100,
    ...     voxelargs=[thetc, correlator, fmri_x, fmritc, os_fmri_x, globalmaxlist, xcorr_y],
    ...     oversampfactor=2,
    ...     debug=True
    ... )
    """
    options = {
        "oversampfactor": 1,
        "interptype": "univariate",
        "debug": False,
    }
    options.update(kwargs)
    oversampfactor = options["oversampfactor"]
    interptype = options["interptype"]
    debug = options["debug"]
    if debug:
        print(f"{oversampfactor=} {interptype=}")
    (thetc, theCorrelator, fmri_x, fmritc, os_fmri_x, theglobalmaxlist, thexcorr_y) = voxelargs
    if oversampfactor >= 1:
        thetc[:] = tide_resample.doresample(fmri_x, fmritc, os_fmri_x, method=interptype)
    else:
        thetc[:] = fmritc
    thexcorr_y, thexcorr_x, theglobalmax = theCorrelator.run(thetc)
    # print(f"_procOneVoxelCorrelation: {thexcorr_x=}")

    return vox, np.mean(thetc), thexcorr_y, thexcorr_x, theglobalmax, theglobalmaxlist


def _packvoxeldata(voxnum: int, voxelargs: list[Any]) -> list[Any]:
    """
    Pack voxel data into a structured list format.

    This function extracts and organizes voxel data from a list of arguments,
    specifically selecting a slice from the fourth element based on the voxel number.

    Parameters
    ----------
    voxnum : int
        The voxel index used to select a specific row from the fourth element
        of voxelargs, which is expected to be a 2D array-like structure.
    voxelargs : list[Any]
        A list containing voxel-related arguments. The expected structure is:
        [arg0, arg1, arg2, array_2d, arg4, arg5, arg6]
        where the fourth element (index 3) should be a 2D array-like object
        from which a row will be selected using voxnum.

    Returns
    -------
    list[Any]
        A list containing the packed voxel data with the following structure:
        [voxelargs[0], voxelargs[1], voxelargs[2],
         voxelargs[3][voxnum, :], voxelargs[4], voxelargs[5], voxelargs[6]]
        where the fourth element is the selected row from the 2D array.

    Notes
    -----
    The function assumes that voxelargs[3] is a 2D array-like structure and
    that voxnum is a valid index for selecting a row from this array.

    Examples
    --------
    >>> voxelargs = [1, 2, 3, [[10, 20], [30, 40]], 5, 6, 7]
    >>> _packvoxeldata(1, voxelargs)
    [1, 2, 3, [30, 40], 5, 6, 7]
    """
    return [
        voxelargs[0],
        voxelargs[1],
        voxelargs[2],
        (voxelargs[3])[voxnum, :],
        voxelargs[4],
        voxelargs[5],
        voxelargs[6],
    ]


def _unpackvoxeldata(retvals: tuple[Any, ...], voxelproducts: list[Any]) -> None:
    """
    Unpack voxel data from retvals into voxelproducts structure.

    Parameters
    ----------
    retvals : tuple[Any, ...]
        Tuple containing voxel data to be unpacked. Expected to contain at least 5 elements
        where:
        - retvals[0]: index/key for first assignment
        - retvals[1]: value for first assignment
        - retvals[2]: array-like data for second assignment
        - retvals[3]: value for third assignment
        - retvals[4]: value for fourth assignment (will be incremented by 0)
    voxelproducts : list[Any]
        List containing voxel data structures where unpacked data will be stored:
        - voxelproducts[0]: dict or array-like structure for first assignment
        - voxelproducts[1]: 2D array-like structure for second assignment
        - voxelproducts[2]: scalar or single value storage
        - voxelproducts[3]: list-like structure for appending fourth assignment

    Returns
    -------
    None
        This function modifies voxelproducts in-place and does not return any value.

    Notes
    -----
    This function performs in-place modifications of the voxelproducts list elements.
    The fourth assignment uses `retvals[4] + 0` which effectively creates a copy of
    the value to ensure no reference issues.

    Examples
    --------
    >>> retvals = (0, 'value1', [1, 2, 3], 42, 10)
    >>> voxelproducts = [{}, [[0]*3], 0, []]
    >>> _unpackvoxeldata(retvals, voxelproducts)
    >>> voxelproducts[0]
    {0: 'value1'}
    >>> voxelproducts[1]
    [[1, 2, 3]]
    >>> voxelproducts[2]
    42
    >>> voxelproducts[3]
    [10]
    """
    (voxelproducts[0])[retvals[0]] = retvals[1]
    (voxelproducts[1])[retvals[0], :] = retvals[2]
    voxelproducts[2] = retvals[3]
    (voxelproducts[3]).append(retvals[4] + 0)


def correlationpass(
    fmridata: NDArray,
    referencetc: NDArray,
    theCorrelator: Any,
    fmri_x: NDArray,
    os_fmri_x: NDArray,
    lagmininpts: int,
    lagmaxinpts: int,
    corrout: NDArray,
    meanval: NDArray,
    nprocs: int = 1,
    alwaysmultiproc: bool = False,
    oversampfactor: int = 1,
    interptype: str = "univariate",
    showprogressbar: bool = True,
    chunksize: int = 1000,
    rt_floattype: np.dtype = np.float64,
    debug: bool = False,
) -> tuple[int, list[float], NDArray]:
    """
    Compute correlation-based similarity function across MRI voxels using multi-processing.

    This function computes a correlation-based similarity measure between a reference time course
    and fMRI data across voxels, using a specified correlator object. It supports both single and
    multi-processing modes and allows for various interpolation and oversampling options.

    Parameters
    ----------
    fmridata : ndarray
        4D fMRI data array of shape (time, x, y, z).
    referencetc : ndarray
        Reference time course of shape (time,).
    theCorrelator : object
        An object implementing the `setreftc` and `setlimits` methods for correlation computation.
    fmri_x : ndarray
        Time points corresponding to fMRI data, shape (time,).
    os_fmri_x : ndarray
        Oversampled time points, shape (oversampled_time,).
    lagmininpts : int
        Minimum lag in samples to consider for correlation.
    lagmaxinpts : int
        Maximum lag in samples to consider for correlation.
    corrout : ndarray
        Output array to store correlation values, shape (time, x, y, z).
    meanval : ndarray
        Array to store mean values, shape (x, y, z).
    nprocs : int, optional
        Number of processes to use for parallel computation. Default is 1.
    alwaysmultiproc : bool, optional
        If True, always use multiprocessing even for single voxel processing. Default is False.
    oversampfactor : int, optional
        Oversampling factor for interpolation. Default is 1.
    interptype : str, optional
        Interpolation type, e.g., 'univariate'. Default is 'univariate'.
    showprogressbar : bool, optional
        Whether to display a progress bar. Default is True.
    chunksize : int, optional
        Size of chunks for multiprocessing. Default is 1000.
    rt_floattype : str, optional
        String representation of floating-point type. Default is 'float64'.
    debug : bool, optional
        If True, enable debug logging. Default is False.

    Returns
    -------
    tuple of (int, list of float, ndarray)
        - Total number of voxels processed.
        - List of global maximum correlation values.
        - Correlation scale array.

    Notes
    -----
    The function uses `tide_genericmultiproc.run_multiproc` to perform multi-voxel correlation
    computations in parallel. It initializes a correlator object and sets the reference time course
    and lag limits before starting the computation.

    Examples
    --------
    >>> import numpy as np
    >>> from some_module import correlationpass, SomeCorrelator
    >>> fmri_data = np.random.rand(100, 64, 64, 32)
    >>> ref_tc = np.random.rand(100)
    >>> correlator = SomeCorrelator()
    >>> fmri_x = np.linspace(0, 100, 100)
    >>> os_fmri_x = np.linspace(0, 100, 200)
    >>> corr_out = np.zeros_like(fmri_data)
    >>> mean_val = np.zeros((64, 64, 32))
    >>> total_voxels, max_vals, corr_scale = correlationpass(
    ...     fmridata=fmri_data,
    ...     referencetc=ref_tc,
    ...     theCorrelator=correlator,
    ...     fmri_x=fmri_x,
    ...     os_fmri_x=os_fmri_x,
    ...     lagmininpts=-10,
    ...     lagmaxinpts=10,
    ...     corrout=corr_out,
    ...     meanval=mean_val,
    ...     nprocs=4,
    ...     debug=False
    ... )
    """
    if debug:
        print(f"calling setreftc in calcsimfunc with length {len(referencetc)}")
    theCorrelator.setreftc(referencetc)
    theCorrelator.setlimits(lagmininpts, lagmaxinpts)
    thetc = np.zeros(np.shape(os_fmri_x), dtype=rt_floattype)
    theglobalmaxlist = []

    # generate a corrscale of the correct length
    dummy = np.zeros(100, dtype=rt_floattype)
    dummy, dummy, dummy, thecorrscale, dummy, dummy = _procOneVoxelCorrelation(
        0,
        _packvoxeldata(
            0, [thetc, theCorrelator, fmri_x, fmridata, os_fmri_x, theglobalmaxlist, dummy]
        ),
        oversampfactor=oversampfactor,
        interptype=interptype,
    )

    inputshape = np.shape(fmridata)
    voxelargs = [thetc, theCorrelator, fmri_x, fmridata, os_fmri_x, theglobalmaxlist, thecorrscale]
    voxelfunc = _procOneVoxelCorrelation
    packfunc = _packvoxeldata
    unpackfunc = _unpackvoxeldata
    voxeltargets = [meanval, corrout, thecorrscale, theglobalmaxlist]
    voxelmask = np.ones_like(fmridata[:, 0])

    volumetotal = tide_genericmultiproc.run_multiproc(
        voxelfunc,
        packfunc,
        unpackfunc,
        voxelargs,
        voxeltargets,
        inputshape,
        voxelmask,
        LGR,
        nprocs,
        alwaysmultiproc,
        showprogressbar,
        chunksize,
        oversampfactor=oversampfactor,
        interptype=interptype,
        debug=debug,
    )
    LGR.info(f"\nSimilarity function calculated on {volumetotal} voxels")

    # garbage collect
    uncollected = gc.collect()
    if uncollected != 0:
        LGR.info(f"garbage collected - unable to collect {uncollected} objects")
    else:
        LGR.info("garbage collected")

    return volumetotal, theglobalmaxlist, thecorrscale
