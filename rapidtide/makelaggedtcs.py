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
from typing import Any

import numpy as np
from numpy.typing import NDArray

import rapidtide.genericmultiproc as tide_genericmultiproc


def _procOneVoxelMakelagtc(
    vox: int,
    voxelargs: list,
    **kwargs: Any,
) -> tuple[int, NDArray]:
    """
    Process a single voxel to compute lag timecourse using lag timecourse generator.

    This function takes a voxel index and associated arguments to compute a lag
    timecourse using the provided lag timecourse generator. The computation involves
    evaluating the generator at timepoints shifted by the specified lag value.

    Parameters
    ----------
    vox : int
        Voxel index identifier used for tracking and debugging purposes.
    voxelargs : list
        List containing three elements:
        1. `lagtcgenerator` - Lag timecourse generator object with `yfromx` method
        2. `thelag` - Lag value to be subtracted from time axis
        3. `timeaxis` - Time axis array for evaluation
    **kwargs : Any
        Additional keyword arguments that can override default options:
        - `rt_floattype` : float type for computations (default: np.float64)
        - `debug` : boolean for debug printing (default: False)

    Returns
    -------
    tuple[int, NDArray]
        Tuple containing:
        - `vox` : Input voxel index
        - `thelagtc` : Computed lag timecourse array

    Notes
    -----
    The lag value is subtracted from the time axis as of 10/18. Alternative approaches
    of adding the lag value resulted in poor performance.

    Examples
    --------
    >>> import numpy as np
    >>>
    >>> # Example usage with mock generator
    >>> class MockGenerator:
    ...     def yfromx(self, x):
    ...         return x**2
    ...
    >>> generator = MockGenerator()
    >>> time_axis = np.array([0, 1, 2, 3, 4])
    >>> voxel_args = [generator, 1.0, time_axis]
    >>> result = _procOneVoxelMakelagtc(0, voxel_args)
    >>> print(result)
    (0, array([0., 1., 4., 9., 16.]))
    """
    # unpack arguments
    options = {
        "rt_floattype": np.float64,
        "debug": False,
    }
    options.update(kwargs)
    rt_floattype = options["rt_floattype"]
    debug = options["debug"]
    (lagtcgenerator, thelag, timeaxis) = voxelargs
    if debug:
        print(f"{vox=}, {thelag=}, {timeaxis=}")

    # question - should maxlag be added or subtracted?  As of 10/18, it is subtracted
    #  potential answer - tried adding, results are terrible.
    thelagtc = (lagtcgenerator.yfromx(timeaxis - thelag)).astype(rt_floattype)

    return (
        vox,
        (thelagtc),
    )


def _packvoxeldata(voxnum: int, voxelargs: list) -> list:
    """
    Pack voxel data into a list format.

    Parameters
    ----------
    voxnum : int
        The index used to select an element from the second element of voxelargs.
    voxelargs : list
        A list containing three elements where:
        - voxelargs[0] is the first element to be returned
        - voxelargs[1] is a list or array from which an element is selected using voxnum
        - voxelargs[2] is the third element to be returned

    Returns
    -------
    list
        A list containing three elements: [voxelargs[0], voxelargs[1][voxnum], voxelargs[2]]

    Notes
    -----
    This function assumes that voxelargs[1] is indexable and that voxnum is a valid index
    for accessing elements in voxelargs[1]. The function does not perform any validation
    of the input parameters.

    Examples
    --------
    >>> _packvoxeldata(1, [10, [20, 30, 40], 50])
    [10, 30, 50]

    >>> _packvoxeldata(0, ['a', ['b', 'c', 'd'], 'e'])
    ['a', 'b', 'e']
    """
    return [voxelargs[0], (voxelargs[1])[voxnum], voxelargs[2]]


def _unpackvoxeldata(retvals: tuple, voxelproducts: list) -> None:
    """
    Unpack voxel data into the specified voxel products array.

    This function takes return values and assigns them to a specific location
    in the voxel products array. It is typically used as part of a larger
    voxel processing pipeline.

    Parameters
    ----------
    retvals : tuple
        A tuple containing the data to be unpacked. The first element is used
        as an index for the voxel products array, and the second element contains
        the actual data to be assigned.
    voxelproducts : list
        A list of arrays where the voxel data will be stored. The function
        modifies the first element of this list in-place.

    Returns
    -------
    None
        This function modifies the voxelproducts list in-place and does not
        return any value.

    Notes
    -----
    The function assumes that retvals[0] is a valid index for the first element
    of voxelproducts and that retvals[1] has compatible dimensions for assignment.

    Examples
    --------
    >>> retvals = (5, [1, 2, 3, 4, 5])
    >>> voxelproducts = [[0] * 10]
    >>> _unpackvoxeldata(retvals, voxelproducts)
    >>> print(voxelproducts[0])
    [0, 0, 0, 0, 0, 1, 2, 3, 4, 5]
    """
    (voxelproducts[0])[retvals[0], :] = retvals[1]


def makelaggedtcs(
    lagtcgenerator: Any,
    timeaxis: NDArray,
    lagmask: NDArray,
    lagtimes: NDArray,
    lagtc: NDArray,
    LGR: logging.Logger | None = None,
    nprocs: int = 1,
    alwaysmultiproc: bool = False,
    showprogressbar: bool = True,
    chunksize: int = 1000,
    rt_floattype: np.dtype = np.dtype(np.float64),
    debug: bool = False,
) -> int:
    """
    Generate lagged timecourses for a set of voxels using multiprocessing.

    This function computes lagged timecourses for each voxel specified in the mask,
    using the provided lag timecourse generator and time axis. It supports
    parallel processing for performance optimization.

    Parameters
    ----------
    lagtcgenerator : Any
        A callable or object that generates lagged timecourses for a single voxel.
    timeaxis : NDArray
        1D array representing the time axis (e.g., TRs or time points).
    lagmask : NDArray
        3D or 4D boolean or integer array defining the voxels to process.
        Non-zero entries indicate voxels to be processed.
    lagtimes : NDArray
        1D array of lag times (in seconds or time units) to be applied.
    lagtc : NDArray
        4D array of shape (ntimepoints, nvoxels, nlags) to store the output lagged
        timecourses. This is updated in-place.
    LGR : logging.Logger, optional
        Logger instance for logging messages. If None, no logging is performed.
    nprocs : int, optional
        Number of processes to use for multiprocessing. Default is 1.
    alwaysmultiproc : bool, optional
        If True, always use multiprocessing even for single voxel processing.
        Default is False.
    showprogressbar : bool, optional
        If True, display a progress bar during processing. Default is True.
    chunksize : int, optional
        Size of chunks to process in each step when using multiprocessing.
        Default is 1000.
    rt_floattype : str, optional
        String representation of the floating-point type.
        Default is `np.float64`.
    debug : bool, optional
        If True, print debug information. Default is False.

    Returns
    -------
    int
        Total number of voxels processed.

    Notes
    -----
    This function uses `tide_genericmultiproc.run_multiproc` internally to
    distribute voxel processing across multiple processes. It is designed for
    efficient batch processing of large 4D datasets.

    Examples
    --------
    >>> import numpy as np
    >>> timeaxis = np.arange(100)
    >>> lagtimes = np.array([0, 1, 2])
    >>> lagmask = np.ones((10, 10, 10), dtype=bool)
    >>> lagtc = np.zeros((100, 1000, 3))
    >>> result = makelaggedtcs(
    ...     lagtcgenerator=my_generator,
    ...     timeaxis=timeaxis,
    ...     lagmask=lagmask,
    ...     lagtimes=lagtimes,
    ...     lagtc=lagtc,
    ...     nprocs=4
    ... )
    >>> print(f"Processed {result} voxels")
    """
    if debug:
        print("makelaggedtcs: Starting")
        print(f"\t{lagtc.shape=}")
        print(f"\t{lagtimes.shape=}")
        print(f"\t{timeaxis.shape=}")

    inputshape = lagtc.shape
    voxelargs = [
        lagtcgenerator,
        lagtimes,
        timeaxis,
    ]
    voxelfunc = _procOneVoxelMakelagtc
    packfunc = _packvoxeldata
    unpackfunc = _unpackvoxeldata
    voxeltargets = [lagtc]

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
        rt_floattype=rt_floattype,
    )
    if LGR is not None:
        LGR.info(f"\nLagged timecourses created for {volumetotal} voxels")

    # garbage collect
    uncollected = gc.collect()
    if uncollected != 0:
        if LGR is not None:
            LGR.info(f"garbage collected - unable to collect {uncollected} objects")
    else:
        if LGR is not None:
            LGR.info("garbage collected")

    if debug:
        print("makelaggedtcs: End\n\n")

    return volumetotal
