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

warnings.simplefilter(action="ignore", category=FutureWarning)
LGR = logging.getLogger("GENERAL")


def _procOneVoxelCoherence(
    vox: int,
    voxelargs: list,
    **kwargs: Any,
) -> tuple[int, NDArray, NDArray, float, float]:
    """
    Process coherence for a single voxel.

    This function computes coherence values for a given voxel using the provided
    coherence calculator and fMRI time course data. It returns the voxel index
    along with coherence values and the location of the maximum coherence.

    Parameters
    ----------
    vox : int
        The voxel index being processed.
    voxelargs : list
        A list containing two elements: the coherence calculator object and
        the fMRI time course data (fmritc).
    **kwargs : Any
        Additional keyword arguments that can override default options:
        - alt : bool, optional (default: False)
            Flag to indicate alternative computation mode.
        - debug : bool, optional (default: False)
            Flag to enable debug printing.

    Returns
    -------
    tuple[int, NDArray, NDArray, float, float]
        A tuple containing:
        - vox : int
            The input voxel index
        - thecoherence_x : NDArray
            X-axis coherence values
        - thecoherence_y : NDArray
            Y-axis coherence values
        - thecoherence_y[maxindex] : float
            Maximum coherence value
        - thecoherence_x[maxindex] : float
            X-coordinate corresponding to maximum coherence

    Notes
    -----
    The function uses the `theCoherer.run()` method to compute coherence values.
    When `alt=True`, the function returns additional dummy values from the
    coherence calculation. The maximum coherence is determined using `np.argmax()`.

    Examples
    --------
    >>> result = _procOneVoxelCoherence(10, [coherer_obj, fmri_data], alt=True)
    >>> voxel_idx, x_vals, y_vals, max_coherence, max_x = result
    """
    options = {
        "alt": False,
        "debug": False,
    }
    options.update(kwargs)
    alt = options["alt"]
    debug = options["debug"]
    (theCoherer, fmritc) = voxelargs
    if debug:
        print(f"{alt=}")
    if alt:
        (
            thecoherence_y,
            thecoherence_x,
            globalmaxindex,
            dummy,
            dummy,
            dummy,
        ) = theCoherer.run(fmritc, trim=True, alt=True)
    else:
        thecoherence_y, thecoherence_x, globalmaxindex = theCoherer.run(fmritc, trim=True)
    maxindex = np.argmax(thecoherence_y)
    return (
        vox,
        thecoherence_x,
        thecoherence_y,
        thecoherence_y[maxindex],
        thecoherence_x[maxindex],
    )


def _packvoxeldata(voxnum: int, voxelargs: list) -> list:
    """
    Pack voxel data for processing.

    Parameters
    ----------
    voxnum : int
        The voxel number to extract from the second element of voxelargs.
    voxelargs : list
        A list containing voxel arguments where:
        - voxelargs[0] is the first voxel argument (returned as-is)
        - voxelargs[1] is a 2D array from which row voxnum is extracted

    Returns
    -------
    list
        A list containing:
        - voxelargs[0] (unchanged)
        - The voxnum-th row of voxelargs[1] as a 1D array

    Notes
    -----
    This function is typically used in voxel-based data processing workflows
    where data needs to be extracted and reorganized for further analysis.

    Examples
    --------
    >>> voxelargs = [10, [[1, 2, 3], [4, 5, 6], [7, 8, 9]]]
    >>> _packvoxeldata(1, voxelargs)
    [10, [4, 5, 6]]
    """
    return [
        voxelargs[0],
        (voxelargs[1])[voxnum, :],
    ]


def _unpackvoxeldata(retvals: tuple, voxelproducts: list) -> None:
    """
    Unpack voxel data from retvals tuple into corresponding voxel product lists.

    This function takes a tuple of voxel data and distributes the values into
    three separate voxel product lists based on the index specified in the first
    element of the retvals tuple.

    Parameters
    ----------
    retvals : tuple
        A tuple containing voxel data where:
        - retvals[0] : int, index for insertion
        - retvals[1] : unused
        - retvals[2] : value to insert into voxelproducts[0]
        - retvals[3] : value to insert into voxelproducts[1]
        - retvals[4] : value to insert into voxelproducts[2]
    voxelproducts : list
        A list of three voxel product arrays/lists where:
        - voxelproducts[0] : first voxel product array
        - voxelproducts[1] : second voxel product array
        - voxelproducts[2] : third voxel product array

    Returns
    -------
    None
        This function modifies the voxelproducts lists in-place and does not return anything.

    Notes
    -----
    The function assumes that retvals contains exactly 5 elements and that
    voxelproducts contains exactly 3 elements. The first element of retvals
    is used as an index to determine the position where values should be inserted
    into each of the three voxel product arrays.

    Examples
    --------
    >>> voxel1 = [0, 0, 0]
    >>> voxel2 = [0, 0, 0]
    >>> voxel3 = [0, 0, 0]
    >>> retvals = (1, None, 10, 20, 30)
    >>> voxelproducts = [voxel1, voxel2, voxel3]
    >>> _unpackvoxeldata(retvals, voxelproducts)
    >>> print(voxel1[1])
    10
    >>> print(voxel2[1])
    20
    >>> print(voxel3[1])
    30
    """
    (voxelproducts[0])[retvals[0]] = retvals[2]
    (voxelproducts[1])[retvals[0]] = retvals[3]
    (voxelproducts[2])[retvals[0]] = retvals[4]


def coherencepass(
    fmridata: NDArray,
    theCoherer: Any,
    coherencefunc: NDArray,
    coherencepeakval: NDArray,
    coherencepeakfreq: NDArray,
    alt: bool = False,
    chunksize: int = 1000,
    nprocs: int = 1,
    alwaysmultiproc: bool = False,
    showprogressbar: bool = True,
    debug: bool = False,
) -> int:
    """
    Perform coherence analysis on fMRI data across voxels using multiprocessing.

    This function applies coherence analysis to each voxel in the input fMRI data,
    storing results in the provided output arrays. It supports parallel processing
    for improved performance and includes optional debugging and progress tracking.

    Parameters
    ----------
    fmridata : NDArray
        Input fMRI data array with shape (time, voxels).
    theCoherer : Any
        Object or function used to perform coherence calculations.
    coherencefunc : NDArray
        Array to store coherence function results for each voxel.
    coherencepeakval : NDArray
        Array to store peak coherence values for each voxel.
    coherencepeakfreq : NDArray
        Array to store peak coherence frequencies for each voxel.
    alt : bool, optional
        If True, use alternative coherence calculation method. Default is False.
    chunksize : int, optional
        Number of voxels to process in each chunk during multiprocessing.
        Default is 1000.
    nprocs : int, optional
        Number of processes to use for multiprocessing. Default is 1.
    alwaysmultiproc : bool, optional
        If True, always use multiprocessing even for small datasets.
        Default is False.
    showprogressbar : bool, optional
        If True, display a progress bar during processing. Default is True.
    debug : bool, optional
        If True, enable debug logging. Default is False.

    Returns
    -------
    int
        Total number of voxels processed.

    Notes
    -----
    This function uses `tide_genericmultiproc.run_multiproc` to distribute
    voxel-wise coherence computations across multiple processes. The results
    are stored directly into the provided output arrays (`coherencefunc`,
    `coherencepeakval`, `coherencepeakfreq`).

    Examples
    --------
    >>> import numpy as np
    >>> fmri_data = np.random.rand(100, 50)
    >>> coherer = SomeCohererClass()
    >>> coherence_func = np.zeros((100, 50))
    >>> peak_val = np.zeros((1, 50))
    >>> peak_freq = np.zeros((1, 50))
    >>> n_voxels = coherencepass(
    ...     fmri_data, coherer, coherence_func, peak_val, peak_freq
    ... )
    >>> print(f"Processed {n_voxels} voxels")
    """
    inputshape = np.shape(fmridata)
    voxelargs = [theCoherer, fmridata]
    voxelfunc = _procOneVoxelCoherence
    packfunc = _packvoxeldata
    unpackfunc = _unpackvoxeldata
    voxeltargets = [coherencefunc, coherencepeakval, coherencepeakfreq]
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
        alt=alt,
        debug=debug,
    )
    LGR.info(f"\nCoherence performed on {volumetotal} voxels")

    # garbage collect
    uncollected = gc.collect()
    if uncollected != 0:
        LGR.info(f"garbage collected - unable to collect {uncollected} objects")
    else:
        LGR.info("garbage collected")

    return volumetotal
