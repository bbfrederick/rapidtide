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
import logging
import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray

import rapidtide.filter as tide_filt
import rapidtide.genericmultiproc as tide_genericmultiproc
import rapidtide.miscmath as tide_math


# note: rawtimecourse has been filtered, but NOT windowed
def _procOneNullCorrelationx(
    vox: int,
    voxelargs: list,
    **kwargs: Any,
) -> tuple[int, float]:
    """
    Process a single voxel to compute the maximum correlation value from a null correlation test.

    This function performs a permutation-based null correlation test for a given voxel. It shuffles
    the reference time course according to the specified method and computes the cross-correlation
    with the original time course. The maximum correlation value is returned along with the voxel index.

    Parameters
    ----------
    vox : int
        The voxel index to process.
    voxelargs : list
        A list containing the following elements in order:
        - `normalizedreftc`: Normalized reference time course.
        - `rawtcfft_r`: Raw FFT magnitude of the reference time course.
        - `rawtcfft_ang`: Raw FFT phase of the reference time course.
        - `theCorrelator`: Correlator object used for cross-correlation.
        - `thefitter`: Fitter object used for fitting the correlation peak.
    **kwargs : Any
        Additional keyword arguments that can override default options:
        - permutationmethod : str, optional
            The method used for shuffling the reference time course.
            Options are 'shuffle' (default) or 'phaserandom'.
        - debug : bool, optional
            If True, prints debug information including the permutation method used.

    Returns
    -------
    tuple[int, float]
        A tuple containing:
        - vox : int
            The voxel index passed as input.
        - maxval : float
            The maximum correlation value obtained from the fitted correlation.

    Notes
    -----
    This function supports two permutation methods:
    - 'shuffle': Randomly shuffles the reference time course.
    - 'phaserandom': Shuffles the phase of the FFT of the reference time course while preserving
      the magnitude.

    Examples
    --------
    >>> result = _procOneNullCorrelationx(
    ...     vox=10,
    ...     voxelargs=[ref_tc, fft_r, fft_ang, correlator, fitter],
    ...     permutationmethod='shuffle',
    ...     debug=True
    ... )
    >>> print(result)
    (10, 0.85)
    """

    options = {
        "permutationmethod": "shuffle",
        "debug": False,
    }
    options.update(kwargs)
    permutationmethod = options["permutationmethod"]
    debug = options["debug"]
    if debug:
        print(f"{permutationmethod=}")
    (
        normalizedreftc,
        rawtcfft_r,
        rawtcfft_ang,
        theCorrelator,
        thefitter,
    ) = voxelargs

    # make a shuffled copy of the regressors
    if permutationmethod == "shuffle":
        permutedtc = np.random.permutation(normalizedreftc)
        # apply the appropriate filter
        # permutedtc = theCorrelator.ncprefilter.apply(Fs, permutedtc)
    elif permutationmethod == "phaserandom":
        permutedtc = tide_filt.ifftfrompolar(rawtcfft_r, np.random.permutation(rawtcfft_ang))
    else:
        print("illegal shuffling method")
        sys.exit()

    # crosscorrelate with original
    thexcorr_y, thexcorr_x, dummy = theCorrelator.run(permutedtc)

    # fit the correlation
    thefitter.setcorrtimeaxis(thexcorr_x)
    (
        maxindex,
        maxlag,
        maxval,
        maxsigma,
        maskval,
        failreason,
        peakstart,
        peakend,
    ) = thefitter.fit(thexcorr_y)

    return vox, maxval


def _packvoxeldata(voxnum: int, voxelargs: list) -> list:
    """
    Pack voxel data into a list format.

    Parameters
    ----------
    voxnum : int
        The voxel number identifier.
    voxelargs : list
        List containing voxel arguments to be packed. Expected to contain at least 5 elements.

    Returns
    -------
    list
        A list containing the first 5 elements from voxelargs in order:
        [voxelargs[0], voxelargs[1], voxelargs[2], voxelargs[3], voxelargs[4]]

    Notes
    -----
    This function currently returns a fixed subset of the input list. For proper functionality,
    the voxnum parameter is not utilized in the current implementation.

    Examples
    --------
    >>> _packvoxeldata(1, [10, 20, 30, 40, 50, 60])
    [10, 20, 30, 40, 50]
    """
    return [voxelargs[0], voxelargs[1], voxelargs[2], voxelargs[3], voxelargs[4]]


def _unpackvoxeldata(retvals: tuple, voxelproducts: list) -> None:
    """
    Unpack voxel data by assigning values to specified indices.

    This function takes return values and assigns them to a specific location
    within a voxel product structure based on the provided indices.

    Parameters
    ----------
    retvals : tuple
        A tuple containing two elements: the first element is the index
        used to access the voxel product, and the second element is the
        value to be assigned.
    voxelproducts : list
        A list of voxel product structures where the assignment will occur.
        The function modifies the first element of this list in-place.

    Returns
    -------
    None
        This function modifies the voxelproducts list in-place and does not
        return any value.

    Notes
    -----
    The function assumes that voxelproducts[0] is a mutable structure (like
    a list or array) that supports item assignment. The first element of
    retvals is used as an index to access voxelproducts[0], and the second
    element of retvals is assigned to that location.

    Examples
    --------
    >>> voxel_data = [[0, 0, 0]]
    >>> _unpackvoxeldata((1, 42), voxel_data)
    >>> voxel_data
    [[0, 42, 0]]
    """
    (voxelproducts[0])[retvals[0]] = retvals[1]


def getNullDistributionData(
    Fs: float,
    theCorrelator: Any,
    thefitter: Any,
    LGR: logging.Logger,
    numestreps: int = 0,
    nprocs: int = 1,
    alwaysmultiproc: bool = False,
    showprogressbar: bool = True,
    chunksize: int = 1000,
    permutationmethod: str = "shuffle",
    rt_floattype: np.dtype = np.float64,
    debug: bool = False,
) -> NDArray:
    """
    Calculate a set of null correlations to determine the distribution of correlation values.

    This function generates a distribution of correlation values by performing permutations
    on the reference time course. The resulting distribution can be used to identify
    spurious correlation thresholds.

    Parameters
    ----------
    Fs : float
        The sample frequency of the raw time course, in Hz.
    theCorrelator : Any
        An object containing the reference time course and related filtering parameters.
    thefitter : Any
        An object used for fitting the correlation data.
    LGR : logging.Logger
        Logger instance for logging messages during execution.
    numestreps : int, optional
        Number of null correlation estimates to compute. Default is 0.
    nprocs : int, optional
        Number of processes to use for multiprocessing. Default is 1.
    alwaysmultiproc : bool, optional
        If True, always use multiprocessing even for small datasets. Default is False.
    showprogressbar : bool, optional
        If True, display a progress bar during computation. Default is True.
    chunksize : int, optional
        Size of chunks for multiprocessing. Default is 1000.
    permutationmethod : str, optional
        Permutation method to use ('shuffle' or other supported methods). Default is 'shuffle'.
    rt_floattype : str, optional
        String representation of the floating-point type. Default is np.float64.
    debug : bool, optional
        If True, enable debug output. Default is False.

    Returns
    -------
    NDArray
        Array of correlation values representing the null distribution.

    Notes
    -----
    This function applies normalization and filtering to the reference time course before
    computing correlations. It supports parallel processing via multiprocessing for
    improved performance when `numestreps` is large.

    Examples
    --------
    >>> import numpy as np
    >>> from some_module import getNullDistributionData
    >>> result = getNullDistributionData(
    ...     Fs=100.0,
    ...     theCorrelator=correlator_obj,
    ...     thefitter=fitter_obj,
    ...     LGR=logging.getLogger(__name__),
    ...     numestreps=1000,
    ...     nprocs=4
    ... )
    >>> print(f"Null correlation distribution shape: {result.shape}")
    """
    inputshape = np.asarray([numestreps])
    normalizedreftc = theCorrelator.ncprefilter.apply(
        Fs,
        tide_math.corrnormalize(
            theCorrelator.reftc,
            windowfunc="None",
            detrendorder=theCorrelator.detrendorder,
        ),
    )
    rawtcfft_r, rawtcfft_ang = tide_filt.polarfft(normalizedreftc)
    corrlist = np.zeros((numestreps), dtype=rt_floattype)
    voxelmask = np.ones((numestreps), dtype=rt_floattype)
    voxelargs = [normalizedreftc, rawtcfft_r, rawtcfft_ang, theCorrelator, thefitter]
    voxelfunc = _procOneNullCorrelationx
    packfunc = _packvoxeldata
    unpackfunc = _unpackvoxeldata
    voxeltargets = [
        corrlist,
    ]

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
        permutationmethod=permutationmethod,
        debug=debug,
    )

    # return the distribution data
    numnonzero = len(np.where(corrlist != 0.0)[0])
    print(
        "{:d} non-zero correlations out of {:d} ({:.2f}%)".format(
            numnonzero, len(corrlist), 100.0 * numnonzero / len(corrlist)
        )
    )
    return corrlist
