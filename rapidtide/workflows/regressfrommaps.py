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

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

import rapidtide.io as tide_io
import rapidtide.linfitfiltpass as tide_linfitfiltpass
import rapidtide.makelaggedtcs as tide_makelagged


def regressfrommaps(
    fmri_data_valid: Any,
    validvoxels: Any,
    initial_fmri_x: Any,
    lagtimes: Any,
    fitmask: Any,
    genlagtc: Any,
    mode: Any,
    outputname: Any,
    oversamptr: Any,
    sLFOfitmean: Any,
    rvalue: Any,
    r2value: Any,
    fitNorm: Any,
    fitcoeff: Any,
    movingsignal: Any,
    lagtc: Any,
    filtereddata: Any,
    LGR: Any,
    TimingLGR: Any,
    regressfiltthreshval: Any,
    saveminimumsLFOfiltfiles: Any,
    nprocs_makelaggedtcs: int = 1,
    nprocs_regressionfilt: int = 1,
    regressderivs: int = 0,
    chunksize: int = 50000,
    showprogressbar: bool = True,
    alwaysmultiproc: bool = False,
    saveEVsandquit: bool = False,
    coefficientsonly: bool = False,
    timemask: Optional[NDArray] = None,
    debug: bool = False,
) -> Tuple[int, NDArray, NDArray]:
    """
    Perform regression analysis on fMRI data using lagged timecourses.

    This function generates voxel-specific regressors from lagged timecourses,
    applies filtering, and performs regression to estimate model coefficients.
    It supports various modes including cross-validation regression (cvrmap),
    and can optionally save intermediate results or quit early.

    Parameters
    ----------
    fmri_data_valid : array-like
        Valid fMRI data to be processed.
    validvoxels : array-like
        Indices or mask of valid voxels.
    initial_fmri_x : array-like
        Initial fMRI timecourse (e.g., stimulus timing).
    lagtimes : array-like
        Time lags to be used for generating lagged regressors.
    fitmask : array-like
        Mask for selecting voxels to fit.
    genlagtc : object
        Generator for lagged timecourses.
    mode : str
        Processing mode (e.g., 'cvrmap').
    outputname : str
        Base name for output files.
    oversamptr : float
        Oversampling rate for timecourse generation.
    sLFOfitmean : array-like
        Mean of sLFO fit values.
    rvalue : array-like
        R-values from regression.
    r2value : array-like
        R-squared values from regression.
    fitNorm : array-like
        Normalization values for fit.
    fitcoeff : array-like
        Coefficients from the fit.
    movingsignal : array-like
        Moving signal components.
    lagtc : array-like
        Lagged timecourses.
    filtereddata : array-like
        Filtered fMRI data.
    LGR : object, optional
        Logger for general logging.
    TimingLGR : object, optional
        Logger for timing information.
    regressfiltthreshval : float
        Threshold for regression filtering.
    saveminimumsLFOfiltfiles : bool
        Whether to save noise removed timeseries.
    nprocs_makelaggedtcs : int, optional
        Number of processes for making lagged timecourses (default is 1).
    nprocs_regressionfilt : int, optional
        Number of processes for regression filtering (default is 1).
    regressderivs : int, optional
        Order of derivatives to include in regressors (default is 0).
    chunksize : int, optional
        Size of chunks for processing (default is 50000).
    showprogressbar : bool, optional
        Whether to show progress bar (default is True).
    alwaysmultiproc : bool, optional
        Force multiprocessing even for small tasks (default is False).
    saveEVsandquit : bool, optional
        Save EVs and quit early (default is False).
    coefficientsonly : bool, optional
        Return only coefficients (default is False).
    timemask : NDArray, optional
       Mask of timepoints to include in regression filtering.
    debug : bool, optional
        Enable debug output (default is False).

    Returns
    -------
    tuple
        If `saveEVsandquit` is True, returns (0, regressorset, evset).
        Otherwise, returns (voxelsprocessed_regressionfilt, regressorset, evset).

    Notes
    -----
    - The function modifies `fitcoeff` in-place when `mode == "cvrmap"` by multiplying by 100.
    - The function uses multiprocessing if `alwaysmultiproc` is True or if `nprocs > 1`.
    - Filtering is performed using `tide_linfitfiltpass.linfitfiltpass`.

    Examples
    --------
    >>> regressfrommaps(
    ...     fmri_data_valid=data,
    ...     validvoxels=voxels,
    ...     initial_fmri_x=stimulus,
    ...     lagtimes=lags,
    ...     fitmask=mask,
    ...     genlagtc=gen,
    ...     mode="cvrmap",
    ...     outputname="output",
    ...     oversamptr=2.0,
    ...     sLFOfitmean=mean,
    ...     rvalue=r_vals,
    ...     r2value=r2_vals,
    ...     fitNorm=norm,
    ...     fitcoeff=coeffs,
    ...     movingsignal=moving,
    ...     lagtc=lagged_tc,
    ...     filtereddata=filtered,
    ...     LGR=logger,
    ...     TimingLGR=timing_logger,
    ...     regressfiltthreshval=0.5,
    ...     saveminimumsLFOfiltfiles=True,
    ...     nprocs_makelaggedtcs=4,
    ...     nprocs_regressionfilt=2,
    ...     regressderivs=2,
    ...     chunksize=10000,
    ...     showprogressbar=True,
    ...     alwaysmultiproc=False,
    ...     saveEVsandquit=False,
    ...     coefficientsonly=False,
    ...     debug=False,
    ... )
    """
    if debug:
        print("regressfrommaps: Starting")
        print(f"\t{nprocs_makelaggedtcs=}")
        print(f"\t{nprocs_regressionfilt=}")
        print(f"\t{regressderivs=}")
        print(f"\t{chunksize=}")
        print(f"\t{showprogressbar=}")
        print(f"\t{alwaysmultiproc=}")
        print(f"\t{mode=}")
        print(f"\t{outputname=}")
        print(f"\t{oversamptr=}")
        print(f"\t{regressfiltthreshval=}")
    rt_floattype = np.float64
    numvalidspatiallocs = np.shape(validvoxels)[0]

    # generate the voxel specific regressors
    if LGR is not None:
        LGR.info("Start lagged timecourse creation")
    if TimingLGR is not None:
        TimingLGR.info("Start lagged timecourse creation")

    voxelsprocessed_makelagged = tide_makelagged.makelaggedtcs(
        genlagtc,
        initial_fmri_x,
        fitmask,
        lagtimes,
        lagtc,
        LGR=LGR,
        nprocs=nprocs_makelaggedtcs,
        alwaysmultiproc=alwaysmultiproc,
        showprogressbar=showprogressbar,
        chunksize=chunksize,
        rt_floattype=rt_floattype,
        debug=debug,
    )
    if timemask is not None:
        lagtc = lagtc * timemask[None, :]

    if debug:
        print(f"\t{lagtimes.shape=}")
        threshmask = np.where(fitmask > 0, 1, 0)
        print(f"\t{np.sum(threshmask)} nonzero mask voxels")
        print(f"\tafter makelaggedtcs: shifted {voxelsprocessed_makelagged} timecourses")
    if LGR is not None:
        LGR.info("End lagged timecourse creation")
    if TimingLGR is not None:
        TimingLGR.info(
            "Lagged timecourse creation end",
            {
                "message2": voxelsprocessed_makelagged,
                "message3": "voxels",
            },
        )

    if regressderivs > 0:
        if debug:
            print(f"adding derivatives up to order {regressderivs} prior to regression")
        regressorset = tide_linfitfiltpass.makevoxelspecificderivs(lagtc, regressderivs)
        baseev = (genlagtc.yfromx(initial_fmri_x)).astype(rt_floattype)
        evset = tide_linfitfiltpass.makevoxelspecificderivs(
            baseev.reshape((1, -1)), regressderivs
        ).reshape((-1, 2))
    else:
        if debug:
            print(f"using raw lagged regressors for regression")
        regressorset = lagtc
        evset = (genlagtc.yfromx(initial_fmri_x)).astype(rt_floattype)
    if debug:
        print(f"{regressorset.shape=}")

    if saveEVsandquit:
        return 0, regressorset, evset

    # now do the filtering
    if LGR is not None:
        LGR.info("Start filtering operation")
    if TimingLGR is not None:
        TimingLGR.info("Start filtering operation")

    voxelsprocessed_regressionfilt = tide_linfitfiltpass.linfitfiltpass(
        numvalidspatiallocs,
        fmri_data_valid,
        regressfiltthreshval,
        regressorset,
        sLFOfitmean,
        rvalue,
        r2value,
        fitcoeff,
        fitNorm,
        movingsignal,
        filtereddata,
        coefficientsonly=coefficientsonly,
        nprocs=nprocs_regressionfilt,
        alwaysmultiproc=alwaysmultiproc,
        showprogressbar=showprogressbar,
        verbose=(LGR is not None),
        chunksize=chunksize,
        rt_floattype=rt_floattype,
        debug=debug,
    )

    if mode == "cvrmap":
        # if we are doing a cvr map, multiply the fitcoeff by 100, so we are in percent
        fitcoeff *= 100.0

    # determine what was removed
    if not coefficientsonly:
        removeddata = fmri_data_valid - filtereddata
        noiseremoved = np.var(removeddata, axis=0)
        if saveminimumsLFOfiltfiles:
            tide_io.writebidstsv(
                f"{outputname}_desc-lfofilterNoiseRemoved_timeseries",
                noiseremoved,
                1.0 / oversamptr,
                starttime=0.0,
                columns=[f"removedbyglm"],
                extraheaderinfo={
                    "Description": "Variance over space of data removed by the sLFO filter at each timepoint"
                },
                append=False,
            )

    if debug:
        print("regressfrommaps: End\n\n")

    return voxelsprocessed_regressionfilt, regressorset, evset
