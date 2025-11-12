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

import rapidtide.calcsimfunc as tide_calcsimfunc
import rapidtide.io as tide_io
import rapidtide.linfitfiltpass as tide_linfitfiltpass
import rapidtide.makelaggedtcs as tide_makelagged
import rapidtide.stats as tide_stats
import rapidtide.util as tide_util


def makeRIPTiDeRegressors(
    initial_fmri_x: Any,
    lagmin: Any,
    lagmax: Any,
    lagtcgenerator: Any,
    LGR: Any,
    nprocs: int = 1,
    alwaysmultiproc: bool = False,
    showprogressbar: bool = True,
    chunksize: int = 1000,
    targetstep: float = 2.5,
    edgepad: int = 0,
    rt_floattype: np.dtype = np.float64,
    debug: bool = False,
) -> Tuple[NDArray, NDArray]:
    """
    Generate regressors for RIPTiDe (Regressors for Inverse Temporal Deconvolution).

    This function creates a set of lagged temporal regressors based on the provided
    parameters, which are used in the context of fMRI data analysis for temporal
    deconvolution. It leverages the `tide_makelagged.makelaggedtcs` function to
    compute the actual regressor matrix.

    Parameters
    ----------
    initial_fmri_x : array_like
        The initial fMRI time series data used to generate the regressors.
    lagmin : float
        The minimum lag (in seconds) to consider for regressor generation.
    lagmax : float
        The maximum lag (in seconds) to consider for regressor generation.
    lagtcgenerator : callable
        A function or callable object that generates lagged time courses.
    LGR : object
        An object containing parameters for the lagged time course generation.
    nprocs : int, optional
        Number of processes to use for parallel computation (default is 1).
    alwaysmultiproc : bool, optional
        If True, always use multiprocessing even for small datasets (default is False).
    showprogressbar : bool, optional
        If True, display a progress bar during computation (default is True).
    chunksize : int, optional
        Size of chunks for processing in multiprocessing (default is 1000).
    targetstep : float, optional
        Target step size (in seconds) between lags (default is 2.5).
    edgepad : int, optional
        Number of padding steps at the beginning and end of the lag range (default is 0).
    rt_floattype : np.dtype, optional
        Data type for the regressor matrix (default is np.float64).
    debug : bool, optional
        If True, print debug information during execution (default is False).

    Returns
    -------
    tuple of (NDArray, NDArray)
        A tuple containing:
        - regressorset : NDArray
            The computed regressor matrix of shape (num_lags, num_timepoints).
        - delaystouse : NDArray
            The array of delay values used for regressor generation.

    Notes
    -----
    This function is intended for use in fMRI data analysis workflows where
    temporal deconvolution is required. The regressors generated can be used
    in subsequent steps such as GLM fitting or temporal filtering.

    Examples
    --------
    >>> import numpy as np
    >>> fmri_data = np.random.rand(100, 50)
    >>> regressors, delays = makeRIPTiDeRegressors(
    ...     initial_fmri_x=fmri_data,
    ...     lagmin=0.0,
    ...     lagmax=10.0,
    ...     lagtcgenerator=my_lag_generator,
    ...     LGR=my_lgr_object,
    ...     nprocs=4,
    ...     debug=True
    ... )
    """
    # make the RIPTiDe evs
    numdelays = int(np.round((lagmax - lagmin) / targetstep, 0))
    numregressors = numdelays + 2 * edgepad
    delaystep = (lagmax - lagmin) / numdelays
    delaystouse = np.linspace(
        lagmin - edgepad * delaystep,
        lagmax + edgepad * delaystep,
        numdelays + 2 * edgepad,
        endpoint=True,
    )
    if debug:
        print(f"{lagmin=}")
        print(f"{lagmax=}")
        print(f"{numdelays=}")
        print(f"{edgepad=}")
        print(f"{numregressors=}")
        print(f"{delaystep=}")
        print(f"{delaystouse=}, {len(delaystouse)}")
        print(f"{len(initial_fmri_x)}")

    regressorset = np.zeros((len(delaystouse), len(initial_fmri_x)), dtype=rt_floattype)

    dummy = tide_makelagged.makelaggedtcs(
        lagtcgenerator,
        initial_fmri_x,
        np.ones_like(delaystouse, dtype=np.float64),
        delaystouse,
        regressorset,
        LGR=LGR,
        nprocs=nprocs,
        alwaysmultiproc=alwaysmultiproc,
        showprogressbar=showprogressbar,
        chunksize=chunksize,
        rt_floattype=rt_floattype,
        debug=debug,
    )

    if debug:
        print(regressorset)

    return regressorset, delaystouse


def calcSimFunc(
    numvalidspatiallocs: Any,
    fmri_data_valid: Any,
    validsimcalcstart: Any,
    validsimcalcend: Any,
    osvalidsimcalcstart: Any,
    osvalidsimcalcend: Any,
    initial_fmri_x: Any,
    os_fmri_x: Any,
    theCorrelator: Any,
    theMutualInformationator: Any,
    cleaned_referencetc: Any,
    corrout: Any,
    regressorset: Any,
    delayvals: Any,
    sLFOfitmean: Any,
    r2value: Any,
    fitcoeff: Any,
    fitNorm: Any,
    meanval: Any,
    corrscale: Any,
    outputname: Any,
    outcorrarray: Any,
    validvoxels: Any,
    nativecorrshape: Any,
    theinputdata: Any,
    theheader: Any,
    lagmininpts: Any,
    lagmaxinpts: Any,
    thepass: Any,
    optiondict: Any,
    LGR: Any,
    TimingLGR: Any,
    similaritymetric: str = "correlation",
    simcalcoffset: int = 0,
    echocancel: bool = False,
    checkpoint: bool = False,
    nprocs: int = 1,
    alwaysmultiproc: bool = False,
    oversampfactor: int = 2,
    interptype: str = "univariate",
    showprogressbar: bool = True,
    chunksize: int = 1000,
    rt_floattype: np.dtype = np.float64,
    mklthreads: int = 1,
    threaddebug: bool = False,
    debug: bool = False,
) -> str:
    """
    Compute similarity metrics (correlation, mutual information, or RIPtiDe) between fMRI data and a reference time series.

    This function performs similarity calculations across voxels using either correlation, mutual information,
    or a hybrid method, depending on the specified `similaritymetric`. It supports multi-processing and can
    optionally save intermediate results.

    Parameters
    ----------
    numvalidspatiallocs : Any
        Number of valid spatial locations in the fMRI data.
    fmri_data_valid : Any
        Valid fMRI data array, typically of shape (n_voxels, n_timepoints).
    validsimcalcstart : Any
        Start index for valid timepoints to use in similarity calculation.
    validsimcalcend : Any
        End index for valid timepoints to use in similarity calculation.
    osvalidsimcalcstart : Any
        Start index for oversampled valid timepoints.
    osvalidsimcalcend : Any
        End index for oversampled valid timepoints.
    initial_fmri_x : Any
        Initial fMRI timepoints (e.g., for correlation).
    os_fmri_x : Any
        Oversampled fMRI timepoints.
    theCorrelator : Any
        Correlator object used for computing correlations.
    theMutualInformationator : Any
        Mutual information calculator object.
    cleaned_referencetc : Any
        Cleaned reference time series.
    corrout : Any
        Output array for storing correlation results.
    regressorset : Any
        Set of regressors for fitting.
    delayvals : Any
        Array of delay values for RIPtiDe calculation.
    sLFOfitmean : Any
        Mean value for fitting.
    r2value : Any
        R² values for model fit.
    fitcoeff : Any
        Fitting coefficients.
    fitNorm : Any
        Normalization values for fitting.
    meanval : Any
        Mean value used in normalization.
    corrscale : Any
        Correlation scale for lag calculation.
    outputname : Any
        Base name for output files.
    outcorrarray : Any
        Array to store correlation output for checkpointing.
    validvoxels : Any
        Indices of valid voxels.
    nativecorrshape : Any
        Shape of the native correlation array.
    theinputdata : Any
        Input data object.
    theheader : Any
        Header information for NIfTI output.
    lagmininpts : Any
        Minimum lag in timepoints.
    lagmaxinpts : Any
        Maximum lag in timepoints.
    thepass : Any
        Pass number for tracking multiple iterations.
    optiondict : Any
        Dictionary of options for saving results.
    LGR : Any
        Logger for general messages.
    TimingLGR : Any
        Logger for timing information.
    similaritymetric : str, optional
        Type of similarity metric to compute. Options are:
        'correlation', 'mutualinfo', 'riptide', or 'hybrid'. Default is 'correlation'.
    simcalcoffset : int, optional
        Offset to subtract from computed lags. Default is 0.
    echocancel : bool, optional
        Whether to cancel echo effects. Default is False.
    checkpoint : bool, optional
        Whether to save intermediate results. Default is False.
    nprocs : int, optional
        Number of processes for multiprocessing. Default is 1.
    alwaysmultiproc : bool, optional
        Force multiprocessing even for single-core cases. Default is False.
    oversampfactor : int, optional
        Oversampling factor for interpolation. Default is 2.
    interptype : str, optional
        Interpolation type. Default is 'univariate'.
    showprogressbar : bool, optional
        Whether to show a progress bar. Default is True.
    chunksize : int, optional
        Size of chunks for processing. Default is 1000.
    rt_floattype : np.dtype, optional
        Rapidtide floating-point data type. Default is np.float64.
    mklthreads : int, optional
        Number of threads for Intel MKL. Default is 1.
    threaddebug : bool, optional
        Enable thread debugging. Default is False.
    debug : bool, optional
        Enable debug mode. Default is False.

    Returns
    -------
    str
        The type of similarity metric used in the calculation.

    Notes
    -----
    - For 'riptide', the function fits linear models to delayed regressors.
    - The function logs timing and processing information using `TimingLGR` and `LGR`.
    - If `checkpoint` is True, intermediate correlation results are saved to disk.
    - This function modifies `corrout` and `outcorrarray` in-place.

    Examples
    --------
    >>> calcSimFunc(
    ...     numvalidspatiallocs=100,
    ...     fmri_data_valid=np.random.rand(100, 100),
    ...     validsimcalcstart=0,
    ...     validsimcalcend=99,
    ...     osvalidsimcalcstart=0,
    ...     osvalidsimcalcend=99,
    ...     initial_fmri_x=np.linspace(0, 1, 100),
    ...     os_fmri_x=np.linspace(0, 1, 100),
    ...     theCorrelator=correlator_obj,
    ...     theMutualInformationator=mi_obj,
    ...     cleaned_referencetc=np.random.rand(100),
    ...     corrout=np.zeros((100, 100)),
    ...     regressorset=np.random.rand(10, 100),
    ...     delayvals=np.array([0, 1, 2]),
    ...     sLFOfitmean=np.mean(np.random.rand(100)),
    ...     r2value=np.zeros(100),
    ...     fitcoeff=np.zeros((100, 1)),
    ...     fitNorm=np.ones(100),
    ...     meanval=np.mean(np.random.rand(100)),
    ...     corrscale=np.arange(100),
    ...     outputname="test_output",
    ...     outcorrarray=np.zeros((100, 100)),
    ...     validvoxels=np.arange(100),
    ...     nativecorrshape=(100, 100),
    ...     theinputdata=input_data_obj,
    ...     theheader=header,
    ...     lagmininpts=-5,
    ...     lagmaxinpts=5,
    ...     thepass=1,
    ...     optiondict={},
    ...     LGR=logging.getLogger(),
    ...     TimingLGR=logging.getLogger(),
    ...     similaritymetric="correlation",
    ...     nprocs=2,
    ...     checkpoint=True,
    ... )
    'Correlation'
    """
    # Step 1 - Correlation step
    if similaritymetric == "mutualinfo":
        similaritytype = "Mutual information"
    elif similaritymetric == "correlation":
        similaritytype = "Correlation"
    elif similaritymetric == "riptide":
        similaritytype = "RIPTiDe"
    else:
        similaritytype = "MI enhanced correlation"
    LGR.info(f"\n\n{similaritytype} calculation, pass {thepass}")
    TimingLGR.info(f"{similaritytype} calculation start, pass {thepass}")

    tide_util.disablemkl(nprocs, debug=threaddebug)
    if similaritymetric == "mutualinfo":
        theMutualInformationator.setlimits(lagmininpts, lagmaxinpts)
        (
            voxelsprocessed_cp,
            theglobalmaxlist,
            trimmedcorrscale,
        ) = tide_calcsimfunc.correlationpass(
            fmri_data_valid[:, validsimcalcstart : validsimcalcend + 1],
            cleaned_referencetc,
            theMutualInformationator,
            initial_fmri_x[validsimcalcstart : validsimcalcend + 1],
            os_fmri_x[osvalidsimcalcstart : osvalidsimcalcend + 1],
            lagmininpts,
            lagmaxinpts,
            corrout,
            meanval,
            nprocs=nprocs,
            alwaysmultiproc=alwaysmultiproc,
            oversampfactor=oversampfactor,
            interptype=interptype,
            showprogressbar=showprogressbar,
            chunksize=chunksize,
            rt_floattype=rt_floattype,
            debug=debug,
        )
    elif (similaritymetric == "correlation") or (similaritymetric == "hybrid"):
        (
            voxelsprocessed_cp,
            theglobalmaxlist,
            trimmedcorrscale,
        ) = tide_calcsimfunc.correlationpass(
            fmri_data_valid[:, validsimcalcstart : validsimcalcend + 1],
            cleaned_referencetc,
            theCorrelator,
            initial_fmri_x[validsimcalcstart : validsimcalcend + 1],
            os_fmri_x[osvalidsimcalcstart : osvalidsimcalcend + 1],
            lagmininpts,
            lagmaxinpts,
            corrout,
            meanval,
            nprocs=nprocs,
            alwaysmultiproc=alwaysmultiproc,
            oversampfactor=oversampfactor,
            interptype=interptype,
            showprogressbar=showprogressbar,
            chunksize=chunksize,
            rt_floattype=rt_floattype,
            debug=debug,
        )
    elif similaritymetric == "riptide":
        # do the linear fit to the comb of delayed regressors
        for thedelay in range(len(delayvals)):
            print(f"Fitting delay {delayvals[thedelay]:.2f}")
            voxelsprocessed_cp = tide_linfitfiltpass.linfitfiltpass(
                numvalidspatiallocs,
                fmri_data_valid[:, validsimcalcstart : validsimcalcend + 1],
                0.0,
                regressorset[thedelay, validsimcalcstart : validsimcalcend + 1],
                sLFOfitmean,
                corrout[:, thedelay],
                r2value,
                fitcoeff,
                fitNorm,
                None,
                None,
                coefficientsonly=True,
                constantevs=True,
                nprocs=nprocs,
                alwaysmultiproc=alwaysmultiproc,
                showprogressbar=showprogressbar,
                verbose=(LGR is not None),
                chunksize=chunksize,
                rt_floattype=rt_floattype,
                debug=debug,
            )
    else:
        print("illegal similarity metric")

    tide_util.enablemkl(mklthreads, debug=threaddebug)

    if similaritymetric != "riptide":
        for i in range(len(theglobalmaxlist)):
            theglobalmaxlist[i] = corrscale[theglobalmaxlist[i]] - simcalcoffset
        namesuffix = "_desc-globallag_hist"
        tide_stats.makeandsavehistogram(
            np.asarray(theglobalmaxlist),
            len(corrscale),
            0,
            outputname + namesuffix,
            displaytitle="Histogram of lag times from global lag calculation",
            therange=(corrscale[0], corrscale[-1]),
            refine=False,
            dictvarname="globallaghist_pass" + str(thepass),
            append=(echocancel or (thepass > 1)),
            thedict=optiondict,
        )

    if checkpoint:
        outcorrarray[:, :] = 0.0
        outcorrarray[validvoxels, :] = corrout[:, :]
        if theinputdata.filetype == "text":
            tide_io.writenpvecs(
                outcorrarray.reshape(nativecorrshape),
                f"{outputname}_corrout_prefit_pass" + str(thepass) + ".txt",
            )
        else:
            savename = f"{outputname}_desc-corroutprefit_pass-" + str(thepass)
            tide_io.savetonifti(outcorrarray.reshape(nativecorrshape), theheader, savename)

    TimingLGR.info(
        f"{similaritytype} calculation end, pass {thepass}",
        {
            "message2": voxelsprocessed_cp,
            "message3": "voxels",
        },
    )

    return similaritytype
