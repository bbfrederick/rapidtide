#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2016-2026 Blaise Frederick
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
import time
import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray

import rapidtide.genericmultiproc as tide_genericmultiproc
import rapidtide.resample as tide_resample

warnings.simplefilter(action="ignore", category=FutureWarning)
LGR = logging.getLogger("GENERAL")


def _resolve_torch_device(torch_module: Any, device: str = "auto") -> Any:
    """
    Resolve a PyTorch device for GPU correlation.

    Parameters
    ----------
    torch_module : Any
        Imported torch module.
    device : str, optional
        Requested device name. Supported values are "auto", "cuda", "mps", and
        "rocm". For ROCm, torch uses the "cuda" device type internally.

    Returns
    -------
    torch.device
        Resolved torch device object.
    """
    requested = device.lower()
    if requested == "auto":
        if torch_module.cuda.is_available():
            return torch_module.device("cuda")
        if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            return torch_module.device("mps")
        raise RuntimeError("No supported GPU backend found (CUDA/ROCm/MPS).")

    if requested == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("Requested CUDA backend, but CUDA is not available.")
        return torch_module.device("cuda")

    if requested == "rocm":
        if not torch_module.cuda.is_available():
            raise RuntimeError("Requested ROCm backend, but GPU backend is not available.")
        if getattr(torch_module.version, "hip", None) is None:
            raise RuntimeError("Requested ROCm backend, but this torch build is not ROCm-enabled.")
        return torch_module.device("cuda")

    if requested == "mps":
        if not (
            hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available()
        ):
            raise RuntimeError("Requested MPS backend, but MPS is not available.")
        return torch_module.device("mps")

    raise ValueError(f"Unknown device '{device}'. Use one of: auto, cuda, rocm, mps.")


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
    thetc, theCorrelator, fmri_x, fmritc, os_fmri_x, theglobalmaxlist, thexcorr_y = voxelargs
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


def correlationpass_cpu(
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
    usegpu: bool = False,
    device: str = "auto",
    batchsize: int | None = None,
    fallback_to_cpu: bool = True,
) -> tuple[int, list[float], NDArray]:
    """
    Dispatch correlation pass to CPU or GPU implementation.

    Parameters
    ----------
    usegpu : bool, optional
        If True, call :func:`correlationpass_gpu`; otherwise call
        :func:`correlationpass_cpu`. Default is False.
    device : str, optional
        GPU backend selector used when ``usegpu=True``. One of: "auto", "cuda",
        "rocm", "mps". Default is "auto".
    batchsize : int | None, optional
        GPU batch size used when ``usegpu=True``
    fallback_to_cpu : bool, optional
        If True, GPU path falls back to CPU when GPU backends are unavailable or
        unsupported for current options.
    """
    if usegpu:
        return correlationpass_gpu(
            fmridata,
            referencetc,
            theCorrelator,
            fmri_x,
            os_fmri_x,
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
            device=device,
            batchsize=chunksize if batchsize is None else batchsize,
            fallback_to_cpu=fallback_to_cpu,
        )

    return correlationpass_cpu(
        fmridata,
        referencetc,
        theCorrelator,
        fmri_x,
        os_fmri_x,
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


def correlationpass_gpu(
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
    device: str = "auto",
    batchsize: int = 1024,
    fallback_to_cpu: bool = True,
) -> tuple[int, list[float], NDArray]:
    """
    GPU-accelerated alternate implementation of :func:`correlationpass`.

    This implementation uses PyTorch to batch cross-correlation on GPU while
    preserving the same API and output structure as the CPU implementation.
    It supports CUDA, ROCm (via torch's CUDA device type), and Apple MPS.

    Notes
    -----
    - This GPU path currently supports correlation weighting ``"None"``.
      Other weighting modes fall back to CPU if ``fallback_to_cpu=True``.
    - Timecourse resampling and preprocessing still use the existing CPU code paths
      to maintain numerical behavior with existing filters.
    """
    if batchsize < 1:
        raise ValueError("batchsize must be >= 1")

    if debug:
        print(f"calling setreftc in calcsimfunc (gpu) with length {len(referencetc)}")
    theCorrelator.setreftc(referencetc)
    theCorrelator.setlimits(lagmininpts, lagmaxinpts)

    corrweighting = str(getattr(theCorrelator, "corrweighting", "None"))
    if corrweighting not in ["None", "phat"]:
        msg = (
            "correlationpass_gpu currently supports corrweighting in {'None','phat'}; "
            f"received '{corrweighting}'"
        )
        if fallback_to_cpu:
            LGR.warning(f"{msg}. Falling back to CPU implementation.")
            return correlationpass_cpu(
                fmridata,
                referencetc,
                theCorrelator,
                fmri_x,
                os_fmri_x,
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
        raise NotImplementedError(msg)

    try:
        import torch
    except ImportError as e:
        if fallback_to_cpu:
            LGR.warning("PyTorch not available; falling back to CPU implementation.")
            return correlationpass_cpu(
                fmridata,
                referencetc,
                theCorrelator,
                fmri_x,
                os_fmri_x,
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
        raise ImportError("correlationpass_gpu requires torch to be installed.") from e

    try:
        torch_device = _resolve_torch_device(torch, device=device)
    except RuntimeError as e:
        if fallback_to_cpu:
            LGR.warning(f"{e} Falling back to CPU implementation.")
            return correlationpass_cpu(
                fmridata,
                referencetc,
                theCorrelator,
                fmri_x,
                os_fmri_x,
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
        raise
    if debug:
        print(f"correlationpass_gpu using device: {torch_device}")

    # Confirm FFT ops actually run on the selected backend. Some backends can
    # appear available but transparently execute unsupported ops on CPU.
    try:
        probe = torch.randn(256, device=torch_device, dtype=torch.float32)
        probe_fft = torch.fft.rfft(probe)
        probe_ifft = torch.fft.irfft(probe_fft, n=256)
        if (probe_fft.device.type != torch_device.type) or (
            probe_ifft.device.type != torch_device.type
        ):
            raise RuntimeError(
                f"FFT ops are not executing on requested device '{torch_device}' "
                f"(rfft on '{probe_fft.device}', irfft on '{probe_ifft.device}')."
            )
    except Exception as e:
        if fallback_to_cpu:
            LGR.warning(f"{e} Falling back to CPU implementation.")
            return correlationpass_cpu(
                fmridata,
                referencetc,
                theCorrelator,
                fmri_x,
                os_fmri_x,
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
        raise

    tc_len = len(os_fmri_x) if oversampfactor >= 1 else len(fmri_x)
    if len(referencetc) != tc_len:
        raise ValueError(
            f"Reference timecourse length ({len(referencetc)}) does not match "
            f"expected length ({tc_len}) for oversampfactor={oversampfactor}."
        )

    # Generate corrscale exactly as the CPU code does.
    thetc = np.zeros(tc_len, dtype=rt_floattype)
    theglobalmaxlist: list[float] = []
    dummy = np.zeros(100, dtype=rt_floattype)
    dummy, dummy, dummy, thecorrscale, dummy, dummy = _procOneVoxelCorrelation(
        0,
        _packvoxeldata(
            0,
            [
                thetc,
                theCorrelator,
                fmri_x,
                fmridata,
                os_fmri_x,
                theglobalmaxlist,
                dummy,
            ],
        ),
        oversampfactor=oversampfactor,
        interptype=interptype,
        debug=debug,
    )

    full_corr_len = 2 * tc_len - 1
    similarityfuncorigin = full_corr_len // 2 + 1
    trim_start = similarityfuncorigin - lagmininpts
    trim_stop = similarityfuncorigin + lagmaxinpts
    trimmed_len = trim_stop - trim_start
    if trimmed_len != corrout.shape[1]:
        raise ValueError(
            f"Trimmed correlation length ({trimmed_len}) does not match corrout width "
            f"({corrout.shape[1]})."
        )

    # FFT-based correlation with reversed reference reproduces fastcorrelate(..., weighting="None").
    ref_reversed = np.ascontiguousarray(theCorrelator.prepreftc[::-1])
    ref_t = torch.as_tensor(ref_reversed, device=torch_device, dtype=torch.float32)
    fft_len = full_corr_len
    ref_fft = torch.fft.rfft(ref_t, n=fft_len)

    numvoxels = int(fmridata.shape[0])
    theglobalmaxlist = []
    preptime = 0.0
    gputime = 0.0

    # Precompute all preprocessed voxel timecourses on CPU.
    t0_pre = time.perf_counter()
    prepped_tc = np.zeros((numvoxels, tc_len), dtype=np.float32)
    do_resample = oversampfactor >= 1 and not (
        (len(fmri_x) == len(os_fmri_x)) and np.array_equal(fmri_x, os_fmri_x)
    )
    for vox in range(numvoxels):
        fmritc = fmridata[vox, :]
        if do_resample:
            thetc_local = tide_resample.doresample(fmri_x, fmritc, os_fmri_x, method=interptype)
        else:
            thetc_local = fmritc + 0.0
        meanval[vox] = np.mean(thetc_local)
        prepped_tc[vox, :] = theCorrelator.preptc(thetc_local)

    preptime += time.perf_counter() - t0_pre

    def _run_gpu_batch(batch_np: NDArray, batch_voxels: NDArray) -> None:
        nonlocal gputime, theglobalmaxlist
        t0 = time.perf_counter()
        batch_t = torch.as_tensor(batch_np, device=torch_device, dtype=torch.float32)
        batch_fft = torch.fft.rfft(batch_t, n=fft_len, dim=-1)
        product = batch_fft * ref_fft.unsqueeze(0)
        if corrweighting == "phat":
            # Match gccproduct(..., weighting='phat') thresholding behavior.
            weighting = torch.abs(product)
            thresh = torch.max(weighting, dim=-1, keepdim=True).values * 0.1
            weighting = torch.maximum(weighting, thresh)
            weighted_product = product / weighting
        else:
            weighted_product = product
        corr_full = torch.fft.irfft(weighted_product, n=fft_len, dim=-1)
        if torch_device.type == "cuda":
            torch.cuda.synchronize(device=torch_device)
        elif torch_device.type == "mps":
            torch.mps.synchronize()
        gputime += time.perf_counter() - t0
        corr_trim = corr_full[:, trim_start:trim_stop]
        global_max_idx = torch.argmax(corr_full, dim=-1)

        corr_trim_cpu = corr_trim.detach().cpu().numpy().astype(rt_floattype, copy=False)
        global_max_idx_cpu = global_max_idx.detach().cpu().numpy()
        for local_idx, vox in enumerate(batch_voxels):
            corrout[vox, :] = corr_trim_cpu[local_idx, :]
            theglobalmaxlist.append(int(global_max_idx_cpu[local_idx]))

    for start in range(0, numvoxels, batchsize):
        stop = min(start + batchsize, numvoxels)
        batch_voxels = np.arange(start, stop, dtype=np.int64)
        _run_gpu_batch(prepped_tc[start:stop, :], batch_voxels)

    LGR.info(f"\nSimilarity function calculated on {numvoxels} voxels (GPU)")
    LGR.info(
        "correlationpass_gpu timing: "
        f"cpu_preprocess={preptime:.3f}s, gpu_corr={gputime:.3f}s, "
        f"batchsize={batchsize}, device={torch_device}"
    )
    uncollected = gc.collect()
    if uncollected != 0:
        LGR.info(f"garbage collected - unable to collect {uncollected} objects")
    else:
        LGR.info("garbage collected")

    return numvoxels, theglobalmaxlist, thecorrscale
