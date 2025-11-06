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
import argparse
import copy
import time
from argparse import Namespace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

import rapidtide.correlate as tide_corr
import rapidtide.filter as tide_filt
import rapidtide.fit as tide_fit
import rapidtide.io as tide_io
import rapidtide.miscmath as tide_math
import rapidtide.resample as tide_resample
import rapidtide.stats as tide_stats
import rapidtide.workflows.parser_funcs as pf

DEFAULT_NUMSPACESTEPS = 1
DEFAULT_NPASSES = 20
DEFAULT_RADIUS = 10.5
DEFAULT_WINDOW_TYPE = "hamming"
DEFAULT_DETREND_ORDER = 3
DEFAULT_AMPTHRESH = 0.3
DEFAULT_MINLAGDIFF = 0.0


def _get_parser() -> Any:
    """
    Create and configure an argument parser for the localflow command-line tool.

    This function sets up an `argparse.ArgumentParser` with a set of predefined
    command-line arguments used to control the behavior of the local flow analysis
    pipeline. It includes options for input/output file handling, reconstruction
    parameters, filtering, windowing, and debugging.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser with all required and optional arguments.

    Notes
    -----
    The parser includes the following argument groups:
    - Required positional arguments: `inputfilename` and `outputroot`
    - Optional arguments for reconstruction parameters such as `npasses`, `radius`,
      `minlagdiff`, `ampthresh`, `gausssigma`, `oversampfac`, `dofit`, `detrendorder`,
      and `nosphere`
    - Miscellaneous options including `noprogressbar` and `debug`

    Examples
    --------
    >>> parser = _get_parser()
    >>> args = parser.parse_args()
    >>> print(args.inputfilename)
    'input.nii.gz'
    """
    # get the command line parameters
    parser = argparse.ArgumentParser(
        prog="localflow",
        description="Calculate local sources of signal.",
        allow_abbrev=False,
    )
    parser.add_argument("inputfilename", type=str, help="The name of the input nifti file.")
    parser.add_argument("outputroot", type=str, help="The root name of the output nifti files.")

    parser.add_argument(
        "--npasses",
        dest="npasses",
        type=int,
        help=f"The number of passes for reconstruction.  Default is {DEFAULT_NPASSES}",
        default=DEFAULT_NPASSES,
    )
    parser.add_argument(
        "--radius",
        dest="radius",
        type=float,
        help=f"The radius around the voxel to check correlations.  Default is {DEFAULT_RADIUS}",
        default=DEFAULT_RADIUS,
    )
    parser.add_argument(
        "--minlagdiff",
        dest="minlagdiff",
        type=float,
        help=f"The minimum lagtime difference threshold to select which diffs to include in reconstruction.  Default is {DEFAULT_MINLAGDIFF}",
        default=DEFAULT_MINLAGDIFF,
    )
    parser.add_argument(
        "--ampthresh",
        dest="ampthresh",
        type=float,
        help=f"The correlation threshold to select which diffs to include in reconstruction.  Default is {DEFAULT_AMPTHRESH}",
        default=DEFAULT_AMPTHRESH,
    )
    parser.add_argument(
        "--gausssigma",
        dest="gausssigma",
        type=float,
        help=(
            "Spatially filter fMRI data prior to analysis "
            "using GAUSSSIGMA in mm.  Set GAUSSSIGMA negative "
            "to set it to half the mean voxel "
            "dimension (a rule of thumb for a good value)."
        ),
        default=0.0,
    )
    parser.add_argument(
        "--oversampfac",
        dest="oversampfactor",
        action="store",
        type=int,
        metavar="OVERSAMPFAC",
        help=(
            "Oversample the fMRI data by the following "
            "integral factor.  Set to -1 for automatic selection (default)."
        ),
        default=-1,
    )
    parser.add_argument(
        "--dofit",
        dest="dofit",
        action="store_true",
        help="Turn on correlation fitting.",
        default=False,
    )
    parser.add_argument(
        "--detrendorder",
        dest="detrendorder",
        action="store",
        type=int,
        metavar="ORDER",
        help=(f"Set order of trend removal (0 to disable). Default is {DEFAULT_DETREND_ORDER}."),
        default=DEFAULT_DETREND_ORDER,
    )
    parser.add_argument(
        "--nosphere",
        dest="dosphere",
        action="store_false",
        help=("Use rectangular rather than spherical reconstruction kernel."),
        default=True,
    )

    pf.addfilteropts(parser, filtertarget="data and regressors", details=True)
    pf.addwindowopts(parser, windowtype=DEFAULT_WINDOW_TYPE)

    misc = parser.add_argument_group("Miscellaneous options")
    misc.add_argument(
        "--noprogressbar",
        dest="showprogressbar",
        action="store_false",
        help=("Will disable showing progress bars (helpful if stdout is going " "to a file)."),
        default=True,
    )
    misc.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Turn on debugging information.",
        default=False,
    )
    return parser


def preprocdata(
    fmridata: Any,
    themask: Any,
    theprefilter: Any,
    oversamplefactor: Any,
    Fs: Any,
    tr: Any,
    detrendorder: int = 3,
    windowfunc: str = "hamming",
    padseconds: int = 0,
    showprogressbar: bool = True,
) -> None:
    """
    Preprocess fMRI data by resampling, filtering, and normalizing voxel time series.

    This function applies a series of preprocessing steps to fMRI data including:
    resampling to a higher temporal resolution, applying a filter, and detrending
    with correlation normalization. It processes each voxel individually based on
    a provided mask.

    Parameters
    ----------
    fmridata : array-like
        4D fMRI data array with shape (nx, ny, nz, nt), where nx, ny, nz are spatial
        dimensions and nt is the number of time points.
    themask : array-like
        3D binary mask array with the same spatial dimensions as `fmridata`. Voxels
        with values > 0 are processed.
    theprefilter : object
        A filter object with an `apply` method that applies a temporal filter to the data.
    oversamplefactor : float
        Factor by which to oversample the data. Must be a positive number.
    Fs : float
        Sampling frequency of the original fMRI data in Hz.
    tr : float
        Repetition time (TR) of the fMRI acquisition in seconds.
    detrendorder : int, optional
        Order of the polynomial used for detrending. Default is 3.
    windowfunc : str, optional
        Window function used for correlation normalization. Default is "hamming".
    padseconds : int, optional
        Number of seconds to pad the resampled signal. Default is 0.
    showprogressbar : bool, optional
        Whether to display a progress bar during voxel processing. Default is True.

    Returns
    -------
    tuple
        A tuple containing:
        - osfmridata_byvox : ndarray
            Resampled and filtered fMRI data for processed voxels, shape (numvoxels, ostimepoints).
        - ostimepoints : int
            Number of time points in the oversampled data.
        - oversamptr : float
            Oversampled repetition time.
        - numvoxelsprocessed : int
            Total number of voxels processed.

    Notes
    -----
    This function modifies the input data in-place during processing. The output includes
    only the voxels that are marked as active in `themask`.

    Examples
    --------
    >>> import numpy as np
    >>> from some_module import preprocdata
    >>> fmri_data = np.random.rand(64, 64, 32, 100)
    >>> mask = np.ones((64, 64, 32))
    >>> filter_obj = SomeFilter()
    >>> result = preprocdata(
    ...     fmri_data, mask, filter_obj, oversamplefactor=2.0, Fs=2.0, tr=2.0
    ... )
    """
    numspatiallocs = fmridata.shape[0] * fmridata.shape[1] * fmridata.shape[2]
    timepoints = fmridata.shape[3]

    initial_fmri_x = np.arange(0.0, timepoints) * tr

    oversamptr = tr / oversamplefactor
    oversampFs = oversamplefactor * Fs
    os_fmri_x = np.arange(0.0, timepoints * oversamplefactor - (oversamplefactor - 1))
    os_fmri_x *= oversamptr
    ostimepoints = len(os_fmri_x)
    fmridata_byvox = fmridata.reshape((numspatiallocs, timepoints))
    themask_byvox = themask.reshape((numspatiallocs))
    osfmridata = np.zeros(
        (fmridata.shape[0], fmridata.shape[1], fmridata.shape[2], ostimepoints), dtype=float
    )
    osfmridata_byvox = osfmridata.reshape((numspatiallocs, ostimepoints))

    numvoxelsprocessed = 0
    for thevoxel in tqdm(
        range(0, numspatiallocs),
        desc="Voxel",
        unit="voxels",
        disable=(not showprogressbar),
    ):
        if themask_byvox[thevoxel] > 0:
            osfmridata_byvox[thevoxel, :] = tide_math.corrnormalize(
                theprefilter.apply(
                    oversampFs,
                    tide_resample.doresample(
                        initial_fmri_x,
                        fmridata_byvox[thevoxel, :],
                        os_fmri_x,
                        padlen=int(oversampFs * padseconds),
                    ),
                ),
                detrendorder=detrendorder,
                windowfunc=windowfunc,
            )
            numvoxelsprocessed += 1
    return osfmridata_byvox, ostimepoints, oversamptr, numvoxelsprocessed


def getcorrloc(
    thedata: Any,
    idx1: Any,
    idx2: Any,
    Fs: Any,
    dofit: bool = False,
    lagmin: float = -12.5,
    lagmax: float = 12.5,
    widthmax: float = 100.0,
    negsearch: float = 15.0,
    possearch: float = 15.0,
    padding: int = 0,
    debug: bool = False,
) -> None:
    """
    Compute the cross-correlation peak between two time series and optionally fit it.

    This function computes the cross-correlation between two time series selected
    from `thedata` using indices `idx1` and `idx2`. It returns the maximum correlation
    value, the corresponding time lag, a mask indicating success, and a failure reason.

    Parameters
    ----------
    thedata : array_like
        Input data array of shape (n_channels, n_samples).
    idx1 : int or array_like
        Index or indices of the first time series in `thedata`.
    idx2 : int or array_like
        Index or indices of the second time series in `thedata`.
    Fs : float
        Sampling frequency of the data.
    dofit : bool, optional
        If True, perform a peak fit on the cross-correlation function. Default is False.
    lagmin : float, optional
        Minimum lag to consider in seconds. Default is -12.5.
    lagmax : float, optional
        Maximum lag to consider in seconds. Default is 12.5.
    widthmax : float, optional
        Maximum width for fitting. Default is 100.0.
    negsearch : float, optional
        Search range for negative lags in seconds. Default is 15.0.
    possearch : float, optional
        Search range for positive lags in seconds. Default is 15.0.
    padding : int, optional
        Zero-padding for FFT-based correlation. Default is 0.
    debug : bool, optional
        If True, print debug information. Default is False.

    Returns
    -------
    tuple
        A tuple of (maxcorr, maxtime, maskval, failreason) where:
        - maxcorr: Maximum correlation value.
        - maxtime: Time lag corresponding to maxcorr in seconds.
        - maskval: Mask indicating fit success (1 = success, 0 = failure).
        - failreason: Numeric code indicating reason for fit failure (0 = no failure).

    Notes
    -----
    - If either time series contains all zeros, the function returns (0.0, 0.0, 0, 0).
    - The function uses `tide_corr.fastcorrelate` for correlation and `tide_fit.simfuncpeakfit`
      for fitting when `dofit=True`.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.rand(10, 1000)
    >>> corr, time, mask, reason = getcorrloc(data, 0, 1, Fs=100, dofit=True)
    >>> print(f"Correlation: {corr}, Lag: {time}s")
    """
    tc1 = thedata[idx1, :]
    tc2 = thedata[idx2, :]
    if np.any(tc1) != 0.0 and np.any(tc2) != 0.0:
        if debug:
            print(f"{idx1=}, {idx2=}")
            print(f"{tc1=}")
            print(f"{tc2=}")

        thesimfunc = tide_corr.fastcorrelate(
            tc1,
            tc2,
            zeropadding=padding,
            usefft=True,
            debug=debug,
        )
        similarityfunclen = len(thesimfunc)
        similarityfuncorigin = similarityfunclen // 2 + 1

        negpoints = int(negsearch * Fs)
        pospoints = int(possearch * Fs)
        trimsimfunc = thesimfunc[
            similarityfuncorigin - negpoints : similarityfuncorigin + pospoints
        ]
        offset = 0.0
        trimtimeaxis = (
            (
                np.arange(0.0, similarityfunclen) * (1.0 / Fs)
                - ((similarityfunclen - 1) * (1.0 / Fs)) / 2.0
            )
            - offset
        )[similarityfuncorigin - negpoints : similarityfuncorigin + pospoints]
        if dofit:
            (
                maxindex,
                maxtime,
                maxcorr,
                maxsigma,
                maskval,
                failreason,
                peakstart,
                peakend,
            ) = tide_fit.simfuncpeakfit(
                trimsimfunc,
                trimtimeaxis,
                useguess=False,
                maxguess=0.0,
                displayplots=False,
                functype="correlation",
                peakfittype="gauss",
                searchfrac=0.5,
                lagmod=1000.0,
                enforcethresh=True,
                allowhighfitamps=False,
                lagmin=lagmin,
                lagmax=lagmax,
                absmaxsigma=1000.0,
                absminsigma=0.25,
                hardlimit=True,
                bipolar=False,
                lthreshval=0.0,
                uthreshval=1.0,
                zerooutbadfit=True,
                debug=False,
            )
        else:
            maxtime = trimtimeaxis[np.argmax(trimsimfunc)]
            maxcorr = np.max(trimsimfunc)
            maskval = 1
            failreason = 0
        if debug:
            print(f"{maxtime=}")
            print(f"{maxcorr=}")
            print(f"{maskval=}")
            print(f"{negsearch=}")
            print(f"{possearch=}")
            print(f"{Fs=}")
            print(f"{len(trimtimeaxis)=}")
            print(trimsimfunc, trimtimeaxis)
        return maxcorr, maxtime, maskval, failreason
    else:
        return 0.0, 0.0, 0, 0


def xyz2index(x: Any, y: Any, z: Any, xsize: Any, ysize: Any, zsize: Any) -> None:
    """
    Convert 3D coordinates to a linear index for a 3D array.

    This function maps 3D coordinates (x, y, z) to a linear index assuming
    row-major order storage of a 3D array with dimensions (xsize, ysize, zsize).

    Parameters
    ----------
    x : Any
        X-coordinate, should be between 0 and xsize-1 inclusive
    y : Any
        Y-coordinate, should be between 0 and ysize-1 inclusive
    z : Any
        Z-coordinate, should be between 0 and zsize-1 inclusive
    xsize : Any
        Size of the array along the x-axis
    ysize : Any
        Size of the array along the y-axis
    zsize : Any
        Size of the array along the z-axis

    Returns
    -------
    int
        Linear index if coordinates are valid (within bounds), -1 otherwise

    Notes
    -----
    The function uses row-major order indexing: index = z + y * zsize + x * zsize * ysize

    Examples
    --------
    >>> xyz2index(1, 2, 3, 10, 10, 10)
    321
    >>> xyz2index(15, 2, 3, 10, 10, 10)
    -1
    """
    if (0 <= x < xsize) and (0 <= y < ysize) and (0 <= z < zsize):
        return int(z) + int(y) * int(zsize) + int(x) * int(zsize * ysize)
    else:
        return -1


def index2xyz(theindex: Any, ysize: Any, zsize: Any) -> None:
    """
    Convert a linear index to 3D coordinates (x, y, z).

    This function maps a 1D index to 3D coordinates within a 3D grid
    with dimensions determined by ysize and zsize. The conversion assumes
    row-major ordering where the index is distributed across the three
    dimensions based on the product of the grid dimensions.

    Parameters
    ----------
    theindex : Any
        The linear index to be converted to 3D coordinates
    ysize : Any
        The size of the grid in the y dimension
    zsize : Any
        The size of the grid in the z dimension

    Returns
    -------
    tuple
        A tuple containing (x, y, z) coordinates corresponding to the input index

    Notes
    -----
    The function assumes that the grid dimensions are such that the total
    number of elements is sufficient to accommodate the given index.
    The conversion follows the formula:
    - x = index // (ysize * zsize)
    - y = (index - x * ysize * zsize) // zsize
    - z = index - x * ysize * zsize - y * zsize

    Examples
    --------
    >>> index2xyz(10, 3, 4)
    (0, 0, 10)

    >>> index2xyz(25, 3, 4)
    (2, 0, 1)
    """
    x = theindex // int(zsize * ysize)
    theindex -= int(x) * int(zsize * ysize)
    y = theindex // int(zsize)
    theindex -= int(y) * int(zsize)
    z = theindex
    return x, y, z


def localflow(args: Any) -> None:
    """
    Perform local flow analysis on fMRI data.

    This function processes fMRI data to compute local correlation and delay information
    across spatial neighbors, followed by a reconstruction step to estimate time delays
    in the signal propagation.

    Parameters
    ----------
    args : Any
        An object containing various arguments for processing, including:
        - inputfilename : str
            Path to the input NIfTI file.
        - outputroot : str
            Root name for output files.
        - gausssigma : float
            Sigma for Gaussian spatial smoothing. If less than 0, automatically computed.
        - oversampfactor : int
            Oversampling factor for preprocessing. If -1, computed automatically.
        - detrendorder : int
            Order of detrending to apply.
        - windowfunc : str
            Window function to use for preprocessing.
        - padseconds : float
            Padding in seconds for preprocessing.
        - showprogressbar : bool
            Whether to show progress bars.
        - dofit : bool
            Whether to fit the correlation.
        - debug : bool
            Whether to enable debug mode.
        - radius : float
            Neighborhood radius in mm.
        - npasses : int
            Number of reconstruction passes.
        - ampthresh : float
            Amplitude threshold for valid correlations.

    Returns
    -------
    None
        This function does not return a value but saves multiple NIfTI files and timing logs.

    Notes
    -----
    The function performs the following steps:
    1. Reads and preprocesses input fMRI data.
    2. Applies spatial filtering if specified.
    3. Prepares data for correlation analysis.
    4. Identifies spatial neighbors within a specified radius.
    5. Computes local correlations and delays.
    6. Reconstructs time delays using iterative averaging.
    7. Saves results as NIfTI files.

    Examples
    --------
    >>> import argparse
    >>> args = argparse.Namespace(
    ...     inputfilename="fmri.nii.gz",
    ...     outputroot="output",
    ...     gausssigma=-1.0,
    ...     oversampfactor=-1,
    ...     detrendorder=1,
    ...     windowfunc="hann",
    ...     padseconds=10.0,
    ...     showprogressbar=True,
    ...     dofit=True,
    ...     debug=False,
    ...     radius=5.0,
    ...     npasses=5,
    ...     ampthresh=0.1
    ... )
    >>> localflow(args)
    """
    # set default variable values
    displayplots = False

    # postprocess filter options
    theobj, theprefilter = pf.postprocessfilteropts(args)

    # save timinginfo
    eventtimes = []
    starttime = time.time()
    thistime = starttime
    eventtimes.append(["Start", 0.0, 0.0, None, None])

    # get the input TR
    inputtr_fromfile, numinputtrs = tide_io.fmritimeinfo(args.inputfilename)
    print("input data: ", numinputtrs, " timepoints, tr = ", inputtr_fromfile)

    input_img, fmridata, input_hdr, thedims, thesizes = tide_io.readfromnifti(args.inputfilename)
    if input_hdr.get_xyzt_units()[1] == "msec":
        tr = thesizes[4] / 1000.0
    else:
        tr = thesizes[4]
    Fs = 1.0 / tr
    print("tr from header =", tr, ", sample frequency is ", Fs)
    thistime = time.time() - starttime
    eventtimes.append(["Read input file", thistime, thistime, None, None])

    xsize, ysize, numslices, timepoints = tide_io.parseniftidims(thedims)
    xdim, ydim, slicethickness, tr = tide_io.parseniftisizes(thesizes)

    numspatiallocs = int(xsize) * int(ysize) * int(numslices)
    fmridata_voxbytime = fmridata.reshape((numspatiallocs, timepoints))
    if args.debug:
        print(f"{fmridata.shape=}")
        print(f"{fmridata_voxbytime.shape=}")

    # make a mask
    meanim = np.mean(fmridata, axis=3)
    themask = np.uint16(tide_stats.makemask(meanim, threshpct=0.1))
    themask_byvox = themask.reshape((numspatiallocs))
    validvoxels = np.where(themask > 0)
    numvalid = len(validvoxels[0])
    print(f"{numvalid} valid")
    output_hdr = copy.deepcopy(input_hdr)
    output_hdr["dim"][4] = 1
    tide_io.savetonifti(
        themask,
        output_hdr,
        f"{args.outputroot}_mask",
        debug=args.debug,
    )
    lasttime = thistime
    thistime = time.time() - starttime
    eventtimes.append(["Made and saved mask", thistime, thistime - lasttime, None, None])

    lasttime = thistime
    thistime = time.time() - starttime
    if args.gausssigma < 0.0:
        # set gausssigma automatically
        args.gausssigma = np.mean([xdim, ydim, slicethickness]) / 2.0
    if args.gausssigma > 0.0:
        eventtimes.append(["Spatial filter start", thistime, thistime - lasttime, None, None])
        print(f"applying gaussian spatial filter to fmri data " f" with sigma={args.gausssigma}")
        for i in tqdm(
            range(timepoints),
            desc="Timepoint",
            unit="timepoints",
            disable=(not args.showprogressbar),
        ):
            fmridata[:, :, :, i] = tide_filt.ssmooth(
                xdim,
                ydim,
                slicethickness,
                args.gausssigma,
                fmridata[:, :, :, i],
            )
        lasttime = thistime
        thistime = time.time() - starttime
        eventtimes.append(
            ["Spatial filter done", thistime, thistime - lasttime, timepoints, "timepoints"]
        )

    # prepare the input data
    lasttime = thistime
    thistime = time.time() - starttime
    eventtimes.append(["Prepare data start", thistime, thistime - lasttime, None, None])
    if args.oversampfactor == -1:
        oversamplefactor = int(np.max([np.ceil(tr / 0.5), 1]))
    else:
        oversamplefactor = args.oversampfactor
    print(f"using an oversample factor of {oversamplefactor}")
    print("Preparing data", flush=True)
    osfmridata_voxbytime, ostimepoints, oversamptr, numvoxelsprocessed = preprocdata(
        fmridata,
        themask,
        theprefilter,
        oversamplefactor,
        Fs,
        tr,
        detrendorder=args.detrendorder,
        windowfunc=args.windowfunc,
        padseconds=args.padseconds,
        showprogressbar=args.showprogressbar,
    )
    print("...done", flush=True)
    print("\n", flush=True)
    output_hdr = copy.deepcopy(input_hdr)
    output_hdr["dim"][4] = ostimepoints
    output_hdr["pixdim"][4] = oversamptr
    tide_io.savetonifti(
        osfmridata_voxbytime.reshape((xsize, ysize, numslices, ostimepoints)),
        output_hdr,
        f"{args.outputroot}_preprocdata",
        debug=args.debug,
    )
    lasttime = thistime
    thistime = time.time() - starttime
    eventtimes.append(
        ["Prepare data done", thistime, thistime - lasttime, numvoxelsprocessed, "voxels"]
    )

    # make list of neighbors
    lasttime = thistime
    thistime = time.time() - starttime
    eventtimes.append(["Find neighbors start", thistime, thistime - lasttime, None, None])
    args.dosphere = True
    xsteps = int(np.ceil(args.radius / xdim))
    ysteps = int(np.ceil(args.radius / ydim))
    zsteps = int(np.ceil(args.radius / slicethickness))

    neighborlist = []
    distancelist = []
    for z in range(-zsteps, zsteps + 1):
        for y in range(-ysteps, ysteps + 1):
            for x in range(-xsteps, xsteps + 1):
                if args.dosphere:
                    distance = np.sqrt(
                        np.square(x * xdim) + np.square(y * ydim) + np.square(z * slicethickness)
                    )
                    if (x != 0 or y != 0 or z != 0) and distance <= args.radius:
                        neighborlist.append((x, y, z))
                        distancelist.append(distance)
                else:
                    if x != 0 or y != 0 or z != 0:
                        neighborlist.append((x, y, z))
    tide_io.writenpvecs(np.transpose(np.asarray(neighborlist)), f"{args.outputroot}_neighbors")
    if args.debug:
        print(f"{len(neighborlist)=}, {neighborlist=}")
    lasttime = thistime
    thistime = time.time() - starttime
    eventtimes.append(
        ["Find neighbors done", thistime, thistime - lasttime, len(neighborlist), "voxels"]
    )

    corrcoeffs = np.zeros((xsize, ysize, numslices, len(neighborlist)), dtype=float)
    delays = np.zeros((xsize, ysize, numslices, len(neighborlist)), dtype=float)
    corrvalid = np.zeros((xsize, ysize, numslices, len(neighborlist)), dtype=int)
    failreason = np.zeros((xsize, ysize, numslices, len(neighborlist)), dtype=int)
    if args.debug:
        print(f"{corrcoeffs.shape=}, {delays.shape=}, {corrvalid.shape=}")
        printfirstdetails = True
    else:
        printfirstdetails = False

    corrcoeffs_byvox = corrcoeffs.reshape((numspatiallocs, len(neighborlist)))
    delays_byvox = delays.reshape((numspatiallocs, len(neighborlist)))
    corrvalid_byvox = corrvalid.reshape((numspatiallocs, len(neighborlist)))
    failreason_byvox = failreason.reshape((numspatiallocs, len(neighborlist)))

    # Find every voxel's valid neighbors
    lasttime = thistime
    thistime = time.time() - starttime
    eventtimes.append(
        ["Generate correlation list start", thistime, thistime - lasttime, None, None]
    )
    print("Generate the correlation list", flush=True)
    indexlist = []
    indexpairs = []
    theindex = 0
    for index1 in tqdm(
        range(numspatiallocs),
        desc="Voxel",
        unit="voxels",
        disable=(not args.showprogressbar),
    ):
        if themask_byvox[index1] > 0:
            # voxel is in the mask
            x, y, z = index2xyz(index1, ysize, numslices)
            for idx, neighbor in enumerate(neighborlist):
                index2 = xyz2index(
                    x + neighbor[0],
                    y + neighbor[1],
                    z + neighbor[2],
                    xsize,
                    ysize,
                    numslices,
                )
                if index2 > 0:
                    # neighbor location is valid
                    if themask_byvox[index2] > 0:
                        # neighbor is in the mask
                        indexpairs.append((index1, index2, idx, theindex + 0))
            theindex += 1
            indexlist.append(index1)
    print("...done", flush=True)
    print("\n", flush=True)
    lasttime = thistime
    thistime = time.time() - starttime
    eventtimes.append(
        [
            "Generate correlation list done",
            thistime,
            thistime - lasttime,
            len(indexpairs),
            "correlation pairs",
        ]
    )
    tide_io.writenpvecs(np.transpose(np.asarray(indexlist)), f"{args.outputroot}_indexlist")

    # Do the correlations
    lasttime = thistime
    thistime = time.time() - starttime
    eventtimes.append(["Do correlations start", thistime, thistime - lasttime, None, None])
    print(f"Process {len(indexpairs)} correlations", flush=True)
    for index1, index2, neighboridx, theindex in tqdm(
        indexpairs,
        desc="Correlation pair",
        unit="pairs",
        disable=(not args.showprogressbar),
    ):
        # print(index1, index2, neighboridx, theindex)
        (
            corrcoeffs_byvox[index1, neighboridx],
            delays_byvox[index1, neighboridx],
            corrvalid_byvox[index1, neighboridx],
            failreason_byvox[index1, neighboridx],
        ) = getcorrloc(
            osfmridata_voxbytime,
            index1,
            index2,
            oversamplefactor * Fs,
            dofit=args.dofit,
            debug=printfirstdetails,
        )
        neighborloc = (
            (neighborlist[neighboridx])[0] + xsteps,
            (neighborlist[neighboridx])[1] + ysteps,
            (neighborlist[neighboridx])[2] + zsteps,
        )
        # print(neighborlist[neighboridx], neighborloc)
    print("...done", flush=True)
    print("\n", flush=True)
    lasttime = thistime
    thistime = time.time() - starttime
    eventtimes.append(
        [
            "Do correlations done",
            thistime,
            thistime - lasttime,
            len(indexpairs),
            "correlation pairs",
        ]
    )

    output_hdr = copy.deepcopy(input_hdr)
    output_hdr["dim"][4] = len(neighborlist)
    tide_io.savetonifti(
        corrcoeffs,
        output_hdr,
        f"{args.outputroot}_corrcoeffs",
        debug=args.debug,
    )
    tide_io.savetonifti(
        delays,
        output_hdr,
        f"{args.outputroot}_delays",
        debug=args.debug,
    )
    tide_io.savetonifti(
        corrvalid,
        output_hdr,
        f"{args.outputroot}_corrvalid",
        debug=args.debug,
    )
    tide_io.savetonifti(
        failreason,
        output_hdr,
        f"{args.outputroot}_failreason",
        debug=args.debug,
    )

    # now reconstruct
    lasttime = thistime
    thistime = time.time() - starttime
    eventtimes.append(["Reconstruction start", thistime, thistime - lasttime, None, None])
    print("Reconstruct", flush=True)
    gain = 0.95
    targetdelay = np.zeros((xsize, ysize, numslices, args.npasses), dtype=float)
    targetdelay_byvox = targetdelay.reshape((numspatiallocs, args.npasses))

    targetdelay[:, :, :, 0] = 0.0 * (np.random.random((xsize, ysize, numslices)) - 0.5)

    numneighbors = np.zeros((xsize, ysize, numslices), dtype=int)
    numneighbors_byvox = numneighbors.reshape((numspatiallocs))
    # loop over passes
    for thepass in tqdm(
        range(1, args.npasses), desc="Pass", unit="passes", disable=(not args.showprogressbar)
    ):
        # loop over voxels
        for thearrayindex, thecoordindex in enumerate(indexlist):
            deltasum = 0.0
            numneighbors_byvox[thecoordindex] = 0
            for whichneighbor in range(len(neighborlist)):
                if (
                    corrvalid_byvox[
                        thecoordindex,
                        whichneighbor,
                    ]
                    > 0
                    and np.fabs(
                        corrcoeffs_byvox[
                            thecoordindex,
                            whichneighbor,
                        ]
                    )
                    > args.ampthresh
                ):
                    thediff = (
                        delays_byvox[
                            thecoordindex,
                            whichneighbor,
                        ]
                        - targetdelay_byvox[thecoordindex, thepass - 1]
                    )
                    thenorm = corrcoeffs_byvox[
                        thecoordindex,
                        whichneighbor,
                    ]
                    numneighbors_byvox[thecoordindex] += 1
                    # deltasum += thediff * thenorm * thenorm / distancelist[whichneighbor]
                    deltasum += thediff * thenorm
            if numneighbors_byvox[thecoordindex] > 0:
                targetdelay_byvox[thecoordindex, thepass] = (
                    gain * targetdelay_byvox[thecoordindex, thepass - 1]
                    + deltasum / numneighbors_byvox[thecoordindex]
                )
    print("...done", flush=True)
    print("\n", flush=True)
    lasttime = thistime
    thistime = time.time() - starttime
    eventtimes.append(
        ["Reconstruction done", thistime, thistime - lasttime, args.npasses - 1, "passes"]
    )

    output_hdr = copy.deepcopy(input_hdr)
    output_hdr["dim"][4] = args.npasses
    tide_io.savetonifti(
        targetdelay,
        output_hdr,
        f"{args.outputroot}_targetdelay",
        debug=args.debug,
    )
    output_hdr["dim"][4] = 1
    tide_io.savetonifti(
        numneighbors,
        output_hdr,
        f"{args.outputroot}_numneighbors",
        debug=args.debug,
    )
    output_hdr["dim"][4] = 1
    tide_io.savetonifti(
        targetdelay[:, :, :, -1],
        output_hdr,
        f"{args.outputroot}_maxtime",
        debug=args.debug,
    )
    formattedtimings = []
    for eventtime in eventtimes:
        if eventtime[3] is not None:
            formattedtimings.append(
                f"{eventtime[1]:.2f}\t{eventtime[2]:.2f}\t{eventtime[0]}\t{eventtime[3]/eventtime[2]:.2f} ({eventtime[4]}/sec)"
            )
        else:
            formattedtimings.append(f"{eventtime[1]:.2f}\t{eventtime[2]:.2f}\t{eventtime[0]}")
        print(formattedtimings[-1])
    tide_io.writevec(formattedtimings, f"{args.outputroot}_formattedruntimings.txt")
