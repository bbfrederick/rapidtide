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
from collections import deque
from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

import rapidtide.fit as tide_fit
import rapidtide.io as tide_io
import rapidtide.peakeval as tide_peakeval
import rapidtide.resample as tide_resample
import rapidtide.simfuncfit as tide_simfuncfit
import rapidtide.util as tide_util
from rapidtide.workflows.delayflow import maskextensionindices
from rapidtide.workflows.resolvedelays import RESOLVEDCHANGEDTHRESH, resolvefromsimfunc

# Delay map quality metrics, recorded per pass in the runoptions.  The constants here
# are the ones every comparison in the delay repair work was scored with, so that
# numbers recorded by a run are directly comparable to the published calibrations.
METRICJUMPTHRESH = 2.0  # seconds; an adjacent-voxel difference above this is a "jump"
METRICMEDIANKERNEL = 5  # neighbourhood for the local median an outlier is measured against
METRICLONGIQRMULT = 2.0  # "long" delay is above median + this many IQRs of the run itself
METRICMINCOMPONENT = 27  # a long-delay blob this size or bigger counts as coherent

_NDIMAGE_TO_NUMPY_PAD_MODE = {
    "reflect": "symmetric",  # d c b a | a b c d | d c b a
    "nearest": "edge",  # a a a a | a b c d | d d d d
    "constant": "constant",
    "wrap": "wrap",
    "mirror": "reflect",  # d c b | a b c d | c b a
}

# Target maximum bytes for the materialised masked-window buffer.
_MAX_CHUNK_BYTES = 256 * 1024 * 1024  # 256 MiB


def _pad_and_view(
    data: NDArray,
    mask: NDArray,
    kernel_shape: tuple[int, ...],
    np_pad_mode: str,
) -> tuple[NDArray, NDArray]:
    """Pad data and mask, then return sliding-window views."""
    pad_widths = tuple((k // 2, k // 2) for k in kernel_shape)
    padded_data = np.pad(data.astype(np.float64), pad_widths, mode=np_pad_mode)
    padded_mask = np.pad(np.asarray(mask, dtype=np.bool_), pad_widths, mode=np_pad_mode)
    windows = np.lib.stride_tricks.sliding_window_view(padded_data, kernel_shape)
    mask_windows = np.lib.stride_tricks.sliding_window_view(padded_mask, kernel_shape)
    return windows, mask_windows


def _nanmedian_chunk(
    windows: NDArray,
    mask_windows: NDArray,
    n_kernel: int,
    start: int,
    end: int,
) -> NDArray:
    """Compute nanmedian for a contiguous slice of flattened voxels."""
    flat_w = windows.reshape(-1, n_kernel)[start:end]
    flat_m = mask_windows.reshape(-1, n_kernel)[start:end]
    masked = np.where(flat_m, flat_w, np.nan)
    return np.nanmedian(masked, axis=1)


def masked_median_filter(
    data: NDArray,
    size: int | tuple[int, ...],
    mask: NDArray | None = None,
    mode: str = "reflect",
) -> NDArray:
    """Median filter with optional mask support.

    When mask is None, delegates to scipy.ndimage.median_filter (C-speed).
    When mask is provided, only voxels where mask is nonzero contribute
    to the median within each filter window.

    Parameters
    ----------
    data : NDArray
        Input array to filter.
    size : int or tuple of int
        Filter kernel size. Scalar applies to all dimensions.
    mask : NDArray or None, optional
        Boolean or integer mask with same shape as data. Nonzero entries
        mark voxels that participate in the median calculation. If None,
        all voxels participate (standard median filter).
    mode : str, optional
        Padding mode matching scipy.ndimage conventions: 'reflect',
        'nearest', 'constant', 'wrap', or 'mirror'. Default is 'reflect'.

    Returns
    -------
    NDArray
        Filtered array with same shape as data, dtype float64.
    """
    if mask is None:
        return ndimage.median_filter(data, size=size, mode=mode)

    if np.isscalar(size):
        kernel_shape = (int(size),) * data.ndim
    else:
        kernel_shape = tuple(int(s) for s in size)

    np_pad_mode = _NDIMAGE_TO_NUMPY_PAD_MODE.get(mode, mode)
    windows, mask_windows = _pad_and_view(data, mask, kernel_shape, np_pad_mode)

    n_voxels = int(np.prod(data.shape))
    n_kernel = int(np.prod(kernel_shape))

    # Process in chunks to cap memory at ~256 MiB
    chunk_size = max(1, _MAX_CHUNK_BYTES // (n_kernel * 8))
    if chunk_size >= n_voxels:
        masked = np.where(
            mask_windows.reshape(n_voxels, n_kernel),
            windows.reshape(n_voxels, n_kernel),
            np.nan,
        )
        result_flat = np.nanmedian(masked, axis=1)
    else:
        result_flat = np.empty(n_voxels, dtype=np.float64)
        for start in range(0, n_voxels, chunk_size):
            end = min(start + chunk_size, n_voxels)
            result_flat[start:end] = _nanmedian_chunk(windows, mask_windows, n_kernel, start, end)

    return result_flat.reshape(data.shape)


def recorddelaymetrics(
    optiondict: dict,
    thepass: Any,
    lagtimes: NDArray,
    fitmask: NDArray,
    validvoxels: NDArray,
    nativespaceshape: tuple,
    numspatiallocs: int,
    despeckle_thresh: float,
    preresolvelags: Optional[NDArray] = None,
    corrout: Optional[NDArray] = None,
    corrscale: Optional[NDArray] = None,
    LGR: Optional[Any] = None,
) -> None:
    """Record delay map quality metrics for this pass into optiondict.

    Every optimisation decision in the delay repair work was made by recomputing
    these numbers offline from the saved maps, which needs the maps shipped around
    and takes minutes per cohort.  They cost a couple of seconds to compute here,
    where the data is already in memory, and land in the runoptions where a whole
    cohort can be tabulated by reading one small JSON per run.

    Metrics recorded, all suffixed ``_pass<N>``:

    map quality
        ``delayoutliers`` / ``delayoutlierfrac``
            voxels further than despeckle_thresh from their local median.  This is
            the headline number the repair comparisons are scored on.
        ``delayjumps`` / ``delayjumpfrac``
            adjacent in-mask voxel pairs differing by more than METRICJUMPTHRESH.
            The damage check: a smoother that flattens real gradients improves the
            outlier count and worsens this one.

    distribution
        ``delayp5`` ... ``delayp99``, ``delayiqr``
            where the delays actually are.  Needed to interpret everything else,
            and free.
        ``delayrailedfrac``
            fraction sitting within one lag step of the search range bounds.  A
            delay pinned at the bound and a genuinely long delay are
            indistinguishable downstream, so it matters whether a cohort is
            hitting the wall.

    long delay structure
        ``delaylongcut`` / ``delaylongfrac`` / ``delaylongcoherent`` /
        ``delaylonglargest`` / ``delaylongdev``
            how much long-delay signal there is, whether it forms connected
            structures (plausibly real) or scattered specks (plausibly error), and
            how far it deviates locally.  This is the cohort characterisation that
            decides whether a smoothness prior is safe to apply.

    resolution effect, only when delay resolution ran this pass
        ``resolveshiftmedian`` / ``resolveshiftposfrac``
            size and sidedness of the changes.  A genuine sidelobe wrap moves
            delays in ONE direction by close to one period; noise repicking moves
            them both ways over a broad range.  This distinguishes them without a
            paired run.
        ``resolveampratio``
            median ratio of similarity at the new lag to similarity at the old one.
            Resolution usually moves to a LOWER peak - that is expected, since the
            wrong peak is generally the taller one - but a collapse here means
            something else is going on.
        ``resolvechgcoherentfrac`` / ``resolvechgscatteredfrac`` / ``resolveselectivity``
            of the long-delay voxels going into resolution, the fraction reassigned
            in coherent blobs versus scattered specks, and their ratio.  A
            selectivity near 1 means resolution cannot tell real long-delay
            structure from a fitting error.  Note this is the single-run analogue
            of the paired-arm measurement: the map going in has already been
            resolved on earlier passes, so it understates the cumulative effect.
    """
    try:
        thesuffix = f"_pass{thepass}"

        themap = np.zeros(numspatiallocs, dtype=float)
        themap[validvoxels] = lagtimes[:]
        themap = themap.reshape(nativespaceshape)
        themask = np.zeros(numspatiallocs, dtype=bool)
        themask[validvoxels] = fitmask[:].astype(bool)
        themask = themask.reshape(nativespaceshape)
        thenummask = int(themask.sum())
        if thenummask < 100:
            return

        # Nearest-in-mask extension, not zero fill.  Zero filling would drag the
        # local median of every surface voxel toward zero and inflate the outlier
        # count there - the same trap that was found in icmrefine.
        theindices = maskextensionindices(themask)
        thelocalmedian = ndimage.median_filter(
            themap[theindices].astype(np.float32), size=METRICMEDIANKERNEL, mode="nearest"
        )
        thedeviation = np.abs(themap - thelocalmedian)
        thenumoutliers = int((themask & (thedeviation > despeckle_thresh)).sum())
        optiondict["delayoutliers" + thesuffix] = thenumoutliers
        optiondict["delayoutlierfrac" + thesuffix] = thenumoutliers / thenummask

        thenumjumps = 0
        thenumpairs = 0
        for theaxis in range(3):
            thediff = np.abs(np.diff(themap, axis=theaxis))
            theboth = np.take(
                themask, range(0, themask.shape[theaxis] - 1), axis=theaxis
            ) & np.take(themask, range(1, themask.shape[theaxis]), axis=theaxis)
            thenumjumps += int((thediff[theboth] > METRICJUMPTHRESH).sum())
            thenumpairs += int(theboth.sum())
        optiondict["delayjumps" + thesuffix] = thenumjumps
        optiondict["delayjumpfrac" + thesuffix] = thenumjumps / max(thenumpairs, 1)

        thevalues = themap[themask]
        theq = np.percentile(thevalues, [5, 25, 50, 75, 95, 99])
        for thename, theval in zip(
            ("delayp5", "delayp25", "delayp50", "delayp75", "delayp95", "delayp99"), theq
        ):
            optiondict[thename + thesuffix] = float(theval)
        theiqr = float(theq[3] - theq[1])
        optiondict["delayiqr" + thesuffix] = theiqr

        thelagmin = optiondict.get("lagmin", None)
        thelagmax = optiondict.get("lagmax", None)
        if thelagmin is not None and thelagmax is not None and corrscale is not None:
            thestep = (
                float(corrscale[1] - corrscale[0]) if len(corrscale) > 1 else 0.0
            )
            optiondict["delayrailedfrac" + thesuffix] = float(
                np.mean(
                    (thevalues <= thelagmin + thestep) | (thevalues >= thelagmax - thestep)
                )
            )

        thelongcut = float(theq[2] + METRICLONGIQRMULT * theiqr)
        thelong = themask & (themap > thelongcut)
        thenumlong = int(thelong.sum())
        optiondict["delaylongcut" + thesuffix] = thelongcut
        optiondict["delaylongfrac" + thesuffix] = thenumlong / thenummask
        if thenumlong > 0:
            thelabels, dummy = ndimage.label(thelong)
            thesizes = np.bincount(thelabels.ravel())
            thesizes[0] = 0
            thecoherent = thelong & np.isin(
                thelabels, np.nonzero(thesizes >= METRICMINCOMPONENT)[0]
            )
            optiondict["delaylongcoherent" + thesuffix] = float(
                thecoherent.sum() / thenumlong
            )
            optiondict["delaylonglargest" + thesuffix] = int(thesizes.max())
            optiondict["delaylongdev" + thesuffix] = float(np.median(thedeviation[thelong]))
        else:
            optiondict["delaylongcoherent" + thesuffix] = 0.0
            optiondict["delaylonglargest" + thesuffix] = 0
            optiondict["delaylongdev" + thesuffix] = 0.0

        if preresolvelags is None:
            return

        thebefore = np.zeros(numspatiallocs, dtype=float)
        thebefore[validvoxels] = preresolvelags[:]
        thebefore = thebefore.reshape(nativespaceshape)
        theshift = themap - thebefore
        thechanged = themask & (np.abs(theshift) > RESOLVEDCHANGEDTHRESH)
        if not np.any(thechanged):
            return
        optiondict["resolveshiftmedian" + thesuffix] = float(
            np.median(np.abs(theshift[thechanged]))
        )
        optiondict["resolveshiftposfrac" + thesuffix] = float(
            np.mean(theshift[thechanged] > 0.0)
        )

        # long-delay classes defined on the map going IN, then tracked through
        thebeforevals = thebefore[themask]
        thebq = np.percentile(thebeforevals, [25, 50, 75])
        thebeforecut = thebq[1] + METRICLONGIQRMULT * (thebq[2] - thebq[0])
        thebeforelong = themask & (thebefore > thebeforecut)
        if np.any(thebeforelong):
            thelabels, dummy = ndimage.label(thebeforelong)
            thesizes = np.bincount(thelabels.ravel())
            thesizes[0] = 0
            thecoh = thebeforelong & np.isin(
                thelabels, np.nonzero(thesizes >= METRICMINCOMPONENT)[0]
            )
            thescat = thebeforelong & ~thecoh
            thecohfrac = float(thechanged[thecoh].mean()) if np.any(thecoh) else np.nan
            thescatfrac = float(thechanged[thescat].mean()) if np.any(thescat) else np.nan
            optiondict["resolvechgcoherentfrac" + thesuffix] = thecohfrac
            optiondict["resolvechgscatteredfrac" + thesuffix] = thescatfrac
            # always recorded, None when the ratio is undefined, so that a tabulation
            # can tell "no coherent long delay to judge" apart from "key missing"
            if thecohfrac and np.isfinite(thecohfrac) and np.isfinite(thescatfrac):
                optiondict["resolveselectivity" + thesuffix] = thescatfrac / thecohfrac
            else:
                optiondict["resolveselectivity" + thesuffix] = None

        # similarity amplitude at the new lag versus the old one, for changed voxels
        if corrout is not None and corrscale is not None and len(corrscale) > 1:
            thechangedflat = thechanged.reshape(-1)[validvoxels]
            if np.any(thechangedflat):
                theidx = np.nonzero(thechangedflat)[0]
                theold = np.array(
                    [
                        np.interp(preresolvelags[i], corrscale, corrout[i, :])
                        for i in theidx
                    ]
                )
                thenew = np.array(
                    [np.interp(lagtimes[i], corrscale, corrout[i, :]) for i in theidx]
                )
                thevalid = np.abs(theold) > 1.0e-9
                if np.any(thevalid):
                    optiondict["resolveampratio" + thesuffix] = float(
                        np.median(thenew[thevalid] / theold[thevalid])
                    )
    except Exception as thereason:
        # instrumentation must never take down a run that is otherwise fine
        if LGR is not None:
            LGR.warning(f"\n*** delay metrics for pass {thepass} FAILED: {thereason} ***")


def fitSimFunc(
    fmri_data_valid: NDArray[np.floating[Any]],
    validsimcalcstart: int,
    validsimcalcend: int,
    osvalidsimcalcstart: int,
    osvalidsimcalcend: int,
    initial_fmri_x: NDArray[np.floating[Any]],
    os_fmri_x: NDArray[np.floating[Any]],
    theMutualInformationator: Any,
    cleaned_referencetc: Any,
    corrout: NDArray[np.floating[Any]],
    outputname: str,
    validvoxels: Any,
    nativespaceshape: Any,
    bidsbasedict: Any,
    numspatiallocs: Any,
    gaussout: Any,
    theinitialdelay: Any,
    windowout: Any,
    R2: Any,
    thesizes: Any,
    internalspaceshape: Any,
    numvalidspatiallocs: Any,
    theinputdata: Any,
    theheader: Any,
    theFitter: Any,
    fitmask: Any,
    lagtimes: Any,
    lagstrengths: Any,
    lagsigma: Any,
    failreason: Any,
    outmaparray: Any,
    trimmedcorrscale: Any,
    similaritytype: Any,
    thepass: Any,
    optiondict: Any,
    LGR: Any,
    TimingLGR: Any,
    simplefit: bool = False,
    upsampfac: int = 8,
    rt_floattype: np.dtype = np.float64,
) -> NDArray | None:
    """
    Perform similarity function fitting and time lag estimation for fMRI data.

    This function conducts either a simple or full fitting process for estimating time lags
    between fMRI signals and a reference time course. It supports hybrid similarity metrics
    and includes optional despeckling and patch shifting steps.

    Parameters
    ----------
    fmri_data_valid : NDArray[np.floating[Any]]
        Valid fMRI data for processing.
    validsimcalcstart : int
        Start index for valid similarity calculation.
    validsimcalcend : int
        End index for valid similarity calculation.
    osvalidsimcalcstart : int
        Start index for oversampled valid similarity calculation.
    osvalidsimcalcend : int
        End index for oversampled valid similarity calculation.
    initial_fmri_x : NDArray[np.floating[Any]]
        Initial fMRI x values.
    os_fmri_x : NDArray[np.floating[Any]]
        Oversampled fMRI x values.
    theMutualInformationator : object
        Mutual information calculator.
    cleaned_referencetc : array_like
        Cleaned reference time course.
    corrout : NDArray[np.floating[Any]]
        Correlation output array.
    outputname : str
        Output filename prefix.
    validvoxels : array_like
        Indices of valid voxels.
    nativespaceshape : tuple
        Native space shape of the data.
    bidsbasedict : dict
        BIDS-based dictionary for output metadata.
    numspatiallocs : int
        Number of spatial locations.
    gaussout : array_like
        Gaussian output array.
    theinitialdelay : float
        Initial delay value.
    windowout : array_like
        Window output array.
    R2 : array_like
        R-squared values.
    thesizes : array_like
        Sizes for processing.
    internalspaceshape : tuple
        Internal space shape.
    numvalidspatiallocs : int
        Number of valid spatial locations.
    theinputdata : object
        Input data object.
    theheader : dict
        Header information.
    theFitter : object
        Fitter object for similarity function fitting.
    fitmask : array_like
        Mask for fitting.
    lagtimes : array_like
        Array to store estimated lag times.
    lagstrengths : array_like
        Array to store lag strengths.
    lagsigma : array_like
        Array to store sigma values for lags.
    failreason : array_like
        Array to store failure reasons.
    outmaparray : array_like
        Output map array.
    trimmedcorrscale : array_like
        Trimmed correlation scale.
    similaritytype : str
        Type of similarity metric used.
    thepass : int
        Current pass number.
    optiondict : dict
        Dictionary of options for processing.
    LGR : object
        Logger for general messages.
    TimingLGR : object
        Logger for timing information.
    simplefit : bool, optional
        If True, perform simple fitting using upsampling. Default is False.
    upsampfac : int, optional
        Upsampling factor for simple fitting. Default is 8.
    rt_floattype : np.dtype, optional
        Real-time floating-point data type. Default is np.float64.

    Returns
    -------
    internaldespeckleincludemask : NDArray[np.floating[Any]] or None
        Mask indicating which voxels were included in despeckling, or None if no despeckling was performed.

    Notes
    -----
    - This function supports both simple and hybrid similarity metrics.
    - Despeckling and patch shifting steps are optional and controlled by `optiondict`.
    - The function modifies `lagtimes`, `lagstrengths`, `lagsigma`, and `fitmask` in-place.

    Examples
    --------
    >>> fitSimFunc(
    ...     fmri_data_valid,
    ...     validsimcalcstart,
    ...     validsimcalcend,
    ...     osvalidsimcalcstart,
    ...     osvalidsimcalcend,
    ...     initial_fmri_x,
    ...     os_fmri_x,
    ...     theMutualInformationator,
    ...     cleaned_referencetc,
    ...     corrout,
    ...     outputname,
    ...     validvoxels,
    ...     nativespaceshape,
    ...     bidsbasedict,
    ...     numspatiallocs,
    ...     gaussout,
    ...     theinitialdelay,
    ...     windowout,
    ...     R2,
    ...     thesizes,
    ...     internalspaceshape,
    ...     numvalidspatiallocs,
    ...     theinputdata,
    ...     theheader,
    ...     theFitter,
    ...     fitmask,
    ...     lagtimes,
    ...     lagstrengths,
    ...     lagsigma,
    ...     failreason,
    ...     outmaparray,
    ...     trimmedcorrscale,
    ...     similaritytype,
    ...     thepass,
    ...     optiondict,
    ...     LGR,
    ...     TimingLGR,
    ...     simplefit=False,
    ...     upsampfac=8,
    ...     rt_floattype="float64",
    ... )
    """
    # Do a peak prefit if doing hybrid
    if optiondict["similaritymetric"] == "hybrid":
        LGR.info(f"\n\nPeak prefit calculation, pass {thepass}")
        TimingLGR.info(f"Peak prefit calculation start, pass {thepass}")

        tide_util.disablemkl(optiondict["nprocs_peakeval"], debug=optiondict["threaddebug"])
        voxelsprocessed_pe, thepeakdict = tide_peakeval.peakevalpass(
            fmri_data_valid[:, validsimcalcstart : validsimcalcend + 1],
            cleaned_referencetc,
            initial_fmri_x[validsimcalcstart : validsimcalcend + 1],
            os_fmri_x[osvalidsimcalcstart : osvalidsimcalcend + 1],
            theMutualInformationator,
            trimmedcorrscale,
            corrout,
            nprocs=optiondict["nprocs_peakeval"],
            alwaysmultiproc=optiondict["alwaysmultiproc"],
            bipolar=optiondict["bipolar"],
            oversampfactor=optiondict["oversampfactor"],
            interptype=optiondict["interptype"],
            showprogressbar=optiondict["showprogressbar"],
            chunksize=optiondict["mp_chunksize"],
            rt_floattype=rt_floattype,
        )
        tide_util.enablemkl(optiondict["mklthreads"], debug=optiondict["threaddebug"])

        TimingLGR.info(
            f"Peak prefit end, pass {thepass}",
            {
                "message2": voxelsprocessed_pe,
                "message3": "voxels",
            },
        )
        mipeaks = np.zeros_like(lagtimes)
        for i in range(numvalidspatiallocs):
            if len(thepeakdict[str(i)]) > 0:
                mipeaks[i] = thepeakdict[str(i)][0][0]
    else:
        thepeakdict = None

    if simplefit:
        basedelay = trimmedcorrscale[0]
        delaystep = (trimmedcorrscale[1] - trimmedcorrscale[0]) / upsampfac
        for thevox in range(numvalidspatiallocs):
            fitmask[thevox] = 1
            upsampcorrout = tide_resample.upsample(
                corrout[thevox, :], 1, upsampfac, intfac=True, dofilt=False
            )
            if optiondict["bipolar"]:
                thismax = np.argmax(np.fabs(upsampcorrout))
            else:
                thismax = np.argmax(upsampcorrout)
            lagtimes[thevox] = basedelay + thismax * delaystep
            lagstrengths[thevox] = upsampcorrout[thismax]
            lagsigma[thevox] = 1.0
        internaldespeckleincludemask = None
    else:
        # Similarity function fitting and time lag estimation
        # write out the current version of the run options
        optiondict["currentstage"] = f"presimfuncfit_pass{thepass}"
        tide_io.writedicttojson(optiondict, f"{outputname}_desc-runoptions_info.json")
        LGR.info(f"\n\nTime lag estimation pass {thepass}")
        TimingLGR.info(f"Time lag estimation start, pass {thepass}")

        theFitter.setfunctype(optiondict["similaritymetric"])
        theFitter.setcorrtimeaxis(trimmedcorrscale)

        # use initial lags if this is a hybrid fit
        if optiondict["similaritymetric"] == "hybrid" and thepeakdict is not None:
            initlags = mipeaks
        else:
            initlags = None

        tide_util.disablemkl(optiondict["nprocs_fitcorr"], debug=optiondict["threaddebug"])
        voxelsprocessed_fc = tide_simfuncfit.fitcorr(
            trimmedcorrscale,
            theFitter,
            corrout,
            fitmask,
            failreason,
            lagtimes,
            lagstrengths,
            lagsigma,
            gaussout,
            windowout,
            R2,
            despeckling=False,
            nprocs=optiondict["nprocs_fitcorr"],
            alwaysmultiproc=optiondict["alwaysmultiproc"],
            fixdelay=optiondict["fixdelay"],
            initialdelayvalue=theinitialdelay,
            showprogressbar=optiondict["showprogressbar"],
            chunksize=optiondict["mp_chunksize"],
            despeckle_thresh=optiondict["despeckle_thresh"],
            initiallags=initlags,
            rt_floattype=rt_floattype,
        )
        tide_util.enablemkl(optiondict["mklthreads"], debug=optiondict["threaddebug"])

        TimingLGR.info(
            f"Time lag estimation end, pass {thepass}",
            {
                "message2": voxelsprocessed_fc,
                "message3": "voxels",
            },
        )

        # Sidelobe unwrapping.
        #
        # When the similarity function has a strong sidelobe, peak picking sometimes
        # lands on it, giving a delay wrong by very nearly one sidelobe period.  These
        # errors are large and, because neighboring voxels see the same waveform, they
        # occur in coherent patches, which a median filter based despeckler struggles
        # with because the neighbors it votes with are wrong too.
        #
        # This is phase unwrapping: the absolute delay is ambiguous modulo the sidelobe
        # period while the local gradient is not.  Each voxel offers candidate delays
        # (the local maxima of its similarity function) and quality guided region
        # growing assigns each the candidate closest to a consensus prediction from its
        # already assigned neighbors.
        #
        # UNCONDITIONAL when requested, and the reason is worth recording because the
        # opposite was assumed for a long time.
        #
        # The original design gated on the measured sidelobe amplitude, on the argument
        # that with no sidelobe there is no periodic ambiguity to resolve and the method
        # degenerates into plain smoothing.  A calibration of 16 HCP runs paired with
        # controls, deliberately spanning the ambiguity range including its bottom
        # quartile, refuted that: resolution improved the delay map in 16 of 16 runs,
        # including all four runs with the LOWEST measurable ambiguity and no sidelobe
        # at all.  A later 26 run UK Biobank calibration agreed, 26 of 26.  Large
        # non-periodic discontinuities fell in every run too, so the gain is not
        # smoothing away real structure.
        #
        # The mechanism is also not what the name suggests.  Across those runs the median
        # change was 1.3 s and only 36% of changes were positive - two sided and small,
        # not the one sided ~one period shift that a genuine sidelobe wrap produces.
        # Only the two runs with a real measured sidelobe showed the periodic signature.
        # So this is a general delay map repair that handles sidelobe wrapping as one
        # special case, and gating it to fire only in that special case was suppressing
        # most of its value.  The gate, and the latch that existed to stop the gate
        # closing again mid-run, are both gone.
        #
        # It runs BEFORE despeckling, and both run.  The order matters and is not
        # symmetric.  Despeckle first then resolve is worse than resolving alone,
        # because resolution can only select among local maxima of the similarity
        # function and so silently discards despeckle's refit values wherever those are
        # not peaks.  Resolve first then despeckle is better than either alone:
        # resolution makes the discrete lobe choice, then despeckle refits the residual
        # local outliers, which resolution cannot do because it cannot produce a
        # non-peak value.  Measured on paired arms, despeckling on top of resolution
        # removed a further 34% of outlier voxels in 16 of 16 HCP runs and 18% in 26 of
        # 26 UK Biobank runs.
        thepreresolvelags = None
        dounwrap = bool(optiondict.get("resolvedelays", False))
        if dounwrap:
            LGR.info(f"\n\n{similaritytype} delay resolution pass {thepass}")
            # Log the sidelobe measurement if there is one.  It no longer decides
            # anything, but it is the natural thing to want alongside the reassignment
            # count, and it is frequently absent - nothing here may assume it is a number.
            theacsidelobeamp = optiondict.get(f"acsidelobeamp_pass{thepass}", None)
            thelag = optiondict.get(f"acsidelobelag_pass{thepass}", None)
            if theacsidelobeamp is not None:
                LGR.info(
                    f"\tsidelobe amplitude {theacsidelobeamp:.3f}"
                    + (f" at {thelag:.3f}s" if thelag is not None else "")
                )
            else:
                LGR.info("\tno sidelobe amplitude measured for this pass")
            TimingLGR.info(f"{similaritytype} delay resolution start, pass {thepass}")
            thexdim, theydim, theslicethickness, dummy = tide_io.parseniftisizes(thesizes)
            # captured unconditionally now: the metrics recorded at the end of the
            # pass need it, not just --saveresolvemaps.  It is a 1-D copy.
            thepreresolvelags = lagtimes.copy()
            lagtimes[:], numunwrapped = resolvefromsimfunc(
                corrout,
                trimmedcorrscale,
                validvoxels,
                nativespaceshape,
                fitmask,
                lagtimes,
                (thexdim, theydim, theslicethickness),
                numpasses=optiondict.get("resolvepasses", 3),
                showprogressbar=optiondict["showprogressbar"],
            )
            optiondict[f"resolved_pass{thepass}"] = numunwrapped
            LGR.info(f"\tresolution reassigned {numunwrapped} voxels")

            # Audit trail.  Delay resolution can move a voxel to a different peak of the
            # similarity function, which is the point, but it is a smoothness prior and
            # a smoothness prior cannot tell a genuinely long delay that varies smoothly
            # from its surroundings apart from a fitting error.  Saving what it changed
            # makes that inspectable instead of invisible.
            #
            # What this is NOT: resolution runs on every pass and the result feeds the
            # next pass's refinement, so the map going into the final pass has already
            # been resolved.  These outputs therefore show what the FINAL pass did, not
            # the cumulative effect of resolution versus not resolving at all.  Nothing
            # computed inside a single run can show the latter - that needs a paired run
            # without --resolvedelays.  Per-pass reassignment counts are in runoptions as
            # resolved_passN.
            if optiondict.get("saveresolvemaps", False) and thepass == optiondict["passes"]:
                if theinputdata.filetype != "text":
                    if theinputdata.filetype == "cifti":
                        timeindex = theheader["dim"][0] - 1
                        spaceindex = theheader["dim"][0]
                        theheader["dim"][timeindex] = 1
                        theheader["dim"][spaceindex] = numspatiallocs
                    else:
                        theheader["dim"][0] = 3
                        theheader["dim"][4] = 1
                        theheader["pixdim"][4] = 1.0
                theshift = lagtimes - thepreresolvelags
                resolvelist = [
                    (
                        thepreresolvelags,
                        "maxtimepreresolve",
                        "map",
                        "second",
                        "Lag time in seconds as it stood immediately before delay "
                        "resolution in the final pass",
                    ),
                    (
                        theshift,
                        "resolveshift",
                        "map",
                        "second",
                        "Change in lag time applied by delay resolution in the final "
                        "pass, in seconds",
                    ),
                    (
                        np.where(np.abs(theshift) > RESOLVEDCHANGEDTHRESH, 1, 0).astype(
                            np.int32
                        ),
                        "resolvechanged",
                        "mask",
                        None,
                        f"Voxels whose delay was changed by more than "
                        f"{RESOLVEDCHANGEDTHRESH} seconds by delay resolution in the "
                        f"final pass",
                    ),
                ]
                tide_io.savemaplist(
                    outputname,
                    resolvelist,
                    validvoxels,
                    nativespaceshape,
                    theheader,
                    bidsbasedict,
                    filetype=theinputdata.filetype,
                    rt_floattype=rt_floattype,
                    cifti_hdr=theinputdata.cifti_hdr,
                )

            TimingLGR.info(
                f"{similaritytype} delay resolution end, pass {thepass}",
                {"message2": numunwrapped, "message3": "voxels"},
            )

        # Correlation time despeckle
        if optiondict["despeckle_passes"] > 0:
            LGR.info(f"\n\n{similaritytype} despeckling pass {thepass}")
            LGR.info(f"\tUsing despeckle_thresh = {optiondict['despeckle_thresh']:.3f}")
            TimingLGR.info(f"{similaritytype} despeckle start, pass {thepass}")

            # find lags that are very different from their neighbors, and refit starting at the median lag for the point
            voxelsprocessed_fc_ds = 0
            despecklingdone = False
            lastnumdespeckled = 1000000
            for despecklepass in range(optiondict["despeckle_passes"]):
                kernel_size = optiondict["despeckle_kernel_size"]
                LGR.info(
                    f"\n\n{similaritytype} despeckling, pass{thepass}, subpass {despecklepass + 1} "
                    f"(kernel={kernel_size})"
                )
                outmaparray *= 0.0
                outmaparray[validvoxels] = lagtimes[:]

                # find voxels to despeckle
                medianmask = outmaparray * 0.0
                medianmask[validvoxels] = fitmask[:]
                medianmask = medianmask.reshape(nativespaceshape)
                medianlags = masked_median_filter(
                    outmaparray.reshape(nativespaceshape), size=kernel_size, mask=medianmask
                ).reshape(numspatiallocs)
                # voxels that we're happy with have initlags set to -1000000.0
                initlags = np.where(
                    np.abs(outmaparray - medianlags) > optiondict["despeckle_thresh"],
                    medianlags,
                    -1000000.0,
                )[validvoxels]

                if len(initlags) > 0:
                    numdespeckled = len(np.where(initlags != -1000000.0)[0])
                    # convergence guard: stop once the flagged count stops falling
                    if lastnumdespeckled > numdespeckled > 0:
                        lastnumdespeckled = numdespeckled
                        tide_util.disablemkl(
                            optiondict["nprocs_fitcorr"], debug=optiondict["threaddebug"]
                        )
                        voxelsprocessed_thispass = tide_simfuncfit.fitcorr(
                            trimmedcorrscale,
                            theFitter,
                            corrout,
                            fitmask,
                            failreason,
                            lagtimes,
                            lagstrengths,
                            lagsigma,
                            gaussout,
                            windowout,
                            R2,
                            despeckling=True,
                            nprocs=optiondict["nprocs_fitcorr"],
                            alwaysmultiproc=optiondict["alwaysmultiproc"],
                            fixdelay=optiondict["fixdelay"],
                            initialdelayvalue=theinitialdelay,
                            showprogressbar=optiondict["showprogressbar"],
                            chunksize=optiondict["mp_chunksize"],
                            despeckle_thresh=optiondict["despeckle_thresh"],
                            initiallags=initlags,
                            rt_floattype=rt_floattype,
                        )
                        tide_util.enablemkl(
                            optiondict["mklthreads"], debug=optiondict["threaddebug"]
                        )
                        # restore full lag range (THANK YOU to Suchita Ganesan for finding and fixing this!!!)
                        theFitter.setrange(optiondict["lagmin"], optiondict["lagmax"])

                        voxelsprocessed_fc_ds += voxelsprocessed_thispass
                        optiondict[
                            "despecklemasksize_pass" + str(thepass) + "_d" + str(despecklepass + 1)
                        ] = voxelsprocessed_thispass
                        optiondict[
                            "despecklemaskpct_pass" + str(thepass) + "_d" + str(despecklepass + 1)
                        ] = (100.0 * voxelsprocessed_thispass / optiondict["corrmasksize"])
                        if optiondict["savedespecklemasks"]:
                            despecklesavemask = np.where(initlags != -1000000.0, 1, 0)
                            despeckleinitlags = np.where(initlags != -1000000.0, initlags, 0)
                            if thepass == optiondict["passes"]:
                                if theinputdata.filetype != "text":
                                    if theinputdata.filetype == "cifti":
                                        timeindex = theheader["dim"][0] - 1
                                        spaceindex = theheader["dim"][0]
                                        theheader["dim"][timeindex] = 1
                                        theheader["dim"][spaceindex] = numspatiallocs
                                    else:
                                        theheader["dim"][0] = 3
                                        theheader["dim"][4] = 1
                                        theheader["pixdim"][4] = 1.0
                                masklist = [
                                    (
                                        despecklesavemask,
                                        f"despeckle_p{thepass}_d{despecklepass + 1}",
                                        "mask",
                                        None,
                                        "Voxels that underwent despeckling",
                                    ),
                                    (
                                        despeckleinitlags,
                                        f"despeckleinitlags_p{thepass}_d{despecklepass + 1}",
                                        "map",
                                        None,
                                        "Target lags for voxels that underwent despeckling",
                                    ),
                                    (
                                        medianlags[validvoxels],
                                        f"despecklemedianlags_p{thepass}_d{despecklepass + 1}",
                                        "map",
                                        None,
                                        "Median filter targets for despeckling",
                                    ),
                                ]
                                tide_io.savemaplist(
                                    outputname,
                                    masklist,
                                    validvoxels,
                                    nativespaceshape,
                                    theheader,
                                    bidsbasedict,
                                    filetype=theinputdata.filetype,
                                    rt_floattype=rt_floattype,
                                    cifti_hdr=theinputdata.cifti_hdr,
                                )
                    else:
                        despecklingdone = True
                else:
                    despecklingdone = True
                if despecklingdone:
                    LGR.info("Nothing left to do! Terminating despeckling")
                    break

            internaldespeckleincludemask = np.where(
                np.abs(outmaparray - medianlags) > optiondict["despeckle_thresh"],
                medianlags,
                0.0,
            )
            if optiondict["savedespecklemasks"] and (optiondict["despeckle_passes"] > 0):
                despecklesavemask = np.where(
                    internaldespeckleincludemask[validvoxels] == 0.0, 0, 1
                )
                if thepass == optiondict["passes"]:
                    if theinputdata.filetype != "text":
                        if theinputdata.filetype == "cifti":
                            timeindex = theheader["dim"][0] - 1
                            spaceindex = theheader["dim"][0]
                            theheader["dim"][timeindex] = 1
                            theheader["dim"][spaceindex] = numspatiallocs
                        else:
                            theheader["dim"][0] = 3
                            theheader["dim"][4] = 1
                            theheader["pixdim"][4] = 1.0
                    masklist = [
                        (
                            despecklesavemask,
                            "despeckle",
                            "mask",
                            None,
                            "Voxels that underwent despeckling in the final pass",
                        )
                    ]
                    tide_io.savemaplist(
                        outputname,
                        masklist,
                        validvoxels,
                        nativespaceshape,
                        theheader,
                        bidsbasedict,
                        filetype=theinputdata.filetype,
                        rt_floattype=rt_floattype,
                        cifti_hdr=theinputdata.cifti_hdr,
                    )
            LGR.info(
                f"\n\n{voxelsprocessed_fc_ds} voxels despeckled in "
                f"{optiondict['despeckle_passes']} passes"
            )
            TimingLGR.info(
                f"{similaritytype} despeckle end, pass {thepass}",
                {
                    "message2": voxelsprocessed_fc_ds,
                    "message3": "voxels",
                },
            )
        else:
            internaldespeckleincludemask = None

        # End of pass.  Record what the delay map looks like now, after both repairs.
        recorddelaymetrics(
            optiondict,
            thepass,
            lagtimes,
            fitmask,
            validvoxels,
            nativespaceshape,
            numspatiallocs,
            optiondict["despeckle_thresh"],
            preresolvelags=thepreresolvelags,
            corrout=corrout,
            corrscale=trimmedcorrscale,
            LGR=LGR,
        )

    return internaldespeckleincludemask
