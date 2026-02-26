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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

import rapidtide.fit as tide_fit
import rapidtide.io as tide_io
import rapidtide.patchmatch as tide_patch
import rapidtide.peakeval as tide_peakeval
import rapidtide.resample as tide_resample
import rapidtide.simfuncfit as tide_simfuncfit
import rapidtide.util as tide_util

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


def _build_peakdict_for_candidates(
    candidate_mask_valid: NDArray[np.bool_],
    corrout: NDArray[np.floating[Any]],
    trimmedcorrscale: NDArray[np.floating[Any]],
    bipolar: bool = False,
) -> dict[str, list[list[float]]]:
    """Build a peak dictionary for candidate (flagged) voxels.

    For each candidate voxel, finds all peaks in its correlation function
    and returns them in the same format as peakevalpass(): {str(vox_idx): [[lag, strength, strength], ...]}.
    """
    peakdict: dict[str, list[list[float]]] = {}
    for vox_idx in np.where(candidate_mask_valid)[0]:
        peaks = tide_fit.getpeaks(trimmedcorrscale, corrout[vox_idx, :], bipolar=bipolar)
        # Convert to peakdict format: [lag, strength, strength]
        peakdict[str(vox_idx)] = [[p[0], p[1], abs(p[1])] for p in peaks]
    return peakdict


def _detect_shifted_patches(
    lagmap_3d: NDArray,
    validmask_3d: NDArray[np.bool_],
    despeckle_thresh: float,
    reference_kernel: int = 9,
    min_patch_size: int = 10,
) -> tuple[NDArray[np.bool_], NDArray]:
    """Detect connected patches of shifted delay values.

    After initial median-filter despeckling removes isolated speckles, large
    patches of wrong-peak selections survive because they fool the small
    median filter.  This function detects them by comparing each voxel to a
    heavily smoothed reference computed with a much larger kernel, then
    labeling connected components of outliers.

    Parameters
    ----------
    lagmap_3d : NDArray
        Lag map in native 3D space.
    validmask_3d : NDArray[np.bool_]
        Boolean mask of valid (fitted) voxels, same shape as lagmap_3d.
    despeckle_thresh : float
        Deviation threshold for flagging voxels.
    reference_kernel : int, optional
        Size of the median filter kernel used to build the large-scale
        reference.  Must be odd.  Default is 9.
    min_patch_size : int, optional
        Minimum number of connected voxels to be considered a patch.
        Smaller clusters are ignored (already handled by regular despeckle).
        Default is 10.

    Returns
    -------
    patch_mask : NDArray[np.bool_]
        Boolean mask (same shape as lagmap_3d) where True marks voxels
        belonging to a detected patch.
    reference : NDArray
        The large-kernel-smoothed reference lag map.
    """
    # Build reference with large median filter — robust to patches as long as
    # the patch is smaller than half the kernel volume.
    reference = masked_median_filter(
        np.where(validmask_3d, lagmap_3d, 0.0),
        size=reference_kernel,
        mode="reflect",
        mask=validmask_3d,
    )
    #reference = ndimage.median_filter(
    #    np.where(validmask_3d, lagmap_3d, 0.0),
    #    size=reference_kernel,
    #    mode="reflect",
    #

    # Find voxels that deviate from the large-scale reference
    deviation = np.abs(lagmap_3d - reference)
    outlier_mask = validmask_3d & (deviation > despeckle_thresh)

    # Label connected components with 26-connectivity
    structure = ndimage.generate_binary_structure(lagmap_3d.ndim, lagmap_3d.ndim)
    labels, n_patches = ndimage.label(outlier_mask, structure=structure)

    # Keep only patches above minimum size
    if n_patches > 0:
        sizes = ndimage.sum_labels(
            np.ones_like(labels, dtype=int), labels, range(1, n_patches + 1)
        )
        small_labels = [i + 1 for i, s in enumerate(sizes) if s < min_patch_size]
        if small_labels:
            labels[np.isin(labels, small_labels)] = 0

    return labels > 0, reference


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
            peakdict=thepeakdict,
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

        # Correlation time despeckle
        if optiondict["despeckle_passes"] > 0:
            LGR.info(f"\n\n{similaritytype} despeckling pass {thepass}")
            LGR.info(f"\tUsing despeckle_thresh = {optiondict['despeckle_thresh']:.3f}")
            TimingLGR.info(f"{similaritytype} despeckle start, pass {thepass}")

            # find lags that are very different from their neighbors, and refit starting at the median lag for the point
            voxelsprocessed_fc_ds = 0
            despecklingdone = False
            lastnumdespeckled = 1000000
            use_multipeak = optiondict.get("despeckle_multipeak", True)
            use_progressive_kernel = optiondict.get("despeckle_progressive_kernel", True)
            use_patch_detection = optiondict.get("despeckle_patch_detection", True)
            patch_refkernel = optiondict.get("despeckle_patch_refkernel", 9)
            patch_minsize = optiondict.get("despeckle_patch_minsize", 10)
            for despecklepass in range(optiondict["despeckle_passes"]):
                # Use larger kernel on later passes to catch medium-sized patches
                if use_progressive_kernel and despecklepass >= 2:
                    kernel_size = 5
                else:
                    kernel_size = 3
                LGR.info(
                    f"\n\n{similaritytype} despeckling subpass {despecklepass + 1} "
                    f"(kernel={kernel_size}, multipeak={use_multipeak})"
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

                # On later passes, detect large connected patches that survive
                # median filtering and add them to the refit candidates
                patches_added = 0
                if use_patch_detection and despecklepass >= 2:
                    lagmap_3d = outmaparray.reshape(nativespaceshape)
                    validmask_3d = np.zeros(nativespaceshape, dtype=bool)
                    validmask_3d.reshape(-1)[validvoxels] = fitmask[:].astype(bool)
                    patch_mask_3d, reference_3d = _detect_shifted_patches(
                        lagmap_3d,
                        validmask_3d,
                        optiondict["despeckle_thresh"],
                        reference_kernel=patch_refkernel,
                        min_patch_size=patch_minsize,
                    )
                    n_patch_voxels = int(patch_mask_3d.sum())
                    if n_patch_voxels > 0:
                        patch_mask_flat = patch_mask_3d.reshape(numspatiallocs)
                        reference_flat = reference_3d.reshape(numspatiallocs)
                        # Add patch voxels as refit candidates (if not already flagged)
                        for i, vox in enumerate(validvoxels):
                            if patch_mask_flat[vox] and initlags[i] == -1000000.0:
                                initlags[i] = reference_flat[vox]
                                patches_added += 1
                        LGR.info(
                            f"\tPatch detection found {n_patch_voxels} voxels in "
                            f"large patches, {patches_added} new candidates added"
                        )
                    else:
                        LGR.info("\tPatch detection found no large patches")

                if len(initlags) > 0:
                    numdespeckled = len(np.where(initlags != -1000000.0)[0])
                    # Bypass convergence guard when patch detection added new
                    # candidates, since the count may increase on that pass
                    if (patches_added > 0 and numdespeckled > 0) or (
                        lastnumdespeckled > numdespeckled > 0
                    ):
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
                            peakdict=thepeakdict,
                            nprocs=optiondict["nprocs_fitcorr"],
                            alwaysmultiproc=optiondict["alwaysmultiproc"],
                            fixdelay=optiondict["fixdelay"],
                            initialdelayvalue=theinitialdelay,
                            showprogressbar=optiondict["showprogressbar"],
                            chunksize=optiondict["mp_chunksize"],
                            despeckle_thresh=optiondict["despeckle_thresh"],
                            initiallags=initlags,
                            multipeak=use_multipeak,
                            rt_floattype=rt_floattype,
                        )
                        tide_util.enablemkl(
                            optiondict["mklthreads"], debug=optiondict["threaddebug"]
                        )

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

        # Patch shifting
        if optiondict["patchshift"]:
            outmaparray *= 0.0
            outmaparray[validvoxels] = eval("lagtimes")[:]
            # new method
            masklist = [
                (
                    outmaparray[validvoxels],
                    f"lagtimes_prepatch_pass{thepass}",
                    "map",
                    None,
                    f"Input lagtimes map prior to patch map generation pass {thepass}",
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

            # create list of anomalous 3D regions that don't match surroundings
            if theinputdata.nim_affine is not None:
                # make an atlas of anomalous patches - each patch shares the same integer value
                step1 = tide_patch.calc_DoG(
                    outmaparray.reshape(nativespaceshape).copy(),
                    theinputdata.nim_affine,
                    thesizes,
                    fwhm=optiondict["patchfwhm"],
                    ratioopt=False,
                    debug=True,
                )
                masklist = [
                    (
                        step1.reshape(internalspaceshape)[validvoxels],
                        f"DoG_pass{thepass}",
                        "map",
                        None,
                        f"DoG map for pass {thepass}",
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
                step2 = tide_patch.invertedflood3D(
                    step1,
                    1,
                )
                masklist = [
                    (
                        step2.reshape(internalspaceshape)[validvoxels],
                        f"invertflood_pass{thepass}",
                        "map",
                        None,
                        f"Inverted flood map for pass {thepass}",
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

                patchmap = tide_patch.separateclusters(
                    step2,
                    sizethresh=optiondict["patchminsize"],
                    debug=True,
                )
                # patchmap = tide_patch.getclusters(
                #   outmaparray.reshape(nativespaceshape),
                #    theinputdata.nim_affine,
                #    thesizes,
                #    fwhm=optiondict["patchfwhm"],
                #    ratioopt=True,
                #    sizethresh=optiondict["patchminsize"],
                #    debug=True,
                # )
                masklist = [
                    (
                        patchmap[validvoxels],
                        f"patch_pass{thepass}",
                        "map",
                        None,
                        f"Patch map for despeckling pass {thepass}",
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

            # now shift the patches to align with the majority of the image
            tide_patch.interppatch(lagtimes, patchmap[validvoxels])

    return internaldespeckleincludemask
