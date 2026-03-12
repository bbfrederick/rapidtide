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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

import rapidtide.fit as tide_fit
import rapidtide.io as tide_io
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


def _detect_shifted_patches(
    lagmap_3d: NDArray,
    validmask_3d: NDArray[np.bool_],
    despeckle_thresh: float,
    reference_kernel: int = 9,
    min_patch_size: int = 10,
    consistency_ratio: float = 0.5,
    use_confidence: bool = False,
    confidence_weight: float = 0.5,
    R2_3d: Optional[NDArray] = None,
    lagstrengths_3d: Optional[NDArray] = None,
) -> tuple[NDArray[np.bool_], NDArray]:
    """Detect connected patches of shifted delay values.

    After initial median-filter despeckling removes isolated speckles, large
    patches of wrong-peak selections survive because they fool the small
    median filter.  This function detects them by comparing each voxel to a
    heavily smoothed reference computed with a much larger kernel, then
    validating connected components against their exterior boundary.

    For each candidate component the function checks:
      1. The median lag inside the patch differs from the one-voxel exterior
         ring by more than ``despeckle_thresh`` (boundary validation).
      2. The standard deviation of lags inside the patch is small relative to
         that offset (consistency check — anomalous patches chose the same
         wrong peak so they are internally uniform).
    Confirmed patches are then grown inward via a constrained flood-fill to
    recover interior voxels that were missed because the smooth reference was
    biased by the patch itself.

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
        Smaller clusters are ignored. Default is 10.
    consistency_ratio : float, optional
        Maximum ratio of (std inside patch) / (offset from exterior) for a
        patch to be confirmed as anomalous.  Lower values require more
        internal consistency.  Default is 0.5.
    use_confidence : bool, optional
        If True, modulate the detection threshold using fit quality metrics
        (R², peak strength).  Regions with poor fit quality are flagged at a
        lower spatial threshold.  Default is False.
    confidence_weight : float, optional
        Weight [0..1] for the confidence modulation.  Only used when
        ``use_confidence`` is True.  Default is 0.5.
    R2_3d : NDArray or None, optional
        R² map in native 3D space.  Used when ``use_confidence`` is True.
    lagstrengths_3d : NDArray or None, optional
        Peak strength map.  Used as secondary confidence metric when provided
        and ``use_confidence`` is True.

    Returns
    -------
    patch_mask : NDArray[np.bool_]
        Boolean mask (same shape as lagmap_3d) where True marks voxels
        belonging to a confirmed anomalous patch, including interior voxels
        recovered by the flood-fill step.
    reference : NDArray
        Reference lag map.  For detected anomalous patch voxels this holds
        the exterior-ring median (a better initial-lag estimate for refitting);
        for all other voxels it holds the large-kernel smoothed reference.
    """
    # Build reference with large median filter.
    reference = masked_median_filter(
        np.where(validmask_3d, lagmap_3d, 0.0),
        size=reference_kernel,
        mode="reflect",
        mask=validmask_3d,
    )

    # Global confidence baseline (used only when use_confidence=True).
    global_mean_R2 = 0.0
    global_mean_strength = 0.0
    if use_confidence:
        if R2_3d is not None:
            valid_R2 = R2_3d[validmask_3d]
            global_mean_R2 = float(np.mean(valid_R2)) if valid_R2.size > 0 else 0.0
        if lagstrengths_3d is not None:
            valid_str = np.abs(lagstrengths_3d[validmask_3d])
            global_mean_strength = float(np.mean(valid_str)) if valid_str.size > 0 else 0.0

    # Initial candidates: voxels that deviate from the smooth reference.
    deviation = np.abs(lagmap_3d - reference)
    outlier_mask = validmask_3d & (deviation > despeckle_thresh)

    structure = ndimage.generate_binary_structure(lagmap_3d.ndim, lagmap_3d.ndim)
    labels, n_patches = ndimage.label(outlier_mask, structure=structure)

    if n_patches == 0:
        return np.zeros_like(validmask_3d, dtype=bool), reference

    confirmed_patch_mask = np.zeros_like(validmask_3d, dtype=bool)
    # ext_reference will hold the exterior-ring median for confirmed patches.
    ext_reference = reference.copy()

    for region_id in range(1, n_patches + 1):
        region_mask = labels == region_id
        region_valid = region_mask & validmask_3d
        if int(np.sum(region_valid)) < min_patch_size:
            continue

        # One-voxel exterior ring.
        dilated = ndimage.binary_dilation(region_mask, structure=structure)
        exterior_ring = dilated & ~region_mask & validmask_3d
        if not np.any(exterior_ring):
            continue

        interior_lags = lagmap_3d[region_valid]
        exterior_lags = lagmap_3d[exterior_ring]
        interior_median = float(np.median(interior_lags))
        exterior_median = float(np.median(exterior_lags))
        interior_std = float(np.std(interior_lags))
        offset = abs(interior_median - exterior_median)

        # Optionally modulate detection threshold by fit quality.
        effective_thresh = despeckle_thresh
        if use_confidence:
            conf_components = []
            if R2_3d is not None and global_mean_R2 > 0.0:
                patch_R2 = float(np.mean(R2_3d[region_valid]))
                conf_components.append(float(np.clip(patch_R2 / global_mean_R2, 0.0, 2.0)))
            if lagstrengths_3d is not None and global_mean_strength > 0.0:
                patch_str = float(np.mean(np.abs(lagstrengths_3d[region_valid])))
                conf_components.append(float(np.clip(patch_str / global_mean_strength, 0.0, 2.0)))
            if conf_components:
                norm_conf = float(np.mean(conf_components))
                # Low confidence → lower threshold (more suspicious).
                effective_thresh = despeckle_thresh * max(
                    0.25, 1.0 - confidence_weight * (1.0 - norm_conf)
                )

        # Boundary validation and consistency check.
        if offset <= effective_thresh:
            continue
        if interior_std >= consistency_ratio * offset:
            continue

        # Confirmed anomalous patch.  Grow inward to recover interior voxels
        # that the smooth reference missed (it was biased by the patch).
        lag_tolerance = max(2.0 * interior_std, 0.5 * despeckle_thresh)
        grown = region_valid.copy()
        for _ in range(50):
            new_dilated = ndimage.binary_dilation(grown, structure=structure)
            candidates = new_dilated & ~grown & validmask_3d
            new_voxels = candidates & (np.abs(lagmap_3d - interior_median) <= lag_tolerance)
            if not np.any(new_voxels):
                break
            grown |= new_voxels

        confirmed_patch_mask |= grown
        ext_reference[grown] = exterior_median

    return confirmed_patch_mask, ext_reference


def _anchor_based_region_growing(
    lagmap_3d: NDArray,
    validmask_3d: NDArray[np.bool_],
    corrout: NDArray[np.floating[Any]],
    validvoxels: NDArray,
    numspatiallocs: int,
    nativespaceshape: tuple,
    trimmedcorrscale: NDArray[np.floating[Any]],
    R2_3d: NDArray,
    fitmask_3d: NDArray,
    dominance_threshold: float = 1.5,
    search_width: float = 5.0,
    min_peak_fraction: float = 0.2,
    bipolar: bool = False,
    debug: bool = False,
) -> Tuple[NDArray[np.bool_], NDArray, int, int]:
    """Anchor-based region growing for robust delay estimation.

    Identifies high-confidence "anchor" voxels where the correlation peak is
    unambiguous (strongly dominant over all other peaks), then propagates lag
    assignments outward via breadth-first search (BFS).  At each frontier voxel
    the peak closest to the spatially extrapolated expected lag is chosen, even if
    it is not the tallest peak.  This allows the algorithm to penetrate artifact
    patches whose true-lag peak has been eroded below a sidelobe, while stalling
    naturally at genuine vascular territory boundaries where no peak exists
    near the extrapolated lag.

    Parameters
    ----------
    lagmap_3d : NDArray
        Current lag map in native 3D space.
    validmask_3d : NDArray[np.bool_]
        Boolean mask of valid (fitted) voxels.
    corrout : NDArray
        Per-voxel correlation functions, shape (numvalidspatiallocs, n_lags).
    validvoxels : NDArray
        Flat 3D indices of valid voxels (length numvalidspatiallocs).
    numspatiallocs : int
        Total number of 3D voxels (product of nativespaceshape).
    nativespaceshape : tuple
        Shape of the 3D volume.
    trimmedcorrscale : NDArray
        Lag axis corresponding to corrout columns.
    R2_3d : NDArray
        R² map in native 3D space.
    fitmask_3d : NDArray
        Fit-success mask in native 3D space (nonzero = successful fit).
    dominance_threshold : float, optional
        Minimum ratio C_max/C_second for a voxel to qualify as an anchor.
        Default is 1.5.
    search_width : float, optional
        Full width (seconds) of the search window around tau_expected when
        looking for the true-lag peak in a frontier voxel's correlation
        function.  Should be less than acsidelobelag to avoid accepting a
        sidelobe.  Default is 5.0.
    min_peak_fraction : float, optional
        A candidate peak must have absolute height >= min_peak_fraction * max
        peak height in that voxel's corrout to be considered.  Filters out
        noise bumps that could be selected when tau_expected falls between two
        genuine territory lags.  Default is 0.2.
    bipolar : bool, optional
        If True, detect both positive and negative peaks.  Default is False.
    debug : bool, optional
        If True, print diagnostic counts.  Default is False.

    Returns
    -------
    refit_mask_3d : NDArray[np.bool_]
        Boolean mask of voxels whose assigned lag differs from current fitted
        lag by more than 0.1 s and should be refitted.
    target_lags_3d : NDArray
        Target lag for each voxel (only meaningful where refit_mask_3d is True).
    n_uncertain : int
        Number of voxels left unchanged because no good peak was found and the
        current fit was also of low quality.
    n_unprocessed : int
        Number of valid voxels that region growing never reached (isolated from
        all anchors).
    """
    numvalidspatiallocs = len(validvoxels)
    shape = nativespaceshape

    # Build reverse lookup: flat 3D index -> valid voxel index (-1 if invalid).
    valid_vox_lookup = np.full(numspatiallocs, -1, dtype=np.int32)
    valid_vox_lookup[validvoxels] = np.arange(numvalidspatiallocs, dtype=np.int32)

    # Precompute peaks and per-voxel max |corrout| (single pass).
    all_peaks: list[list] = []
    dominance_valid = np.zeros(numvalidspatiallocs, dtype=np.float32)
    max_corrout_valid = np.max(np.abs(corrout), axis=1)  # shape: (numvalidspatiallocs,)
    for i in range(numvalidspatiallocs):
        peaks = tide_fit.getpeaks(trimmedcorrscale, corrout[i, :], bipolar=bipolar)
        all_peaks.append(peaks)
        if len(peaks) == 0:
            dominance_valid[i] = 0.0
        elif len(peaks) == 1:
            dominance_valid[i] = np.inf
        else:
            peaks_by_strength = sorted(peaks, key=lambda p: abs(p[1]), reverse=True)
            second = abs(peaks_by_strength[1][1])
            dominance_valid[i] = (
                abs(peaks_by_strength[0][1]) / second if second > 1e-10 else np.inf
            )

    # Map dominance into 3D space.
    dominance_3d = np.zeros(shape, dtype=np.float32)
    dominance_3d.reshape(-1)[validvoxels] = dominance_valid

    # Identify anchor voxels: high dominance + high R² + clean fit.
    valid_r2 = R2_3d[validmask_3d]
    r2_thresh = float(np.percentile(valid_r2[valid_r2 > 0], 70)) if np.any(valid_r2 > 0) else 0.0
    anchor_3d = (
        validmask_3d
        & (dominance_3d >= dominance_threshold)
        & (R2_3d >= r2_thresh)
        & (fitmask_3d > 0)
    )
    n_anchors = int(anchor_3d.sum())
    if debug:
        print(f"  _anchor_based_region_growing: {n_anchors} anchors (R²≥{r2_thresh:.3f})")

    if n_anchors == 0:
        n_unprocessed = int(validmask_3d.sum())
        return np.zeros(shape, dtype=bool), lagmap_3d.copy(), 0, n_unprocessed

    # Initialise assigned-lag map; anchors are pre-assigned their current lags.
    _UNASSIGNED = -1.0e9
    assigned_3d = np.full(shape, _UNASSIGNED, dtype=np.float64)
    assigned_3d[anchor_3d] = lagmap_3d[anchor_3d]
    processed_3d = anchor_3d.copy()

    # Face-adjacent 3D offsets (6-connectivity).
    face_offsets = [
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ]

    # Seed BFS queue with all valid unprocessed neighbors of anchors.
    in_queue = np.zeros(shape, dtype=bool)
    queue: deque[tuple[int, int, int]] = deque()
    anchor_coords = np.argwhere(anchor_3d)
    for ax, ay, az in anchor_coords:
        for dx, dy, dz in face_offsets:
            nx, ny, nz = ax + dx, ay + dy, az + dz
            if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                if (
                    validmask_3d[nx, ny, nz]
                    and not processed_3d[nx, ny, nz]
                    and not in_queue[nx, ny, nz]
                ):
                    queue.append((nx, ny, nz))
                    in_queue[nx, ny, nz] = True

    half_window = search_width / 2.0
    n_uncertain = 0
    n_territory = 0

    while queue:
        x, y, z = queue.popleft()

        # Another path may have already processed this voxel.
        if processed_3d[x, y, z]:
            continue

        # tau_expected = median lag of all already-processed face-adjacent neighbors.
        neighbor_lags = []
        for dx, dy, dz in face_offsets:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                if processed_3d[nx, ny, nz]:
                    neighbor_lags.append(assigned_3d[nx, ny, nz])

        if not neighbor_lags:
            # No processed neighbor yet; re-queue for later (can happen if the
            # seeding voxel's anchor neighbor was processed after this was enqueued
            # via a different path — rare but possible in concurrent BFS seeding).
            queue.append((x, y, z))
            continue

        tau_expected = float(np.median(neighbor_lags))

        # Look up precomputed peaks for this voxel.
        flat_idx = int(np.ravel_multi_index((x, y, z), shape))
        valid_idx = valid_vox_lookup[flat_idx]
        peaks = all_peaks[valid_idx]
        min_height = min_peak_fraction * max_corrout_valid[valid_idx]

        nearby = [
            p for p in peaks if abs(p[0] - tau_expected) <= half_window and abs(p[1]) >= min_height
        ]

        if nearby:
            # Choose the peak closest to tau_expected regardless of height.
            best = min(nearby, key=lambda p: abs(p[0] - tau_expected))
            assigned_3d[x, y, z] = best[0]
        else:
            # No peak near tau_expected.
            dom = float(dominance_3d[x, y, z])
            r2 = float(R2_3d[x, y, z])
            if dom >= dominance_threshold and r2 >= r2_thresh and fitmask_3d[x, y, z] > 0:
                # Territory boundary: high-quality fit at a genuinely different lag.
                # Accept current lag and seed the adjacent territory from here.
                assigned_3d[x, y, z] = lagmap_3d[x, y, z]
                n_territory += 1
            else:
                # Uncertain: poor fit and no nearby peak — leave lag unchanged.
                assigned_3d[x, y, z] = lagmap_3d[x, y, z]
                n_uncertain += 1

        processed_3d[x, y, z] = True

        # Enqueue newly reachable unprocessed valid neighbors.
        for dx, dy, dz in face_offsets:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                if (
                    validmask_3d[nx, ny, nz]
                    and not processed_3d[nx, ny, nz]
                    and not in_queue[nx, ny, nz]
                ):
                    queue.append((nx, ny, nz))
                    in_queue[nx, ny, nz] = True

    n_unprocessed = int(np.sum(validmask_3d & ~processed_3d))
    if debug:
        n_corrected = int(
            np.sum(processed_3d & validmask_3d & (np.abs(assigned_3d - lagmap_3d) > 0.1))
        )
        print(
            f"  region growing complete: {n_corrected} reassigned, "
            f"{n_territory} territory boundaries, {n_uncertain} uncertain, "
            f"{n_unprocessed} unprocessed"
        )

    # Leave unprocessed voxels at their current lag.
    unprocessed_mask = validmask_3d & ~processed_3d
    assigned_3d[unprocessed_mask] = lagmap_3d[unprocessed_mask]

    # Refit mask: assigned lag differs from current fitted lag by more than
    # twice the lag axis step (handles discrete quantization of getpeaks output).
    lag_step = (
        float(trimmedcorrscale[1] - trimmedcorrscale[0]) if len(trimmedcorrscale) > 1 else 0.1
    )
    refit_tol = max(0.5, 2.0 * lag_step)
    refit_mask_3d = validmask_3d & (np.abs(assigned_3d - lagmap_3d) > refit_tol)

    return refit_mask_3d, assigned_3d, n_uncertain, n_unprocessed


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

        # Correlation time despeckle
        if optiondict["despeckle_passes"] > 0:
            LGR.info(f"\n\n{similaritytype} despeckling pass {thepass}")
            LGR.info(f"\tUsing despeckle_thresh = {optiondict['despeckle_thresh']:.3f}")
            TimingLGR.info(f"{similaritytype} despeckle start, pass {thepass}")

            # find lags that are very different from their neighbors, and refit starting at the median lag for the point
            voxelsprocessed_fc_ds = 0
            despecklingdone = False
            lastnumdespeckled = 1000000
            use_patch_detection = optiondict.get("despeckle_patch_detection", False)
            patch_refkernel = optiondict.get("despeckle_patch_refkernel", 9)
            patch_minsize = optiondict.get("despeckle_patch_minsize", 10)
            for despecklepass in range(optiondict["despeckle_passes"]):
                kernel_size = 3
                LGR.info(
                    f"\n\n{similaritytype} despeckling subpass {despecklepass + 1} "
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

                # On later passes, detect large connected patches that survive
                # median filtering and add them to the refit candidates
                patches_added = 0
                if use_patch_detection and despecklepass >= 2:
                    lagmap_3d = outmaparray.reshape(nativespaceshape)
                    validmask_3d = np.zeros(nativespaceshape, dtype=bool)
                    validmask_3d.reshape(-1)[validvoxels] = fitmask[:].astype(bool)
                    use_conf = optiondict.get("despeckle_patch_use_confidence", False)
                    R2_3d_ds = lagstrengths_3d_ds = None
                    if use_conf:
                        R2_3d_ds = np.zeros(nativespaceshape)
                        R2_3d_ds.reshape(-1)[validvoxels] = R2[:]
                        lagstrengths_3d_ds = np.zeros(nativespaceshape)
                        lagstrengths_3d_ds.reshape(-1)[validvoxels] = lagstrengths[:]
                    patch_mask_3d, reference_3d = _detect_shifted_patches(
                        lagmap_3d,
                        validmask_3d,
                        optiondict["despeckle_thresh"],
                        reference_kernel=patch_refkernel,
                        min_patch_size=patch_minsize,
                        consistency_ratio=optiondict.get("despeckle_patch_consistency_ratio", 0.5),
                        use_confidence=use_conf,
                        confidence_weight=optiondict.get("despeckle_patch_confidence_weight", 0.5),
                        R2_3d=R2_3d_ds,
                        lagstrengths_3d=lagstrengths_3d_ds,
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

        # Patch shift correction: detect anomalous patches and refit them.
        # This runs after all despeckle passes and catches self-consistent
        # patches of voxels that all chose the same wrong correlation peak.
        if optiondict["patchshift"]:
            LGR.info(f"\n\nPatch shift correction pass {thepass}")
            TimingLGR.info(f"Patch shift correction start, pass {thepass}")

            outmaparray *= 0.0
            outmaparray[validvoxels] = lagtimes[:]
            lagmap_3d = outmaparray.reshape(nativespaceshape)
            validmask_3d = np.zeros(nativespaceshape, dtype=bool)
            validmask_3d.reshape(-1)[validvoxels] = fitmask[:].astype(bool)

            use_conf = optiondict.get("despeckle_patch_use_confidence", False)
            R2_3d_ps = lagstrengths_3d_ps = None
            if use_conf:
                R2_3d_ps = np.zeros(nativespaceshape)
                R2_3d_ps.reshape(-1)[validvoxels] = R2[:]
                lagstrengths_3d_ps = np.zeros(nativespaceshape)
                lagstrengths_3d_ps.reshape(-1)[validvoxels] = lagstrengths[:]

            patch_mask_3d, patch_reference_3d = _detect_shifted_patches(
                lagmap_3d,
                validmask_3d,
                optiondict["despeckle_thresh"],
                reference_kernel=optiondict.get("despeckle_patch_refkernel", 9),
                min_patch_size=optiondict.get("despeckle_patch_minsize", 10),
                consistency_ratio=optiondict.get("despeckle_patch_consistency_ratio", 0.5),
                use_confidence=use_conf,
                confidence_weight=optiondict.get("despeckle_patch_confidence_weight", 0.5),
                R2_3d=R2_3d_ps,
                lagstrengths_3d=lagstrengths_3d_ps,
            )

            n_patch_voxels = int(patch_mask_3d.sum())
            LGR.info(f"\tPatch detection found {n_patch_voxels} anomalous voxels")

            if n_patch_voxels > 0:
                patch_mask_flat = patch_mask_3d.reshape(numspatiallocs)
                reference_flat = patch_reference_3d.reshape(numspatiallocs)
                initlags_ps = np.full(numvalidspatiallocs, -1000000.0)
                for i, vox in enumerate(validvoxels):
                    if patch_mask_flat[vox]:
                        initlags_ps[i] = reference_flat[vox]

                tide_util.disablemkl(optiondict["nprocs_fitcorr"], debug=optiondict["threaddebug"])
                voxelsprocessed_ps = tide_simfuncfit.fitcorr(
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
                    initiallags=initlags_ps,
                    rt_floattype=rt_floattype,
                )
                tide_util.enablemkl(optiondict["mklthreads"], debug=optiondict["threaddebug"])
                LGR.info(f"\tPatch shift corrected {voxelsprocessed_ps} voxels")

                if optiondict.get("savedespecklemasks", False) and thepass == optiondict["passes"]:
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
                            patch_mask_flat[validvoxels].astype(np.int32),
                            f"patchmask_p{thepass}",
                            "mask",
                            None,
                            f"Anomalous patch voxels for pass {thepass}",
                        ),
                        (
                            reference_flat[validvoxels],
                            f"patchreference_p{thepass}",
                            "map",
                            None,
                            f"Reference lag targets for patch shift pass {thepass}",
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

            TimingLGR.info(
                f"Patch shift correction end, pass {thepass}",
                {
                    "message2": n_patch_voxels,
                    "message3": "voxels detected",
                },
            )

        # Robust delay estimation via anchor-based region growing.
        # This post-pass selects correlation peaks by spatial consistency rather
        # than peak height, allowing it to correct artifact patches even where
        # the true-lag peak has been eroded below the sidelobe peak.  Territory
        # boundaries (true vascular discontinuities) are preserved naturally.
        # Disabled by default; enable with --robustdelay.
        if optiondict.get("robustdelay", False):
            LGR.info(f"\n\nRobust delay estimation (anchor-based region growing), pass {thepass}")
            TimingLGR.info(f"Robust delay estimation start, pass {thepass}")

            outmaparray *= 0.0
            outmaparray[validvoxels] = lagtimes[:]
            lagmap_3d_rd = outmaparray.reshape(nativespaceshape)

            validmask_3d_rd = np.zeros(nativespaceshape, dtype=bool)
            validmask_3d_rd.reshape(-1)[validvoxels] = fitmask[:].astype(bool)

            R2_3d_rd = np.zeros(nativespaceshape)
            R2_3d_rd.reshape(-1)[validvoxels] = R2[:]

            fitmask_3d_rd = np.zeros(nativespaceshape, dtype=np.uint16)
            fitmask_3d_rd.reshape(-1)[validvoxels] = fitmask[:]

            passsuffix = "_pass" + str(thepass)
            acsidelobelag = optiondict.get("acsidelobelag" + passsuffix, 10.0)

            refit_mask_3d, target_lags_3d, n_uncertain, n_unprocessed = (
                _anchor_based_region_growing(
                    lagmap_3d_rd,
                    validmask_3d_rd,
                    corrout,
                    validvoxels,
                    numspatiallocs,
                    nativespaceshape,
                    trimmedcorrscale,
                    R2_3d_rd,
                    fitmask_3d_rd,
                    dominance_threshold=optiondict.get("robustdelay_dominance_threshold", 1.5),
                    search_width=optiondict.get("robustdelay_search_width", 5.0),
                    min_peak_fraction=optiondict.get("robustdelay_min_peak_fraction", 0.2),
                    bipolar=optiondict.get("bipolar", False),
                    debug=optiondict.get("debug", False),
                )
            )

            n_refit = int(refit_mask_3d.sum())
            LGR.info(
                f"\tRobust delay: {n_refit} voxels reassigned, "
                f"{n_uncertain} uncertain, {n_unprocessed} unreachable"
            )

            if n_refit > 0:
                refit_mask_flat = refit_mask_3d.reshape(numspatiallocs)
                target_lags_flat = target_lags_3d.reshape(numspatiallocs)
                initlags_rd = np.full(numvalidspatiallocs, -1000000.0)
                for i, vox in enumerate(validvoxels):
                    if refit_mask_flat[vox]:
                        initlags_rd[i] = target_lags_flat[vox]

                tide_util.disablemkl(optiondict["nprocs_fitcorr"], debug=optiondict["threaddebug"])
                voxelsprocessed_rd = tide_simfuncfit.fitcorr(
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
                    initiallags=initlags_rd,
                    rt_floattype=rt_floattype,
                )
                tide_util.enablemkl(optiondict["mklthreads"], debug=optiondict["threaddebug"])
                LGR.info(f"\tRobust delay refitted {voxelsprocessed_rd} voxels")

                if optiondict.get("savedespecklemasks", False) and thepass == optiondict["passes"]:
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
                            refit_mask_flat[validvoxels].astype(np.int32),
                            f"robustdelay_reassigned_p{thepass}",
                            "mask",
                            None,
                            f"Voxels reassigned by robust delay estimation, pass {thepass}",
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

            TimingLGR.info(
                f"Robust delay estimation end, pass {thepass}",
                {
                    "message2": n_refit,
                    "message3": "voxels reassigned",
                },
            )

    return internaldespeckleincludemask
