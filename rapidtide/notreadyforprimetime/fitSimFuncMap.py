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

import rapidtide.io as tide_io
import rapidtide.patchmatch as tide_patch
import rapidtide.peakeval as tide_peakeval
import rapidtide.resample as tide_resample
import rapidtide.simfuncfit as tide_simfuncfit
import rapidtide.util as tide_util


def _nan_median(vals: NDArray[np.floating[Any]]) -> float:
    return float(np.nanmedian(vals))


def _build_despeckle_targets(
    lagmap_flat: NDArray[np.floating[Any]],
    validmask_flat: NDArray[np.bool_],
    nativespaceshape: tuple[int, ...],
    base_thresh: float,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], NDArray[np.bool_]]:
    lagmap_nan = np.full(lagmap_flat.shape, np.nan, dtype=np.float64)
    lagmap_nan[validmask_flat] = lagmap_flat[validmask_flat]
    lagvol = lagmap_nan.reshape(nativespaceshape)

    local_median = ndimage.generic_filter(lagvol, _nan_median, size=3, mode="nearest").reshape(
        lagmap_flat.shape
    )
    fixed_thresh = np.full(lagmap_flat.shape, base_thresh, dtype=np.float64)
    finite = np.isfinite(local_median)
    candidates = validmask_flat & finite & (np.abs(lagmap_flat - local_median) > fixed_thresh)
    return local_median, fixed_thresh, candidates


def _refine_candidates_with_confidence(
    spatial_candidates: NDArray[np.bool_],
    lagmap_flat: NDArray[np.floating[Any]],
    local_median: NDArray[np.floating[Any]],
    threshold_map: NDArray[np.floating[Any]],
    validvoxels: NDArray[np.integer[Any]],
    lagstrengths: NDArray[np.floating[Any]],
    lagsigma: NDArray[np.floating[Any]],
    R2: NDArray[np.floating[Any]],
    failreason: NDArray[np.integer[Any]],
    min_r2: float = 0.2,
    min_strength: float = 0.2,
    max_sigma: float = 1.0e30,
    strong_outlier_factor: float = 2.0,
) -> tuple[NDArray[np.bool_], dict[str, int]]:
    """Restrict spatial outlier candidates to low-confidence fits, with a strong-outlier override."""
    conf_low = (
        (R2 < min_r2)
        | (np.abs(lagstrengths) < min_strength)
        | (lagsigma > max_sigma)
        | (failreason != 0)
    )
    conf_low_flat = np.zeros(spatial_candidates.shape, dtype=bool)
    conf_low_flat[validvoxels] = conf_low
    strong_outlier = np.abs(lagmap_flat - local_median) > (strong_outlier_factor * threshold_map)
    combined = spatial_candidates & (conf_low_flat | strong_outlier)
    info = {
        "spatial": int(np.sum(spatial_candidates[validvoxels])),
        "conf_low": int(np.sum(conf_low)),
        "strong_outlier": int(np.sum(strong_outlier[validvoxels])),
        "combined": int(np.sum(combined[validvoxels])),
    }
    return combined, info


def _safe_nanmedian(vals: NDArray[np.floating[Any]], default: float = 0.0) -> float:
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return default
    return float(np.median(finite))


def _optimize_despeckle_labels_icm(
    lagmap_flat: NDArray[np.floating[Any]],
    candidate_mask_flat: NDArray[np.bool_],
    validmask_flat: NDArray[np.bool_],
    validvoxels: NDArray[np.integer[Any]],
    peakdict: Optional[dict[str, list[list[float]]]],
    nativespaceshape: tuple[int, ...],
    max_candidates: int = 3,
    max_iters: int = 3,
    data_weight: float = 1.0,
    smooth_weight: float = 1.0,
    min_change: float = 1.0e-6,
) -> tuple[NDArray[np.floating[Any]], dict[str, float]]:
    """
    Optimize lag labels using explicit peak candidates and local smoothness.

    For each candidate voxel, choose among top-K peak lag labels (plus current lag)
    using ICM updates on:
      E = -data_weight * quality + smooth_weight * |lag - neighborhood_median|
    """
    label_map = lagmap_flat.astype(np.float64).copy()
    flat_to_valid = np.full(lagmap_flat.shape, -1, dtype=np.int64)
    flat_to_valid[validvoxels] = np.arange(len(validvoxels), dtype=np.int64)
    label_candidates: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    n_with_peaks = 0
    n_with_fallback = 0
    total_cands = 0

    for flat_idx in np.where(candidate_mask_flat)[0]:
        valid_idx = int(flat_to_valid[flat_idx])
        if valid_idx < 0:
            continue
        current_lag = float(label_map[flat_idx])
        peak_entries = [] if peakdict is None else peakdict.get(str(valid_idx), [])
        cand_lags = []
        cand_qual = []
        if len(peak_entries) > 0:
            n_with_peaks += 1
            for p in peak_entries[:max_candidates]:
                cand_lags.append(float(p[0]))
                # Prefer MI quality if present, otherwise use correlation strength
                qual = float(p[2]) if len(p) > 2 else float(np.abs(p[1]))
                cand_qual.append(qual)
        else:
            n_with_fallback += 1

        # Always include current lag as a candidate to allow no-op local optimum
        cand_lags.append(current_lag)
        cand_qual.append(float(np.max(cand_qual)) if len(cand_qual) > 0 else 0.0)

        unique_lags = list(dict.fromkeys(cand_lags))
        unique_qual = []
        for ul in unique_lags:
            idx = cand_lags.index(ul)
            unique_qual.append(cand_qual[idx])
        cands = np.array(unique_lags, dtype=np.float64)
        quals = np.array(unique_qual, dtype=np.float64)
        if np.max(quals) > np.min(quals):
            quals = (quals - np.min(quals)) / (np.max(quals) - np.min(quals))
        else:
            quals *= 0.0
        label_candidates[int(flat_idx)] = (cands, quals)
        total_cands += len(cands)

    if len(label_candidates) == 0:
        return label_map, {
            "changed": 0.0,
            "iters": 0.0,
            "vox_with_peaks": float(n_with_peaks),
            "vox_fallback": float(n_with_fallback),
            "mean_candidates": 0.0,
        }

    iters_run = 0
    for _ in range(max_iters):
        iters_run += 1
        lagmap_nan = np.full(label_map.shape, np.nan, dtype=np.float64)
        lagmap_nan[validmask_flat] = label_map[validmask_flat]
        neigh_median = ndimage.generic_filter(
            lagmap_nan.reshape(nativespaceshape), _nan_median, size=3, mode="nearest"
        ).reshape(label_map.shape)
        n_changed_iter = 0
        for flat_idx, (cands, quals) in label_candidates.items():
            target = neigh_median[flat_idx]
            if not np.isfinite(target):
                target = label_map[flat_idx]
            energies = (-data_weight * quals) + (smooth_weight * np.abs(cands - target))
            best = int(np.argmin(energies))
            new_lag = float(cands[best])
            if np.abs(new_lag - label_map[flat_idx]) > min_change:
                label_map[flat_idx] = new_lag
                n_changed_iter += 1
        if n_changed_iter == 0:
            break

    changed_total = int(np.sum(np.abs(label_map - lagmap_flat) > min_change))
    n_candidate_vox = max(len(label_candidates), 1)
    return label_map, {
        "changed": float(changed_total),
        "iters": float(iters_run),
        "vox_with_peaks": float(n_with_peaks),
        "vox_fallback": float(n_with_fallback),
        "mean_candidates": float(total_cands / n_candidate_vox),
    }


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
            validmask_flat = np.zeros(numspatiallocs, dtype=bool)
            validmask_flat[validvoxels] = True
            legacy_mode = optiondict["despeckle_legacy_mode"]
            confidence_mode = optiondict.get("despeckle_confidence_mode", True)
            last_candidates = None
            last_lagtimes = lagtimes.copy()
            corrstep = (
                np.abs(trimmedcorrscale[1] - trimmedcorrscale[0])
                if len(trimmedcorrscale) > 1
                else optiondict["despeckle_thresh"] * 0.1
            )
            lag_change_tol = optiondict.get("despeckle_lag_change_tol", 0.25 * corrstep)
            lastnumdespeckled = 1000000
            medianlags = np.zeros(numspatiallocs, dtype=np.float64)
            candidate_mask_flat = np.zeros(numspatiallocs, dtype=bool)
            for despecklepass in range(optiondict["despeckle_passes"]):
                LGR.info(f"\n\n{similaritytype} despeckling subpass {despecklepass + 1}")
                if legacy_mode:
                    lagmap_flat = np.zeros(numspatiallocs, dtype=np.float64)
                    lagmap_flat[validvoxels] = lagtimes[:]
                    medianlags = ndimage.median_filter(
                        lagmap_flat.reshape(nativespaceshape), 3
                    ).reshape(numspatiallocs)
                    fixed_thresh = np.full(numspatiallocs, optiondict["despeckle_thresh"], dtype=np.float64)
                    candidate_mask_flat = np.zeros(numspatiallocs, dtype=bool)
                    candidate_mask_flat[validvoxels] = (
                        np.abs(lagmap_flat[validvoxels] - medianlags[validvoxels])
                        > optiondict["despeckle_thresh"]
                    )
                else:
                    lagmap_flat = np.zeros(numspatiallocs, dtype=np.float64)
                    lagmap_flat[validvoxels] = lagtimes[:]
                    (
                        medianlags,
                        fixed_thresh,
                        candidate_mask_flat,
                    ) = _build_despeckle_targets(
                        lagmap_flat,
                        validmask_flat,
                        nativespaceshape,
                        optiondict["despeckle_thresh"],
                    )
                    if confidence_mode:
                        candidate_mask_flat, confinfo = _refine_candidates_with_confidence(
                            candidate_mask_flat,
                            lagmap_flat,
                            medianlags,
                            fixed_thresh,
                            validvoxels,
                            lagstrengths,
                            lagsigma,
                            R2,
                            failreason,
                            min_r2=optiondict.get("despeckle_min_r2", 0.2),
                            min_strength=optiondict.get("despeckle_min_strength", 0.2),
                            max_sigma=optiondict.get("despeckle_max_sigma", 1.0e3),
                            strong_outlier_factor=optiondict.get(
                                "despeckle_strong_outlier_factor", 2.0
                            ),
                        )
                        LGR.info(
                            "\tconfidence filter: "
                            f"spatial={confinfo['spatial']}, "
                            f"lowconf={confinfo['conf_low']}, "
                            f"strong={confinfo['strong_outlier']}, "
                            f"combined={confinfo['combined']}"
                        )
                numdespeckled = int(np.sum(candidate_mask_flat[validvoxels]))
                LGR.info(
                    f"\tidentified {numdespeckled} candidates "
                    f"(median threshold={_safe_nanmedian(fixed_thresh):.4f})"
                )
                if numdespeckled == 0:
                    LGR.info("Nothing left to do! Terminating despeckling")
                    break
                if legacy_mode:
                    if not (lastnumdespeckled > numdespeckled > 0):
                        LGR.info("Legacy stop criterion met. Terminating despeckling")
                        break
                    lastnumdespeckled = numdespeckled
                else:
                    if last_candidates is not None and np.array_equal(candidate_mask_flat, last_candidates):
                        LGR.info("Candidate mask unchanged from previous pass. Terminating despeckling")
                        break

                if legacy_mode:
                    initlags = np.where(candidate_mask_flat, medianlags, -1000000.0)[validvoxels]
                    tide_util.disablemkl(optiondict["nprocs_fitcorr"], debug=optiondict["threaddebug"])
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
                        rt_floattype=rt_floattype,
                    )
                    tide_util.enablemkl(optiondict["mklthreads"], debug=optiondict["threaddebug"])
                else:
                    lagmap_after, icminfo = _optimize_despeckle_labels_icm(
                        lagmap_flat,
                        candidate_mask_flat,
                        validmask_flat,
                        validvoxels,
                        thepeakdict,
                        nativespaceshape,
                        max_candidates=int(optiondict.get("despeckle_peak_candidates", 3)),
                        max_iters=int(optiondict.get("despeckle_label_maxiters", 3)),
                        data_weight=float(optiondict.get("despeckle_label_data_weight", 1.0)),
                        smooth_weight=float(optiondict.get("despeckle_label_smooth_weight", 1.0)),
                        min_change=float(optiondict.get("despeckle_label_minchange", 1.0e-6)),
                    )
                    lagtimes[:] = lagmap_after[validvoxels]
                    voxelsprocessed_thispass = int(icminfo["changed"])
                    LGR.info(
                        "\tlabel optimization: "
                        f"changed={int(icminfo['changed'])}, "
                        f"iters={int(icminfo['iters'])}, "
                        f"vox_with_peaks={int(icminfo['vox_with_peaks'])}, "
                        f"vox_fallback={int(icminfo['vox_fallback'])}, "
                        f"mean_candidates={icminfo['mean_candidates']:.2f}"
                    )

                voxelsprocessed_fc_ds += voxelsprocessed_thispass
                optiondict[
                    "despecklemasksize_pass" + str(thepass) + "_d" + str(despecklepass + 1)
                ] = voxelsprocessed_thispass
                optiondict[
                    "despecklemaskpct_pass" + str(thepass) + "_d" + str(despecklepass + 1)
                ] = (100.0 * voxelsprocessed_thispass / optiondict["corrmasksize"])

                if not legacy_mode:
                    max_lag_change = np.max(np.abs(lagtimes - last_lagtimes))
                    LGR.info(f"\tmax lag change after despeckle subpass: {max_lag_change:.5f} s")
                    if max_lag_change <= lag_change_tol:
                        LGR.info(
                            f"Max lag change ({max_lag_change:.5f}) <= tolerance ({lag_change_tol:.5f}). "
                            "Terminating despeckling"
                        )
                        break
                    last_lagtimes[:] = lagtimes[:]
                    last_candidates = candidate_mask_flat.copy()

            internaldespeckleincludemask = np.where(
                candidate_mask_flat,
                medianlags.astype(np.float64),
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
