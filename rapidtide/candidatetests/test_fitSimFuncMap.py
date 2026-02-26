#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from rapidtide.notreadyforprimetime.fitSimFuncMap import (
    _optimize_despeckle_labels_icm,
    _refine_candidates_with_confidence,
)


def test_confidence_filter_rejects_high_confidence_mild_outlier():
    spatial = np.array([False, False, True, False, False], dtype=bool)
    lagmap = np.array([0.0, 0.0, 6.0, 0.0, 0.0], dtype=float)
    med = np.zeros_like(lagmap)
    thresh = np.full_like(lagmap, 5.0)
    validvox = np.arange(5, dtype=int)

    lagstrengths = np.array([0.8, 0.8, 0.8, 0.8, 0.8], dtype=float)
    lagsigma = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
    r2 = np.array([0.9, 0.9, 0.9, 0.9, 0.9], dtype=float)
    failreason = np.zeros(5, dtype=np.uint16)

    combined, info = _refine_candidates_with_confidence(
        spatial,
        lagmap,
        med,
        thresh,
        validvox,
        lagstrengths,
        lagsigma,
        r2,
        failreason,
        min_r2=0.2,
        min_strength=0.2,
        max_sigma=10.0,
        strong_outlier_factor=2.0,
    )
    assert info["spatial"] == 1
    assert info["combined"] == 0
    assert not combined[2]


def test_confidence_filter_keeps_low_confidence_outlier():
    spatial = np.array([False, False, True, False, False], dtype=bool)
    lagmap = np.array([0.0, 0.0, 6.0, 0.0, 0.0], dtype=float)
    med = np.zeros_like(lagmap)
    thresh = np.full_like(lagmap, 5.0)
    validvox = np.arange(5, dtype=int)

    lagstrengths = np.array([0.8, 0.8, 0.8, 0.8, 0.8], dtype=float)
    lagsigma = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
    r2 = np.array([0.9, 0.9, 0.1, 0.9, 0.9], dtype=float)
    failreason = np.zeros(5, dtype=np.uint16)

    combined, info = _refine_candidates_with_confidence(
        spatial,
        lagmap,
        med,
        thresh,
        validvox,
        lagstrengths,
        lagsigma,
        r2,
        failreason,
        min_r2=0.2,
        min_strength=0.2,
        max_sigma=10.0,
        strong_outlier_factor=2.0,
    )
    assert info["spatial"] == 1
    assert info["combined"] == 1
    assert combined[2]


def test_confidence_filter_keeps_strong_outlier_even_if_high_confidence():
    spatial = np.array([False, False, True, False, False], dtype=bool)
    lagmap = np.array([0.0, 0.0, 12.0, 0.0, 0.0], dtype=float)
    med = np.zeros_like(lagmap)
    thresh = np.full_like(lagmap, 5.0)
    validvox = np.arange(5, dtype=int)

    lagstrengths = np.array([0.8, 0.8, 0.8, 0.8, 0.8], dtype=float)
    lagsigma = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
    r2 = np.array([0.9, 0.9, 0.9, 0.9, 0.9], dtype=float)
    failreason = np.zeros(5, dtype=np.uint16)

    combined, info = _refine_candidates_with_confidence(
        spatial,
        lagmap,
        med,
        thresh,
        validvox,
        lagstrengths,
        lagsigma,
        r2,
        failreason,
        min_r2=0.2,
        min_strength=0.2,
        max_sigma=10.0,
        strong_outlier_factor=2.0,
    )
    assert info["spatial"] == 1
    assert info["strong_outlier"] == 1
    assert info["combined"] == 1
    assert combined[2]


def test_optimize_despeckle_labels_icm_selects_spatially_consistent_peak():
    # 1D spatial layout with middle voxel as candidate
    lagmap_flat = np.array([0.0, 4.0, 0.0], dtype=float)
    candidate_mask_flat = np.array([False, True, False], dtype=bool)
    validmask_flat = np.array([True, True, True], dtype=bool)
    validvoxels = np.array([0, 1, 2], dtype=int)
    peakdict = {"1": [[4.0, 0.8, 0.9], [0.0, 0.5, 0.7]]}

    optimized, info = _optimize_despeckle_labels_icm(
        lagmap_flat,
        candidate_mask_flat,
        validmask_flat,
        validvoxels,
        peakdict,
        nativespaceshape=(3,),
        max_candidates=2,
        max_iters=4,
        data_weight=0.0,
        smooth_weight=1.0,
    )
    assert optimized[1] == 0.0
    assert int(info["changed"]) == 1


def test_optimize_despeckle_labels_icm_keeps_current_when_no_candidates():
    lagmap_flat = np.array([1.0, 2.0, 3.0], dtype=float)
    candidate_mask_flat = np.array([False, False, False], dtype=bool)
    validmask_flat = np.array([True, True, True], dtype=bool)
    validvoxels = np.array([0, 1, 2], dtype=int)
    optimized, info = _optimize_despeckle_labels_icm(
        lagmap_flat,
        candidate_mask_flat,
        validmask_flat,
        validvoxels,
        peakdict=None,
        nativespaceshape=(3,),
    )
    assert np.allclose(optimized, lagmap_flat)
    assert int(info["changed"]) == 0
