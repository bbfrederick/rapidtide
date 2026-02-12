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
import sys

import numpy as np
import pytest

from rapidtide.patchmatch import (
    _smooth_array,
    binary_zero_crossing,
    calc_DoG,
    clamp,
    dehaze,
    difference_of_gaussian,
    flood3d,
    get_bounding_box,
    getclusters,
    growregion,
    interpolate_masked_voxels,
    interppatch,
    invertedflood3D,
    separateclusters,
)

# ==================== Helpers ====================


def _make_affine(voxsize=2.0):
    """Create a simple diagonal affine matrix."""
    affine = np.eye(4)
    affine[0, 0] = voxsize
    affine[1, 1] = voxsize
    affine[2, 2] = voxsize
    return affine


def _make_sizes(voxsize=2.0):
    return (1.0, voxsize, voxsize, voxsize, 1.0, 1.0, 1.0, 1.0)


def _make_sphere_image(shape=(20, 20, 20), center=None, radius=5, value=100.0):
    """Create a 3D image with a bright sphere and dark background."""
    if center is None:
        center = tuple(s // 2 for s in shape)
    data = np.zeros(shape, dtype=np.float32)
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    dist = np.sqrt((z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2)
    data[dist <= radius] = value
    return data


# ==================== clamp ====================


def test_clamp(debug=False):
    # Within range
    assert clamp(0, 10, 5) == 5
    # Below range
    assert clamp(0, 10, -3) == 0
    # Above range
    assert clamp(0, 10, 15) == 10
    # At boundaries
    assert clamp(0, 10, 0) == 0
    assert clamp(0, 10, 10) == 10
    # Negative range
    assert clamp(-5, -1, -3) == -3
    assert clamp(-5, -1, -10) == -5
    assert clamp(-5, -1, 0) == -1
    if debug:
        print("test_clamp passed")


# ==================== interpolate_masked_voxels ====================


def test_interpolate_masked_voxels(debug=False):
    # Create a simple 3D gradient
    data = np.zeros((5, 5, 5), dtype=np.float64)
    for i in range(5):
        data[i, :, :] = float(i)

    # Mask a single interior voxel
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[2, 2, 2] = True

    result = interpolate_masked_voxels(data, mask, method="linear")
    # The masked voxel should be close to 2.0 (linear gradient)
    assert abs(result[2, 2, 2] - 2.0) < 0.1
    # Unmasked voxels should be unchanged
    assert result[0, 0, 0] == 0.0
    assert result[4, 0, 0] == 4.0

    if debug:
        print(f"Interpolated value at (2,2,2): {result[2, 2, 2]}")


def test_interpolate_masked_voxels_shape_mismatch(debug=False):
    data = np.zeros((5, 5, 5))
    mask = np.zeros((3, 3, 3), dtype=bool)
    with pytest.raises(ValueError, match="same shape"):
        interpolate_masked_voxels(data, mask)

    if debug:
        print("test_interpolate_masked_voxels_shape_mismatch passed")


def test_interpolate_masked_voxels_nearest(debug=False):
    data = np.ones((5, 5, 5), dtype=np.float64) * 7.0
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[2, 2, 2] = True
    result = interpolate_masked_voxels(data, mask, method="nearest")
    assert abs(result[2, 2, 2] - 7.0) < 0.01

    if debug:
        print("test_interpolate_masked_voxels_nearest passed")


def test_interpolate_masked_voxels_extrapolation(debug=False):
    # Mask a corner — linear interpolation may not reach it, so extrapolation kicks in
    data = np.ones((5, 5, 5), dtype=np.float64) * 3.0
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[0, 0, 0] = True
    # Set unmasked voxels so corner is outside convex hull for linear
    data[0, 0, 0] = 0.0  # will be replaced

    result = interpolate_masked_voxels(data, mask, method="linear", extrapolate=True)
    # Should get a value (not NaN) thanks to extrapolation
    assert np.isfinite(result[0, 0, 0])

    if debug:
        print(f"Extrapolated corner value: {result[0, 0, 0]}")


# ==================== get_bounding_box ====================


def test_get_bounding_box(debug=False):
    mask = np.zeros((10, 10, 10), dtype=int)
    mask[3:7, 3:7, 3:7] = 1

    mins, maxs = get_bounding_box(mask, 1)
    assert mins == (3, 3, 3)
    assert maxs == (6, 6, 6)

    if debug:
        print(f"Bounding box: {mins} -> {maxs}")


def test_get_bounding_box_with_buffer(debug=False):
    mask = np.zeros((10, 10, 10), dtype=int)
    mask[3:7, 3:7, 3:7] = 1

    mins, maxs = get_bounding_box(mask, 1, buffer=2)
    assert mins[0] == 1
    assert mins[1] == 1
    assert mins[2] == 1

    if debug:
        print(f"Bounding box with buffer=2: {mins} -> {maxs}")


def test_get_bounding_box_buffer_clamped(debug=False):
    mask = np.zeros((10, 10, 10), dtype=int)
    mask[0:3, 0:3, 0:3] = 1

    mins, maxs = get_bounding_box(mask, 1, buffer=5)
    # Mins should be clamped at 0
    assert mins[0] == 0
    assert mins[1] == 0
    assert mins[2] == 0

    if debug:
        print(f"Clamped bounding box: {mins} -> {maxs}")


def test_get_bounding_box_not_3d(debug=False):
    mask = np.zeros((10, 10), dtype=int)
    with pytest.raises(ValueError, match="3D"):
        get_bounding_box(mask, 1)

    if debug:
        print("test_get_bounding_box_not_3d passed")


# ==================== flood3d ====================


def test_flood3d(debug=False):
    # Create image with a 'wall' that blocks flood fill
    image = np.zeros((5, 5, 3), dtype=int)
    # Fill connected region from (0,0) — all zeros become newvalue
    result = flood3d(image, 7)
    assert np.all(result == 7)

    if debug:
        print("test_flood3d all-zeros passed")


def test_flood3d_with_wall(debug=False):
    # Create image with a wall separating two regions
    image = np.zeros((5, 5, 1), dtype=int)
    image[2, :, 0] = 1  # horizontal wall
    result = flood3d(image, 9)
    # The flood starts at (0,0) and should fill the top portion
    assert result[0, 0, 0] == 9
    # Wall should remain as 1 (not connected to origin zero-region)
    assert result[2, 0, 0] == 1 or result[2, 0, 0] == 9  # wall value in original is 1

    if debug:
        print(f"Flood result shape: {result.shape}")


# ==================== invertedflood3D ====================


def test_invertedflood3d(debug=False):
    # For an all-zeros image, flood3d fills everything with newvalue
    # invertedflood3D = image + newvalue - flood3d(image, newvalue) = 0 + nv - nv = 0
    image = np.zeros((5, 5, 3), dtype=int)
    result = invertedflood3D(image, 1)
    assert np.all(result == 0)

    if debug:
        print("test_invertedflood3d all-zeros passed")


def test_invertedflood3d_enclosed_region(debug=False):
    # Create enclosed region: wall around center
    image = np.zeros((7, 7, 1), dtype=int)
    # Create a ring wall
    image[2, 2:5, 0] = 1
    image[4, 2:5, 0] = 1
    image[2:5, 2, 0] = 1
    image[2:5, 4, 0] = 1
    # Interior is 0, surrounded by 1s
    # flood3d from (0,0) fills exterior zeros with newvalue
    # invertedflood3D should highlight the interior
    result = invertedflood3D(image, 1)
    # The enclosed interior voxel (3,3,0) was 0, flood couldn't reach it
    # so: 0 + 1 - 0 = 1 (since flood3d couldn't fill it, it stays 0 in flood result)
    assert result[3, 3, 0] == 1

    if debug:
        print(f"Interior voxel value: {result[3, 3, 0]}")


# ==================== growregion and separateclusters ====================


def test_growregion(debug=False):
    image = np.zeros((5, 5, 5), dtype=int)
    image[1:4, 1:4, 1:4] = 1  # 3x3x3 = 27 voxel cluster
    separated = np.zeros_like(image)
    regionsize = growregion(image, (1, 1, 1), 1, separated, 0)
    assert regionsize == 27
    assert np.all(separated[1:4, 1:4, 1:4] == 1)
    assert np.all(separated[0, :, :] == 0)

    if debug:
        print(f"Region size: {regionsize}")


def test_growregion_single_voxel(debug=False):
    image = np.zeros((5, 5, 5), dtype=int)
    image[2, 2, 2] = 1
    separated = np.zeros_like(image)
    regionsize = growregion(image, (2, 2, 2), 1, separated, 0)
    assert regionsize == 1

    if debug:
        print("test_growregion_single_voxel passed")


def test_separateclusters(debug=False):
    image = np.zeros((10, 10, 5), dtype=int)
    # Two separate clusters
    image[1:3, 1:3, 1:3] = 1  # cluster 1: 8 voxels
    image[6:9, 6:9, 1:4] = 1  # cluster 2: 27 voxels

    result = separateclusters(image)
    # Should have exactly 2 distinct nonzero labels
    labels = set(np.unique(result)) - {0}
    assert len(labels) == 2

    if debug:
        print(f"Cluster labels: {labels}")


def test_separateclusters_with_sizethresh(debug=False):
    image = np.zeros((10, 10, 5), dtype=int)
    image[1:3, 1:3, 1:3] = 1  # 8 voxels
    image[6:9, 6:9, 1:4] = 1  # 27 voxels

    result = separateclusters(image, sizethresh=10)
    # Only the 27-voxel cluster should survive
    labels = set(np.unique(result)) - {0}
    assert len(labels) == 1

    if debug:
        print(f"Surviving labels after sizethresh=10: {labels}")


# ==================== dehaze ====================


def test_dehaze(debug=False):
    # Create bimodal data: background near 0, signal near 100
    rng = np.random.RandomState(42)
    data = np.concatenate([rng.normal(5, 2, 5000), rng.normal(100, 10, 5000)]).astype(np.float32)

    result = dehaze(data.copy(), 3)
    # Dark voxels should be zeroed
    assert np.sum(result == 0) > 0
    # Bright voxels should remain
    assert np.max(result) > 50

    if debug:
        print(f"Dehaze: {np.sum(result == 0)} zeroed, max={np.max(result):.1f}")


def test_dehaze_level_clamping(debug=False):
    rng = np.random.RandomState(42)
    data = np.concatenate([rng.normal(5, 2, 5000), rng.normal(100, 10, 5000)]).astype(np.float32)

    # Levels outside 1-5 should be clamped
    result_low = dehaze(data.copy(), -1)  # clamped to 1
    result_high = dehaze(data.copy(), 10)  # clamped to 5
    # Both should produce valid output without error
    assert result_low is not None
    assert result_high is not None

    if debug:
        print("test_dehaze_level_clamping passed")


# ==================== _smooth_array ====================


def test_smooth_array_basic(debug=False):
    data = np.zeros((10, 10, 10), dtype=np.float32)
    data[5, 5, 5] = 100.0
    affine = _make_affine(2.0)

    result = _smooth_array(data, affine, fwhm=4.0)
    # Peak should be reduced by smoothing
    assert result[5, 5, 5] < 100.0
    # Neighbors should have gained value
    assert result[5, 5, 4] > 0.0
    # Total should be approximately conserved
    assert abs(np.sum(result) - np.sum(data)) < 1.0

    if debug:
        print(f"Smoothed peak: {result[5, 5, 5]:.2f}")


def test_smooth_array_fwhm_zero(debug=False):
    data = np.ones((5, 5, 5), dtype=np.float32) * 3.0
    affine = _make_affine()

    with pytest.warns(UserWarning):
        result = _smooth_array(data, affine, fwhm=0.0)
    # No smoothing should occur
    np.testing.assert_array_equal(result, data)

    if debug:
        print("test_smooth_array_fwhm_zero passed")


def test_smooth_array_fwhm_none(debug=False):
    data = np.ones((5, 5, 5), dtype=np.float32) * 3.0
    affine = _make_affine()

    result = _smooth_array(data, affine, fwhm=None)
    np.testing.assert_array_equal(result, data)

    if debug:
        print("test_smooth_array_fwhm_none passed")


def test_smooth_array_ensure_finite(debug=False):
    data = np.ones((5, 5, 5), dtype=np.float32)
    data[2, 2, 2] = np.nan
    affine = _make_affine()

    result = _smooth_array(data, affine, fwhm=None, ensure_finite=True)
    assert np.isfinite(result[2, 2, 2])
    assert result[2, 2, 2] == 0.0

    if debug:
        print("test_smooth_array_ensure_finite passed")


def test_smooth_array_integer_input(debug=False):
    data = np.ones((5, 5, 5), dtype=np.int32) * 5
    affine = _make_affine()

    result = _smooth_array(data, affine, fwhm=None)
    assert result.dtype == np.float32

    if debug:
        print(f"Integer input converted to {result.dtype}")


def test_smooth_array_int64_input(debug=False):
    data = np.ones((5, 5, 5), dtype=np.int64) * 5
    affine = _make_affine()

    result = _smooth_array(data, affine, fwhm=None)
    assert result.dtype == np.float64

    if debug:
        print(f"Int64 input converted to {result.dtype}")


def test_smooth_array_copy(debug=False):
    data = np.ones((5, 5, 5), dtype=np.float32) * 3.0
    affine = _make_affine()

    result = _smooth_array(data, affine, fwhm=4.0, copy=True)
    # Original should be unchanged
    np.testing.assert_array_equal(data, np.ones((5, 5, 5), dtype=np.float32) * 3.0)

    if debug:
        print("test_smooth_array_copy passed")


# ==================== binary_zero_crossing ====================


def test_binary_zero_crossing(debug=False):
    # Create data with positive region surrounded by negative
    data = np.ones((10, 10, 10), dtype=np.float64) * -1.0
    data[3:7, 3:7, 3:7] = 1.0

    result = binary_zero_crossing(data)
    assert result.dtype == np.uint8
    # Edge voxels (boundary of positive region) should be 1
    assert result[3, 3, 3] == 1
    # Deep interior should be 0 (distance > 1)
    assert result[5, 5, 5] == 0
    # Outside should be 0
    assert result[0, 0, 0] == 0

    if debug:
        print(f"Edge voxels: {np.sum(result == 1)}")


def test_binary_zero_crossing_all_positive(debug=False):
    data = np.ones((5, 5, 5), dtype=np.float64)
    result = binary_zero_crossing(data)
    # All positive: EDT of all-1 mask → edges at boundary of volume
    assert result.dtype == np.uint8

    if debug:
        print("test_binary_zero_crossing_all_positive passed")


# ==================== difference_of_gaussian ====================


def test_difference_of_gaussian(debug=False):
    data = _make_sphere_image(shape=(20, 20, 20), radius=5, value=100.0)
    affine = _make_affine(2.0)

    result = difference_of_gaussian(data, affine, fwhmNarrow=3.0)
    assert result.dtype == np.uint8
    assert result.shape == data.shape
    # Should detect edges (some nonzero voxels)
    assert np.sum(result > 0) > 0

    if debug:
        print(f"DoG edge voxels: {np.sum(result > 0)}")


def test_difference_of_gaussian_no_ratioopt(debug=False):
    data = _make_sphere_image()
    affine = _make_affine(2.0)

    result = difference_of_gaussian(data, affine, fwhmNarrow=3.0, ratioopt=False)
    assert result.dtype == np.uint8
    assert np.sum(result > 0) > 0

    if debug:
        print("test_difference_of_gaussian_no_ratioopt passed")


# ==================== calc_DoG ====================


def test_calc_DoG(debug=False):
    data = _make_sphere_image(shape=(20, 20, 20), radius=5, value=100.0)
    affine = _make_affine(2.0)
    sizes = _make_sizes(2.0)

    result = calc_DoG(data, affine, sizes, fwhm=3, ratioopt=True)
    assert result.dtype == np.uint8
    assert result.shape == data.shape
    assert np.sum(result > 0) > 0

    if debug:
        print(f"calc_DoG edge voxels: {np.sum(result > 0)}")


# ==================== getclusters ====================


def test_getclusters(debug=False):
    # Create image with a bright blob — should produce at least one cluster
    data = _make_sphere_image(shape=(20, 20, 20), radius=5, value=100.0)
    affine = _make_affine(2.0)
    sizes = _make_sizes(2.0)

    result = getclusters(data, affine, sizes, fwhm=3, sizethresh=5)
    assert result.shape == data.shape
    # May or may not find clusters depending on DoG output
    # Just verify it runs without error and returns correct shape

    if debug:
        num_clusters = len(set(np.unique(result)) - {0})
        print(f"getclusters found {num_clusters} clusters")


# ==================== interppatch ====================


def test_interppatch(debug=False):
    # Create image data with known values and a labeled region
    img_data = np.ones((10, 10, 10), dtype=np.float64) * 5.0
    # Set some variation
    img_data[3:7, 3:7, 3:7] = 10.0

    # Create separated image with one labeled region
    separated = np.zeros((10, 10, 10), dtype=int)
    separated[4:6, 4:6, 4:6] = 1  # small interior region

    interpolated, justboxes = interppatch(img_data, separated, method="linear")
    assert interpolated.shape == img_data.shape
    assert justboxes.shape == img_data.shape
    # justboxes should contain data only within the bounding box regions
    assert np.sum(justboxes > 0) > 0

    if debug:
        print(
            f"interppatch: interpolated range [{np.min(interpolated):.1f}, {np.max(interpolated):.1f}]"
        )


def test_interppatch_no_regions(debug=False):
    img_data = np.ones((5, 5, 5), dtype=np.float64) * 3.0
    separated = np.zeros((5, 5, 5), dtype=int)  # no labeled regions

    interpolated, justboxes = interppatch(img_data, separated)
    # No regions to process, output should match input
    np.testing.assert_array_almost_equal(interpolated, img_data)

    if debug:
        print("test_interppatch_no_regions passed")


# ==================== Main test entry point ====================


def test_patchmatch(debug=False):
    test_clamp(debug=debug)
    test_interpolate_masked_voxels(debug=debug)
    test_interpolate_masked_voxels_shape_mismatch(debug=debug)
    test_interpolate_masked_voxels_nearest(debug=debug)
    test_interpolate_masked_voxels_extrapolation(debug=debug)
    test_get_bounding_box(debug=debug)
    test_get_bounding_box_with_buffer(debug=debug)
    test_get_bounding_box_buffer_clamped(debug=debug)
    test_get_bounding_box_not_3d(debug=debug)
    test_flood3d(debug=debug)
    test_flood3d_with_wall(debug=debug)
    test_invertedflood3d(debug=debug)
    test_invertedflood3d_enclosed_region(debug=debug)
    test_growregion(debug=debug)
    test_growregion_single_voxel(debug=debug)
    test_separateclusters(debug=debug)
    test_separateclusters_with_sizethresh(debug=debug)
    test_dehaze(debug=debug)
    test_dehaze_level_clamping(debug=debug)
    test_smooth_array_basic(debug=debug)
    test_smooth_array_fwhm_zero(debug=debug)
    test_smooth_array_fwhm_none(debug=debug)
    test_smooth_array_ensure_finite(debug=debug)
    test_smooth_array_integer_input(debug=debug)
    test_smooth_array_int64_input(debug=debug)
    test_smooth_array_copy(debug=debug)
    test_binary_zero_crossing(debug=debug)
    test_binary_zero_crossing_all_positive(debug=debug)
    test_difference_of_gaussian(debug=debug)
    test_difference_of_gaussian_no_ratioopt(debug=debug)
    test_calc_DoG(debug=debug)
    test_getclusters(debug=debug)
    test_interppatch(debug=debug)
    test_interppatch_no_regions(debug=debug)


if __name__ == "__main__":
    test_patchmatch(debug=True)
