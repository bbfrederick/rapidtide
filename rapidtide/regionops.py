#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   Copyright 2026-2026 Blaise Frederick
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
from typing import Optional

import numpy as np


def _neighbor_offsets(connectivity: int) -> np.ndarray:
    """Return integer neighbor offsets for a requested 3D connectivity."""
    if connectivity == 6:
        return np.array(
            [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], dtype=int
        )
    if connectivity == 18:
        nbrs = []
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if (dx, dy, dz) != (0, 0, 0) and (abs(dz) + abs(dy) + abs(dx) <= 2):
                        nbrs.append((dx, dy, dz))
        return np.array(nbrs, dtype=int)
    if connectivity == 26:
        nbrs = []
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if (dx, dy, dz) != (0, 0, 0):
                        nbrs.append((dx, dy, dz))
        return np.array(nbrs, dtype=int)
    raise ValueError("connectivity must be 6, 18, or 26")


def _coerce_tensor_field(
    anisotropy_field: Optional[np.ndarray], expected_shape: tuple[int, int, int]
) -> Optional[np.ndarray]:
    """
    Convert a tensor field into an ``(X, Y, Z, 3, 3)`` array.

    Accepted input layouts are ``(X, Y, Z, 3, 3)``, ``(X, Y, Z, 6)`` with
    ``[xx, yy, zz, xy, xz, yz]`` ordering, and ``(X, Y, Z, 9)`` storing the
    flattened 3x3 tensor.
    """
    if anisotropy_field is None:
        return None

    field = np.asarray(anisotropy_field, dtype=np.float64)
    if field.shape[:3] != expected_shape:
        raise ValueError(
            f"anisotropy_field spatial shape {field.shape[:3]} does not match mask shape "
            f"{expected_shape}"
        )

    if field.ndim == 5 and field.shape[3:] == (3, 3):
        return field

    tensor_field = np.zeros(expected_shape + (3, 3), dtype=np.float64)
    if field.ndim == 4 and field.shape[3] == 6:
        tensor_field[..., 0, 0] = field[..., 0]
        tensor_field[..., 1, 1] = field[..., 1]
        tensor_field[..., 2, 2] = field[..., 2]
        tensor_field[..., 0, 1] = tensor_field[..., 1, 0] = field[..., 3]
        tensor_field[..., 0, 2] = tensor_field[..., 2, 0] = field[..., 4]
        tensor_field[..., 1, 2] = tensor_field[..., 2, 1] = field[..., 5]
        return tensor_field

    if field.ndim == 4 and field.shape[3] == 9:
        return field.reshape(expected_shape + (3, 3))

    raise ValueError(
        "anisotropy_field must have shape (X, Y, Z, 3, 3), (X, Y, Z, 6), or (X, Y, Z, 9)"
    )


def _directional_preference_score(tensor: np.ndarray, step: np.ndarray) -> float:
    """
    Convert a local tensor and step direction into a bounded anisotropic score.

    The score ranges from 0 to 1 for positive semidefinite tensors and is larger
    when the step aligns with high-conductance / low-penalty directions.
    """
    stepvec = np.asarray(step, dtype=np.float64)
    stepnorm = np.linalg.norm(stepvec)
    if stepnorm == 0.0:
        return 1.0
    unitstep = stepvec / stepnorm
    symtensor = 0.5 * (tensor + tensor.T)
    trace = float(np.trace(symtensor))
    if trace <= 0.0:
        return 1.0
    preference = float(unitstep @ symtensor @ unitstep)
    return float(np.clip(preference / trace, 0.0, 1.0))


def partition_3d(
    mask: np.ndarray,
    n_regions: int,
    connectivity: int = 6,
    seed: Optional[int] = None,
    balance_alpha: float = 0.0,
    jitter: float = 0.0,
    anisotropy_field: Optional[np.ndarray] = None,
    anisotropy_strength: float = 0.0,
) -> np.ndarray:
    """
    Partition a 3D mask into N random simply connected regions.

    Parameters
    ----------
    mask : (X, Y, Z) uint16 array
        True for voxels inside the domain.
    n_regions : int
        Number of regions.
    connectivity : {6, 18, 26}
        Neighborhood definition.
    seed : int or None
        RNG seed for reproducibility.
    balance_alpha : float >= 0
        If > 0, biases growth to balance volumes:
        p(i) ∝ (|R_i| + 1)^(-alpha).
    jitter : float >= 0
        If > 0, adds small random priority to reduce lattice artifacts.
    anisotropy_field : array-like or None
        Optional 3D tensor field that biases growth along preferred directions.
        Accepted layouts are ``(X, Y, Z, 3, 3)``, ``(X, Y, Z, 6)`` with
        ``[xx, yy, zz, xy, xz, yz]`` ordering, and ``(X, Y, Z, 9)``.
    anisotropy_strength : float >= 0
        Strength of the anisotropic weighting. Larger values more strongly favor
        low-penalty / high-conductance directions defined by ``anisotropy_field``.

    Returns
    -------
    labels : (X, Y, Z) int array
        Region labels in [0, n_regions-1], -1 outside mask.
    """
    assert mask.ndim == 3 and mask.dtype == np.uint16
    rng = np.random.default_rng(seed)

    X, Y, Z = mask.shape
    labels = -np.ones((X, Y, Z), dtype=np.int32)
    tensor_field = _coerce_tensor_field(anisotropy_field, (X, Y, Z))
    if anisotropy_strength < 0.0:
        raise ValueError("anisotropy_strength must be >= 0")

    # --- neighbor offsets ---
    nbrs = _neighbor_offsets(connectivity)

    # --- helper to check bounds ---
    def in_bounds(x, y, z):
        return (0 <= z < Z) and (0 <= y < Y) and (0 <= x < X)

    # --- choose N random seed voxels inside mask ---
    coords = np.argwhere(mask > 0)
    if len(coords) < n_regions:
        raise ValueError("mask has fewer voxels than n_regions")

    seed_idx = rng.choice(len(coords), size=n_regions, replace=False)
    seeds = coords[seed_idx]

    # --- initialize ---
    frontier = deque()
    region_sizes = np.zeros(n_regions, dtype=np.int64)

    for i, (x, y, z) in enumerate(seeds):
        labels[x, y, z] = i
        region_sizes[i] = 1
        # push neighbors to frontier
        for dx, dy, dz in nbrs:
            xx, yy, zz = x + dx, y + dy, z + dz
            if in_bounds(xx, yy, zz) and mask[xx, yy, zz] and labels[xx, yy, zz] == -1:
                frontier.append((xx, yy, zz))

    # optional jitter: maintain a parallel random priority queue via shuffling chunks
    def maybe_shuffle(q: deque, prob=0.1):
        if jitter > 0 and rng.random() < prob:
            tmp = list(q)
            rng.shuffle(tmp)
            q.clear()
            q.extend(tmp)

    # --- growth loop ---
    while frontier:
        x, y, z = frontier.popleft()
        if labels[x, y, z] != -1:
            continue

        # gather neighboring labels
        neigh_labels = []
        anisotropic_scores = {}
        local_tensor = None if tensor_field is None else tensor_field[x, y, z, :, :]
        for dx, dy, dz in nbrs:
            xx, yy, zz = x + dx, y + dy, z + dz
            if in_bounds(xx, yy, zz):
                li = labels[xx, yy, zz]
                if li != -1:
                    neigh_labels.append(li)
                    if local_tensor is not None:
                        score = _directional_preference_score(local_tensor, np.array([dx, dy, dz]))
                        anisotropic_scores.setdefault(int(li), []).append(score)

        if not neigh_labels:
            # not yet reachable; re-enqueue
            frontier.append((x, y, z))
            continue

        # unique labels
        uniq = np.unique(neigh_labels)

        # choose label (optionally balanced)
        weights = np.ones(len(uniq), dtype=np.float64)
        if balance_alpha > 0:
            sizes = region_sizes[uniq] + 1.0
            weights *= sizes ** (-balance_alpha)
        if local_tensor is not None and anisotropy_strength > 0.0:
            direction_weights = np.ones(len(uniq), dtype=np.float64)
            for idx, label in enumerate(uniq):
                mean_score = np.mean(anisotropic_scores.get(int(label), [1.0 / 3.0]))
                penalty = 1.0 - mean_score
                direction_weights[idx] = np.exp(-anisotropy_strength * penalty)
            weights *= direction_weights
        if np.any(weights > 0):
            probs = weights / weights.sum()
            chosen = rng.choice(uniq, p=probs)
        else:
            chosen = uniq[rng.integers(len(uniq))]

        labels[x, y, z] = int(chosen)
        region_sizes[chosen] += 1

        # expand frontier
        for dx, dy, dz in nbrs:
            xx, yy, zz = (
                x + dx,
                y + dy,
                z + dz,
            )
            if in_bounds(xx, yy, zz) and mask[xx, yy, zz] and labels[xx, yy, zz] == -1:
                frontier.append((xx, yy, zz))

        maybe_shuffle(frontier)

    return labels
