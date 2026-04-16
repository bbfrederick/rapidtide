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
from typing import Optional, Tuple

import numpy as np


def partition_3d(
    mask: np.ndarray,
    n_regions: int,
    connectivity: int = 6,
    seed: Optional[int] = None,
    balance_alpha: float = 0.0,
    jitter: float = 0.0,
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

    Returns
    -------
    labels : (X, Y, Z) int array
        Region labels in [0, n_regions-1], -1 outside mask.
    """
    assert mask.ndim == 3 and mask.dtype == np.uint16
    rng = np.random.default_rng(seed)

    X, Y, Z = mask.shape
    labels = -np.ones((X, Y, Z), dtype=np.int32)

    # --- neighbor offsets ---
    if connectivity == 6:
        nbrs = np.array(
            [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], dtype=int
        )
    elif connectivity == 18:
        nbrs = []
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if (dx, dy, dz) != (0, 0, 0) and (abs(dz) + abs(dy) + abs(dx) <= 2):
                        nbrs.append((dx, dy, dz))
        nbrs = np.array(nbrs, dtype=int)
    elif connectivity == 26:
        nbrs = []
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if (dx, dy, dz) != (0, 0, 0):
                        nbrs.append((dx, dy, dz))
        nbrs = np.array(nbrs, dtype=int)
    else:
        raise ValueError("connectivity must be 6, 18, or 26")

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
        for dx, dy, dz in nbrs:
            xx, yy, zz = x + dx, y + dy, z + dz
            if in_bounds(xx, yy, zz):
                li = labels[xx, yy, zz]
                if li != -1:
                    neigh_labels.append(li)

        if not neigh_labels:
            # not yet reachable; re-enqueue
            frontier.append((x, y, z))
            continue

        # unique labels
        uniq = np.unique(neigh_labels)

        # choose label (optionally balanced)
        if balance_alpha > 0:
            sizes = region_sizes[uniq] + 1.0
            probs = sizes ** (-balance_alpha)
            probs = probs / probs.sum()
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
