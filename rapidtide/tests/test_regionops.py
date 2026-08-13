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

import numpy as np
import pytest

from rapidtide.regionops import (
    _coerce_tensor_field,
    _directional_preference_score,
    _neighbor_offsets,
    partition_3d,
)


def _countcomponents(themembership, theoffsets):
    """Count the connected components of a boolean volume under given offsets.

    Parameters
    ----------
    themembership : NDArray
        Boolean volume, True for voxels belonging to the region under test.
    theoffsets : NDArray
        Integer neighbour offsets, as returned by _neighbor_offsets.

    Returns
    -------
    int
        The number of connected components.
    """
    theremaining = {tuple(int(c) for c in thevoxel) for thevoxel in np.argwhere(themembership)}
    theshape = themembership.shape
    thecomponents = 0
    while theremaining:
        thecomponents += 1
        thequeue = deque([theremaining.pop()])
        while thequeue:
            thex, they, thez = thequeue.popleft()
            for thedx, thedy, thedz in theoffsets:
                theneighbor = (thex + thedx, they + thedy, thez + thedz)
                if not all(0 <= theneighbor[i] < theshape[i] for i in range(3)):
                    continue
                if theneighbor in theremaining:
                    theremaining.discard(theneighbor)
                    thequeue.append(theneighbor)
    return thecomponents


def test_coerce_tensor_field_from_six_component_layout():
    tensor_components = np.zeros((2, 3, 4, 6), dtype=np.float64)
    tensor_components[..., 0] = 4.0
    tensor_components[..., 1] = 2.0
    tensor_components[..., 2] = 1.0
    tensor_components[..., 3] = 0.5
    tensor_components[..., 4] = 0.25
    tensor_components[..., 5] = 0.125

    tensor_field = _coerce_tensor_field(tensor_components, (2, 3, 4))

    assert tensor_field.shape == (2, 3, 4, 3, 3)
    np.testing.assert_allclose(
        tensor_field[0, 0, 0],
        [[4.0, 0.5, 0.25], [0.5, 2.0, 0.125], [0.25, 0.125, 1.0]],
    )


def test_coerce_tensor_field_rejects_mismatched_shape():
    tensor_components = np.zeros((2, 3, 5, 6), dtype=np.float64)

    with pytest.raises(ValueError):
        _coerce_tensor_field(tensor_components, (2, 3, 4))


def test_directional_preference_score_favors_major_axis():
    tensor = np.diag([9.0, 1.0, 1.0])

    x_score = _directional_preference_score(tensor, np.array([1.0, 0.0, 0.0]))
    y_score = _directional_preference_score(tensor, np.array([0.0, 1.0, 0.0]))

    assert x_score > y_score


def test_neighbor_offsets_match_their_connectivity_definitions():
    """6, 18 and 26 connectivity are face, face-plus-edge, and face-edge-plus-corner
    neighbourhoods respectively, which is exactly a cut on the L1 distance."""
    thesix = _neighbor_offsets(6)
    theeighteen = _neighbor_offsets(18)
    thetwentysix = _neighbor_offsets(26)

    assert len(thesix) == 6
    assert len(theeighteen) == 18
    assert len(thetwentysix) == 26

    # faces only, edges and faces, then everything
    assert set(np.abs(thesix).sum(axis=1)) == {1}
    assert set(np.abs(theeighteen).sum(axis=1)) == {1, 2}
    assert set(np.abs(thetwentysix).sum(axis=1)) == {1, 2, 3}

    # no offset is the origin, none is repeated, and each set nests in the next
    for theoffsets in (thesix, theeighteen, thetwentysix):
        theset = {tuple(int(c) for c in theoffset) for theoffset in theoffsets}
        assert len(theset) == len(theoffsets)
        assert (0, 0, 0) not in theset
        # symmetry: the growth loop relies on a voxel's neighbour having it back
        assert all(tuple(-np.array(theoffset)) in theset for theoffset in theset)
    assert {tuple(o) for o in thesix} < {tuple(o) for o in theeighteen}
    assert {tuple(o) for o in theeighteen} < {tuple(o) for o in thetwentysix}


def test_neighbor_offsets_rejects_other_connectivities():
    """Anything other than the three defined neighbourhoods is a caller error."""
    for thebadvalue in (0, 4, 8, 27, -6):
        with pytest.raises(ValueError, match="connectivity must be 6, 18, or 26"):
            _neighbor_offsets(thebadvalue)


def test_coerce_tensor_field_passes_none_through():
    """No field means no anisotropy, which has to stay distinguishable from a field
    of zeros - the latter would make every direction equally unattractive."""
    assert _coerce_tensor_field(None, (2, 3, 4)) is None


def test_coerce_tensor_field_accepts_full_and_flattened_layouts():
    """The (X, Y, Z, 3, 3) layout is returned untouched and the (X, Y, Z, 9) layout
    is folded into it in row major order."""
    theshape = (2, 3, 4)
    thefull = np.arange(np.prod(theshape) * 9, dtype=np.float64).reshape(theshape + (3, 3))
    theresult = _coerce_tensor_field(thefull, theshape)
    assert theresult.shape == theshape + (3, 3)
    np.testing.assert_array_equal(theresult, thefull)

    theflat = thefull.reshape(theshape + (9,))
    theresult = _coerce_tensor_field(theflat, theshape)
    assert theresult.shape == theshape + (3, 3)
    np.testing.assert_array_equal(theresult, thefull)


def test_coerce_tensor_field_rejects_an_unusable_trailing_dimension():
    """Only 6, 9, or a trailing (3, 3) can be interpreted as a tensor."""
    for thebadfield in (
        np.zeros((2, 3, 4, 5)),
        np.zeros((2, 3, 4, 7)),
        np.zeros((2, 3, 4, 3, 2)),
    ):
        with pytest.raises(ValueError, match="anisotropy_field must have shape"):
            _coerce_tensor_field(thebadfield, (2, 3, 4))


def test_coerce_tensor_field_six_component_layout_is_symmetric():
    """The 6 component layout stores only the upper triangle, so the result has to
    come back symmetric or the off diagonal terms are being dropped."""
    thecomponents = np.zeros((2, 2, 2, 6), dtype=np.float64)
    thecomponents[..., 3:] = [0.5, 0.25, 0.125]
    thetensor = _coerce_tensor_field(thecomponents, (2, 2, 2))[0, 0, 0]
    np.testing.assert_allclose(thetensor, thetensor.T)


def test_directional_preference_score_handles_degenerate_input():
    """A zero step and a non positive trace both mean "no directional information",
    which must score as neutral rather than dividing by zero."""
    assert _directional_preference_score(np.diag([9.0, 1.0, 1.0]), np.zeros(3)) == 1.0
    assert _directional_preference_score(np.zeros((3, 3)), np.array([1.0, 0.0, 0.0])) == 1.0
    assert _directional_preference_score(np.diag([-1.0, -1.0, -1.0]), np.array([1.0, 0, 0])) == 1.0


def test_directional_preference_score_is_bounded_and_scale_free():
    """The score is documented as lying in [0, 1] for positive semidefinite tensors,
    and as a ratio to the trace it must not care about the tensor's overall scale."""
    thetensor = np.diag([9.0, 3.0, 1.0])
    therng = np.random.default_rng(4)
    for dummy in range(50):
        thestep = therng.normal(size=3)
        thescore = _directional_preference_score(thetensor, thestep)
        assert 0.0 <= thescore <= 1.0
    # scaling the tensor leaves the score alone, and step length does not matter
    for thescale in (0.01, 1.0, 100.0):
        assert np.isclose(
            _directional_preference_score(thescale * thetensor, np.array([1.0, 0.0, 0.0])),
            _directional_preference_score(thetensor, np.array([5.0, 0.0, 0.0])),
        )
    # An asymmetric tensor scores the same as its transpose.  Note this is a property
    # of the quadratic form itself, not of the explicit symmetrisation in the code:
    # u @ A @ u discards the antisymmetric part of A, and trace(A) equals
    # trace(0.5 * (A + A.T)), so the symmetrisation cannot change the ratio.  It is
    # documentation rather than arithmetic, and no test can distinguish it.
    theasymmetric = np.array([[2.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    thestep = np.array([1.0, 1.0, 0.0])
    assert np.isclose(
        _directional_preference_score(theasymmetric, thestep),
        _directional_preference_score(theasymmetric.T, thestep),
    )


def test_partition_3d_rejects_impossible_requests():
    """Guard rails: more regions than voxels, a negative anisotropy strength, and an
    undefined connectivity are all caller errors rather than silent degradation."""
    themask = np.ones((2, 2, 2), dtype=np.uint16)
    with pytest.raises(ValueError, match="fewer voxels than n_regions"):
        partition_3d(themask, n_regions=9, seed=0)
    with pytest.raises(ValueError, match="anisotropy_strength must be >= 0"):
        partition_3d(themask, n_regions=2, seed=0, anisotropy_strength=-1.0)
    with pytest.raises(ValueError, match="connectivity must be 6, 18, or 26"):
        partition_3d(themask, n_regions=2, seed=0, connectivity=7)


def test_partition_3d_labels_exactly_the_mask():
    """Every in mask voxel gets a region and every out of mask voxel keeps -1.  A
    partition that leaks outside the mask, or leaves holes inside it, is not one."""
    therng = np.random.default_rng(7)
    themask = np.zeros((9, 8, 7), dtype=np.uint16)
    themask[1:8, 1:7, 1:6] = 1
    # punch out a void so the domain is not a plain box
    themask[3:5, 3:5, 2:4] = 0

    thelabels = partition_3d(themask, n_regions=5, connectivity=6, seed=int(therng.integers(1000)))

    assert thelabels.shape == themask.shape
    assert np.all(thelabels[themask == 0] == -1), "labels leaked outside the mask"
    assert np.all(thelabels[themask > 0] >= 0), "in mask voxels were left unassigned"
    assert np.all(thelabels[themask > 0] < 5), "a label outside the requested range appeared"
    assert len(np.unique(thelabels[themask > 0])) == 5


def test_partition_3d_regions_are_connected():
    """The headline promise is N *simply connected* regions.  A region that comes
    back as two islands is a partition, but not the one that was asked for."""
    for theconnectivity in (6, 18, 26):
        themask = np.zeros((10, 9, 8), dtype=np.uint16)
        themask[1:9, 1:8, 1:7] = 1
        thelabels = partition_3d(
            themask, n_regions=4, connectivity=theconnectivity, seed=20 + theconnectivity
        )
        theoffsets = _neighbor_offsets(theconnectivity)
        for theregion in range(4):
            thecomponents = _countcomponents(thelabels == theregion, theoffsets)
            assert thecomponents == 1, (
                f"region {theregion} came back in {thecomponents} pieces at "
                f"connectivity {theconnectivity}"
            )


def test_partition_3d_is_reproducible_from_its_seed():
    """Same seed, same partition; different seed, different partition.  Without the
    first there is no reproducible analysis, and without the second the seed is not
    actually reaching the RNG."""
    themask = np.ones((7, 7, 7), dtype=np.uint16)
    thefirst = partition_3d(themask, n_regions=4, seed=99)
    thesecond = partition_3d(themask, n_regions=4, seed=99)
    theother = partition_3d(themask, n_regions=4, seed=100)

    np.testing.assert_array_equal(thefirst, thesecond)
    assert not np.array_equal(thefirst, theother), "the seed is not reaching the RNG"


def test_partition_3d_balance_alpha_evens_out_region_sizes():
    """balance_alpha biases growth by (size + 1)**-alpha, so raising it has to make
    the region volumes more equal.  It is the only knob that does, so if it stopped
    working nothing else would reveal it."""
    themask = np.ones((12, 12, 12), dtype=np.uint16)
    thespreads = {}
    for thealpha in (0.0, 2.0):
        theranges = []
        for theseed in range(6):
            thelabels = partition_3d(themask, n_regions=5, seed=theseed, balance_alpha=thealpha)
            thesizes = np.bincount(thelabels[themask > 0], minlength=5)
            theranges.append(thesizes.max() - thesizes.min())
        thespreads[thealpha] = float(np.mean(theranges))

    assert thespreads[2.0] < thespreads[0.0], (
        f"balancing did not even out the regions: spread {thespreads[0.0]:.0f} unbalanced "
        f"versus {thespreads[2.0]:.0f} balanced"
    )


def test_partition_3d_jitter_still_produces_a_valid_partition():
    """jitter reshuffles the frontier to break up lattice artefacts.  It changes the
    outcome, but it must not break the invariants."""
    themask = np.ones((8, 8, 8), dtype=np.uint16)
    theplain = partition_3d(themask, n_regions=4, seed=5, jitter=0.0)
    thejittered = partition_3d(themask, n_regions=4, seed=5, jitter=1.0)

    assert not np.array_equal(theplain, thejittered), "jitter had no effect"
    assert np.all(thejittered[themask > 0] >= 0)
    assert len(np.unique(thejittered[themask > 0])) == 4
    theoffsets = _neighbor_offsets(6)
    for theregion in range(4):
        assert _countcomponents(thejittered == theregion, theoffsets) == 1


def test_partition_3d_survives_weights_underflowing_to_zero():
    """A large anisotropy_strength drives exp(-strength * penalty) to zero for every
    candidate, leaving no weight to normalise.  The fall back to a uniform choice has
    to keep the partition valid rather than dividing by zero."""
    themask = np.ones((6, 6, 6), dtype=np.uint16)
    thetensorfield = np.zeros((6, 6, 6, 6), dtype=np.float64)
    thetensorfield[..., 0] = 9.0
    thetensorfield[..., 1] = 1.0
    thetensorfield[..., 2] = 1.0

    thelabels = partition_3d(
        themask,
        n_regions=3,
        seed=3,
        anisotropy_field=thetensorfield,
        anisotropy_strength=5000.0,
    )
    assert np.all(thelabels >= 0)
    assert len(np.unique(thelabels)) == 3


def test_partition_3d_anisotropy_elongates_regions_along_the_preferred_axis():
    """The point of the tensor field is to bias growth directionally.  A field
    favouring x must produce regions that are longer in x than in y, or the field is
    being accepted and then ignored."""
    themask = np.ones((14, 14, 6), dtype=np.uint16)
    thetensorfield = np.zeros((14, 14, 6, 6), dtype=np.float64)
    thetensorfield[..., 0] = 25.0  # xx
    thetensorfield[..., 1] = 1.0  # yy
    thetensorfield[..., 2] = 1.0  # zz

    theisotropicratios, theanisotropicratios = [], []
    for theseed in range(8):
        for thestrength, thebucket in ((0.0, theisotropicratios), (12.0, theanisotropicratios)):
            thelabels = partition_3d(
                themask,
                n_regions=4,
                seed=theseed,
                anisotropy_field=thetensorfield,
                anisotropy_strength=thestrength,
            )
            for theregion in range(4):
                thecoords = np.argwhere(thelabels == theregion)
                if len(thecoords) < 10:
                    continue
                thexextent = np.ptp(thecoords[:, 0]) + 1
                theyextent = np.ptp(thecoords[:, 1]) + 1
                thebucket.append(thexextent / theyextent)

    assert np.mean(theanisotropicratios) > np.mean(theisotropicratios), (
        f"anisotropy did not elongate the regions: x/y extent ratio "
        f"{np.mean(theisotropicratios):.2f} isotropic versus "
        f"{np.mean(theanisotropicratios):.2f} anisotropic"
    )


def test_partition_3d_accepts_anisotropy_field():
    mask = np.ones((5, 5, 5), dtype=np.uint16)
    tensor_field = np.zeros((5, 5, 5, 6), dtype=np.float64)
    tensor_field[..., 0] = 9.0
    tensor_field[..., 1] = 1.0
    tensor_field[..., 2] = 1.0

    labels = partition_3d(
        mask,
        n_regions=4,
        connectivity=6,
        seed=1234,
        balance_alpha=0.5,
        jitter=0.0,
        anisotropy_field=tensor_field,
        anisotropy_strength=3.0,
    )

    assert labels.shape == mask.shape
    assert np.all(labels[mask > 0] >= 0)
    assert len(np.unique(labels[mask > 0])) == 4
