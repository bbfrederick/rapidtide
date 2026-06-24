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
import numpy as np
import pytest

from rapidtide.regionops import (
    _coerce_tensor_field,
    _directional_preference_score,
    partition_3d,
)


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
