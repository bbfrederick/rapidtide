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
import numpy as np

import rapidtide.workflows.unwrapdelay as uw

SIDELOBE = 12.0


def _waveform(t):
    """A waveform with a deliberate sidelobe one SIDELOBE period out."""
    return np.exp(-((t / 3.0) ** 2)) + 0.6 * np.exp(-(((t - SIDELOBE) / 3.0) ** 2))


def _makemovie(shape=(12, 12, 8), voxdim=2.0, speed=6.0, numlags=90, lagstep=0.5):
    """A linear ramp delay field, so the truth is known everywhere."""
    lags = -5.0 + np.arange(numlags) * lagstep
    grids = np.indices(shape).astype(float)
    tau = grids[0] * voxdim / speed
    corr = _waveform(lags[None, None, None, :] - tau[..., None])
    mask = np.ones(shape, dtype=np.uint16)
    return corr, lags, tau, mask, (voxdim, voxdim, voxdim)


def test_findcandidatepeaks(debug=False):
    """Both the main lobe and the sidelobe must be found, main lobe first."""
    corr, lags, tau, mask, vox = _makemovie()
    cl, ca = uw.findcandidatepeaks(corr, lags, 4)
    # strongest candidate should be the true delay
    err = np.abs(cl[..., 0] - tau)
    assert np.median(err) < 0.5 * (
        lags[1] - lags[0]
    ), f"main lobe not found, median err {np.median(err)}"
    # second candidate should be the sidelobe
    gap = cl[..., 1] - cl[..., 0]
    good = np.isfinite(gap)
    assert np.median(np.abs(gap[good] - SIDELOBE)) < 1.0, "sidelobe not found as second candidate"


def test_unwrap_recovers_injected_sidelobe_errors(debug=False):
    """Corrupt voxels so the sidelobe outranks the main lobe; unwrapping must fix them."""
    corr, lags, tau, mask, vox = _makemovie()
    cl, ca = uw.findcandidatepeaks(corr, lags, 4)

    rng = np.random.RandomState(3)
    corrupt = rng.rand(*mask.shape) < 0.2
    cl2, ca2 = cl.copy(), ca.copy()
    idx = np.array(np.nonzero(corrupt)).T
    for i, j, k in idx:
        cl2[i, j, k, [0, 1]] = cl2[i, j, k, [1, 0]]
        ca2[i, j, k, [0, 1]] = ca2[i, j, k, [1, 0]]

    naive = np.abs(cl2[..., 0] - tau) > 2.0
    assert naive.sum() > 10, "the injection did not actually break naive peak picking"

    # flow free smoothness prior is enough here, as on real data
    got, changed, conf = uw.unwrapdelaymap(cl2, ca2, None, mask, vox, showprogressbar=False)
    bad = np.abs(got - tau) > 2.0
    if debug:
        print(f"naive bad {naive.sum()}, unwrapped bad {bad.sum()}")
    assert (
        bad.sum() < 0.1 * naive.sum()
    ), f"unwrapping fixed only {naive.sum() - bad.sum()} of {naive.sum()} injected errors"


def test_unwrap_with_flow_prior(debug=False):
    """The flow prior path must also run and recover the injected errors."""
    corr, lags, tau, mask, vox = _makemovie()
    cl, ca = uw.findcandidatepeaks(corr, lags, 4)
    rng = np.random.RandomState(5)
    corrupt = rng.rand(*mask.shape) < 0.2
    for i, j, k in np.array(np.nonzero(corrupt)).T:
        cl[i, j, k, [0, 1]] = cl[i, j, k, [1, 0]]
        ca[i, j, k, [0, 1]] = ca[i, j, k, [1, 0]]
    # true velocity: flow along +x at the known speed
    vel = np.zeros(mask.shape + (3,))
    vel[..., 0] = 6.0
    got, changed, conf = uw.unwrapdelaymap(cl, ca, vel, mask, vox, showprogressbar=False)
    bad = (np.abs(got - tau) > 2.0).sum()
    assert bad < 0.1 * corrupt.sum(), f"{bad} voxels still wrong after flow guided unwrap"


def test_unwrapdelay(debug=False):
    test_findcandidatepeaks(debug=debug)
    test_unwrap_recovers_injected_sidelobe_errors(debug=debug)
    test_unwrap_with_flow_prior(debug=debug)


if __name__ == "__main__":
    test_unwrapdelay(debug=True)
