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
import os
import tempfile

import nibabel as nib
import numpy as np

import rapidtide.workflows.delayflow as df

# ==================== Helpers ====================


def _make_radial_dataset(tmpdir, shape=(32, 32, 28), voxdim=2.0, truespeed=5.0, noise=0.05):
    """Make a synthetic delay map with an analytically known flow field.

    Radial outflow from a point source at constant speed c gives tau = r / c, so
    grad(tau) = rhat / c, the speed is c everywhere, and the flow direction is
    radially outward.  Every quantity delayflow computes has a closed form here.
    """
    rng = np.random.RandomState(1234)
    thecenter = [(thedim - 1) / 2.0 for thedim in shape]
    thegrids = np.indices(shape).astype(float)
    theoffsets = [(thegrids[i] - thecenter[i]) * voxdim for i in range(3)]
    theradius = np.sqrt(sum(theoffset**2 for theoffset in theoffsets))

    outerradius = 0.9 * np.min(thecenter) * voxdim
    themask = ((theradius < outerradius) & (theradius > 3.0 * voxdim)).astype(np.uint16)
    tau = (theradius / truespeed + rng.normal(0.0, noise, theradius.shape)) * themask
    cbv = np.clip(0.8 + rng.normal(0.0, 0.02, theradius.shape), 0.0, 1.0) * themask

    theaffine = np.diag([voxdim, voxdim, voxdim, 1.0])
    thenames = {}
    for thedata, thedesc, thesuffix in [
        (tau, "maxtime", "map"),
        (cbv, "maxcorrsq", "map"),
        (themask, "corrfit", "mask"),
    ]:
        thename = os.path.join(tmpdir, f"sub-01_desc-{thedesc}_{thesuffix}.nii.gz")
        nib.save(nib.Nifti1Image(thedata.astype(np.float32), theaffine), thename)
        thenames[thedesc] = thename

    truedirection = np.stack(
        [theoffset / np.maximum(theradius, 1.0e-9) for theoffset in theoffsets], axis=-1
    )
    return thenames, themask, truedirection, truespeed


def _baseargs(delayfile, outputroot):
    args = df._get_parser().parse_args([delayfile, outputroot])
    args.showprogressbar = False
    args.debug = False
    return args


# ==================== Unit tests ====================


def test_priorityflood(debug=False):
    """Filling should remove interior local minima but leave a monotone surface alone."""
    themask = np.ones((8, 8, 8), dtype=np.uint16)
    thegrids = np.indices((8, 8, 8)).astype(float)

    # a plane, which has no pits at all - filling should be nearly a no-op
    theplane = thegrids[0].copy()
    thefilled = df.priorityflood(theplane, themask, showprogressbar=False)
    assert np.allclose(thefilled, theplane, atol=1.0e-4), "filling perturbed a pit free surface"

    # now punch a pit into it and confirm the pit gets raised
    thepitted = theplane.copy()
    thepitted[4, 4, 4] = -10.0
    thefilled = df.priorityflood(thepitted, themask, showprogressbar=False)
    assert thefilled[4, 4, 4] > thepitted[4, 4, 4], "pit was not filled"
    assert np.all(thefilled >= thepitted - 1.0e-9), "filling lowered the surface"


def test_estimategradient(debug=False):
    """A known linear ramp must give back its own slope."""
    theshape = (16, 16, 16)
    voxdims = (2.0, 3.0, 4.0)
    themask = np.ones(theshape, dtype=np.uint16)
    theweight = np.ones(theshape, dtype=np.float64)
    thegrids = np.indices(theshape).astype(float)

    theslopes = (0.05, -0.02, 0.01)
    tau = sum(theslopes[i] * thegrids[i] * voxdims[i] for i in range(3))

    gradx, grady, gradz, fitvalid = df.estimategradient(tau, theweight, themask, voxdims, 8.0)

    # check only the interior, where the fit neighborhood is complete
    theinterior = (slice(4, -4), slice(4, -4), slice(4, -4))
    for thegradient, theslope, thename in [
        (gradx, theslopes[0], "x"),
        (grady, theslopes[1], "y"),
        (gradz, theslopes[2], "z"),
    ]:
        thevalue = thegradient[theinterior]
        assert np.allclose(
            thevalue, theslope, atol=1.0e-6
        ), f"{thename} gradient is {thevalue.mean()}, should be {theslope}"
    assert np.all(fitvalid[theinterior] > 0), "interior fits flagged invalid"


def test_labelterritories(debug=False):
    """Two separated sources should give two territories."""
    theshape = (20, 8, 8)
    themask = np.ones(theshape, dtype=np.uint16)
    thegrids = np.indices(theshape).astype(float)
    # a V shaped surface, so voxels drain toward one of the two ends
    tau = np.abs(thegrids[0] - 9.5)
    thelabels, thesizes = df.labelterritories(
        tau, themask, (1.0, 1.0, 1.0), minterritorysize=1, showprogressbar=False
    )
    assert len(thesizes) >= 2, f"expected at least 2 territories, got {len(thesizes)}"
    # the two halves must not share a label
    assert thelabels[0, 4, 4] != thelabels[-1, 4, 4], "the two arms were merged"


# ==================== Integration test ====================


def delayflow_integration(debug=False):
    """Run delayflow end to end and check it recovers the known flow field."""
    if debug:
        print("delayflow_integration")
    with tempfile.TemporaryDirectory() as tmpdir:
        thenames, themask, truedirection, truespeed = _make_radial_dataset(tmpdir)
        outputroot = os.path.join(tmpdir, "output")

        args = _baseargs(thenames["maxtime"], outputroot)
        args.debug = debug
        args.seedstride = 6
        df.delayflow(args)

        # the CBV proxy and mask should have been autodetected from the delay map name
        assert args.cbvfile == thenames["maxcorrsq"], "did not autodetect the CBV proxy"
        assert args.maskfile == thenames["corrfit"], "did not autodetect the mask"

        for thedesc, thesuffix in [
            ("flowfit", "mask"),
            ("tausmoothed", "map"),
            ("gradmag", "map"),
            ("speed", "map"),
            ("speedresolved", "mask"),
            ("velocity", "map"),
            ("direction", "map"),
            ("flowaccum", "map"),
            ("territories", "map"),
            ("fluxdivergence", "map"),
        ]:
            thename = f"{outputroot}_desc-{thedesc}_{thesuffix}.nii.gz"
            assert os.path.exists(thename), f"{thename} not created"

        thevalid = nib.load(f"{outputroot}_desc-flowfit_mask.nii.gz").get_fdata() > 0
        speed = nib.load(f"{outputroot}_desc-speed_map.nii.gz").get_fdata()
        direction = nib.load(f"{outputroot}_desc-direction_map.nii.gz").get_fdata()

        # speed should come back close to the truth
        themedianspeed = np.median(speed[thevalid])
        if debug:
            print(f"{themedianspeed=}, {truespeed=}")
        assert (
            np.abs(themedianspeed - truespeed) / truespeed < 0.15
        ), f"median speed {themedianspeed} is not close to {truespeed}"

        # and direction should be radially outward
        thedots = np.sum(direction * truedirection, axis=-1)[thevalid]
        theangles = np.degrees(np.arccos(np.clip(thedots, -1.0, 1.0)))
        themedianangle = np.median(theangles)
        if debug:
            print(f"{themedianangle=}")
        assert themedianangle < 10.0, f"median direction error {themedianangle} deg is too large"

        # flow accumulation must grow outward for a radial source
        accumulation = nib.load(f"{outputroot}_desc-flowaccum_map.nii.gz").get_fdata()
        thegrids = np.indices(themask.shape).astype(float)
        thecenter = [(thedim - 1) / 2.0 for thedim in themask.shape]
        theradius = np.sqrt(sum(((thegrids[i] - thecenter[i]) * 2.0) ** 2 for i in range(3)))
        theinner = thevalid & (theradius < 15.0)
        theouter = thevalid & (theradius > 20.0)
        assert (
            accumulation[theouter].mean() > accumulation[theinner].mean()
        ), "flow accumulation does not increase downstream"


def delayflow_speedceiling(debug=False):
    """Flow faster than the resolvable limit must be flagged, not silently reported."""
    if debug:
        print("delayflow_speedceiling")
    with tempfile.TemporaryDirectory() as tmpdir:
        # 200 mm/s, far above the ceiling set by fitradius/delaynoise
        thenames, themask, _, _ = _make_radial_dataset(tmpdir, truespeed=200.0, noise=0.1)
        outputroot = os.path.join(tmpdir, "output")

        args = _baseargs(thenames["maxtime"], outputroot)
        args.debug = debug
        args.dostreamlines = False
        args.dodivergence = False
        df.delayflow(args)

        thevalid = nib.load(f"{outputroot}_desc-flowfit_mask.nii.gz").get_fdata() > 0
        resolved = nib.load(f"{outputroot}_desc-speedresolved_mask.nii.gz").get_fdata() > 0
        thefraction = np.sum(resolved) / np.sum(thevalid)
        if debug:
            print(f"{thefraction=}")
        assert thefraction < 0.25, (
            f"{100.0 * thefraction:.1f}% of voxels claimed a resolvable speed, but the true "
            f"speed is well above the ceiling"
        )


def test_delayflow(debug=False):
    test_priorityflood(debug=debug)
    test_estimategradient(debug=debug)
    test_labelterritories(debug=debug)
    delayflow_integration(debug=debug)
    delayflow_speedceiling(debug=debug)


if __name__ == "__main__":
    test_delayflow(debug=True)
