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

import rapidtide.workflows.corrflow as cf

# ==================== Helpers ====================


def _correlationwaveform(t):
    """A correlation shaped waveform: a gaussian damped cosine peaking at zero."""
    return np.exp(-((t / 2.5) ** 2)) * np.cos(2.0 * np.pi * t / 7.0)


def _make_radial_movie(
    tmpdir, shape=(28, 28, 24), voxdim=2.0, truespeed=5.0, noise=0.0, numlags=26
):
    """Make a synthetic 4D similarity function with an analytically known flow field.

    The movie is I(x, l) = f(l - tau(x)) with tau = r/c, so the correlation peak
    propagates radially outward at speed c and the true velocity is c * rhat.
    """
    rng = np.random.RandomState(20260728)
    lagstart, lagstep = -2.0, 0.4

    thecenter = [(thedim - 1) / 2.0 for thedim in shape]
    thegrids = np.indices(shape).astype(float)
    theoffsets = [(thegrids[i] - thecenter[i]) * voxdim for i in range(3)]
    theradius = np.sqrt(sum(theoffset**2 for theoffset in theoffsets))

    outerradius = 0.85 * np.min(thecenter) * voxdim
    themask = ((theradius < outerradius) & (theradius > 2.5 * voxdim)).astype(np.uint16)
    tau = theradius / truespeed
    lags = lagstart + np.arange(numlags) * lagstep

    corr = _correlationwaveform(lags[np.newaxis, np.newaxis, np.newaxis, :] - tau[..., np.newaxis])
    amp = np.clip(1.0 - 0.012 * theradius, 0.15, 1.0)
    corr = corr * amp[..., np.newaxis] * themask[..., np.newaxis]
    if noise > 0:
        corr = corr + rng.normal(0.0, noise, corr.shape) * themask[..., np.newaxis]

    theaffine = np.diag([voxdim, voxdim, voxdim, 1.0])
    theaffine[:3, 3] = [-thecenter[i] * voxdim for i in range(3)]

    thenames = {}
    theimg = nib.Nifti1Image(corr.astype(np.float32), theaffine)
    theimg.header["pixdim"][4] = lagstep
    theimg.header["toffset"] = lagstart
    thenames["corrout"] = os.path.join(tmpdir, "sub-01_desc-corrout_info.nii.gz")
    nib.save(theimg, thenames["corrout"])

    thenames["mask"] = os.path.join(tmpdir, "sub-01_desc-corrfit_mask.nii.gz")
    nib.save(nib.Nifti1Image(themask, theaffine), thenames["mask"])

    truedirection = np.stack(
        [theoffset / np.maximum(theradius, 1.0e-9) for theoffset in theoffsets], axis=-1
    )
    return thenames, themask, truedirection, truespeed, lags


def _baseargs(corrfile, outputroot):
    args = cf._get_parser().parse_args([corrfile, outputroot])
    args.showprogressbar = False
    args.debug = False
    return args


# ==================== Unit tests ====================


def test_getlagaxis(debug=False):
    """The lag axis must come back out of the header exactly as written."""
    theimg = nib.Nifti1Image(np.zeros((2, 2, 2, 10), dtype=np.float32), np.eye(4))
    theimg.header["pixdim"][4] = 0.25
    theimg.header["toffset"] = -1.5
    thelags = cf.getlagaxis(theimg.header, 10)
    assert np.isclose(thelags[0], -1.5), f"lag start is {thelags[0]}, should be -1.5"
    assert np.isclose(thelags[1] - thelags[0], 0.25), "lag step is wrong"
    assert len(thelags) == 10, "wrong number of lags"

    # a file with no lag step is not a corrout file, and should say so
    theimg.header["pixdim"][4] = 0.0
    try:
        cf.getlagaxis(theimg.header, 10)
        assert False, "should have raised on a nonpositive lag step"
    except ValueError:
        pass


def test_peaklag(debug=False):
    """The peak lag map must recover a known peak position to subsample accuracy."""
    theshape = (6, 6, 6)
    thelags = np.linspace(-2.0, 8.0, 26)
    themask = np.ones(theshape, dtype=np.uint16)

    # a known, deliberately non grid aligned peak
    truetau = 3.17
    corr = _correlationwaveform(
        thelags[np.newaxis, np.newaxis, np.newaxis, :] - truetau
    ) * np.ones(theshape + (1,))

    thepeaks = cf.peaklag(corr, thelags, themask)
    theerror = np.abs(thepeaks - truetau).max()
    thestep = thelags[1] - thelags[0]
    if debug:
        print(f"peak lag error {theerror} s, lag step {thestep} s")
    assert theerror < 0.5 * thestep, (
        f"peak lag error {theerror} s exceeds half a lag step ({0.5 * thestep} s) - "
        f"the parabolic refinement is not working"
    )


def test_opticalflow_recovers_known_field(debug=False):
    """The core estimator must recover a known radial flow field on clean data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        thenames, themask, truedirection, truespeed, thelags = _make_radial_movie(tmpdir)
        corrdata = nib.load(thenames["corrout"]).get_fdata()

        velocity, therank, thecoherence, theresidual = cf.computeopticalflow(
            corrdata,
            themask,
            (2.0, 2.0, 2.0),
            thelags,
            showprogressbar=False,
        )

        thevalid = themask > 0
        speed = np.linalg.norm(velocity, axis=-1)
        thedirection = velocity / np.maximum(speed[..., np.newaxis], 1.0e-9)
        theangles = np.degrees(
            np.arccos(np.clip(np.sum(thedirection * truedirection, axis=-1)[thevalid], -1.0, 1.0))
        )
        themedianspeed = np.median(speed[thevalid])
        themedianangle = np.median(theangles)
        if debug:
            print(f"{themedianspeed=} (true {truespeed}), {themedianangle=}")

        assert (
            np.abs(themedianspeed - truespeed) / truespeed < 0.2
        ), f"median speed {themedianspeed} is not close to {truespeed}"
        assert themedianangle < 10.0, f"median direction error {themedianangle} deg is too large"

        # For a coherent travelling wavefront the structure tensor is rank one - the
        # gradient is everywhere parallel to grad(tau).  If this starts coming back
        # as rank 3, the eigenvalue threshold has been loosened and the velocity is
        # picking up noise along the near null directions.
        themedianrank = np.median(therank[thevalid])
        if debug:
            print(f"{themedianrank=}")
        assert themedianrank <= 2, (
            f"median structure tensor rank is {themedianrank}; a clean travelling "
            f"wavefront should be close to rank 1"
        )


def test_lowered_eigenvalue_threshold_is_worse(debug=False):
    """Guard the counterintuitive result that admitting more eigenvectors hurts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        thenames, themask, truedirection, truespeed, thelags = _make_radial_movie(tmpdir)
        corrdata = nib.load(thenames["corrout"]).get_fdata()
        thevalid = themask > 0

        theerrors = {}
        for thethreshold in [1.0e-4, 0.1]:
            velocity = cf.computeopticalflow(
                corrdata,
                themask,
                (2.0, 2.0, 2.0),
                thelags,
                mineigenvalue=thethreshold,
                showprogressbar=False,
            )[0]
            speed = np.linalg.norm(velocity, axis=-1)
            thedirection = velocity / np.maximum(speed[..., np.newaxis], 1.0e-9)
            theerrors[thethreshold] = np.median(
                np.degrees(
                    np.arccos(
                        np.clip(np.sum(thedirection * truedirection, axis=-1)[thevalid], -1.0, 1.0)
                    )
                )
            )
        if debug:
            print(f"direction error by threshold: {theerrors}")
        assert theerrors[0.1] < theerrors[1.0e-4], (
            f"a permissive eigenvalue threshold ({theerrors[1.0e-4]:.2f} deg) should be "
            f"worse than a strict one ({theerrors[0.1]:.2f} deg), not better"
        )


# ==================== Integration test ====================


def corrflow_integration(debug=False):
    """Run corrflow end to end and check it recovers the known flow field."""
    if debug:
        print("corrflow_integration")
    with tempfile.TemporaryDirectory() as tmpdir:
        thenames, themask, truedirection, truespeed, thelags = _make_radial_movie(tmpdir)
        outputroot = os.path.join(tmpdir, "output")

        args = _baseargs(thenames["corrout"], outputroot)
        args.debug = debug
        args.seedstride = 5
        cf.corrflow(args)

        assert args.maskfile == thenames["mask"], "did not autodetect the mask"

        for thedesc, thesuffix in [
            ("flowfit", "mask"),
            ("cbvproxy", "map"),
            ("velocity", "map"),
            ("direction", "map"),
            ("speed", "map"),
            ("flowrank", "map"),
            ("flowcoherence", "map"),
            ("flowresidual", "map"),
            ("fluxdivergence", "map"),
            ("orderingfield", "map"),
            ("flowaccum", "map"),
            ("territories", "map"),
        ]:
            thename = f"{outputroot}_desc-{thedesc}_{thesuffix}.nii.gz"
            assert os.path.exists(thename), f"{thename} not created"

        thevalid = nib.load(f"{outputroot}_desc-flowfit_mask.nii.gz").get_fdata() > 0
        speed = nib.load(f"{outputroot}_desc-speed_map.nii.gz").get_fdata()
        direction = nib.load(f"{outputroot}_desc-direction_map.nii.gz").get_fdata()

        themedianspeed = np.median(speed[thevalid])
        theangles = np.degrees(
            np.arccos(np.clip(np.sum(direction * truedirection, axis=-1)[thevalid], -1.0, 1.0))
        )
        if debug:
            print(f"{themedianspeed=}, median angle {np.median(theangles)}")
        assert np.abs(themedianspeed - truespeed) / truespeed < 0.2, "speed is off"
        assert np.median(theangles) < 10.0, "direction is off"

        # streamlines should run downstream, same invariant as delayflow
        thetrk = f"{outputroot}_desc-flow_streamlines.trk"
        assert os.path.exists(thetrk), "streamline file not created"
        theloaded = nib.streamlines.load(thetrk)
        thepervertex = theloaded.tractogram.data_per_point
        for thename in ["arrivaltime", "speed", "territory"]:
            assert thename in thepervertex, f"{thename} missing from the trk scalars"
        thesteps = np.concatenate(
            [np.diff(np.asarray(a).ravel()) for a in thepervertex["arrivaltime"]]
        )
        thefraction = np.mean(thesteps > 0)
        if debug:
            print(f"fraction of streamline steps increasing in tau: {thefraction}")
        assert (
            thefraction > 0.90
        ), f"only {100.0 * thefraction:.1f}% of streamline steps increase in arrival time"


def corrflow_solvers(debug=False):
    """Both solvers must work on clean data; tls should not attenuate the speed."""
    if debug:
        print("corrflow_solvers")
    with tempfile.TemporaryDirectory() as tmpdir:
        thenames, themask, truedirection, truespeed, thelags = _make_radial_movie(tmpdir)
        corrdata = nib.load(thenames["corrout"]).get_fdata()
        thevalid = themask > 0

        thespeeds = {}
        for thesolver in ["ols", "tls"]:
            velocity = cf.computeopticalflow(
                corrdata,
                themask,
                (2.0, 2.0, 2.0),
                thelags,
                solver=thesolver,
                showprogressbar=False,
            )[0]
            thespeeds[thesolver] = np.median(np.linalg.norm(velocity, axis=-1)[thevalid])
        if debug:
            print(f"median speed by solver: {thespeeds} (true {truespeed})")
        for thesolver, thespeed in thespeeds.items():
            assert (
                np.abs(thespeed - truespeed) / truespeed < 0.2
            ), f"{thesolver} speed {thespeed} is not close to {truespeed}"


def test_corrflow(debug=False):
    test_getlagaxis(debug=debug)
    test_peaklag(debug=debug)
    test_opticalflow_recovers_known_field(debug=debug)
    test_lowered_eigenvalue_threshold_is_worse(debug=debug)
    corrflow_integration(debug=debug)
    corrflow_solvers(debug=debug)


if __name__ == "__main__":
    test_corrflow(debug=True)
