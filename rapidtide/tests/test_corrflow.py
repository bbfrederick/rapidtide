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


def test_oversamplelagaxis_passthrough(debug=False):
    """A factor of one or less leaves the movie and its lag axis untouched."""
    if debug:
        print("oversamplelagaxis_passthrough")
    thelags = np.linspace(-2.0, 8.0, 26)
    corr = np.random.RandomState(1).randn(3, 3, 3, 26)
    for thefactor in [1, 0, -2]:
        outcorr, outlags = cf.oversamplelagaxis(corr, thelags, thefactor)
        assert outcorr is corr, f"factor {thefactor} should not copy the data"
        assert outlags is thelags


def test_oversamplelagaxis_interpolates(debug=False):
    """Oversampling refines the lag axis without moving the correlation peak.

    The interpolation is meant to recover the underlying continuous function, so a peak
    at a known lag must stay where it was rather than drift.
    """
    if debug:
        print("oversamplelagaxis_interpolates")
    thelags = np.linspace(-2.0, 8.0, 26)
    truetau = 3.17
    corr = _correlationwaveform(
        thelags[np.newaxis, np.newaxis, np.newaxis, :] - truetau
    ) * np.ones((4, 4, 4, 1))

    thefactor = 4
    outcorr, outlags = cf.oversamplelagaxis(corr, thelags, thefactor)
    assert len(outlags) == (len(thelags) - 1) * thefactor + 1
    assert outcorr.shape == corr.shape[:3] + (len(outlags),)
    # the endpoints are preserved exactly
    assert np.isclose(outlags[0], thelags[0])
    assert np.isclose(outlags[-1], thelags[-1])
    # and the finer axis puts the peak closer to the truth than the original did
    themask = np.ones((4, 4, 4), dtype=np.uint16)
    coarseerror = np.abs(cf.peaklag(corr, thelags, themask) - truetau).max()
    fineerror = np.abs(cf.peaklag(outcorr, outlags, themask) - truetau).max()
    if debug:
        print(f"coarse error {coarseerror}, fine error {fineerror}")
    assert fineerror <= coarseerror + 1.0e-9


def test_opticalflow_without_correlation_weighting(debug=False):
    """The unweighted structure tensor path still recovers the known field."""
    if debug:
        print("opticalflow_without_correlation_weighting")
    with tempfile.TemporaryDirectory() as tmpdir:
        thenames, themask, truedirection, truespeed, thelags = _make_radial_movie(tmpdir)
        corrdata = nib.load(thenames["corrout"]).get_fdata()
        velocity = cf.computeopticalflow(
            corrdata,
            themask,
            (2.0, 2.0, 2.0),
            thelags,
            weightbycorr=False,
            showprogressbar=False,
        )[0]
        thespeed = np.median(np.linalg.norm(velocity, axis=-1)[themask > 0])
        if debug:
            print(f"unweighted median speed {thespeed} (true {truespeed})")
        assert np.abs(thespeed - truespeed) / truespeed < 0.3


def test_opticalflow_empty_mask(debug=False):
    """An empty mask produces empty output rather than failing."""
    if debug:
        print("opticalflow_empty_mask")
    theshape = (10, 10, 10)
    thelags = np.linspace(-2.0, 8.0, 12)
    corrdata = np.zeros(theshape + (12,))
    themask = np.zeros(theshape, dtype=np.uint16)
    velocity, rank, residual, cbv = cf.computeopticalflow(
        corrdata, themask, (2.0, 2.0, 2.0), thelags, showprogressbar=False, debug=True
    )
    assert velocity.shape == theshape + (3,)
    assert np.all(velocity == 0.0)
    assert np.all(rank == 0)


# ==================== corrflow() option and validation tests ====================


def test_corrflow_rejects_too_few_lags(debug=False):
    """A movie with too few lag samples cannot support a flow estimate."""
    if debug:
        print("corrflow_rejects_too_few_lags")
    with tempfile.TemporaryDirectory() as tmpdir:
        theimg = nib.Nifti1Image(np.zeros((8, 8, 8, 4), dtype=np.float32), np.eye(4))
        theimg.header["pixdim"][4] = 0.4
        theimg.header["toffset"] = -2.0
        thepath = os.path.join(tmpdir, "sub-01_desc-corrout_info.nii.gz")
        nib.save(theimg, thepath)
        args = _baseargs(thepath, os.path.join(tmpdir, "out"))
        try:
            cf.corrflow(args)
            assert False, "should have raised on a movie with too few lags"
        except ValueError as thedetail:
            assert "lag samples" in str(thedetail)


def test_corrflow_rejects_mismatched_mask(debug=False):
    """A mask that does not match the movie geometry is rejected."""
    if debug:
        print("corrflow_rejects_mismatched_mask")
    with tempfile.TemporaryDirectory() as tmpdir:
        thenames, themask, truedirection, truespeed, thelags = _make_radial_movie(tmpdir)
        badmask = nib.Nifti1Image(np.ones((5, 5, 5), dtype=np.uint16), np.eye(4))
        badmaskpath = os.path.join(tmpdir, "badmask.nii.gz")
        nib.save(badmask, badmaskpath)
        args = _baseargs(thenames["corrout"], os.path.join(tmpdir, "out"))
        args.maskfile = badmaskpath
        try:
            cf.corrflow(args)
            assert False, "should have raised on a mismatched mask"
        except ValueError as thedetail:
            assert "mask dimensions" in str(thedetail)


def test_corrflow_rejects_tiny_mask(debug=False):
    """A CBV threshold that leaves almost nothing behind is rejected."""
    if debug:
        print("corrflow_rejects_tiny_mask")
    with tempfile.TemporaryDirectory() as tmpdir:
        thenames, themask, truedirection, truespeed, thelags = _make_radial_movie(tmpdir)
        args = _baseargs(thenames["corrout"], os.path.join(tmpdir, "out"))
        # nothing can survive a threshold above the normalized CBV maximum
        args.cbvthresh = 10.0
        try:
            cf.corrflow(args)
            assert False, "should have raised on a nearly empty mask"
        except ValueError as thedetail:
            assert "fewer than 100 valid voxels" in str(thedetail)


def test_corrflow_autodiscovers_mask(debug=False):
    """A companion corrfit mask next to the movie is picked up automatically."""
    if debug:
        print("corrflow_autodiscovers_mask")
    with tempfile.TemporaryDirectory() as tmpdir:
        thenames, themask, truedirection, truespeed, thelags = _make_radial_movie(tmpdir)
        outputroot = os.path.join(tmpdir, "automask")
        args = _baseargs(thenames["corrout"], outputroot)
        assert args.maskfile is None
        cf.corrflow(args)
        derived = nib.load(f"{outputroot}_desc-flowfit_mask.nii.gz").get_fdata()
        # the discovered mask bounds the analysis mask
        assert np.all(derived[themask == 0] == 0)


def test_corrflow_derives_mask_when_none_given(debug=False):
    """With no mask file to find, the mask comes from the nonzero correlation functions."""
    if debug:
        print("corrflow_derives_mask_when_none_given")
    with tempfile.TemporaryDirectory() as tmpdir:
        thenames, themask, truedirection, truespeed, thelags = _make_radial_movie(tmpdir)
        # remove the companion mask so autodiscovery finds nothing
        os.remove(thenames["mask"])
        outputroot = os.path.join(tmpdir, "nomask")
        args = _baseargs(thenames["corrout"], outputroot)
        args.maskfile = None
        cf.corrflow(args)
        themaskpath = f"{outputroot}_desc-flowfit_mask.nii.gz"
        assert os.path.isfile(themaskpath)
        derived = nib.load(themaskpath).get_fdata()
        assert np.sum(derived) >= 100


def test_corrflow_maxspeed_clipping(debug=False):
    """A low speed ceiling clips the velocity field to that magnitude."""
    if debug:
        print("corrflow_maxspeed_clipping")
    with tempfile.TemporaryDirectory() as tmpdir:
        thenames, themask, truedirection, truespeed, thelags = _make_radial_movie(tmpdir)
        outputroot = os.path.join(tmpdir, "clipped")
        args = _baseargs(thenames["corrout"], outputroot)
        args.maskfile = thenames["mask"]
        # well below the true speed, so the clip has to engage
        args.maxspeed = 1.0
        cf.corrflow(args)
        thespeedpath = f"{outputroot}_desc-speed_map.nii.gz"
        assert os.path.isfile(thespeedpath)
        thespeed = nib.load(thespeedpath).get_fdata()
        assert np.max(thespeed) <= 1.0 + 1.0e-5, f"speed reached {np.max(thespeed)}"


def test_corrflow_with_lag_oversampling(debug=False):
    """Running with lag oversampling produces the same outputs on a finer axis."""
    if debug:
        print("corrflow_with_lag_oversampling")
    with tempfile.TemporaryDirectory() as tmpdir:
        thenames, themask, truedirection, truespeed, thelags = _make_radial_movie(tmpdir)
        outputroot = os.path.join(tmpdir, "oversamp")
        args = _baseargs(thenames["corrout"], outputroot)
        args.maskfile = thenames["mask"]
        args.lagoversamp = 2
        cf.corrflow(args)
        assert os.path.isfile(f"{outputroot}_desc-speed_map.nii.gz")


def test_corrflow_explicit_delayfile(debug=False):
    """An explicit ordering field is read instead of being derived from the movie."""
    if debug:
        print("corrflow_explicit_delayfile")
    with tempfile.TemporaryDirectory() as tmpdir:
        thenames, themask, truedirection, truespeed, thelags = _make_radial_movie(tmpdir)
        corrimg = nib.load(thenames["corrout"])
        # a simple ramp, distinguishable from anything the movie would produce
        thedelay = np.zeros(themask.shape, dtype=np.float32)
        thedelay[:] = np.arange(themask.shape[0], dtype=np.float32)[:, None, None]
        delaypath = os.path.join(tmpdir, "sub-01_desc-maxtime_map.nii.gz")
        nib.save(nib.Nifti1Image(thedelay, corrimg.affine), delaypath)

        outputroot = os.path.join(tmpdir, "withdelay")
        args = _baseargs(thenames["corrout"], outputroot)
        args.maskfile = thenames["mask"]
        args.delayfile = delaypath
        cf.corrflow(args)
        thefieldpath = f"{outputroot}_desc-orderingfield_map.nii.gz"
        assert os.path.isfile(thefieldpath)
        thefield = nib.load(thefieldpath).get_fdata()
        # inside the mask the saved field is the ramp we handed in
        np.testing.assert_allclose(
            thefield[themask > 0], thedelay[themask > 0], rtol=1e-5, atol=1e-5
        )


def test_corrflow_rejects_mismatched_delayfile(debug=False):
    """A delay map that does not match the movie geometry is rejected."""
    if debug:
        print("corrflow_rejects_mismatched_delayfile")
    with tempfile.TemporaryDirectory() as tmpdir:
        thenames, themask, truedirection, truespeed, thelags = _make_radial_movie(tmpdir)
        baddelay = nib.Nifti1Image(np.zeros((5, 5, 5), dtype=np.float32), np.eye(4))
        baddelaypath = os.path.join(tmpdir, "baddelay.nii.gz")
        nib.save(baddelay, baddelaypath)
        args = _baseargs(thenames["corrout"], os.path.join(tmpdir, "out"))
        args.maskfile = thenames["mask"]
        args.delayfile = baddelaypath
        try:
            cf.corrflow(args)
            assert False, "should have raised on a mismatched delay map"
        except ValueError as thedetail:
            assert "delay map dimensions" in str(thedetail)


def test_corrflow_autodiscovers_delayfile(debug=False):
    """A companion maxtime map next to the movie is picked up automatically."""
    if debug:
        print("corrflow_autodiscovers_delayfile")
    with tempfile.TemporaryDirectory() as tmpdir:
        thenames, themask, truedirection, truespeed, thelags = _make_radial_movie(tmpdir)
        corrimg = nib.load(thenames["corrout"])
        thedelay = np.zeros(themask.shape, dtype=np.float32)
        thedelay[:] = np.arange(themask.shape[0], dtype=np.float32)[:, None, None]
        # the root is the corrout name with the _desc-corrout_info suffix stripped
        nib.save(
            nib.Nifti1Image(thedelay, corrimg.affine),
            os.path.join(tmpdir, "sub-01_desc-maxtime_map.nii.gz"),
        )
        outputroot = os.path.join(tmpdir, "auto")
        args = _baseargs(thenames["corrout"], outputroot)
        args.maskfile = thenames["mask"]
        assert args.delayfile is None
        cf.corrflow(args)
        thefield = nib.load(f"{outputroot}_desc-orderingfield_map.nii.gz").get_fdata()
        np.testing.assert_allclose(
            thefield[themask > 0], thedelay[themask > 0], rtol=1e-5, atol=1e-5
        )


def test_corrflow_nonstandard_filename_root(debug=False):
    """A movie whose name lacks the corrout suffix still resolves a usable root."""
    if debug:
        print("corrflow_nonstandard_filename_root")
    with tempfile.TemporaryDirectory() as tmpdir:
        thenames, themask, truedirection, truespeed, thelags = _make_radial_movie(tmpdir)
        oddname = os.path.join(tmpdir, "justamovie.nii.gz")
        os.rename(thenames["corrout"], oddname)
        outputroot = os.path.join(tmpdir, "oddroot")
        args = _baseargs(oddname, outputroot)
        args.maskfile = thenames["mask"]
        cf.corrflow(args)
        assert os.path.isfile(f"{outputroot}_desc-speed_map.nii.gz")


# ==================== main() tests ====================


def test_corrflow_without_depression_filling(debug=False):
    """With filling disabled the ordering field is used directly for the topology steps."""
    if debug:
        print("corrflow_without_depression_filling")
    with tempfile.TemporaryDirectory() as tmpdir:
        thenames, themask, truedirection, truespeed, thelags = _make_radial_movie(tmpdir)
        outputroot = os.path.join(tmpdir, "nofill")
        args = _baseargs(thenames["corrout"], outputroot)
        args.maskfile = thenames["mask"]
        args.dofill = False
        cf.corrflow(args)
        assert os.path.isfile(f"{outputroot}_desc-orderingfield_map.nii.gz")
        assert os.path.isfile(f"{outputroot}_desc-flowaccum_map.nii.gz")


def test_opticalflow_planar_wave_is_rank_one(debug=False):
    """A single planar wavefront leaves the higher rank classes empty.

    The tls solver loops over rank classes 1, 2 and 3 and skips any that no voxel falls
    into; a purely planar wave is rank one everywhere, so the other two are empty.  This
    runs under tls specifically, since the default ols solver takes a different path.
    """
    if debug:
        print("opticalflow_planar_wave_is_rank_one")
    theshape = (14, 14, 14)
    voxdim = 2.0
    numlags = 24
    thelags = -2.0 + np.arange(numlags) * 0.4
    truespeed = 5.0
    # tau depends on x only, so the spatial gradient points along one axis everywhere
    thex = np.arange(theshape[0])[:, None, None] * voxdim
    tau = np.broadcast_to(thex, theshape) / truespeed
    corr = _correlationwaveform(thelags[None, None, None, :] - tau[..., None])
    # mask only the interior, so no voxel sees an edge and every one is rank one
    themask = np.zeros(theshape, dtype=np.uint16)
    themask[3:-3, 3:-3, 3:-3] = 1

    velocity, therank, thecoherence, theresidual = cf.computeopticalflow(
        corr, themask, (voxdim, voxdim, voxdim), thelags, solver="tls", showprogressbar=False
    )
    thevalid = themask > 0
    assert np.all(therank[thevalid] == 1), "a planar wave should be rank one throughout"
    if debug:
        print(f"planar median speed {np.median(np.linalg.norm(velocity, axis=-1)[thevalid])}")


def test_main_runs_a_full_analysis(debug=False):
    """The entrypoint parses a real command line and runs the analysis."""
    if debug:
        print("main_runs_a_full_analysis")
    import sys

    with tempfile.TemporaryDirectory() as tmpdir:
        thenames, themask, truedirection, truespeed, thelags = _make_radial_movie(tmpdir)
        outputroot = os.path.join(tmpdir, "viamain")
        oldargv = sys.argv
        sys.argv = [
            "corrflow",
            thenames["corrout"],
            outputroot,
            "--maskfile",
            thenames["mask"],
            "--noprogressbar",
        ]
        try:
            cf.main()
        finally:
            sys.argv = oldargv
        assert os.path.isfile(f"{outputroot}_desc-speed_map.nii.gz")


def test_main_missing_args(debug=False):
    """The entrypoint prints help and re-raises when the command line is unusable."""
    if debug:
        print("main_missing_args")
    import sys

    oldargv = sys.argv
    sys.argv = ["corrflow"]
    try:
        cf.main()
        assert False, "should have raised SystemExit"
    except SystemExit:
        pass
    finally:
        sys.argv = oldargv


def test_corrflow(debug=False):
    test_getlagaxis(debug=debug)
    test_peaklag(debug=debug)
    test_oversamplelagaxis_passthrough(debug=debug)
    test_oversamplelagaxis_interpolates(debug=debug)
    test_opticalflow_recovers_known_field(debug=debug)
    test_opticalflow_without_correlation_weighting(debug=debug)
    test_opticalflow_empty_mask(debug=debug)
    test_lowered_eigenvalue_threshold_is_worse(debug=debug)
    corrflow_integration(debug=debug)
    corrflow_solvers(debug=debug)
    test_corrflow_rejects_too_few_lags(debug=debug)
    test_corrflow_rejects_mismatched_mask(debug=debug)
    test_corrflow_rejects_tiny_mask(debug=debug)
    test_corrflow_autodiscovers_mask(debug=debug)
    test_corrflow_derives_mask_when_none_given(debug=debug)
    test_corrflow_without_depression_filling(debug=debug)
    test_opticalflow_planar_wave_is_rank_one(debug=debug)
    test_corrflow_maxspeed_clipping(debug=debug)
    test_corrflow_with_lag_oversampling(debug=debug)
    test_corrflow_explicit_delayfile(debug=debug)
    test_corrflow_rejects_mismatched_delayfile(debug=debug)
    test_corrflow_autodiscovers_delayfile(debug=debug)
    test_corrflow_nonstandard_filename_root(debug=debug)
    test_main_runs_a_full_analysis(debug=debug)
    test_main_missing_args(debug=debug)


if __name__ == "__main__":
    test_corrflow(debug=True)
