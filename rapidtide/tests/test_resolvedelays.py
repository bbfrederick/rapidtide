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
import argparse
import os
import tempfile

import nibabel as nb
import numpy as np

import rapidtide.io as tide_io
import rapidtide.workflows.resolvedelays as uw

SIDELOBE = 12.0


def _waveform(t, mainamp=1.0, sideamp=0.6):
    """A waveform with a deliberate sidelobe one SIDELOBE period out.

    Parameters
    ----------
    t : NDArray
        Lag values relative to the true delay, in seconds.
    mainamp : float, optional
        Amplitude of the true peak.  Default is 1.0.
    sideamp : float, optional
        Amplitude of the sidelobe.  Swapping these two is how the tests plant a
        voxel whose strongest peak is the wrong one.  Default is 0.6.

    Returns
    -------
    NDArray
        The similarity function, same shape as t.
    """
    return mainamp * np.exp(-((t / 3.0) ** 2)) + sideamp * np.exp(-(((t - SIDELOBE) / 3.0) ** 2))


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


def test_findcandidatepeaks_refines_peaks_off_the_sample_grid(debug=False):
    """Parabolic refinement must beat the sample spacing.

    A delay is only useful to a fraction of the lag step, so the three point
    parabolic interpolation is doing real work here, not cosmetics.  Plant peaks
    deliberately between samples and require the recovered location to be an order
    of magnitude better than rounding to the nearest sample would give.
    """
    thelagaxis = np.linspace(-20.0, 20.0, 161)  # 0.25 s spacing
    thelagstep = float(thelagaxis[1] - thelagaxis[0])
    # offsets chosen to sit at assorted fractions of a sample, including the worst
    # case of exactly half way between two samples
    theoffsets = np.array([0.0, 0.1, 0.125, 0.2, 0.5]) * thelagstep
    thetruelags = 3.0 + theoffsets

    thecorr = np.array([np.exp(-(((thelagaxis - thelag) / 2.0) ** 2)) for thelag in thetruelags])
    thecandidatelags, dummy = uw.findcandidatepeaks(thecorr, thelagaxis, 2)

    theerrors = np.abs(thecandidatelags[:, 0] - thetruelags)
    if debug:
        print(f"lag step {thelagstep}, errors {theerrors}")
    assert np.all(theerrors < 0.1 * thelagstep), (
        f"refined peaks are off by {theerrors}, which is not better than the "
        f"{thelagstep} s sample spacing"
    )


def test_findcandidatepeaks_orders_by_amplitude_and_pads_with_nan(debug=False):
    """Index 0 must be what naive peak picking would choose, and unused candidate
    slots must be NaN rather than a stale or zero delay."""
    thelagaxis = np.linspace(-20.0, 20.0, 161)

    def thebump(theamp, theloc):
        return theamp * np.exp(-(((thelagaxis - theloc) / 2.0) ** 2))

    # one voxel with three peaks in a deliberately unsorted amplitude order, and
    # one voxel with a single peak
    thecorr = np.array(
        [
            thebump(0.5, -10.0) + thebump(0.9, 0.0) + thebump(0.7, 10.0),
            thebump(0.9, 4.0),
        ]
    )
    thecandidatelags, thecandidateamps = uw.findcandidatepeaks(thecorr, thelagaxis, 5)

    if debug:
        print(f"lags {thecandidatelags}\namps {thecandidateamps}")

    # strongest first, regardless of where they sit on the lag axis
    assert np.allclose(thecandidatelags[0, :3], [0.0, 10.0, -10.0], atol=0.05)
    thefiniteamps = thecandidateamps[0, :3]
    assert np.all(np.diff(thefiniteamps) < 0), "candidates are not sorted strongest first"

    # slots beyond the number of real peaks are NaN in BOTH arrays
    assert np.all(np.isnan(thecandidatelags[0, 3:]))
    assert np.all(np.isnan(thecandidateamps[0, 3:]))
    assert np.isfinite(thecandidatelags[1, 0])
    assert np.all(np.isnan(thecandidatelags[1, 1:]))


def test_findcandidatepeaks_does_not_treat_array_ends_as_peaks(debug=False):
    """A maximum sitting on the first or last lag sample is not a resolvable peak:
    it has no neighbour on one side, so there is nothing to interpolate and no
    evidence it is a local maximum at all."""
    thelagaxis = np.linspace(-20.0, 20.0, 161)
    # monotonically rising, so the only maximum is the very last sample
    thecorr = np.linspace(0.0, 1.0, 161)[None, :]

    thecandidatelags, thecandidateamps = uw.findcandidatepeaks(thecorr, thelagaxis, 3)
    if debug:
        print(f"lags {thecandidatelags}, amps {thecandidateamps}")
    assert np.all(np.isnan(thecandidatelags)), "an array end was reported as a peak"


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
    got, changed, conf = uw.resolvedelaymap(cl2, ca2, None, mask, vox, showprogressbar=False)
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
    got, changed, conf = uw.resolvedelaymap(cl, ca, vel, mask, vox, showprogressbar=False)
    bad = (np.abs(got - tau) > 2.0).sum()
    assert bad < 0.1 * corrupt.sum(), f"{bad} voxels still wrong after flow guided unwrap"


def test_icmrefine_ignores_voxels_outside_the_mask(debug=False):
    """icmrefine forms its prior by median filtering the current solution.  Every voxel
    of this small mask is a surface voxel, so if the filter sees zeros outside the mask
    the prior collapses toward zero and each voxel re-snaps to the WRONG candidate.

    The delay here is uniform, so a smoothness prior has nothing to fix and the correct
    answer is to change nothing at all."""
    theshape = (7, 7, 7)
    thetruth, thedecoy = 8.0, 2.0

    themask = np.zeros(theshape, dtype=bool)
    themask[2:4, 2:4, 2:4] = True  # a 2x2x2 block, 8 voxels in a 27 voxel neighbourhood

    thetau = np.where(themask, thetruth, 0.0)
    thecandidates = np.zeros(theshape + (2,), dtype=float)
    thecandidates[..., 0] = thetruth
    thecandidates[..., 1] = thedecoy

    theresult, thenumchanged = uw.icmrefine(thetau, thecandidates, themask, numpasses=3)

    if debug:
        print(f"{theresult[themask]=}, {thenumchanged=}")
    # zero filling drags the prior to 0.0, which is nearer the decoy at 2.0 than the
    # truth at 8.0, so every voxel would flip
    assert np.allclose(theresult[themask], thetruth), (
        f"boundary voxels re-snapped to {np.unique(theresult[themask])} instead of "
        f"{thetruth}; the prior is seeing outside the mask"
    )
    assert thenumchanged == 0, f"{thenumchanged} assignments changed on a uniform field"


def test_icmrefine_with_one_pass_is_a_noop(debug=False):
    """numpasses counts the region grow as pass 1, so numpasses=1 must mean no
    refinement at all rather than one refinement."""
    theshape = (6, 6, 6)
    themask = np.ones(theshape, dtype=np.uint16)
    # tau sits at 8.0, which is NOT one of the candidates, so any refinement pass at
    # all is forced to move it.  A fixed point would hide an off by one in the count.
    thetau = np.full(theshape, 8.0)
    thecandidates = np.zeros(theshape + (2,), dtype=float)
    thecandidates[..., 0] = 2.0
    thecandidates[..., 1] = 7.0

    theresult, thenumchanged = uw.icmrefine(thetau, thecandidates, themask, numpasses=1)
    if debug:
        print(f"one pass -> {np.unique(theresult)}, changed {thenumchanged}")
    assert np.array_equal(theresult, thetau), "numpasses=1 refined anyway"
    assert thenumchanged == 0

    # two passes means exactly one refinement, which snaps to the nearer candidate
    theresult, thenumchanged = uw.icmrefine(thetau, thecandidates, themask, numpasses=2)
    if debug:
        print(f"two passes -> {np.unique(theresult)}, changed {thenumchanged}")
    assert np.allclose(theresult, 7.0), "the single refinement did not happen"
    assert thenumchanged == theshape[0] * theshape[1] * theshape[2]


def test_icmrefine_stops_early_once_nothing_changes(debug=False):
    """The loop breaks as soon as a pass changes nothing, so a converged problem
    must cost the same whether asked for 3 passes or 30."""
    theshape = (6, 6, 6)
    themask = np.ones(theshape, dtype=np.uint16)
    thecandidates = np.zeros(theshape + (2,), dtype=float)
    thecandidates[..., 0] = 5.0
    thecandidates[..., 1] = -5.0
    # start everything on the wrong candidate; one pass fixes it, the rest are free
    thetau = np.full(theshape, -5.0)

    thefew, thefewchanged = uw.icmrefine(thetau, thecandidates, themask, numpasses=3)
    themany, themanychanged = uw.icmrefine(thetau, thecandidates, themask, numpasses=30)
    if debug:
        print(f"3 passes changed {thefewchanged}, 30 passes changed {themanychanged}")
    assert np.array_equal(thefew, themany)
    assert thefewchanged == themanychanged, "extra passes kept counting changes after convergence"


def test_resolvedelaymap_tolerates_voxels_with_no_candidates(debug=False):
    """A flat similarity function offers no local maximum at all.  Such a voxel
    cannot be assigned a delay, but it must not stall the region grow or crash it."""
    theshape = (6, 6, 6)
    thelagaxis = np.linspace(-20.0, 20.0, 161)
    thecorr = np.zeros(theshape + (len(thelagaxis),))
    thecorr[:, :, :] = np.exp(-(((thelagaxis - 3.0) / 2.0) ** 2))
    # a solid flat block: no local maxima anywhere in it
    thecorr[1:3, 1:3, 1:3, :] = 0.0

    thecandidatelags, thecandidateamps = uw.findcandidatepeaks(thecorr, thelagaxis, 4)
    theflat = np.zeros(theshape, dtype=bool)
    theflat[1:3, 1:3, 1:3] = True
    assert np.all(np.isnan(thecandidatelags[theflat])), "the flat block still had candidates"

    themask = np.ones(theshape, dtype=np.uint16)
    thetau, thechanged, theconfidence = uw.resolvedelaymap(
        thecandidatelags, thecandidateamps, None, themask, (2.0, 2.0, 2.0), showprogressbar=False
    )
    if debug:
        print(f"flat block values {np.unique(thetau[theflat])}")
        print(f"good voxel values {np.unique(np.round(thetau[~theflat], 3))}")

    # the voxels that do have a peak are still resolved correctly
    assert np.allclose(thetau[~theflat], 3.0, atol=0.05)
    # and the candidateless ones come back as zero rather than NaN
    assert np.all(np.isfinite(thetau))
    assert np.allclose(thetau[theflat], 0.0)


def _makehardcase(shape=(12, 12, 8), speed=2.0, noise=0.05, numlags=90, lagstep=0.5, seed=100):
    """A case hard enough that the region grow can actually get things wrong.

    The easy synthetic cases in this file are unwrapped perfectly by region growing
    alone, which makes them useless for testing the guards that exist to handle hard
    data - every guard looks unnecessary when nothing is hard.  This case is steep
    (a fast delay ramp, so genuine neighbour to neighbour differences are large),
    noisy (so the similarity functions grow spurious extra candidates), and wrapped
    in a coherent block rather than as isolated speckles, which is the failure mode
    the module docstring identifies as the one median filtering cannot repair.

    Parameters
    ----------
    shape : tuple of int
        Native space shape.
    speed : float
        Apparent propagation speed in mm/s; lower means a steeper delay ramp.
    noise : float
        Standard deviation of the noise added to the similarity functions.
    numlags : int
        Number of lag samples.
    lagstep : float
        Lag spacing, in seconds.
    seed : int
        Seed for the noise.

    Returns
    -------
    thecandidatelags, thecandidateamps : NDArray
        As returned by findcandidatepeaks.
    thetau : NDArray
        The true delay of every voxel.
    themask : NDArray
        An all ones mask.
    thevoxdims : tuple of float
        The voxel dimensions in mm.
    """
    thelagaxis = -5.0 + np.arange(numlags) * lagstep
    thetau = np.indices(shape).astype(float)[0] * 2.0 / speed
    thecorrupt = np.zeros(shape, dtype=bool)
    thecorrupt[4:9, 4:9, 2:6] = True

    thecorr = np.where(
        thecorrupt[..., None],
        _waveform(thelagaxis[None, None, None, :] - thetau[..., None], 0.9, 1.0),
        _waveform(thelagaxis[None, None, None, :] - thetau[..., None], 1.0, 0.6),
    )
    thecorr = thecorr + np.random.RandomState(seed).normal(0.0, noise, thecorr.shape)
    thecandidatelags, thecandidateamps = uw.findcandidatepeaks(
        thecorr, thelagaxis, uw.DEFAULT_MAXCANDIDATES
    )
    return (
        thecandidatelags,
        thecandidateamps,
        thetau,
        np.ones(shape, dtype=np.uint16),
        (2.0, 2.0, 2.0),
    )


def test_resolvedelaymap_predicts_by_consensus_not_from_one_neighbour(debug=False):
    """Each voxel is predicted from the median over ALL its assigned neighbours, not
    from the single neighbour that happened to pop off the heap.

    This is the change the module docstring credits with taking new wraps from 1696
    down to 952 on HCP data, turning a statistical tie with despeckling into a 37%
    win.  It is invisible on easy data - region growing gets those right either way -
    so it has to be measured on a case where a single wrapped neighbour really can
    carry a correct voxel off with it.
    """
    thelags, theamps, thetau, themask, thevoxdims = _makehardcase()
    thenaivebad = int((np.abs(np.nan_to_num(thelags[..., 0]) - thetau) > 2.0).sum())

    thetauout = uw.resolvedelaymap(
        thelags, theamps, None, themask, thevoxdims, maxdeltatau=0.0, showprogressbar=False
    )[0]
    thebad = int((np.abs(thetauout - thetau) > 2.0).sum())
    if debug:
        print(f"naive wrong {thenaivebad}, region grown wrong {thebad}")

    assert thenaivebad > 50, "the hard case is not actually hard for naive peak picking"
    # consensus prediction leaves a handful wrong here; predicting from a single
    # neighbour leaves an order of magnitude more
    assert thebad < 15, (
        f"{thebad} voxels wrong after unwrapping - consensus prediction is not holding "
        f"the solution together on coherent wrapped patches"
    )


def test_resolvedelaymap_confidence_is_the_gap_not_the_peak_height(debug=False):
    """Growth order is set by how UNAMBIGUOUS a voxel is, not by how strong its peak
    is.  Those are different things, and confusing them inverts the whole method.

    A voxel with one dominant peak is trustworthy however small that peak is; a voxel
    with two nearly tied peaks is a coin flip however tall they are.  Here the wrapped
    block is deliberately the tallest thing in the volume while also being the most
    ambiguous, so ordering by peak height seeds the grow on the wrapped voxels and
    propagates them across the whole volume.
    """
    theshape = (12, 12, 8)
    thenumlags = 90
    thelagaxis = -5.0 + np.arange(thenumlags) * 0.5
    thetau = np.indices(theshape).astype(float)[0] * 2.0 / 6.0
    thecorrupt = np.zeros(theshape, dtype=bool)
    thecorrupt[4:9, 4:9, 2:6] = True

    # wrapped: peaks of 1.2 and 1.1, so the tallest peak but a gap of only 0.1
    # correct: peaks of 1.0 and 0.6, so a shorter peak but a gap of 0.4
    thecorr = np.where(
        thecorrupt[..., None],
        _waveform(thelagaxis[None, None, None, :] - thetau[..., None], 1.1, 1.2),
        _waveform(thelagaxis[None, None, None, :] - thetau[..., None], 1.0, 0.6),
    )
    thelags, theamps = uw.findcandidatepeaks(thecorr, thelagaxis, uw.DEFAULT_MAXCANDIDATES)
    themask = np.ones(theshape, dtype=np.uint16)

    # confirm the fixture really is the awkward way round
    assert theamps[thecorrupt][:, 0].mean() > theamps[~thecorrupt][:, 0].mean()
    thegapwrapped = float((theamps[..., 0] - theamps[..., 1])[thecorrupt].mean())
    thegapcorrect = float((theamps[..., 0] - theamps[..., 1])[~thecorrupt].mean())
    assert thegapwrapped < thegapcorrect

    thegrown = uw.resolvedelaymap(
        thelags, theamps, None, themask, (2.0, 2.0, 2.0), maxdeltatau=0.0, showprogressbar=False
    )[0]
    thebad = int((np.abs(thegrown - thetau) > 2.0).sum())
    if debug:
        print(
            f"wrapped peak height {theamps[thecorrupt][:, 0].mean():.2f} with gap "
            f"{thegapwrapped:.2f}; correct height {theamps[~thecorrupt][:, 0].mean():.2f} "
            f"with gap {thegapcorrect:.2f}; wrong after unwrapping {thebad}"
        )
    assert thebad == 0, (
        f"{thebad} voxels wrong - the grow was seeded by peak height rather than by "
        f"how unambiguous each voxel is"
    )


def test_resolvedelaymap_confidence_floor_actually_gates_the_sources(debug=False):
    """--minconfidence must really stop low confidence voxels predicting for their
    neighbours, which means the map it produces has to differ from the ungated one.

    The floor defaults to off because it does not IMPROVE anything - the module
    docstring records that every nonzero setting is neutral or worse - but "does not
    help" is not the same as "does nothing", and the option is kept precisely so the
    finding is not re-derived.  If it silently stopped having any effect, that record
    would quietly become untestable.
    """
    thelags, theamps, thetau, themask, thevoxdims = _makehardcase()

    theungated = uw.resolvedelaymap(
        thelags, theamps, None, themask, thevoxdims, maxdeltatau=0.0, showprogressbar=False
    )[0]
    thegated, thechanged, theconfidence = uw.resolvedelaymap(
        thelags,
        theamps,
        None,
        themask,
        thevoxdims,
        maxdeltatau=0.0,
        minconfidence=0.9,
        showprogressbar=False,
    )
    thediffering = int((np.abs(thegated - theungated) > 1.0e-9).sum())
    if debug:
        print(f"a 0.9 confidence floor changed {thediffering} voxels")

    assert thediffering > 0, "the confidence floor had no effect at all"
    # the confidence map is a real per voxel quantity, not a constant
    assert np.ptp(theconfidence[themask > 0]) > 0.0


def test_resolvedelaymap_has_no_absolute_reference(debug=False):
    """Unwrapping resolves delays RELATIVE to each other, not against any absolute
    truth, so a confidently wrapped majority takes the whole volume with it.

    This is inherent, not a defect - the same property every phase unwrapper has -
    but it is worth pinning, because it is the failure mode to suspect if a run
    comes back with every delay shifted by close to one sidelobe period.  The
    algorithm's protection is confidence ordering: it only holds when the wrapped
    voxels are the ambiguous ones, which on real data is what makes them wrap.
    """
    theshape = (8, 8, 6)
    thenumlags = 90
    thelagaxis = -5.0 + np.arange(thenumlags) * 0.5
    thetau = np.indices(theshape).astype(float)[0] * 2.0 / 6.0
    therng = np.random.RandomState(7)
    thecorrupt = therng.rand(*theshape) < 0.25

    # equal confidence: the wrapped voxels are just as sure of themselves as the
    # correct ones, so nothing anchors the correct branch
    thecorr = np.where(
        thecorrupt[..., None],
        _waveform(thelagaxis[None, None, None, :] - thetau[..., None], 0.6, 1.0),
        _waveform(thelagaxis[None, None, None, :] - thetau[..., None], 1.0, 0.6),
    )
    thelags, theamps = uw.findcandidatepeaks(thecorr, thelagaxis, uw.DEFAULT_MAXCANDIDATES)
    themask = np.ones(theshape, dtype=np.uint16)

    thegapcorrect = float((theamps[..., 0] - theamps[..., 1])[~thecorrupt].mean())
    thegapwrapped = float((theamps[..., 0] - theamps[..., 1])[thecorrupt].mean())
    # tied to well within the spread the region grow could discriminate on; they are
    # not bit identical only because the two lobes overlap slightly
    assert (
        abs(thegapcorrect - thegapwrapped) < 0.01
    ), f"the confidences were not actually tied: {thegapcorrect} vs {thegapwrapped}"

    theresult = uw.resolvedelaymap(
        thelags, theamps, None, themask, (2.0, 2.0, 2.0), maxdeltatau=0.0, showprogressbar=False
    )[0]
    theresult = uw.icmrefine(theresult, thelags, themask, numpasses=uw.DEFAULT_NUMPASSES)[0]

    theoffsets = theresult - thetau
    if debug:
        print(f"tied confidence {thegapcorrect:.2f}; offsets {np.unique(np.round(theoffsets, 2))}")

    # whichever branch it picked, it must be SELF CONSISTENT: one offset everywhere,
    # either zero or one full sidelobe period.  A mixture would mean the region grow
    # is not propagating a single solution, which would be a real bug.
    assert np.ptp(theoffsets) < 0.5, "the solution is not internally consistent"
    assert np.isclose(np.median(theoffsets), 0.0, atol=0.5) or np.isclose(
        np.median(theoffsets), SIDELOBE, atol=0.5
    ), "the offset is neither zero nor one sidelobe period"


def _makeflatcase(
    shape=(8, 8, 6), numlags=90, lagstep=0.5, corruptfraction=0.25, seed=7, noise=0.05
):
    """Build rapidtide's flat (numvalidspatiallocs, numlags) layout with planted wraps.

    Parameters
    ----------
    shape : tuple of int
        Native space shape.
    numlags : int
        Number of lag samples.
    lagstep : float
        Lag spacing, in seconds.
    corruptfraction : float
        Fraction of voxels whose sidelobe is made to outrank the main lobe.  Those
        voxels get a 0.9 main lobe against a 1.0 sidelobe, so their peaks are nearly
        tied and their confidence is low, while a correct voxel has a 1.0 against a
        0.6 and is confidently right.  That asymmetry matters: see
        test_resolvedelaymap_has_no_absolute_reference for what happens without it.
    seed : int
        Seed for the corruption pattern.
    noise : float
        Noise added to the similarity functions.  Without it the region grow lands
        on a fixed point and the refinement passes have nothing to do, which would
        make any test comparing paths that include refinement vacuous.

    Returns
    -------
    thecorrflat : NDArray
        Similarity functions, shape (numvoxels, numlags).
    thelagaxis : NDArray
        Lag values, in seconds.
    thetruth : NDArray
        The true delay of every voxel, flattened.
    thecorrupt : NDArray
        Boolean, flattened, True where a wrap was planted.
    """
    thelagaxis = -5.0 + np.arange(numlags) * lagstep
    thegrids = np.indices(shape).astype(float)
    thetau = thegrids[0] * 2.0 / 6.0

    therng = np.random.RandomState(seed)
    thecorrupt = therng.rand(*shape) < corruptfraction

    thecorr = np.where(
        thecorrupt[..., None],
        _waveform(thelagaxis[None, None, None, :] - thetau[..., None], 0.9, 1.0),
        _waveform(thelagaxis[None, None, None, :] - thetau[..., None], 1.0, 0.6),
    )
    thecorr = thecorr + np.random.RandomState(seed + 500).normal(0.0, noise, thecorr.shape)
    thenumvoxels = int(np.prod(shape))
    return (
        thecorr.reshape(thenumvoxels, numlags),
        thelagaxis,
        thetau.reshape(thenumvoxels),
        thecorrupt.reshape(thenumvoxels),
    )


def test_resolvefromsimfunc_matches_the_four_dimensional_path(debug=False):
    """The flat adapter must give bit identical results to running the algorithm on
    the 4D volume.

    This is the whole point of resolvefromsimfunc: it exists purely to avoid
    materialising a 450 MB volume, so it is only correct if it changes nothing.  The
    reshape and scatter it does to get there are exactly where an axis ordering slip
    would hide.
    """
    theshape = (8, 8, 6)
    thecorrflat, thelagaxis, thetruth, dummy = _makeflatcase(shape=theshape)
    thenumvoxels = int(np.prod(theshape))
    thevalidvoxels = np.arange(thenumvoxels)
    thefitmask = np.ones(thenumvoxels, dtype=np.uint16)
    thelagtimes = np.zeros(thenumvoxels, dtype=np.float64)
    thevoxdims = (2.0, 2.0, 2.0)

    theflatresult, dummy2 = uw.resolvefromsimfunc(
        thecorrflat,
        thelagaxis,
        thevalidvoxels,
        theshape,
        thefitmask,
        thelagtimes,
        thevoxdims,
        showprogressbar=False,
    )

    # the same computation, done the long way on the 4D volume
    thecorr4d = thecorrflat.reshape(theshape + (thecorrflat.shape[-1],))
    thelags4d, theamps4d = uw.findcandidatepeaks(thecorr4d, thelagaxis, uw.DEFAULT_MAXCANDIDATES)
    themask4d = np.ones(theshape, dtype=np.uint16)
    thegrown = uw.resolvedelaymap(
        thelags4d, theamps4d, None, themask4d, thevoxdims, maxdeltatau=0.0, showprogressbar=False
    )[0]
    thevolresult, thenumrefined = uw.icmrefine(
        thegrown, thelags4d, themask4d, numpasses=uw.DEFAULT_NUMPASSES
    )

    # the refinement has to actually do something, or the comparison would pass just
    # as happily against an adapter that skipped it
    assert thenumrefined > 0, "icmrefine changed nothing, so this comparison is vacuous"

    if debug:
        print(
            f"refinement moved {thenumrefined} assignments; "
            f"max discrepancy {np.max(np.abs(theflatresult - thevolresult.reshape(-1)))}"
        )
    assert np.array_equal(theflatresult, thevolresult.reshape(thenumvoxels))

    # and it actually repaired the planted wraps, or the comparison is vacuous
    thenaive = np.abs(thelags4d[..., 0].reshape(thenumvoxels) - thetruth) > 2.0
    thefixed = np.abs(theflatresult - thetruth) > 2.0
    assert thenaive.sum() > 20, "no wraps were planted"
    assert thefixed.sum() < 0.1 * thenaive.sum()


def test_resolvefromsimfunc_leaves_unfit_voxels_untouched(debug=False):
    """Voxels outside the fit mask have no delay worth resolving, so whatever they
    were carrying must survive unchanged."""
    theshape = (8, 8, 6)
    thecorrflat, thelagaxis, thetruth, dummy = _makeflatcase(shape=theshape)
    thenumvoxels = int(np.prod(theshape))

    thefitmask = np.ones(thenumvoxels, dtype=np.uint16)
    theunfit = np.zeros(thenumvoxels, dtype=bool)
    theunfit[::7] = True
    thefitmask[theunfit] = 0

    # a recognisable sentinel that resolution must not overwrite
    thelagtimes = np.zeros(thenumvoxels, dtype=np.float64)
    thelagtimes[theunfit] = -999.0

    thenew, dummy2 = uw.resolvefromsimfunc(
        thecorrflat,
        thelagaxis,
        np.arange(thenumvoxels),
        theshape,
        thefitmask,
        thelagtimes,
        (2.0, 2.0, 2.0),
        showprogressbar=False,
    )
    if debug:
        print(f"unfit values {np.unique(thenew[theunfit])}")
    assert np.all(thenew[theunfit] == -999.0), "resolution overwrote out of mask voxels"
    # the in mask ones did get resolved
    assert np.any(thenew[~theunfit] != 0.0)
    # and the input array was not modified in place
    assert np.all(thelagtimes[theunfit] == -999.0)


def test_resolvefromsimfunc_keeps_unfit_voxels_out_of_the_region_grow(debug=False):
    """Out of mask voxels must not merely be excluded from the OUTPUT, they must be
    excluded from the unwrap itself.

    Writing the old values back over them at the end hides the difference, so this
    walls a block of good voxels in behind a shell of unfit ones carrying a wildly
    wrong delay.  If the fit mask is not honoured when the volume mask is built, the
    shell joins the region grow and drags the voxels behind it.
    """
    theshape = (10, 10, 8)
    thenumlags, thelagstep = 90, 0.5
    thelagaxis = -5.0 + np.arange(thenumlags) * thelagstep
    thenumvoxels = int(np.prod(theshape))

    # a hollow shell of unfit voxels around an interior block of good ones
    theshell = np.zeros(theshape, dtype=bool)
    theshell[2:8, 2:8, 1:7] = True
    theshell[3:7, 3:7, 2:6] = False
    theinterior = np.zeros(theshape, dtype=bool)
    theinterior[3:7, 3:7, 2:6] = True

    thetau = np.zeros(theshape)
    thecorr = _waveform(thelagaxis[None, None, None, :] - thetau[..., None], 1.0, 0.6)
    # the shell is centred a long way off, and confidently so
    theshellcorr = _waveform(
        thelagaxis[None, None, None, :] - np.full(theshape, 24.0)[..., None], 1.0, 0.6
    )
    thecorr = np.where(theshell[..., None], theshellcorr, thecorr)

    thefitmask = np.ones(thenumvoxels, dtype=np.uint16)
    thefitmask[theshell.reshape(-1)] = 0

    thenew, dummy = uw.resolvefromsimfunc(
        thecorr.reshape(thenumvoxels, thenumlags),
        thelagaxis,
        np.arange(thenumvoxels),
        theshape,
        thefitmask,
        np.zeros(thenumvoxels, dtype=np.float64),
        (2.0, 2.0, 2.0),
        showprogressbar=False,
    )
    theinteriorvalues = thenew.reshape(theshape)[theinterior]
    if debug:
        print(f"interior delays {np.unique(np.round(theinteriorvalues, 2))}")

    assert np.allclose(
        theinteriorvalues, 0.0, atol=0.3
    ), "the unfit shell took part in the unwrap and dragged the interior with it"


def test_resolvefromsimfunc_counts_only_changes_above_the_threshold(debug=False):
    """The reassignment count is the number of voxels that moved by more than
    RESOLVEDCHANGEDTHRESH.  Re-snapping to a candidate nudges nearly every voxel by
    a hair, so counting any change at all would report the whole brain every pass.
    """
    theshape = (8, 8, 6)
    thecorrflat, thelagaxis, dummy, dummy2 = _makeflatcase(shape=theshape)
    thenumvoxels = int(np.prod(theshape))
    thevalidvoxels = np.arange(thenumvoxels)
    thefitmask = np.ones(thenumvoxels, dtype=np.uint16)
    theargs = (thelagaxis, thevalidvoxels, theshape, thefitmask)

    # what resolution settles on, whatever it is
    theresolved, dummy3 = uw.resolvefromsimfunc(
        thecorrflat, *theargs, np.zeros(thenumvoxels), (2.0, 2.0, 2.0), showprogressbar=False
    )

    # feeding that back means nothing moved at all
    dummy4, thesame = uw.resolvefromsimfunc(
        thecorrflat, *theargs, theresolved.copy(), (2.0, 2.0, 2.0), showprogressbar=False
    )
    # the constant is shared with fitSimFuncMap's resolvechanged audit mask, so the
    # two cannot be allowed to drift apart; pin it rather than comparing it to itself
    assert uw.RESOLVEDCHANGEDTHRESH == 0.5
    # a nudge smaller than the threshold is still nothing
    thenudge = theresolved + 0.4
    dummy5, thenudged = uw.resolvefromsimfunc(
        thecorrflat, *theargs, thenudge, (2.0, 2.0, 2.0), showprogressbar=False
    )
    # a shove larger than it counts every voxel
    theshove = theresolved + 0.6
    dummy6, theshoved = uw.resolvefromsimfunc(
        thecorrflat, *theargs, theshove, (2.0, 2.0, 2.0), showprogressbar=False
    )

    if debug:
        print(f"identical {thesame}, nudged {thenudged}, shoved {theshoved}")
    assert thesame == 0
    assert thenudged == 0, "a sub threshold nudge was counted as a reassignment"
    assert theshoved == thenumvoxels


def _writecorrfile(thedir, theroot, thecorr, thelagaxis, thevoxdim=2.0):
    """Write a similarity function to a rapidtide style corrout NIfTI.

    Parameters
    ----------
    thedir : str
        Directory to write into.
    theroot : str
        Filename root, without the _desc-corrout_info suffix.
    thecorr : NDArray
        The 4D similarity function.
    thelagaxis : NDArray
        The lag values, in seconds.  Written as pixdim[4] and toffset, which is
        where getlagaxis reads them back from.
    thevoxdim : float
        Isotropic voxel size, in mm.

    Returns
    -------
    str
        Full path of the file written.
    """
    thename = os.path.join(thedir, f"{theroot}_desc-corrout_info.nii.gz")
    theaffine = np.diag([thevoxdim, thevoxdim, thevoxdim, 1.0])
    theimage = nb.Nifti1Image(thecorr.astype(np.float32), theaffine)
    theimage.header["pixdim"][4] = float(thelagaxis[1] - thelagaxis[0])
    theimage.header["toffset"] = float(thelagaxis[0])
    nb.save(theimage, thename)
    return thename


def test_resolvedelays_commandline_repairs_a_wrapped_map(debug=False):
    """Smoke test of the standalone tool: it must read a corrout file, find its mask
    by convention, repair the wraps, and write its four maps."""
    theshape = (8, 8, 6)
    thenumlags, thelagstep = 90, 0.5
    thelagaxis = -5.0 + np.arange(thenumlags) * thelagstep
    thecorrflat, dummy, thetruthflat, thecorruptflat = _makeflatcase(
        shape=theshape, numlags=thenumlags, lagstep=thelagstep
    )
    thecorr = thecorrflat.reshape(theshape + (thenumlags,))
    thetruth = thetruthflat.reshape(theshape)

    with tempfile.TemporaryDirectory() as thedir:
        thecorrname = _writecorrfile(thedir, "sub-TEST", thecorr, thelagaxis)
        # the conventionally named mask, which the tool must find without being told
        nb.save(
            nb.Nifti1Image(np.ones(theshape, dtype=np.uint16), np.diag([2.0, 2.0, 2.0, 1.0])),
            os.path.join(thedir, "sub-TEST_desc-corrfit_mask.nii.gz"),
        )
        theoutputroot = os.path.join(thedir, "out")
        theargs = argparse.Namespace(
            corrfile=thecorrname,
            outputroot=theoutputroot,
            maskfile=None,
            prior="smooth",
            maxcandidates=uw.DEFAULT_MAXCANDIDATES,
            numpasses=uw.DEFAULT_NUMPASSES,
            minconfidence=0.0,
            maxdeltatau=uw.DEFAULT_MAXDELTATAU,
            fitradius=uw.DEFAULT_FITRADIUS,
            lagoversamp=1,
            showprogressbar=False,
            debug=False,
        )
        uw.resolvedelays(theargs)

        theexpected = [
            "_desc-maxtimeresolved_map",
            "_desc-resolvechanged_mask",
            "_desc-resolveconfidence_map",
            "_desc-maxtimenaive_map",
        ]
        for thesuffix in theexpected:
            assert os.path.isfile(
                f"{theoutputroot}{thesuffix}.nii.gz"
            ), f"{thesuffix} was not written"

        dummy2, theresolved, dummy3, dummy4, dummy5 = tide_io.readfromnifti(
            f"{theoutputroot}_desc-maxtimeresolved_map.nii.gz"
        )
        dummy6, thenaive, dummy7, dummy8, dummy9 = tide_io.readfromnifti(
            f"{theoutputroot}_desc-maxtimenaive_map.nii.gz"
        )

    theresolved = theresolved.reshape(theshape)
    thenaive = thenaive.reshape(theshape)
    thenaivebad = int((np.abs(thenaive - thetruth) > 2.0).sum())
    theresolvedbad = int((np.abs(theresolved - thetruth) > 2.0).sum())
    if debug:
        print(f"naive wrong {thenaivebad}, resolved wrong {theresolvedbad}")

    # the naive map must really be broken, and the resolved one must really fix it
    assert thenaivebad > 20, "the planted wraps did not survive the file round trip"
    assert theresolvedbad < 0.1 * thenaivebad


def test_resolvedelays_derives_the_root_and_falls_back_to_a_ptp_mask(debug=False):
    """With no mask file present the tool must synthesise one from the data rather
    than failing, and it must still strip the corrout suffix to find its root."""
    theshape = (6, 6, 5)
    thenumlags, thelagstep = 60, 0.5
    thelagaxis = -5.0 + np.arange(thenumlags) * thelagstep
    thegrids = np.indices(theshape).astype(float)
    thetau = thegrids[0] * 2.0 / 6.0
    thecorr = _waveform(thelagaxis[None, None, None, :] - thetau[..., None])
    # a dead slab with no variation at all, which the ptp fallback must exclude
    thecorr[0, :, :, :] = 0.0

    with tempfile.TemporaryDirectory() as thedir:
        thecorrname = _writecorrfile(thedir, "sub-NOMASK", thecorr, thelagaxis)
        theoutputroot = os.path.join(thedir, "out")
        uw.resolvedelays(
            argparse.Namespace(
                corrfile=thecorrname,
                outputroot=theoutputroot,
                maskfile=None,
                prior="smooth",
                maxcandidates=uw.DEFAULT_MAXCANDIDATES,
                numpasses=1,
                minconfidence=0.0,
                maxdeltatau=uw.DEFAULT_MAXDELTATAU,
                fitradius=uw.DEFAULT_FITRADIUS,
                lagoversamp=1,
                showprogressbar=False,
                debug=False,
            )
        )
        dummy, theresolved, dummy2, dummy3, dummy4 = tide_io.readfromnifti(
            f"{theoutputroot}_desc-maxtimeresolved_map.nii.gz"
        )

    theresolved = theresolved.reshape(theshape)
    if debug:
        print(f"dead slab {np.unique(theresolved[0])}, live slab min {theresolved[1:].min()}")
    # the dead slab is outside the synthesised mask, so it holds nothing
    assert np.allclose(theresolved[0], 0.0)
    # and the live voxels were resolved to their ramp values
    assert np.allclose(theresolved[1:], thetau[1:], atol=0.3)


def test_resolvedelays_strips_the_extension_when_the_name_is_unconventional(debug=False):
    """The root is used to guess the mask filename, so it has to be derived even when
    the input is not named XXX_desc-corrout_info.nii.gz.  Falling back to plain
    extension stripping is what lets an arbitrarily named file still find its mask.
    """
    theshape = (6, 6, 5)
    thenumlags = 60
    thelagaxis = -5.0 + np.arange(thenumlags) * 0.5
    thetau = np.indices(theshape).astype(float)[0] * 2.0 / 6.0
    thecorr = _waveform(thelagaxis[None, None, None, :] - thetau[..., None])

    with tempfile.TemporaryDirectory() as thedir:
        # deliberately NOT the conventional corrout name
        thecorrname = os.path.join(thedir, "oddlynamed.nii.gz")
        theimage = nb.Nifti1Image(thecorr.astype(np.float32), np.diag([2.0, 2.0, 2.0, 1.0]))
        theimage.header["pixdim"][4] = 0.5
        theimage.header["toffset"] = -5.0
        nb.save(theimage, thecorrname)
        # the mask it should find, named off the stripped root
        themaskvolume = np.ones(theshape, dtype=np.uint16)
        themaskvolume[0] = 0
        nb.save(
            nb.Nifti1Image(themaskvolume, np.diag([2.0, 2.0, 2.0, 1.0])),
            os.path.join(thedir, "oddlynamed_desc-corrfit_mask.nii.gz"),
        )
        theoutputroot = os.path.join(thedir, "out")
        uw.resolvedelays(
            argparse.Namespace(
                corrfile=thecorrname,
                outputroot=theoutputroot,
                maskfile=None,
                prior="smooth",
                maxcandidates=uw.DEFAULT_MAXCANDIDATES,
                numpasses=1,
                minconfidence=0.0,
                maxdeltatau=uw.DEFAULT_MAXDELTATAU,
                fitradius=uw.DEFAULT_FITRADIUS,
                lagoversamp=1,
                showprogressbar=False,
                debug=False,
            )
        )
        dummy, theresolved, dummy2, dummy3, dummy4 = tide_io.readfromnifti(
            f"{theoutputroot}_desc-maxtimeresolved_map.nii.gz"
        )

    theresolved = theresolved.reshape(theshape)
    if debug:
        print(f"excluded slab {np.unique(theresolved[0])}")
    # the mask really was found and applied: its excluded slab is empty
    assert np.allclose(theresolved[0], 0.0), "the conventionally named mask was not picked up"
    assert np.allclose(theresolved[1:], thetau[1:], atol=0.3)


def test_get_parser_defaults_match_the_module_constants(debug=False):
    """The help text quotes the DEFAULT_ constants, so the two must not drift apart."""
    theparser = uw._get_parser()
    # corrfile is validated for existence as it is parsed, so it has to be real
    with tempfile.NamedTemporaryFile(suffix=".nii.gz") as thefile:
        theargs = theparser.parse_args([thefile.name, "someroot"])
    if debug:
        print(theargs)
    assert theargs.maxcandidates == uw.DEFAULT_MAXCANDIDATES
    assert theargs.numpasses == uw.DEFAULT_NUMPASSES
    assert theargs.minconfidence == uw.DEFAULT_MINCONFIDENCE
    assert theargs.maxdeltatau == uw.DEFAULT_MAXDELTATAU
    assert theargs.fitradius == uw.DEFAULT_FITRADIUS

    # and the constants themselves are the documented values.  numpasses in
    # particular is 3 because the docstring measures the benefit as concentrated in
    # the first two or three refinements, with slow damage accumulating after.
    assert uw.DEFAULT_NUMPASSES == 3
    assert uw.DEFAULT_MAXCANDIDATES == 6
    assert uw.DEFAULT_MINCONFIDENCE == 0.0
    # the smooth prior is the default because flow was not measurably better
    assert theargs.prior == "smooth"
    assert theargs.showprogressbar is True
    assert theargs.lagoversamp == 1


def test_resolvedelays(debug=False):
    test_findcandidatepeaks(debug=debug)
    test_findcandidatepeaks_refines_peaks_off_the_sample_grid(debug=debug)
    test_findcandidatepeaks_orders_by_amplitude_and_pads_with_nan(debug=debug)
    test_findcandidatepeaks_does_not_treat_array_ends_as_peaks(debug=debug)
    test_unwrap_recovers_injected_sidelobe_errors(debug=debug)
    test_unwrap_with_flow_prior(debug=debug)
    test_icmrefine_ignores_voxels_outside_the_mask(debug=debug)
    test_icmrefine_with_one_pass_is_a_noop(debug=debug)
    test_icmrefine_stops_early_once_nothing_changes(debug=debug)
    test_resolvedelaymap_tolerates_voxels_with_no_candidates(debug=debug)
    test_resolvedelaymap_predicts_by_consensus_not_from_one_neighbour(debug=debug)
    test_resolvedelaymap_confidence_is_the_gap_not_the_peak_height(debug=debug)
    test_resolvedelaymap_confidence_floor_actually_gates_the_sources(debug=debug)
    test_resolvedelaymap_has_no_absolute_reference(debug=debug)
    test_resolvefromsimfunc_matches_the_four_dimensional_path(debug=debug)
    test_resolvefromsimfunc_leaves_unfit_voxels_untouched(debug=debug)
    test_resolvefromsimfunc_keeps_unfit_voxels_out_of_the_region_grow(debug=debug)
    test_resolvefromsimfunc_counts_only_changes_above_the_threshold(debug=debug)
    test_resolvedelays_commandline_repairs_a_wrapped_map(debug=debug)
    test_resolvedelays_derives_the_root_and_falls_back_to_a_ptp_mask(debug=debug)
    test_resolvedelays_strips_the_extension_when_the_name_is_unconventional(debug=debug)
    test_get_parser_defaults_match_the_module_constants(debug=debug)


if __name__ == "__main__":
    test_resolvedelays(debug=True)
