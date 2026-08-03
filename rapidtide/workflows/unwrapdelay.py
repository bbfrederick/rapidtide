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
"""
Resolve sidelobe ambiguity in a rapidtide delay map by phase unwrapping.

The problem
-----------
The similarity function of an LFO signal has sidelobes spaced by the dominant
period of the waveform (rapidtide writes its own estimate to
XXX_autocorr_sidelobetime_passN.txt).  When a sidelobe happens to exceed the main
lobe in a voxel, peak picking assigns a delay that is wrong by very nearly one
full period.  The errors are large, and because neighboring voxels see the same
waveform they tend to fail together, in coherent patches rather than as isolated
speckles - which is exactly what a median filter based despeckler cannot repair,
since the neighbors it would vote with are wrong too.

Two things to check before believing anything this program tells you
--------------------------------------------------------------------
1) MEASURE YOUR SIDELOBE.  It is a property of the LFO spectrum of the particular
   acquisition, not a constant.  rapidtide writes its estimate to
   XXX_autocorr_sidelobetime_passN.txt, and leaves it None when it cannot find
   one; the autocorrelation itself is saved as XXX_desc-autocorr_timeseries.  If
   the sidelobe amplitude is negligible there is no periodic ambiguity to
   resolve, and any large delay changes are being driven by something else -
   usually noise repicking the peak more or less at random within the search
   range.  This program will still happily "repair" those, but what it is then
   doing is smoothing, not unwrapping.

   The signature is diagnostic and is reported at the end of a run.  A genuine
   periodic alias produces delay changes that are NARROW and ONE SIDED, centred
   on the sidelobe lag.  Noise driven repicking produces changes that are BROAD
   and TWO SIDED, spread over the search range.

2) LONG DELAYS ARE NOT AUTOMATICALLY WRONG.  Vascular pathology can produce
   genuine delays of tens of seconds.  What distinguishes real long transit from
   a repicking error is not the magnitude but the spatial behaviour: real delay
   varies smoothly from the surrounding tissue, while an error jumps.  Note that
   the smoothness prior used here cannot tell the difference at a discontinuity,
   so on a patient with focal delay pathology this program may erase exactly the
   finding of interest.  Inspect the changed mask before trusting the output.

The framing
-----------
This is phase unwrapping.  The absolute delay is ambiguous modulo the sidelobe
period, while the local GRADIENT of the delay is not.  Unwrapping means
integrating a trustworthy gradient to resolve an ambiguous absolute value, which
is the same problem solved in InSAR and in MRI field mapping.

How much does the flow actually help?  Less than expected - see the measured
numbers below.  The sidelobe period is about 13.8 s in the test data while the
median delay difference between adjacent voxels is 0.24 s, a ratio of nearly 60.
The ambiguity being resolved is therefore enormous compared to the local
gradient, so almost any sensible local prior picks the right candidate, and
predicting simply that a voxel resembles its neighbors does nearly as well as
predicting from the velocity field.  The flow prior should matter more when the
sidelobe spacing is small relative to the local delay gradient; on this data it
does not, so --prior defaults to "smooth", which needs no optical flow at all
and is much faster.

Where the gradient comes from
-----------------------------
From corrflow.  Its optical flow velocity is estimated differentially from the
whole similarity function, so it never picks a peak and therefore cannot pick the
wrong one.  There is a further piece of luck that makes it especially suitable:

    A sidelobe of a travelling wave travels at the SAME velocity as the main
    lobe, because it is the same waveform shifted in lag.

So sidelobes reinforce the velocity estimate rather than corrupting it.  They
corrupt only the absolute delay, which is precisely the quantity we are trying to
repair.  That asymmetry is the whole basis of this program.

Given the velocity, the eikonal relation inverts cleanly.  corrflow solves
v = grad(tau)/|grad(tau)|**2, so::

    grad(tau) = v / |v|**2

and the predicted delay change over a step d is simply d . grad(tau).

What actually reduced the error rate
------------------------------------
Plain region growing propagates its own mistakes: a voxel that has been assigned a
wrapped value will drag its correct neighbors to match, so the method creates new
wrapped voxels even as it fixes old ones.  On HCP data with a strong sidelobe
(13.2 s, amplitude 0.25) the original single neighbor version fixed 8712 of 9591
wrapped voxels but created 1696 new ones, netting 2575 - statistically a tie with
rapidtide's own despeckling at 2664.

Two guards were tried.  Only one worked:

- CONSENSUS PREDICTION (kept, always on).  Predict from the median over EVERY
  already assigned neighbor rather than from the single neighbor that happened to
  pop off the heap.  New wraps fell 1696 -> 952 and the net went 2575 -> 1685,
  which is 37% better than despeckling rather than tied with it.  A single wrapped
  neighbor can no longer carry a voxel on its own.

- CONFIDENCE FLOOR (--minconfidence, kept but defaults to off).  Refusing to let
  low confidence voxels predict for their neighbors sounded right and does
  nothing useful: net totals of 1685, 2449, 1982, 1706, 1685 for floors of 0.0,
  0.25, 0.5, 0.75, 0.9.  Every nonzero setting is neutral or worse, because
  excluding sources shrinks the consensus and makes the median LESS robust, which
  is the opposite of the intent.  The option is retained only so that this does
  not get re-derived.

The algorithm
-------------
Quality guided region growing, the standard unwrapping approach.  Each voxel
offers a set of candidate delays - the local maxima of its similarity function.
Growth starts from the least ambiguous voxel and proceeds in order of confidence,
predicting each new voxel's delay from its already assigned neighbors and
snapping to the nearest candidate.  Processing the confident voxels first means
the reliable regions are unwrapped before the ambiguous ones, so errors do not
propagate outward from a bad seed.

This is a proof of concept.
"""

import argparse
import os
import time
from heapq import heappop, heappush
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter
from tqdm import tqdm

import rapidtide.io as tide_io
import rapidtide.workflows.parser_funcs as pf
from rapidtide.workflows.corrflow import computeopticalflow, getlagaxis, oversamplelagaxis
from rapidtide.workflows.delayflow import makeoutputheader, neighboroffsets

DEFAULT_MAXCANDIDATES = 6
DEFAULT_MAXDELTATAU = 3.0
DEFAULT_MINSPEED = 0.5
DEFAULT_MINCONFIDENCE = 0.0
DEFAULT_NUMPASSES = 3
DEFAULT_AMBIGRATIO = 0.8
DEFAULT_FITRADIUS = 6.0


def _get_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for the unwrapdelay command line tool.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser with all required and optional arguments.
    """
    parser = argparse.ArgumentParser(
        prog="unwrapdelay",
        description=(
            "Resolve sidelobe ambiguity in a rapidtide delay map by unwrapping it "
            "against the optical flow velocity field from corrflow."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "corrfile",
        type=lambda x: pf.is_valid_file(parser, x),
        help="The rapidtide similarity function, XXX_desc-corrout_info.nii.gz.",
    )
    parser.add_argument("outputroot", type=str, help="The root name of the output files.")
    parser.add_argument(
        "--maskfile",
        dest="maskfile",
        type=lambda x: pf.is_valid_file(parser, x),
        metavar="FILE",
        help="The analysis mask.  Defaults to XXX_desc-corrfit_mask.nii.gz.",
        default=None,
    )
    parser.add_argument(
        "--prior",
        dest="prior",
        type=str,
        choices=["smooth", "flow"],
        help=(
            'Which local prior to predict each voxel\'s delay with.  "smooth" predicts '
            "that a voxel resembles its neighbors and needs no optical flow, so it is "
            'much faster.  "flow" predicts using grad(tau) = v/|v|**2 from corrflow.  '
            "The default is smooth because on the test data the flow prior was not "
            "measurably better - the sidelobe period dwarfs the local delay gradient, "
            "so the ambiguity is easy to resolve either way.  Try flow if your "
            "sidelobe spacing is small relative to your delay gradients."
        ),
        default="smooth",
    )
    parser.add_argument(
        "--maxcandidates",
        dest="maxcandidates",
        type=int,
        metavar="N",
        help=(
            f"Consider at most this many candidate peaks per voxel, strongest first.  "
            f"Default is {DEFAULT_MAXCANDIDATES}."
        ),
        default=DEFAULT_MAXCANDIDATES,
    )
    parser.add_argument(
        "--numpasses",
        dest="numpasses",
        type=int,
        metavar="N",
        help=(
            f"Number of passes.  Pass 1 is the region grow; later passes re-snap every "
            f"voxel to the candidate nearest a smoothed version of the current "
            f"solution.  Note that simply rerunning the region grow does nothing - it "
            f"is deterministic and takes no delay map as input - so iteration only "
            f"means anything in this feedback form.  Returns diminish sharply and the "
            f"smoothness prior slowly starts inventing structure, so more is not "
            f"better.  Default is {DEFAULT_NUMPASSES}."
        ),
        default=DEFAULT_NUMPASSES,
    )
    parser.add_argument(
        "--minconfidence",
        dest="minconfidence",
        type=float,
        metavar="QUANTILE",
        help=(
            f"Confidence floor, as a quantile (0-1) of the per voxel peak ambiguity "
            f"gap.  Voxels below it are still assigned, but are not trusted to predict "
            f"for their neighbors.  Region growing otherwise propagates its own "
            f"mistakes - a voxel assigned a wrapped value drags its correct neighbors "
            f"along - which is where newly wrapped voxels come from.  Default is "
            f"{DEFAULT_MINCONFIDENCE} (off)."
        ),
        default=DEFAULT_MINCONFIDENCE,
    )
    parser.add_argument(
        "--maxdeltatau",
        dest="maxdeltatau",
        type=float,
        metavar="SECONDS",
        help=(
            f"Clip the predicted delay change between adjacent voxels to this, in "
            f"seconds.  This keeps a wild velocity estimate from dragging the unwrap "
            f"off course.  Default is {DEFAULT_MAXDELTATAU}."
        ),
        default=DEFAULT_MAXDELTATAU,
    )
    parser.add_argument(
        "--fitradius",
        dest="fitradius",
        type=float,
        metavar="RADIUS",
        help=f"Radius for the optical flow fit, in mm.  Default is {DEFAULT_FITRADIUS}.",
        default=DEFAULT_FITRADIUS,
    )
    parser.add_argument(
        "--lagoversamp",
        dest="lagoversamp",
        type=int,
        metavar="N",
        help="Interpolate the lag axis N times finer before estimating flow.  Default is 1.",
        default=1,
    )
    parser.add_argument(
        "--noprogressbar",
        dest="showprogressbar",
        action="store_false",
        help="Disable progress bars.",
        default=True,
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Turn on debugging information.",
        default=False,
    )
    return parser


def findcandidatepeaks(
    corrdata: NDArray, lagaxis: NDArray, maxcandidates: int = DEFAULT_MAXCANDIDATES
) -> Tuple[NDArray, NDArray]:
    """
    Find the candidate delays in every voxel: the local maxima of the similarity function.

    Peaks are refined by three point parabolic interpolation and returned sorted by
    amplitude, strongest first, so index 0 is what naive peak picking would choose.

    Parameters
    ----------
    corrdata : NDArray
        The 4D similarity function, with lag along the last axis.
    lagaxis : NDArray
        The lag values, in seconds.
    maxcandidates : int, optional
        Maximum number of candidates to keep per voxel.

    Returns
    -------
    candidatelags : NDArray
        Shape (..., maxcandidates), the candidate delays in seconds.  Slots with no
        candidate hold NaN.
    candidateamps : NDArray
        Shape (..., maxcandidates), the corresponding peak amplitudes, NaN where
        there is no candidate.
    """
    thelagstep = float(lagaxis[1] - lagaxis[0])
    numlags = len(lagaxis)

    # interior local maxima
    islocalmax = np.zeros(corrdata.shape, dtype=bool)
    islocalmax[..., 1:-1] = (corrdata[..., 1:-1] > corrdata[..., :-2]) & (
        corrdata[..., 1:-1] >= corrdata[..., 2:]
    )

    theamplitudes = np.where(islocalmax, corrdata, -np.inf)
    theorder = np.argsort(-theamplitudes, axis=-1)[..., :maxcandidates]

    thechosenamps = np.take_along_axis(theamplitudes, theorder, axis=-1)
    thevalid = np.isfinite(thechosenamps)

    # parabolic refinement of each chosen peak
    thesafeindex = np.clip(theorder, 1, numlags - 2)
    theleft = np.take_along_axis(corrdata, thesafeindex - 1, axis=-1)
    thecenter = np.take_along_axis(corrdata, thesafeindex, axis=-1)
    theright = np.take_along_axis(corrdata, thesafeindex + 1, axis=-1)
    thedenominator = theleft - 2.0 * thecenter + theright
    theshift = np.where(
        np.abs(thedenominator) > 1.0e-12,
        0.5
        * (theleft - theright)
        / np.where(np.abs(thedenominator) > 1.0e-12, thedenominator, 1.0),
        0.0,
    )
    theshift = np.clip(theshift, -0.5, 0.5)

    candidatelags = np.where(
        thevalid, lagaxis[np.clip(theorder, 0, numlags - 1)] + theshift * thelagstep, np.nan
    )
    candidateamps = np.where(thevalid, thechosenamps, np.nan)
    return candidatelags, candidateamps


def icmrefine(
    tau: NDArray,
    candidatelags: NDArray,
    themask: NDArray,
    numpasses: int = DEFAULT_NUMPASSES,
    kernelsize: int = 3,
) -> Tuple[NDArray, int]:
    """
    Iteratively re-snap every voxel to the candidate nearest a smoothed solution.

    The region growing pass is deterministic and takes no delay map as input, so
    simply running it twice changes nothing.  Useful iteration means feeding the
    solution back as the prior: smooth the current delay map, then reassign every
    voxel to whichever of its candidates lies closest to that smoothed field.  This
    is iterated conditional modes on a smoothness regularised labelling problem,
    and it converges because the label set is finite.

    It helps, with sharply diminishing returns.  On HCP data the wrapped voxel
    count runs 1685, 1463, 1354, 1272 over the first three iterations and then
    creeps down to 1000 by iteration 20 - but the count of large NON periodic jumps
    rises slowly over the same span (24949 to 25227), which is the signature of the
    smoothness prior starting to invent structure rather than repair it.  Most of
    the benefit is in the first two or three passes, which is why that is the
    default; running to convergence trades a little more wrap reduction against
    slowly accumulating damage elsewhere.

    Parameters
    ----------
    tau : NDArray
        The current delay map.
    candidatelags : NDArray
        Candidate delays, as returned by findcandidatepeaks.
    themask : NDArray
        The 3D mask of valid voxels.
    numpasses : int, optional
        Number of refinement iterations.  1 means the region grow only.
    kernelsize : int, optional
        Size of the median filter used to form the smoothed prior.

    Returns
    -------
    tau : NDArray
        The refined delay map.
    thetotalchanged : int
        How many voxel assignments changed across all iterations.
    """
    thetotalchanged = 0
    for thepass in range(max(numpasses - 1, 0)):
        thesmoothed = median_filter(
            np.where(themask > 0, tau, 0.0), size=kernelsize, mode="nearest"
        )
        thedistances = np.where(
            np.isfinite(candidatelags), np.abs(candidatelags - thesmoothed[..., None]), np.inf
        )
        thebest = np.argmin(thedistances, axis=-1)
        thenew = np.take_along_axis(np.nan_to_num(candidatelags), thebest[..., None], axis=-1)[
            ..., 0
        ]
        thenew = np.where(themask > 0, thenew, 0.0)
        thenumchanged = int(np.sum((themask > 0) & (np.abs(thenew - tau) > 1.0e-6)))
        tau = thenew
        thetotalchanged += thenumchanged
        if thenumchanged == 0:
            break
    return tau, thetotalchanged


def unwrapdelaymap(
    candidatelags: NDArray,
    candidateamps: NDArray,
    velocity: Optional[NDArray],
    themask: NDArray,
    voxdims: Tuple[float, float, float],
    maxdeltatau: float = DEFAULT_MAXDELTATAU,
    minspeed: float = DEFAULT_MINSPEED,
    minconfidence: float = DEFAULT_MINCONFIDENCE,
    showprogressbar: bool = True,
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Unwrap the delay map by quality guided region growing against a velocity field.

    Each voxel is assigned the candidate delay closest to the value predicted from
    its already assigned neighbors, where the prediction uses grad(tau) = v/|v|**2
    from the optical flow field.  Voxels are visited in order of confidence, so
    unambiguous regions are unwrapped first and cannot be dragged off course by an
    ambiguous seed.

    Parameters
    ----------
    candidatelags, candidateamps : NDArray
        Candidate delays and amplitudes, as returned by findcandidatepeaks.
    velocity : NDArray or None
        The 4D optical flow velocity field, in mm/s.  Pass None for the flow free
        smoothness prior, which predicts that a voxel resembles its neighbors.
    themask : NDArray
        The 3D mask of valid voxels.
    voxdims : tuple of float
        The voxel dimensions in mm.
    maxdeltatau : float, optional
        Clip on the predicted delay change per step, in seconds.
    minspeed : float, optional
        Floor on |v| when forming grad(tau) = v/|v|**2, to keep near stationary
        voxels from predicting an enormous delay change.
    minconfidence : float, optional
        Confidence floor, expressed as a quantile of the confidence distribution
        over the mask.  Voxels below it are still assigned, but are not trusted as
        prediction sources for their neighbors.  This exists because plain region
        growing propagates its own mistakes: a voxel that was assigned a wrapped
        value will happily drag its correct neighbors to match.  0.0 disables it.
    showprogressbar : bool, optional
        Show a progress bar.

    Returns
    -------
    tau : NDArray
        The unwrapped delay map, in seconds.
    thechanged : NDArray
        Mask of voxels assigned something other than their strongest peak.
    theconfidence : NDArray
        The per voxel ambiguity gap used to order the growth.
    """
    theshape = themask.shape
    paddedshape = tuple(np.array(theshape) + 2)
    offsets, distances = neighboroffsets(paddedshape, voxdims)

    def pad(thearray, fill=0.0):
        if thearray.ndim == 3:
            return np.pad(thearray, 1, mode="constant", constant_values=fill).reshape(-1)
        return np.pad(
            thearray, ((1, 1), (1, 1), (1, 1), (0, 0)), mode="constant", constant_values=fill
        ).reshape(-1, thearray.shape[-1])

    paddedmask = np.pad(themask > 0, 1, mode="constant", constant_values=False).reshape(-1)
    paddedlags = pad(candidatelags, np.nan)
    paddedamps = pad(candidateamps, np.nan)

    # grad(tau) = v / |v|**2, with a floor on the speed
    if velocity is None:
        paddedgrad = np.zeros((paddedmask.shape[0], 3), dtype=np.float64)
    else:
        thespeed = np.maximum(np.linalg.norm(velocity, axis=-1), minspeed)
        gradtau = velocity / (thespeed**2)[..., np.newaxis]
        paddedgrad = pad(gradtau, 0.0)

    # confidence is the gap between the best and second best peak: a voxel with one
    # dominant peak is unambiguous, one with two similar peaks is a coin flip
    thebest = paddedamps[:, 0]
    thesecond = paddedamps[:, 1] if paddedamps.shape[1] > 1 else np.full_like(thebest, np.nan)
    theconfidence = np.where(np.isnan(thesecond), thebest, thebest - thesecond)
    theconfidence = np.nan_to_num(theconfidence, nan=-np.inf)

    # confidence floor: below this a voxel may be assigned, but is not trusted to
    # predict for anyone else
    if minconfidence > 0.0:
        thefinite = theconfidence[paddedmask & np.isfinite(theconfidence)]
        thefloor = float(np.quantile(thefinite, minconfidence)) if len(thefinite) else -np.inf
    else:
        thefloor = -np.inf
    istrusted = theconfidence >= thefloor

    # displacement to each neighbor, in mm
    thedisplacements = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                thedisplacements.append(
                    np.array([di * voxdims[0], dj * voxdims[1], dk * voxdims[2]])
                )
    thedisplacements = np.array(thedisplacements)

    tau = np.full(paddedmask.shape, np.nan, dtype=np.float64)
    assigned = np.zeros(paddedmask.shape, dtype=bool)
    validindices = np.nonzero(paddedmask)[0]
    numvalid = len(validindices)

    theheap: list = []
    remaining = set(int(i) for i in validindices)

    with tqdm(
        total=numvalid, desc="Unwrapping", unit="voxels", disable=(not showprogressbar)
    ) as thebar:
        while remaining:
            # start a new region at the most confident unassigned voxel
            theseed = max(remaining, key=lambda i: theconfidence[i])
            tau[theseed] = paddedlags[theseed, 0]
            assigned[theseed] = True
            remaining.discard(theseed)
            thebar.update(1)
            heappush(theheap, (-theconfidence[theseed], theseed))

            while theheap:
                dummy, thisindex = heappop(theheap)
                for theoffset, thedisplacement in zip(offsets, thedisplacements):
                    neighborindex = int(thisindex + theoffset)
                    if not paddedmask[neighborindex] or assigned[neighborindex]:
                        continue

                    # Predict by consensus over EVERY already assigned neighbor of
                    # the target, not just the one that happened to pop.  A single
                    # wrapped neighbor can otherwise drag a correct voxel with it,
                    # which is where the newly wrapped voxels come from.  Trusted
                    # neighbors are used alone if any exist.
                    thesources = neighborindex - offsets
                    theusable = assigned[thesources]
                    thepreferred = theusable & istrusted[thesources]
                    if np.any(thepreferred):
                        theusable = thepreferred
                    if not np.any(theusable):
                        continue
                    themidgrads = 0.5 * (paddedgrad[thesources] + paddedgrad[neighborindex])
                    thedeltas = np.clip(
                        np.sum(-thedisplacements * themidgrads, axis=1),
                        -maxdeltatau,
                        maxdeltatau,
                    )
                    theprediction = float(np.median((tau[thesources] + thedeltas)[theusable]))

                    thecandidates = paddedlags[neighborindex]
                    thevalid = np.isfinite(thecandidates)
                    if not np.any(thevalid):
                        continue
                    thedistances = np.where(
                        thevalid, np.abs(thecandidates - theprediction), np.inf
                    )
                    tau[neighborindex] = thecandidates[int(np.argmin(thedistances))]
                    assigned[neighborindex] = True
                    remaining.discard(neighborindex)
                    thebar.update(1)
                    heappush(theheap, (-theconfidence[neighborindex], neighborindex))

    thechanged = assigned & np.isfinite(tau) & (np.abs(tau - paddedlags[:, 0]) > 1.0e-6)

    def unpad(thearray):
        return thearray.reshape(paddedshape)[1:-1, 1:-1, 1:-1]

    return (
        np.nan_to_num(unpad(tau)) * (themask > 0),
        np.uint16(unpad(thechanged)) * np.uint16(themask > 0),
        unpad(np.where(np.isfinite(theconfidence), theconfidence, 0.0)),
    )


def ambiguousfraction(
    corrout: NDArray,
    lagaxis: NDArray,
    fitmask: NDArray,
    separation: float,
    ratio: float = DEFAULT_AMBIGRATIO,
    maxcandidates: int = DEFAULT_MAXCANDIDATES,
) -> float:
    """
    Fraction of voxels facing a genuine choice between two similarity function peaks.

    This is the quantity unwrapping actually acts on, and it is not the same thing
    as the autocorrelation sidelobe amplitude.  The sidelobe is a property of the
    REGRESSOR; wrapping is a property of INDIVIDUAL VOXELS.  Measured on HCP data,
    runs where rapidtide reports no sidelobe at all still had 4-5 percent of voxels
    with a near-tied competing peak at a period-like separation - roughly half the
    rate of the strong-sidelobe runs, but nowhere near zero, and unwrapping fixed
    thousands of voxels on exactly those runs.  Gating on the regressor statistic
    therefore asks the wrong question.

    A voxel counts as ambiguous when its second strongest peak is within `ratio` of
    the strongest and lies more than `separation` seconds away.  The separation
    requirement is what distinguishes a real alternative from a shoulder of the same
    peak; passing rapidtide's own despeckle_thresh keeps this from introducing a new
    tuned constant.

    Parameters
    ----------
    corrout : NDArray
        The similarity function, with lag along the last axis.  Works on either the
        4D volume or rapidtide's (numvalidspatiallocs, numlags) layout.
    lagaxis : NDArray
        The lag values, in seconds.
    fitmask : NDArray
        Mask of voxels to consider, matching corrout's leading axes.
    separation : float
        Minimum lag separation, in seconds, for a competing peak to count.
    ratio : float, optional
        Minimum second/first amplitude ratio for a competing peak to count.
    maxcandidates : int, optional
        Number of candidate peaks to examine.

    Returns
    -------
    float
        The fraction of masked voxels that are ambiguous, 0.0 to 1.0.
    """
    thecandidatelags, thecandidateamps = findcandidatepeaks(corrout, lagaxis, maxcandidates)
    if thecandidatelags.shape[-1] < 2:
        return 0.0
    thebest = thecandidateamps[..., 0]
    thesecond = thecandidateamps[..., 1]
    themask = np.asarray(fitmask) > 0
    if not themask.any():
        return 0.0
    theusable = themask & np.isfinite(thesecond) & (thebest > 0)
    theratio = np.where(theusable, thesecond / np.maximum(thebest, 1.0e-9), 0.0)
    thegap = np.abs(thecandidatelags[..., 1] - thecandidatelags[..., 0])
    theambiguous = theusable & (theratio > ratio) & (thegap > separation)
    return float(theambiguous[themask].mean())


def unwrapfromsimfunc(
    corrout: NDArray,
    lagaxis: NDArray,
    validvoxels: Any,
    nativespaceshape: Any,
    fitmask: NDArray,
    lagtimes: NDArray,
    voxdims: Tuple[float, float, float],
    maxcandidates: int = DEFAULT_MAXCANDIDATES,
    numpasses: int = DEFAULT_NUMPASSES,
    showprogressbar: bool = True,
) -> Tuple[NDArray, int]:
    """
    Run the unwrap on rapidtide's internal flat arrays, for in-pipeline use.

    This is the adapter that lets fitSimFuncMap call the unwrapper without
    materialising the whole similarity function as a 4D volume.  findcandidatepeaks
    operates along the last axis and is otherwise shape agnostic, so it runs
    directly on the (numvalidspatiallocs, numlags) array; only the much smaller
    candidate arrays (numvalidspatiallocs, maxcandidates) are expanded to native
    space.  For a typical HCP run that is 21 MB rather than 450 MB.

    Parameters
    ----------
    corrout : NDArray
        The similarity function, shape (numvalidspatiallocs, numlags).
    lagaxis : NDArray
        The lag values, in seconds (rapidtide's trimmedcorrscale).
    validvoxels : array-like
        Indices of the valid voxels within the flattened native space.
    nativespaceshape : tuple
        The 3D shape of the volume.
    fitmask : NDArray
        Fit success mask, indexed like lagtimes.
    lagtimes : NDArray
        The current delay estimates, indexed over valid voxels.
    voxdims : tuple of float
        The voxel dimensions in mm.
    maxcandidates : int, optional
        Maximum candidate peaks per voxel.
    numpasses : int, optional
        Region grow plus this many minus one ICM refinement passes.
    showprogressbar : bool, optional
        Show a progress bar.

    Returns
    -------
    newlagtimes : NDArray
        The unwrapped delay estimates, indexed like lagtimes.
    numchanged : int
        How many voxels were reassigned by more than half a second.
    """
    nativespaceshape = tuple(int(thedim) for thedim in nativespaceshape)
    numspatiallocs = int(np.prod(nativespaceshape))

    thecandidatelags, thecandidateamps = findcandidatepeaks(corrout, lagaxis, maxcandidates)
    thenumcandidates = thecandidatelags.shape[-1]

    fulllags = np.full((numspatiallocs, thenumcandidates), np.nan, dtype=np.float64)
    fullamps = np.full((numspatiallocs, thenumcandidates), np.nan, dtype=np.float64)
    fulllags[validvoxels, :] = thecandidatelags
    fullamps[validvoxels, :] = thecandidateamps
    fulllags = fulllags.reshape(nativespaceshape + (thenumcandidates,))
    fullamps = fullamps.reshape(nativespaceshape + (thenumcandidates,))

    themask = np.zeros(numspatiallocs, dtype=np.uint16)
    themask[validvoxels] = np.uint16(np.asarray(fitmask) > 0)
    themask = themask.reshape(nativespaceshape)

    tau = unwrapdelaymap(
        fulllags,
        fullamps,
        None,
        themask,
        voxdims,
        maxdeltatau=0.0,
        showprogressbar=showprogressbar,
    )[0]
    if numpasses > 1:
        tau = icmrefine(tau, fulllags, themask, numpasses=numpasses)[0]

    newlagtimes = np.array(lagtimes, dtype=lagtimes.dtype, copy=True)
    theunwrapped = tau.reshape(numspatiallocs)[validvoxels]
    thevalid = np.asarray(fitmask) > 0
    newlagtimes[thevalid] = theunwrapped[thevalid]
    numchanged = int(np.sum(np.abs(newlagtimes - lagtimes) > 0.5))
    return newlagtimes, numchanged


def unwrapdelay(args: argparse.Namespace) -> None:
    """
    Unwrap a rapidtide delay map against the corrflow velocity field.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command line arguments.

    Returns
    -------
    None
    """
    starttime = time.time()
    theroot = args.corrfile
    for theending in ["_desc-corrout_info.nii.gz", "_desc-corrout_info.nii"]:
        if theroot.endswith(theending):
            theroot = theroot[: -len(theending)]
            break
    else:
        theroot, _ = tide_io.niftisplitext(theroot)

    print(f"reading similarity function from {args.corrfile}")
    corr_img, corr_data, corr_hdr, corr_dims, corr_sizes = tide_io.readfromnifti(args.corrfile)
    xsize, ysize, numslices, numlags = tide_io.parseniftidims(corr_dims)
    xdim, ydim, slicethickness, dummy = tide_io.parseniftisizes(corr_sizes)
    voxdims = (xdim, ydim, slicethickness)
    corrdata = np.nan_to_num(corr_data).astype(np.float64)
    lagaxis = getlagaxis(corr_hdr, numlags)

    if args.maskfile is None:
        thecandidate = f"{theroot}_desc-corrfit_mask.nii.gz"
        if os.path.isfile(thecandidate):
            args.maskfile = thecandidate
    if args.maskfile is not None:
        print(f"reading mask from {args.maskfile}")
        dummy, mask_data, mask_hdr, dummy2, dummy3 = tide_io.readfromnifti(args.maskfile)
        themask = np.uint16(mask_data.reshape((xsize, ysize, numslices)) > 0)
    else:
        themask = np.uint16(np.ptp(corrdata, axis=-1) > 0.0)
    print(f"{int(np.sum(themask))} voxels in the mask")

    if args.lagoversamp > 1:
        corrdata, lagaxis = oversamplelagaxis(corrdata, lagaxis, args.lagoversamp)

    if args.prior == "flow":
        print("estimating the flow field with corrflow")
        velocity = computeopticalflow(
            corrdata,
            themask,
            voxdims,
            lagaxis,
            fitradius=args.fitradius,
            showprogressbar=args.showprogressbar,
        )[0]
    else:
        velocity = None

    print("finding candidate peaks")
    candidatelags, candidateamps = findcandidatepeaks(corrdata, lagaxis, args.maxcandidates)
    thenumcandidates = np.sum(np.isfinite(candidatelags), axis=-1)
    print(
        f"  median candidates per voxel: {np.median(thenumcandidates[themask > 0]):.1f}, "
        f"{100.0 * np.mean(thenumcandidates[themask > 0] > 1):.1f}% of voxels are ambiguous"
    )

    print("unwrapping")
    tau, thechanged, theconfidence = unwrapdelaymap(
        candidatelags,
        candidateamps,
        velocity,
        themask,
        voxdims,
        maxdeltatau=(args.maxdeltatau if args.prior == "flow" else 0.0),
        minconfidence=args.minconfidence,
        showprogressbar=args.showprogressbar,
    )
    if args.numpasses > 1:
        tau, thenumrefined = icmrefine(tau, candidatelags, themask, numpasses=args.numpasses)
        print(f"  {args.numpasses - 1} refinement passes changed {thenumrefined} assignments")
        thechanged = np.uint16(
            (themask > 0) & (np.abs(tau - np.nan_to_num(candidatelags[..., 0])) > 1.0e-6)
        )

    thenumchanged = int(np.sum(thechanged))
    print(
        f"  reassigned {thenumchanged} voxels "
        f"({100.0 * thenumchanged / max(int(np.sum(themask)), 1):.2f}%)"
    )

    # Diagnostic: does this actually look like a periodic alias?  A real sidelobe
    # ambiguity moves delays by close to one period, in one direction.  Noise
    # repicking the peak inside the search range moves them both ways, broadly.
    thenaive = np.nan_to_num(candidatelags[..., 0])
    thechanges = (tau - thenaive)[(themask > 0) & (thechanged > 0)]
    if len(thechanges) > 20:
        thepositive = float(np.mean(thechanges > 0))
        theiqr = float(
            np.percentile(np.abs(thechanges), 75) - np.percentile(np.abs(thechanges), 25)
        )
        themedian = float(np.median(np.abs(thechanges)))
        print(f"  change magnitude: median {themedian:.2f} s, IQR width {theiqr:.2f} s")
        print(f"  {100.0 * thepositive:.0f}% of changes are positive")
        if thepositive > 0.9 or thepositive < 0.1:
            print("  -> one sided and periodic looking: consistent with sidelobe aliasing")
        else:
            print(
                "  -> WARNING: changes go both ways, which is NOT the signature of a "
                "periodic\n     sidelobe alias.  Check your measured sidelobe amplitude "
                "(rapidtide leaves\n     acsidelobelag None when it finds none).  If there "
                "is no sidelobe, this run is\n     smoothing noise driven peak instability, "
                "not unwrapping - and it may be\n     erasing genuine long delays.  Real "
                "vascular pathology can produce delays of\n     tens of seconds, "
                "distinguished from error by varying smoothly, not by size."
            )

    output_hdr = makeoutputheader(corr_hdr, numvols=1)
    tide_io.savetonifti(
        tau.astype(np.float32),
        output_hdr,
        f"{args.outputroot}_desc-maxtimeunwrapped_map",
        debug=args.debug,
    )
    tide_io.savetonifti(
        np.uint16(candidatelags[..., 0] * 0 + thechanged),
        output_hdr,
        f"{args.outputroot}_desc-unwrapchanged_mask",
        debug=args.debug,
    )
    tide_io.savetonifti(
        theconfidence.astype(np.float32),
        output_hdr,
        f"{args.outputroot}_desc-unwrapconfidence_map",
        debug=args.debug,
    )
    tide_io.savetonifti(
        np.nan_to_num(candidatelags[..., 0]).astype(np.float32) * (themask > 0),
        output_hdr,
        f"{args.outputroot}_desc-maxtimenaive_map",
        debug=args.debug,
    )
    print(f"done in {time.time() - starttime:.1f} seconds")


def main() -> None:
    try:
        args = _get_parser().parse_args()
    except SystemExit:
        _get_parser().print_help()
        raise
    unwrapdelay(args)


if __name__ == "__main__":
    main()
