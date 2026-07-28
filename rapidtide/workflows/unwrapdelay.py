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
from tqdm import tqdm

import rapidtide.io as tide_io
import rapidtide.workflows.parser_funcs as pf
from rapidtide.workflows.corrflow import computeopticalflow, getlagaxis, oversamplelagaxis
from rapidtide.workflows.delayflow import makeoutputheader, neighboroffsets

DEFAULT_MAXCANDIDATES = 6
DEFAULT_MAXDELTATAU = 3.0
DEFAULT_MINSPEED = 0.5
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


def unwrapdelaymap(
    candidatelags: NDArray,
    candidateamps: NDArray,
    velocity: Optional[NDArray],
    themask: NDArray,
    voxdims: Tuple[float, float, float],
    maxdeltatau: float = DEFAULT_MAXDELTATAU,
    minspeed: float = DEFAULT_MINSPEED,
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
                    # predict using the average gradient over the step
                    themidgrad = 0.5 * (paddedgrad[thisindex] + paddedgrad[neighborindex])
                    thedelta = float(np.dot(thedisplacement, themidgrad))
                    thedelta = float(np.clip(thedelta, -maxdeltatau, maxdeltatau))
                    theprediction = tau[thisindex] + thedelta

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
        showprogressbar=args.showprogressbar,
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
