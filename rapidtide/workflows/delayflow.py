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
Convert a rapidtide delay (maxtime) map into a blood flow map.

The theory
----------
The delay map tau(x) is an arrival time field: it says when the sLFO waveform
reaches each voxel.  A parcel of blood carrying the waveform along a trajectory
x(t) arrives at x(t) at time t, so tau(x(t)) = t.  Differentiating along the
trajectory gives the governing equation of this entire workflow::

    grad(tau) . v = 1

That is the eikonal equation.  It says the component of the velocity along the
arrival time gradient is 1/|grad(tau)|, and that flow points toward INCREASING
delay.  Unlike diffusion tractography we therefore get a true direction, not
just an orientation - the sign of the gradient tells us upstream from
downstream for free.

The catch: this is one scalar equation in three unknowns.  Any velocity
component tangent to the isochrone surfaces is invisible to us.  This is the
aperture problem, and it is a property of the measurement, not of the
algorithm.  What we actually recover is the normal velocity of the arrival time
wavefront, which equals the true blood velocity only where flow is
perpendicular to the isochrones.  The minimum norm solution is::

    v = grad(tau) / |grad(tau)|**2

What this program does
----------------------
1) Robust gradient estimation by local weighted plane fitting of tau, weighted
   by a CBV/quality proxy (maxcorrsq).
2) Velocity and speed from the eikonal relation, with an explicit speed ceiling
   derived from the delay noise level (in fast vessels |grad(tau)| is
   indistinguishable from zero and the speed estimate blows up).
3) Depression filling (priority flood) of tau, which removes the spurious local
   extrema that noise creates.  Without this, streamlines spiral into fake
   sinks and the territory map shatters into hundreds of fragments.
4) Multiple flow direction (MFD) routing and flow accumulation, borrowed from
   terrain analysis - treat tau as an elevation surface and let blood run
   downhill.  High accumulation identifies trunk draining vessels.
5) Catchment labeling, which gives vascular territories as a byproduct.
6) Streamline tractography of the velocity field.
7) A divergence of flux diagnostic.  Blood is conserved, so div(cbv * v) should
   be near zero away from true inlets and outlets.  We do not (yet) impose this
   as a constraint, but we report it, because it is the single best indicator of
   where the flow estimate is untrustworthy.

Interpretive caveat
-------------------
A voxel contains arterial, capillary, and venous compartments with very
different delays, and the measured lag is a CBV weighted blend in which the
venous compartment dominates BOLD weighting.  The flow map produced here is
therefore largely a VENOUS DRAINAGE map in parenchyma, with arterial structure
visible only in and near large arteries.  Bear that in mind when looking at the
pictures.

This is a proof of concept.
"""

import argparse
import copy
import os
import time
from heapq import heappop, heappush
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import correlate, distance_transform_edt, map_coordinates
from tqdm import tqdm

import rapidtide.filter as tide_filt
import rapidtide.io as tide_io
import rapidtide.workflows.parser_funcs as pf

DEFAULT_FITRADIUS = 6.0
DEFAULT_GAUSSSIGMA = -1.0
DEFAULT_DELAYNOISE = 0.1
DEFAULT_MFDEXPONENT = 1.1
DEFAULT_CBVTHRESH = 0.05
DEFAULT_FILLEPSILON = 1.0e-7
DEFAULT_STEPSIZE = 0.5
DEFAULT_MAXSTEPS = 2000
DEFAULT_MINLENGTH = 5.0
DEFAULT_SEEDSTRIDE = 2
DEFAULT_MINTERRITORYSIZE = 20


def _get_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for the delayflow command line tool.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser with all required and optional arguments.
    """
    parser = argparse.ArgumentParser(
        prog="delayflow",
        description=(
            "Infer a blood flow map from a rapidtide delay map by treating the delay "
            "map as an arrival time field and solving the eikonal relation "
            "grad(tau).v = 1."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "delayfile",
        type=lambda x: pf.is_valid_file(parser, x),
        help=(
            "The name of the rapidtide delay map, normally "
            "XXX_desc-maxtime_map.nii.gz.  Units are assumed to be seconds."
        ),
    )
    parser.add_argument("outputroot", type=str, help="The root name of the output files.")

    parser.add_argument(
        "--cbvfile",
        dest="cbvfile",
        type=lambda x: pf.is_valid_file(parser, x),
        metavar="FILE",
        help=(
            "A relative CBV proxy map, used to weight the gradient fit, to seed the "
            "flow accumulation, and to form the flux for the divergence diagnostic.  "
            "Defaults to XXX_desc-maxcorrsq_map.nii.gz alongside the delay map, if it "
            "exists.  Note that maxcorrsq is doing double duty here: it is a rough "
            "relative blood volume proxy AND a delay reliability weight, since it is "
            "the fraction of variance explained by the fit."
        ),
        default=None,
    )
    parser.add_argument(
        "--maskfile",
        dest="maskfile",
        type=lambda x: pf.is_valid_file(parser, x),
        metavar="FILE",
        help=(
            "Restrict the analysis to this mask.  Defaults to "
            "XXX_desc-corrfit_mask.nii.gz alongside the delay map if it exists, "
            "otherwise to voxels where the delay map is nonzero and finite."
        ),
        default=None,
    )
    parser.add_argument(
        "--cbvthresh",
        dest="cbvthresh",
        type=float,
        metavar="VALUE",
        help=(
            f"Exclude voxels whose CBV proxy is below this value (after normalization "
            f"to the 98th percentile).  Default is {DEFAULT_CBVTHRESH}."
        ),
        default=DEFAULT_CBVTHRESH,
    )
    parser.add_argument(
        "--gausssigma",
        dest="gausssigma",
        type=float,
        metavar="SIGMA",
        help=(
            "Spatially smooth the delay map with a gaussian kernel of this sigma (mm) "
            "before differentiating.  Set to a negative value to use half the mean "
            "voxel dimension, set to 0.0 to disable.  Default is to autoset."
        ),
        default=DEFAULT_GAUSSSIGMA,
    )
    parser.add_argument(
        "--fitradius",
        dest="fitradius",
        type=float,
        metavar="RADIUS",
        help=(
            f"The radius, in mm, of the neighborhood used for the local plane fit that "
            f"estimates grad(tau).  This sets the effective spatial scale of the flow "
            f"map, and together with --delaynoise it sets the speed ceiling.  Default "
            f"is {DEFAULT_FITRADIUS}."
        ),
        default=DEFAULT_FITRADIUS,
    )
    parser.add_argument(
        "--delaynoise",
        dest="delaynoise",
        type=float,
        metavar="SECONDS",
        help=(
            f"The approximate standard deviation of the per voxel delay estimate, in "
            f"seconds.  Gradients smaller than delaynoise/fitradius are considered "
            f"unresolved, and the speeds derived from them are flagged and capped.  "
            f"Default is {DEFAULT_DELAYNOISE}."
        ),
        default=DEFAULT_DELAYNOISE,
    )
    parser.add_argument(
        "--mfdexponent",
        dest="mfdexponent",
        type=float,
        metavar="P",
        help=(
            f"The exponent applied to the slope when partitioning flow among downstream "
            f"neighbors in the multiple flow direction routing.  Larger values "
            f"concentrate flow along the steepest path; p=1.1 is the standard terrain "
            f"analysis value.  Default is {DEFAULT_MFDEXPONENT}."
        ),
        default=DEFAULT_MFDEXPONENT,
    )
    parser.add_argument(
        "--minterritorysize",
        dest="minterritorysize",
        type=int,
        metavar="NVOXELS",
        help=(
            f"Territories smaller than this many voxels are set to zero in the "
            f"territory map.  Default is {DEFAULT_MINTERRITORYSIZE}."
        ),
        default=DEFAULT_MINTERRITORYSIZE,
    )
    parser.add_argument(
        "--nofill",
        dest="dofill",
        action="store_false",
        help=(
            "Skip the depression filling step.  Routing and territory labeling will "
            "then be corrupted by every noise induced local extremum in the delay map, "
            "so this is really only useful for seeing what filling buys you."
        ),
        default=True,
    )
    parser.add_argument(
        "--nostreamlines",
        dest="dostreamlines",
        action="store_false",
        help="Skip streamline tracking.",
        default=True,
    )
    parser.add_argument(
        "--nodivergence",
        dest="dodivergence",
        action="store_false",
        help="Skip the divergence of flux diagnostic.",
        default=True,
    )
    parser.add_argument(
        "--seedstride",
        dest="seedstride",
        type=int,
        metavar="N",
        help=(
            f"Seed streamlines from every Nth voxel in each dimension.  Default is "
            f"{DEFAULT_SEEDSTRIDE}."
        ),
        default=DEFAULT_SEEDSTRIDE,
    )
    parser.add_argument(
        "--stepsize",
        dest="stepsize",
        type=float,
        metavar="MM",
        help=f"Streamline integration step size in mm.  Default is {DEFAULT_STEPSIZE}.",
        default=DEFAULT_STEPSIZE,
    )
    parser.add_argument(
        "--maxsteps",
        dest="maxsteps",
        type=int,
        metavar="N",
        help=f"Maximum number of steps per streamline.  Default is {DEFAULT_MAXSTEPS}.",
        default=DEFAULT_MAXSTEPS,
    )
    parser.add_argument(
        "--minlength",
        dest="minlength",
        type=float,
        metavar="MM",
        help=f"Discard streamlines shorter than this, in mm.  Default is {DEFAULT_MINLENGTH}.",
        default=DEFAULT_MINLENGTH,
    )
    parser.add_argument(
        "--noprogressbar",
        dest="showprogressbar",
        action="store_false",
        help="Will disable showing progress bars (helpful if stdout is going to a file).",
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


def makeoutputheader(inputheader: Any, numvols: int = 1) -> Any:
    """
    Make an output nifti header with the right number of volumes.

    Parameters
    ----------
    inputheader : nifti header
        The header to copy.
    numvols : int, optional
        The number of volumes in the output.  Default is 1.

    Returns
    -------
    nifti header
        A copy of the input header, set up for a 3D or 4D output.
    """
    theheader = copy.deepcopy(inputheader)
    if numvols > 1:
        theheader["dim"][0] = 4
        theheader["dim"][4] = numvols
        theheader["pixdim"][4] = 1.0
    else:
        theheader["dim"][0] = 3
        theheader["dim"][4] = 1
    return theheader


def neighboroffsets(
    paddedshape: Tuple[int, int, int], voxdims: Tuple[float, float, float]
) -> Tuple[NDArray, NDArray]:
    """
    Precompute the flat index offsets and physical distances of the 26 neighbors.

    Working with flat indices into a volume padded by one voxel on every side means
    that every voxel in the mask is guaranteed to have all 26 neighbors in range, so
    no bounds checking is needed in the inner loops.

    Parameters
    ----------
    paddedshape : tuple of int
        The shape of the padded volume.
    voxdims : tuple of float
        The voxel dimensions in mm.

    Returns
    -------
    offsets : NDArray
        The flat index offset of each of the 26 neighbors.
    distances : NDArray
        The physical distance to each of the 26 neighbors, in mm.
    """
    ny, nz = paddedshape[1], paddedshape[2]
    offsets = []
    distances = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                offsets.append(di * ny * nz + dj * nz + dk)
                distances.append(
                    np.sqrt(
                        (di * voxdims[0]) ** 2 + (dj * voxdims[1]) ** 2 + (dk * voxdims[2]) ** 2
                    )
                )
    return np.array(offsets, dtype=np.int64), np.array(distances, dtype=np.float64)


def priorityflood(
    elevation: NDArray,
    themask: NDArray,
    epsilon: float = DEFAULT_FILLEPSILON,
    showprogressbar: bool = True,
) -> NDArray:
    """
    Fill the depressions in an elevation surface using the priority flood algorithm.

    This is the Barnes et al. priority flood with an epsilon gradient.  After
    filling, every voxel in the mask has a strictly descending path to the mask
    boundary, which is exactly the condition needed for the flow routing and
    catchment labeling to be well defined.

    Note the sign convention.  Priority flood removes local MINIMA of whatever
    surface you hand it.  To remove the spurious local maxima of tau (dead end
    sinks that break downstream accumulation) call this with elevation=-tau.  To
    remove the spurious local minima of tau (fake sources that shatter the
    territory map) call it with elevation=+tau.

    Parameters
    ----------
    elevation : NDArray
        The 3D elevation surface to fill.
    themask : NDArray
        The 3D mask of valid voxels.
    epsilon : float, optional
        The tiny increment added along filled paths to guarantee strict descent.
    showprogressbar : bool, optional
        Show a progress bar.

    Returns
    -------
    NDArray
        The depression filled elevation surface, on the original grid.
    """
    paddedmask = np.pad(themask > 0, 1, mode="constant", constant_values=False)
    paddedelev = np.pad(elevation, 1, mode="constant", constant_values=0.0)
    paddedshape = paddedmask.shape

    maskflat = paddedmask.reshape(-1)
    elevflat = paddedelev.reshape(-1).astype(np.float64)
    offsets, _ = neighboroffsets(paddedshape, (1.0, 1.0, 1.0))

    filled = np.full(elevflat.shape, np.inf, dtype=np.float64)
    visited = np.zeros(elevflat.shape, dtype=bool)

    # seed the queue with every masked voxel that touches the outside of the mask
    validindices = np.nonzero(maskflat)[0]
    theheap: List[Tuple[float, int]] = []
    for theindex in validindices:
        if not np.all(maskflat[theindex + offsets]):
            filled[theindex] = elevflat[theindex]
            visited[theindex] = True
            heappush(theheap, (float(elevflat[theindex]), int(theindex)))

    if len(theheap) == 0:
        # no boundary at all - nothing sensible to do, hand back what we were given
        return elevation.copy()

    numvalid = len(validindices)
    with tqdm(
        total=numvalid,
        desc="Filling",
        unit="voxels",
        disable=(not showprogressbar),
    ) as thebar:
        thebar.update(len(theheap))
        while theheap:
            thiselev, thisindex = heappop(theheap)
            for theoffset in offsets:
                neighborindex = thisindex + theoffset
                if maskflat[neighborindex] and not visited[neighborindex]:
                    visited[neighborindex] = True
                    filled[neighborindex] = max(elevflat[neighborindex], thiselev + epsilon)
                    heappush(theheap, (float(filled[neighborindex]), int(neighborindex)))
                    thebar.update(1)

    filled[~visited] = elevflat[~visited]
    return filled.reshape(paddedshape)[1:-1, 1:-1, 1:-1]


def estimategradient(
    tau: NDArray,
    theweight: NDArray,
    themask: NDArray,
    voxdims: Tuple[float, float, float],
    fitradius: float,
    ridge: float = 1.0e-8,
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Estimate grad(tau) by local weighted plane fitting.

    At each voxel we fit tau ~ a + b.d over all masked neighbors within fitradius,
    where d is the offset in mm, minimizing sum(w * (a + b.d - tau)**2).  This is
    far more robust than finite differencing, and because the weights include the
    mask it degrades gracefully at the edge of the brain instead of manufacturing
    an enormous spurious gradient there.

    The trick that makes this fast is that every moment of the normal equations is
    a correlation of either the mask or mask*tau with a fixed small kernel, so the
    whole volume is handled with 14 correlations plus one batched 4x4 solve.

    Parameters
    ----------
    tau : NDArray
        The 3D arrival time field, in seconds.
    theweight : NDArray
        A per voxel weight (here, the CBV/quality proxy).  Voxels with unreliable
        delays should be downweighted, which is precisely what maxcorrsq does.
    themask : NDArray
        The 3D mask of valid voxels.
    voxdims : tuple of float
        The voxel dimensions in mm.
    fitradius : float
        The radius of the fitting neighborhood, in mm.
    ridge : float, optional
        A small ridge term added to the normal equations for stability.

    Returns
    -------
    gradx, grady, gradz : NDArray
        The components of grad(tau), in s/mm.
    fitvalid : NDArray
        A mask of voxels where the fit was well conditioned.
    """
    # build the kernel coordinate grids, in mm
    halfwidths = [max(int(np.ceil(fitradius / thedim)), 1) for thedim in voxdims]
    thegrids = np.meshgrid(
        *[
            np.arange(-halfwidth, halfwidth + 1) * thedim
            for halfwidth, thedim in zip(halfwidths, voxdims)
        ],
        indexing="ij",
    )
    dx, dy, dz = thegrids
    theradius = np.sqrt(dx**2 + dy**2 + dz**2)

    # gaussian weights, truncated at fitradius
    thekernel = np.exp(-0.5 * (theradius / (fitradius / 2.0)) ** 2)
    thekernel[theradius > fitradius] = 0.0

    maskeddata = (themask > 0).astype(np.float64)
    weighteddata = maskeddata * np.nan_to_num(theweight)
    weightedtau = weighteddata * np.nan_to_num(tau)

    def moment(thedata: NDArray, thekern: NDArray) -> NDArray:
        """
        Accumulate one moment of the normal equations over the fitting neighborhood.

        Parameters
        ----------
        thedata : NDArray
            The 3D volume to accumulate, already multiplied by the fit weights.
        thekern : NDArray
            The neighborhood kernel, premultiplied by whatever combination of dx, dy,
            and dz this particular moment requires.

        Returns
        -------
        NDArray
            The moment at every voxel, on the original grid.
        """
        return correlate(thedata, thekern, mode="constant", cval=0.0)

    # moments of the design matrix
    S = moment(weighteddata, thekernel)
    Sx = moment(weighteddata, thekernel * dx)
    Sy = moment(weighteddata, thekernel * dy)
    Sz = moment(weighteddata, thekernel * dz)
    Sxx = moment(weighteddata, thekernel * dx * dx)
    Sxy = moment(weighteddata, thekernel * dx * dy)
    Sxz = moment(weighteddata, thekernel * dx * dz)
    Syy = moment(weighteddata, thekernel * dy * dy)
    Syz = moment(weighteddata, thekernel * dy * dz)
    Szz = moment(weighteddata, thekernel * dz * dz)

    # moments of the data
    T = moment(weightedtau, thekernel)
    Tx = moment(weightedtau, thekernel * dx)
    Ty = moment(weightedtau, thekernel * dy)
    Tz = moment(weightedtau, thekernel * dz)

    validvoxels = np.nonzero(themask > 0)
    numvalid = len(validvoxels[0])

    thematrix = np.zeros((numvalid, 4, 4), dtype=np.float64)
    thematrix[:, 0, 0] = S[validvoxels]
    thematrix[:, 0, 1] = thematrix[:, 1, 0] = Sx[validvoxels]
    thematrix[:, 0, 2] = thematrix[:, 2, 0] = Sy[validvoxels]
    thematrix[:, 0, 3] = thematrix[:, 3, 0] = Sz[validvoxels]
    thematrix[:, 1, 1] = Sxx[validvoxels]
    thematrix[:, 1, 2] = thematrix[:, 2, 1] = Sxy[validvoxels]
    thematrix[:, 1, 3] = thematrix[:, 3, 1] = Sxz[validvoxels]
    thematrix[:, 2, 2] = Syy[validvoxels]
    thematrix[:, 2, 3] = thematrix[:, 3, 2] = Syz[validvoxels]
    thematrix[:, 3, 3] = Szz[validvoxels]

    therhs = np.zeros((numvalid, 4), dtype=np.float64)
    therhs[:, 0] = T[validvoxels]
    therhs[:, 1] = Tx[validvoxels]
    therhs[:, 2] = Ty[validvoxels]
    therhs[:, 3] = Tz[validvoxels]

    # ridge regularize, scaled to the size of the problem, then solve them all at once
    thescale = np.maximum(np.abs(np.trace(thematrix, axis1=1, axis2=2)), 1.0e-30)
    thematrix += ridge * thescale[:, np.newaxis, np.newaxis] * np.eye(4)[np.newaxis, :, :]

    thesolution = np.linalg.solve(thematrix, therhs[:, :, np.newaxis])[:, :, 0]
    thecondition = np.linalg.cond(thematrix)

    gradx = np.zeros(tau.shape, dtype=np.float64)
    grady = np.zeros(tau.shape, dtype=np.float64)
    gradz = np.zeros(tau.shape, dtype=np.float64)
    fitvalid = np.zeros(tau.shape, dtype=np.uint16)

    gradx[validvoxels] = thesolution[:, 1]
    grady[validvoxels] = thesolution[:, 2]
    gradz[validvoxels] = thesolution[:, 3]
    fitvalid[validvoxels] = np.where(np.isfinite(thecondition) & (thecondition < 1.0e8), 1, 0)

    return gradx, grady, gradz, fitvalid


def flowaccumulation(
    tau: NDArray,
    themask: NDArray,
    voxdims: Tuple[float, float, float],
    seedweight: NDArray,
    mfdexponent: float = DEFAULT_MFDEXPONENT,
    showprogressbar: bool = True,
) -> NDArray:
    """
    Compute flow accumulation by multiple flow direction routing on the delay map.

    This is straight out of terrain analysis, with tau playing the role of
    elevation and blood playing the role of water.  Each voxel hands its
    accumulated load to all of its downstream (later arriving) neighbors,
    partitioned in proportion to slope**mfdexponent.  Multiple flow directions,
    rather than a single steepest descent direction, is the right choice here
    because vascular trees bifurcate.

    Because we process voxels in order of increasing tau, every upstream
    contribution has already arrived by the time a voxel is handled, so a single
    pass suffices.  That ordering is a valid topological sort only if tau has no
    interior local maxima, which is what the depression filling guarantees.

    Seeding each voxel with its blood volume rather than with 1.0 means the result
    approximates total upstream blood volume rather than a voxel count, which is a
    considerably better trunk vessel detector.

    Parameters
    ----------
    tau : NDArray
        The 3D (depression filled) arrival time field, in seconds.
    themask : NDArray
        The 3D mask of valid voxels.
    voxdims : tuple of float
        The voxel dimensions in mm.
    seedweight : NDArray
        The per voxel starting load, normally the relative CBV.
    mfdexponent : float, optional
        The slope exponent for flow partitioning.
    showprogressbar : bool, optional
        Show a progress bar.

    Returns
    -------
    NDArray
        The flow accumulation map.
    """
    paddedmask = np.pad(themask > 0, 1, mode="constant", constant_values=False).reshape(-1)
    paddedtau = np.pad(tau, 1, mode="constant", constant_values=0.0).reshape(-1)
    paddedaccum = np.pad(seedweight, 1, mode="constant", constant_values=0.0).reshape(-1)
    paddedshape = tuple(np.array(themask.shape) + 2)
    offsets, distances = neighboroffsets(paddedshape, voxdims)

    validindices = np.nonzero(paddedmask)[0]
    processorder = validindices[np.argsort(paddedtau[validindices])]

    for thisindex in tqdm(
        processorder,
        desc="Routing",
        unit="voxels",
        disable=(not showprogressbar),
    ):
        neighborindices = thisindex + offsets
        thedrop = paddedtau[neighborindices] - paddedtau[thisindex]
        isdownstream = paddedmask[neighborindices] & (thedrop > 0.0)
        if not np.any(isdownstream):
            continue
        theslopes = np.where(isdownstream, thedrop / distances, 0.0)
        theweights = theslopes**mfdexponent
        theweights /= np.sum(theweights)
        paddedaccum[neighborindices] += paddedaccum[thisindex] * theweights

    return paddedaccum.reshape(paddedshape)[1:-1, 1:-1, 1:-1]


def labelterritories(
    tau: NDArray,
    themask: NDArray,
    voxdims: Tuple[float, float, float],
    minterritorysize: int = DEFAULT_MINTERRITORYSIZE,
    showprogressbar: bool = True,
) -> Tuple[NDArray, Dict[int, int]]:
    """
    Label vascular territories by steepest ascent to a common source.

    Each voxel is assigned to the same territory as its steepest UPSTREAM neighbor
    (the masked neighbor with the lowest tau).  Voxels with no lower neighbor are
    sources, and start new territories.  Processing in order of increasing tau
    guarantees that any lower neighbor has already been labeled, so again one pass
    is enough.

    This is the classic watershed by steepest descent, and it inherits the classic
    watershed failure mode: every noise induced local minimum of tau becomes a
    spurious source.  Depression filling upstream of this call is what keeps the
    output from shattering.

    Parameters
    ----------
    tau : NDArray
        The 3D (depression filled) arrival time field, in seconds.
    themask : NDArray
        The 3D mask of valid voxels.
    voxdims : tuple of float
        The voxel dimensions in mm.
    minterritorysize : int, optional
        Territories with fewer voxels than this are set to zero.
    showprogressbar : bool, optional
        Show a progress bar.

    Returns
    -------
    thelabels : NDArray
        The integer territory label map.
    thesizes : dict
        A mapping from surviving label to voxel count.
    """
    paddedmask = np.pad(themask > 0, 1, mode="constant", constant_values=False).reshape(-1)
    paddedtau = np.pad(tau, 1, mode="constant", constant_values=0.0).reshape(-1)
    paddedshape = tuple(np.array(themask.shape) + 2)
    offsets, distances = neighboroffsets(paddedshape, voxdims)

    paddedlabels = np.zeros(paddedmask.shape, dtype=np.int32)
    validindices = np.nonzero(paddedmask)[0]
    processorder = validindices[np.argsort(paddedtau[validindices])]

    nextlabel = 1
    for thisindex in tqdm(
        processorder,
        desc="Labeling",
        unit="voxels",
        disable=(not showprogressbar),
    ):
        neighborindices = thisindex + offsets
        therise = paddedtau[thisindex] - paddedtau[neighborindices]
        isupstream = paddedmask[neighborindices] & (therise > 0.0)
        if np.any(isupstream):
            theslopes = np.where(isupstream, therise / distances, -np.inf)
            paddedlabels[thisindex] = paddedlabels[neighborindices[np.argmax(theslopes)]]
        else:
            paddedlabels[thisindex] = nextlabel
            nextlabel += 1

    # drop the runts
    theuniquelabels, thecounts = np.unique(paddedlabels[validindices], return_counts=True)
    thesizes = {}
    for thelabel, thecount in zip(theuniquelabels, thecounts):
        if thelabel > 0:
            if thecount < minterritorysize:
                paddedlabels[paddedlabels == thelabel] = 0
            else:
                thesizes[int(thelabel)] = int(thecount)

    return paddedlabels.reshape(paddedshape)[1:-1, 1:-1, 1:-1], thesizes


def tracestreamlines(
    direction: NDArray,
    themask: NDArray,
    voxdims: Tuple[float, float, float],
    seedstride: int = DEFAULT_SEEDSTRIDE,
    stepsize: float = DEFAULT_STEPSIZE,
    maxsteps: int = DEFAULT_MAXSTEPS,
    minlength: float = DEFAULT_MINLENGTH,
    showprogressbar: bool = True,
) -> List[NDArray]:
    """
    Trace streamlines through the flow direction field by RK4 integration.

    Each seed is tracked both downstream and forward from upstream, and the two
    halves are spliced, so a seed placed in the middle of a vessel recovers the
    whole path rather than half of it.  Tracking terminates on leaving the mask,
    on the direction field going undefined, or on a sharp reversal (which is what
    a residual local extremum looks like from the inside).

    Integration is done in voxel index space with steps scaled by the voxel
    dimensions, so anisotropic voxels are handled correctly.

    Parameters
    ----------
    direction : NDArray
        A 4D array of unit flow direction vectors, in mm space.
    themask : NDArray
        The 3D mask of valid voxels.
    voxdims : tuple of float
        The voxel dimensions in mm.
    seedstride : int, optional
        Seed from every Nth voxel in each dimension.
    stepsize : float, optional
        Integration step size in mm.
    maxsteps : int, optional
        Maximum steps per streamline half.
    minlength : float, optional
        Discard streamlines shorter than this, in mm.
    showprogressbar : bool, optional
        Show a progress bar.

    Returns
    -------
    list of NDArray
        The streamlines, as arrays of voxel coordinates.
    """
    thevoxdims = np.array(voxdims, dtype=np.float64)
    maskeddata = (themask > 0).astype(np.float64)

    def interpdirection(thepoint: NDArray) -> Optional[NDArray]:
        """
        Trilinearly interpolate the unit flow direction at an arbitrary point.

        Parameters
        ----------
        thepoint : NDArray
            The point to evaluate, in voxel index space.

        Returns
        -------
        NDArray or None
            The unit direction vector, or None if the point has left the mask or the
            direction field is degenerate there.  A None return is the signal to stop
            integrating.
        """
        thecoords = thepoint.reshape(3, 1)
        if map_coordinates(maskeddata, thecoords, order=1, mode="constant", cval=0.0)[0] < 0.5:
            return None
        thevector = np.array(
            [
                map_coordinates(direction[:, :, :, i], thecoords, order=1, mode="nearest")[0]
                for i in range(3)
            ]
        )
        thenorm = np.linalg.norm(thevector)
        if thenorm < 1.0e-6:
            return None
        return thevector / thenorm

    def integrate(startpoint: NDArray, thesign: float) -> List[NDArray]:
        """
        Integrate one half of a streamline away from a seed point by RK4.

        Parameters
        ----------
        startpoint : NDArray
            The seed point, in voxel index space.
        thesign : float
            The travel direction: 1.0 to follow the flow downstream, -1.0 to walk
            back upstream against it.

        Returns
        -------
        list of NDArray
            The points of this half of the path, in voxel index space, beginning with
            startpoint.  Terminates on leaving the mask, on the direction field going
            undefined, on a sharp reversal, or after maxsteps steps.
        """
        thepath = [startpoint.copy()]
        thepoint = startpoint.copy()
        lastvector = None
        for _ in range(maxsteps):
            # RK4 in voxel index space, with the step scaled into mm
            k1 = interpdirection(thepoint)
            if k1 is None:
                break
            thestep = thesign * stepsize / thevoxdims
            k2 = interpdirection(thepoint + 0.5 * thestep * k1)
            if k2 is None:
                break
            k3 = interpdirection(thepoint + 0.5 * thestep * k2)
            if k3 is None:
                break
            k4 = interpdirection(thepoint + thestep * k3)
            if k4 is None:
                break
            thevector = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            thenorm = np.linalg.norm(thevector)
            if thenorm < 1.0e-6:
                break
            thevector /= thenorm
            # a sharp reversal means we have fallen into a residual local extremum
            if lastvector is not None and np.dot(thevector, lastvector) < 0.0:
                break
            lastvector = thevector
            thepoint = thepoint + thestep * thevector
            thepath.append(thepoint.copy())
        return thepath

    seedmask = np.zeros(themask.shape, dtype=bool)
    seedmask[::seedstride, ::seedstride, ::seedstride] = True
    theseeds = np.array(np.nonzero((themask > 0) & seedmask)).T

    thestreamlines = []
    for theseed in tqdm(
        theseeds,
        desc="Tracking",
        unit="seeds",
        disable=(not showprogressbar),
    ):
        startpoint = theseed.astype(np.float64)
        downstream = integrate(startpoint, 1.0)
        upstream = integrate(startpoint, -1.0)
        thepath = np.array(upstream[::-1] + downstream[1:])
        if len(thepath) < 2:
            continue
        thelength = np.sum(np.linalg.norm(np.diff(thepath, axis=0) * thevoxdims, axis=1))
        if thelength >= minlength:
            thestreamlines.append(thepath)

    return thestreamlines


def maskextensionindices(themask: NDArray) -> Tuple[NDArray, ...]:
    """
    Build an index array that maps every voxel to its nearest in mask voxel.

    Anything that interpolates or differentiates near the edge of a masked volume
    needs an explicit story about what lies outside, because the default answer -
    zero - is always wrong and never raises an error.  Indexing a volume with the
    result of this function extends it outside the mask by nearest in mask value,
    which is the right story for both interpolation and differentiation.

    Parameters
    ----------
    themask : NDArray
        The 3D mask of valid voxels.

    Returns
    -------
    tuple of NDArray
        An index tuple suitable for fancy indexing a volume of the same shape.
    """
    return tuple(distance_transform_edt(themask == 0, return_indices=True, return_distances=False))


def samplealongstreamlines(
    thestreamlines: List[NDArray],
    thevolumes: Dict[str, NDArray],
    themask: Optional[NDArray] = None,
    nearestneighbor: Optional[List[str]] = None,
) -> Dict[str, List[NDArray]]:
    """
    Sample scalar volumes at every vertex of every streamline.

    Tractography viewers colour streamlines by local orientation by default, which
    is the right convention for diffusion tractography, where orientation really is
    all you have.  Here it throws away the one thing this method gives you that dMRI
    cannot - the SIGN of the direction - so an artery and the vein running
    antiparallel beside it render identically.  Attaching per vertex scalars lets
    you colour by arrival time instead, which makes propagation visible in a static
    rendering and disambiguates upstream from downstream.

    Pass themask to extend each volume to its nearest in mask value before sampling.
    This matters more than it sounds like it should.  Streamlines terminate AT the
    mask boundary, and the output volumes are zero outside the mask, so without the
    extension the trilinear interpolation blends real values with those zeros over
    the last vertex or two of every single streamline.  Colouring by arrival time
    would then paint the tip of each streamline as the EARLIEST arriving point in
    the dataset, at exactly the location where it is in fact the latest.

    Parameters
    ----------
    thestreamlines : list of NDArray
        The streamlines, in voxel coordinates.
    thevolumes : dict
        A mapping from scalar name to 3D volume.  Names are truncated to 20
        characters, which is the trk format limit.
    themask : NDArray, optional
        The analysis mask.  If supplied, each volume is extended outside the mask by
        nearest in mask value before sampling.
    nearestneighbor : list of str, optional
        Names that should be sampled with nearest neighbor rather than linear
        interpolation.  Use this for anything categorical - interpolating integer
        territory labels would manufacture meaningless intermediate values.

    Returns
    -------
    dict
        A mapping from scalar name to a list of (npoints, 1) arrays, one per
        streamline, suitable for handing to nibabel as data_per_point.
    """
    if nearestneighbor is None:
        nearestneighbor = []

    # one distance transform gives the nearest in mask voxel for the whole volume,
    # which we then reuse to extend every scalar
    theextensionindices = None
    if themask is not None:
        theextensionindices = maskextensionindices(themask)

    thedata: Dict[str, List[NDArray]] = {}
    for thename, thevolume in thevolumes.items():
        theorder = 0 if thename in nearestneighbor else 1
        if theextensionindices is not None:
            thevolume = thevolume[theextensionindices]
        thesamples = []
        for thestreamline in thestreamlines:
            thevalues = map_coordinates(thevolume, thestreamline.T, order=theorder, mode="nearest")
            thesamples.append(thevalues.astype(np.float32).reshape(-1, 1))
        thedata[thename[:20]] = thesamples
    return thedata


def savestreamlines(
    thestreamlines: List[NDArray],
    theaffine: NDArray,
    thedims: Tuple[int, int, int],
    voxdims: Tuple[float, float, float],
    thefilename: str,
    datapervertex: Optional[Dict[str, List[NDArray]]] = None,
    debug: bool = False,
) -> bool:
    """
    Write streamlines out as a trk file for viewing.

    Parameters
    ----------
    thestreamlines : list of NDArray
        The streamlines, in voxel coordinates.
    theaffine : NDArray
        The voxel to RAS mm affine of the input image.
    thedims : tuple of int
        The volume dimensions.
    voxdims : tuple of float
        The voxel dimensions in mm.
    thefilename : str
        The output filename.
    datapervertex : dict, optional
        Per vertex scalars, as returned by samplealongstreamlines.  These become
        trk scalars, and are what viewers use to colour the streamlines by
        something more informative than local orientation.
    debug : bool, optional
        Enable debug output.

    Returns
    -------
    bool
        True if the file was written.
    """
    try:
        import nibabel as nib
        from nibabel.streamlines import Field, Tractogram

        theheader = {
            Field.VOXEL_TO_RASMM: theaffine.copy(),
            Field.VOXEL_SIZES: tuple(voxdims),
            Field.DIMENSIONS: tuple(int(thedim) for thedim in thedims),
            Field.VOXEL_ORDER: "".join(nib.aff2axcodes(theaffine)),
        }
        thetractogram = Tractogram(
            thestreamlines,
            data_per_point=(datapervertex if datapervertex else None),
            affine_to_rasmm=theaffine,
        )
        nib.streamlines.save(thetractogram, thefilename, header=theheader)
        return True
    except Exception as thexception:
        print(f"WARNING: could not write streamlines to {thefilename}: {thexception}")
        if debug:
            raise
        return False


def delayflow(args: argparse.Namespace) -> None:
    """
    Infer a blood flow map from a rapidtide delay map.

    Reads the delay map and its companion CBV proxy and mask, estimates the velocity
    field from the eikonal relation, then routes, labels, and tracks it, writing all
    of the maps described in the module docstring to args.outputroot.

    Note that this mutates args in place: cbvfile, maskfile, and gausssigma are
    filled in with the values actually used when they were left unset on the command
    line, so that they get recorded in the run options json.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command line arguments, as produced by _get_parser().

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the delay map is not 3D, if the CBV proxy or mask dimensions do not match
        the delay map, or if fewer than 100 voxels survive masking.
    """
    starttime = time.time()
    therunoptions: Dict[str, Any] = {}
    for thekey, thevalue in vars(args).items():
        if isinstance(thevalue, (str, int, float, bool, type(None))):
            therunoptions[thekey] = thevalue

    # figure out what the delay map is called, so we can guess the companion files
    thedelayroot = args.delayfile
    for theending in ["_desc-maxtime_map.nii.gz", "_desc-maxtimerefined_map.nii.gz"]:
        if thedelayroot.endswith(theending):
            thedelayroot = thedelayroot[: -len(theending)]
            break
    else:
        thedelayroot, _ = tide_io.niftisplitext(thedelayroot)

    ##########################################################################
    # read the delay map
    ##########################################################################
    print(f"reading delay map from {args.delayfile}")
    (
        delay_img,
        delay_data,
        delay_hdr,
        delay_dims,
        delay_sizes,
    ) = tide_io.readfromnifti(args.delayfile)
    xsize, ysize, numslices, timepoints = tide_io.parseniftidims(delay_dims)
    xdim, ydim, slicethickness, tr = tide_io.parseniftisizes(delay_sizes)
    voxdims = (xdim, ydim, slicethickness)
    if timepoints > 1:
        raise ValueError("the delay map must be a 3D file")
    tau = np.nan_to_num(delay_data.reshape((xsize, ysize, numslices))).astype(np.float64)
    print(f"delay map is {xsize}x{ysize}x{numslices}, voxel size {xdim}x{ydim}x{slicethickness}")

    ##########################################################################
    # read the CBV proxy
    ##########################################################################
    if args.cbvfile is None:
        thecandidate = f"{thedelayroot}_desc-maxcorrsq_map.nii.gz"
        if os.path.isfile(thecandidate):
            args.cbvfile = thecandidate
            print(f"using {thecandidate} as the CBV proxy")
    if args.cbvfile is not None:
        print(f"reading CBV proxy from {args.cbvfile}")
        (
            cbv_img,
            cbv_data,
            cbv_hdr,
            cbv_dims,
            cbv_sizes,
        ) = tide_io.readfromnifti(args.cbvfile)
        if not tide_io.checkspacematch(delay_hdr, cbv_hdr):
            raise ValueError("CBV proxy dimensions do not match delay map dimensions")
        cbv = np.nan_to_num(cbv_data.reshape((xsize, ysize, numslices))).astype(np.float64)
        cbv = np.clip(cbv, 0.0, None)
    else:
        print("no CBV proxy found - using uniform weighting")
        cbv = np.ones((xsize, ysize, numslices), dtype=np.float64)
    therunoptions["cbvfile"] = args.cbvfile

    ##########################################################################
    # read or construct the mask
    ##########################################################################
    if args.maskfile is None:
        thecandidate = f"{thedelayroot}_desc-corrfit_mask.nii.gz"
        if os.path.isfile(thecandidate):
            args.maskfile = thecandidate
            print(f"using {thecandidate} as the mask")
    if args.maskfile is not None:
        print(f"reading mask from {args.maskfile}")
        (
            mask_img,
            mask_data,
            mask_hdr,
            mask_dims,
            mask_sizes,
        ) = tide_io.readfromnifti(args.maskfile)
        if not tide_io.checkspacematch(delay_hdr, mask_hdr):
            raise ValueError("mask dimensions do not match delay map dimensions")
        themask = np.uint16(mask_data.reshape((xsize, ysize, numslices)) > 0)
    else:
        print("no mask found - masking on nonzero delay values")
        themask = np.uint16((tau != 0.0) & np.isfinite(tau))
    therunoptions["maskfile"] = args.maskfile

    # normalize the CBV proxy to its 98th percentile inside the mask, then use it to
    # trim voxels whose delay estimate is too poorly determined to be worth anything
    if np.any(themask > 0):
        thenorm = np.percentile(cbv[themask > 0], 98.0)
        if thenorm > 0.0:
            cbv /= thenorm
    cbv = np.clip(cbv, 0.0, None)
    themask = np.uint16((themask > 0) & (cbv >= args.cbvthresh))
    numvalid = int(np.sum(themask))
    print(f"{numvalid} voxels in the analysis mask")
    if numvalid < 100:
        raise ValueError("fewer than 100 valid voxels - check your mask and CBV threshold")

    output_hdr = makeoutputheader(delay_hdr, numvols=1)
    tide_io.savetonifti(
        themask, output_hdr, f"{args.outputroot}_desc-flowfit_mask", debug=args.debug
    )

    ##########################################################################
    # spatially smooth the delay map
    ##########################################################################
    if args.gausssigma < 0.0:
        args.gausssigma = np.mean([xdim, ydim, slicethickness]) / 2.0
        therunoptions["gausssigma"] = args.gausssigma
    if args.gausssigma > 0.0:
        print(f"applying gaussian spatial filter with sigma={args.gausssigma}")
        # smooth in a mask aware way (normalized convolution), so that the zeros
        # outside the brain do not drag the edge values down and manufacture a
        # gigantic spurious gradient at the brain boundary
        maskeddata = (themask > 0).astype(np.float64)
        thenumerator = tide_filt.ssmooth(
            xdim, ydim, slicethickness, args.gausssigma, tau * maskeddata
        )
        thedenominator = tide_filt.ssmooth(xdim, ydim, slicethickness, args.gausssigma, maskeddata)
        safedenominator = np.where(thedenominator > 1.0e-6, thedenominator, 1.0)
        smoothedtau = np.where(thedenominator > 1.0e-6, thenumerator / safedenominator, tau)
        smoothedtau = np.where(themask > 0, smoothedtau, 0.0)
    else:
        smoothedtau = tau * (themask > 0)

    tide_io.savetonifti(
        smoothedtau.astype(np.float32),
        output_hdr,
        f"{args.outputroot}_desc-tausmoothed_map",
        debug=args.debug,
    )

    ##########################################################################
    # estimate the gradient and turn it into a velocity
    ##########################################################################
    print(f"estimating grad(tau) by local plane fitting, radius={args.fitradius} mm")
    gradx, grady, gradz, fitvalid = estimategradient(
        smoothedtau, cbv, themask, voxdims, args.fitradius
    )
    gradmag = np.sqrt(gradx**2 + grady**2 + gradz**2)

    # Anything below this gradient is indistinguishable from zero given the noise on
    # the delay estimate, so the corresponding speed is a lower bound, not a
    # measurement.  In big vessels essentially everything lands here.
    mingradient = args.delaynoise / args.fitradius
    speedceiling = 1.0 / mingradient
    print(f"minimum resolvable |grad(tau)| is {mingradient:.5f} s/mm")
    print(f"speeds above {speedceiling:.2f} mm/s are unresolved and will be capped")

    speedvalid = np.uint16((themask > 0) & (fitvalid > 0) & (gradmag > mingradient))
    numresolved = int(np.sum(speedvalid))
    print(
        f"{numresolved} of {numvalid} voxels ({100.0 * numresolved / numvalid:.1f}%) "
        f"have a resolvable speed"
    )

    # v = grad(tau) / |grad(tau)|**2.  Note the SQUARED denominator - direction from
    # the gradient, speed from its reciprocal.
    safegradmag = np.where(gradmag > mingradient, gradmag, mingradient)
    speed = np.where(themask > 0, 1.0 / safegradmag, 0.0)
    velocity = np.zeros((xsize, ysize, numslices, 3), dtype=np.float64)
    direction = np.zeros((xsize, ysize, numslices, 3), dtype=np.float64)
    for i, thecomponent in enumerate([gradx, grady, gradz]):
        direction[:, :, :, i] = np.where(gradmag > 0.0, thecomponent / safegradmag, 0.0)
        velocity[:, :, :, i] = direction[:, :, :, i] * speed

    vector_hdr = makeoutputheader(delay_hdr, numvols=3)
    tide_io.savetonifti(
        gradmag.astype(np.float32),
        output_hdr,
        f"{args.outputroot}_desc-gradmag_map",
        debug=args.debug,
    )
    tide_io.savetonifti(
        speed.astype(np.float32),
        output_hdr,
        f"{args.outputroot}_desc-speed_map",
        debug=args.debug,
    )
    tide_io.savetonifti(
        speedvalid, output_hdr, f"{args.outputroot}_desc-speedresolved_mask", debug=args.debug
    )
    tide_io.savetonifti(
        velocity.astype(np.float32),
        vector_hdr,
        f"{args.outputroot}_desc-velocity_map",
        debug=args.debug,
    )
    tide_io.savetonifti(
        direction.astype(np.float32),
        vector_hdr,
        f"{args.outputroot}_desc-direction_map",
        debug=args.debug,
    )

    ##########################################################################
    # divergence of flux diagnostic
    ##########################################################################
    if args.dodivergence:
        print("computing div(cbv * v) as a conservation diagnostic")
        # Blood is conserved, so div(cbv * v) should be near zero except at true
        # inlets and outlets.  We are not imposing this - just reporting where the
        # eikonal solution violates it, which is where the tangential velocity we
        # cannot see is largest.
        flux = velocity * cbv[:, :, :, np.newaxis]
        divergence = np.zeros((xsize, ysize, numslices), dtype=np.float64)
        for i in range(3):
            thegradients = estimategradient(
                flux[:, :, :, i], cbv, themask, voxdims, args.fitradius
            )
            divergence += thegradients[i]
        tide_io.savetonifti(
            divergence.astype(np.float32),
            output_hdr,
            f"{args.outputroot}_desc-fluxdivergence_map",
            debug=args.debug,
        )

    ##########################################################################
    # depression filling
    ##########################################################################
    if args.dofill:
        # Fill pits of -tau to kill spurious local MAXIMA (dead end sinks that break
        # downstream accumulation)...
        print("filling depressions for downstream routing")
        taudown = -priorityflood(-smoothedtau, themask, showprogressbar=args.showprogressbar)
        # ...and pits of +tau to kill spurious local MINIMA (fake sources that
        # shatter the territory map).  Same algorithm, opposite sign.
        print("filling depressions for upstream labeling")
        tauup = priorityflood(smoothedtau, themask, showprogressbar=args.showprogressbar)
        tide_io.savetonifti(
            np.where(themask > 0, taudown, 0.0).astype(np.float32),
            output_hdr,
            f"{args.outputroot}_desc-taufilleddown_map",
            debug=args.debug,
        )
        tide_io.savetonifti(
            np.where(themask > 0, tauup, 0.0).astype(np.float32),
            output_hdr,
            f"{args.outputroot}_desc-taufilledup_map",
            debug=args.debug,
        )
    else:
        print("skipping depression filling")
        taudown = smoothedtau
        tauup = smoothedtau

    ##########################################################################
    # flow accumulation
    ##########################################################################
    print("computing flow accumulation")
    accumulation = flowaccumulation(
        taudown,
        themask,
        voxdims,
        cbv * (themask > 0),
        mfdexponent=args.mfdexponent,
        showprogressbar=args.showprogressbar,
    )
    accumulation = np.where(themask > 0, accumulation, 0.0)
    logaccumulation = np.where(accumulation > 0.0, np.log10(accumulation + 1.0), 0.0)
    tide_io.savetonifti(
        accumulation.astype(np.float32),
        output_hdr,
        f"{args.outputroot}_desc-flowaccum_map",
        debug=args.debug,
    )
    tide_io.savetonifti(
        logaccumulation.astype(np.float32),
        output_hdr,
        f"{args.outputroot}_desc-logflowaccum_map",
        debug=args.debug,
    )

    ##########################################################################
    # vascular territories
    ##########################################################################
    print("labeling vascular territories")
    territories, thesizes = labelterritories(
        tauup,
        themask,
        voxdims,
        minterritorysize=args.minterritorysize,
        showprogressbar=args.showprogressbar,
    )
    print(f"found {len(thesizes)} territories with at least {args.minterritorysize} voxels")
    tide_io.savetonifti(
        territories.astype(np.int32),
        output_hdr,
        f"{args.outputroot}_desc-territories_map",
        debug=args.debug,
    )
    thesortedsizes = sorted(thesizes.items(), key=lambda thepair: thepair[1], reverse=True)
    with open(f"{args.outputroot}_desc-territorysizes_info.tsv", "w") as thefile:
        thefile.write("label\tnumvoxels\n")
        for thelabel, thesize in thesortedsizes:
            thefile.write(f"{thelabel}\t{thesize}\n")

    ##########################################################################
    # streamlines
    ##########################################################################
    if args.dostreamlines:
        print("tracking streamlines")
        thestreamlines = tracestreamlines(
            direction,
            themask,
            voxdims,
            seedstride=args.seedstride,
            stepsize=args.stepsize,
            maxsteps=args.maxsteps,
            minlength=args.minlength,
            showprogressbar=args.showprogressbar,
        )
        print(f"kept {len(thestreamlines)} streamlines")
        if len(thestreamlines) > 0:
            # Attach per vertex scalars so that viewers can colour by something more
            # useful than local orientation.  Colouring by arrivaltime is the one to
            # reach for first: it renders the propagation of the wavefront directly,
            # and unlike orientation colouring it distinguishes a vessel from its
            # antiparallel neighbor.
            print("sampling scalars along streamlines")
            datapervertex = samplealongstreamlines(
                thestreamlines,
                {
                    "arrivaltime": smoothedtau,
                    "speed": speed,
                    "logflowaccum": logaccumulation,
                    "gradmag": gradmag,
                    "territory": territories.astype(np.float64),
                },
                themask=themask,
                nearestneighbor=["territory"],
            )
            savestreamlines(
                thestreamlines,
                delay_img.affine,
                (xsize, ysize, numslices),
                voxdims,
                f"{args.outputroot}_desc-flow_streamlines.trk",
                datapervertex=datapervertex,
                debug=args.debug,
            )
            # a density map is a useful sanity check even if the trk will not load
            density = np.zeros((xsize, ysize, numslices), dtype=np.float64)
            for thestreamline in thestreamlines:
                theindices = np.round(thestreamline).astype(int)
                theindices = theindices[
                    np.all(theindices >= 0, axis=1)
                    & (theindices[:, 0] < xsize)
                    & (theindices[:, 1] < ysize)
                    & (theindices[:, 2] < numslices)
                ]
                np.add.at(density, (theindices[:, 0], theindices[:, 1], theindices[:, 2]), 1.0)
            tide_io.savetonifti(
                density.astype(np.float32),
                output_hdr,
                f"{args.outputroot}_desc-streamlinedensity_map",
                debug=args.debug,
            )

    ##########################################################################
    # wrap up
    ##########################################################################
    therunoptions["numvalidvoxels"] = numvalid
    therunoptions["numresolvedvoxels"] = numresolved
    therunoptions["mingradient"] = float(mingradient)
    therunoptions["speedceiling"] = float(speedceiling)
    therunoptions["numterritories"] = len(thesizes)
    therunoptions["runtime"] = time.time() - starttime
    tide_io.writedicttojson(therunoptions, f"{args.outputroot}_desc-runoptions_info.json")
    print(f"done in {time.time() - starttime:.1f} seconds")


def main() -> None:
    """
    Command line entry point for delayflow.

    Parses the command line and runs the workflow, printing the help text before
    reraising if parsing fails.

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        If argument parsing fails, or if --help was requested.
    """
    try:
        args = _get_parser().parse_args()
    except SystemExit:
        _get_parser().print_help()
        raise
    delayflow(args)


if __name__ == "__main__":
    main()
