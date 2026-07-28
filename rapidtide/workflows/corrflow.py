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
Infer a blood flow map directly from the 4D similarity function movie.

Why not just use the delay map
------------------------------
delayflow starts from the delay map tau(x) and solves the eikonal relation
grad(tau).v = 1.  That requires committing to a single peak location per voxel,
and peak finding is the fragile step on noisy data - a correlation function with
a marginal or split peak produces a delay estimate whose error is not small and
Gaussian but categorical, and differentiating a field of those is unpleasant.

This program goes back one step and works on the similarity function movie
itself, before any peak has been picked.

The two are solving the same equation
-------------------------------------
If the correlation movie is I(x, l) = f(l - tau(x)) for some waveform f, where l
is the lag axis, then::

    dI/dl  = f'(l - tau)
    grad(I) = -f'(l - tau) grad(tau)

so the optical flow brightness constancy constraint becomes::

    dI/dl + v . grad(I) = f' (1 - v . grad(tau)) = 0    ==>    v . grad(tau) = 1

which is exactly the eikonal relation that delayflow solves.  The difference is
entirely in the estimation: optical flow accumulates the constraint over the
whole waveform and over every lag, so it never forms tau and never inherits a
peak fitting failure.

What it does NOT fix: the constraint is identical, so the aperture problem is
identical.  Velocity tangent to the isochrone surfaces is invisible here too.
Anyone expecting optical flow to recover the full 3D velocity will be
disappointed for exactly the same reason as before.

Method
------
Three dimensional Lucas-Kanade.  At each voxel we solve, over a spatial
neighborhood and over all lags at once::

    min_v  sum w (dI/dl + v . grad(I))**2

whose normal equations are the structure tensor system::

    [ sum w Ix Ix   sum w Ix Iy   sum w Ix Iz ] [vx]         [ sum w Ix Il ]
    [ sum w Ix Iy   sum w Iy Iy   sum w Iy Iz ] [vy]  =  -   [ sum w Iy Il ]
    [ sum w Ix Iz   sum w Iy Iz   sum w Iz Iz ] [vz]         [ sum w Iz Il ]

Accumulating over lag rather than solving per lag frame and averaging is the
right thing to do here: the vasculature does not move, so there is one static
velocity field, and every lag is independent evidence about it.

The eigenvalues of the structure tensor are the natural confidence measure, and
they diagnose the aperture problem directly.  The system is solved through its
eigendecomposition, keeping only eigenvalues above a threshold relative to the
largest - that is, by pseudoinverse rather than by direct inversion.

Do not be tempted to lower that threshold.  For a coherent travelling wavefront
the tensor is genuinely rank ONE pointwise: I = f(l - tau) gives grad(I) =
-f' grad(tau), so the gradient is everywhere parallel to grad(tau) and the data
constrain the velocity along exactly one direction.  That is a total aperture
problem, and it does not matter, because the velocity we are entitled to
recover - the minimum norm eikonal solution - lies along grad(tau) too.  Keeping
only the dominant eigenvector IS the correct normal flow answer.  Admitting the
near null eigenvectors does not add information, it just divides noise by a
small number and sprays it into the tangential directions.  Empirically, on a
noiseless synthetic radial wave, a threshold of 1e-4 gives a median direction
error of 13 degrees, while 0.1 gives 1.6 degrees.

So the rank map should be read as a description of the local geometry, not as a
quality score.  Rank 1 is the expected, healthy regime for a clean propagating
wavefront.  Rank 0 means no usable information.  Ranks 2 and 3 mean the
neighborhood contains structure in more than one direction, which is real at
bifurcations and confluences, but is also what noise looks like.

Kernel balance, and why the speed comes out low
-----------------------------------------------
The velocity is a ratio of two estimated derivatives, v = -I_l / I_x, and each is
computed with its own smoothing kernel.  Gaussian smoothing attenuates a
frequency component by exp(-sigma**2 omega**2 / 2), so if the two kernels have
different effective widths the ratio - and therefore the speed - is biased.

For a wavefront travelling at speed c, a spatial kernel of derivsigma voxels is
equivalent to a temporal kernel of derivsigma * voxdim / c seconds.  The two are
balanced when::

    lagsmooth * lagstep  ==  derivsigma * voxdim / c

Below balance the temporal kernel is too wide, I_l is over attenuated, and the
speed reads LOW.  This is measurable and quantitatively predictive: on a
synthetic wave at 20 mm/s with 2 mm voxels and a 0.4 s lag step, the balance
condition calls for derivsigma = 4, and the recovered/true speed ratio goes
0.76 at derivsigma 1, 0.83 at 2, 1.08 at 4, and overshoots to 1.46 at 6.

Two ways to restore balance, and they are NOT equivalent:

1) Widen the spatial kernel (--derivsigma).  Effective, but it blurs the flow
   field, and the correct value depends on the local speed, so a single global
   value cannot be right everywhere.
2) Narrow the temporal kernel by interpolating the lag axis (--lagoversamp).
   Cheap and exact, but it SATURATES - on the same test it goes 0.76, 0.85,
   0.87, 0.88 for factors 1, 2, 4, 8, and never reaches 1.

So interpolation alone will not fix a badly unbalanced fit.  The proper solution
is to balance adaptively, using a first pass speed estimate to set derivsigma per
voxel and iterating; that is not implemented yet.

Reused machinery
----------------
Everything downstream of the velocity field is shared with delayflow -
streamline tracking, per vertex scalar sampling, the trk writer, and the
divergence of flux diagnostic.  The routing, depression filling and territory
labeling steps need a scalar field, which optical flow does not produce, so a
peak lag map is derived from the movie (or read from a file) purely to provide
the ordering they require.  Note that this peak map is used only for topology,
never for velocity.

This is a proof of concept.
"""

import argparse
import copy
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.ndimage import correlate1d, gaussian_filter1d
from tqdm import tqdm

import rapidtide.io as tide_io
import rapidtide.workflows.parser_funcs as pf
from rapidtide.workflows.delayflow import (
    estimategradient,
    flowaccumulation,
    labelterritories,
    makeoutputheader,
    maskextensionindices,
    priorityflood,
    samplealongstreamlines,
    savestreamlines,
    tracestreamlines,
)

DEFAULT_FITRADIUS = 6.0
DEFAULT_DERIVSIGMA = 1.0
DEFAULT_LAGSMOOTH = 1.0
DEFAULT_LAGOVERSAMP = 1
DEFAULT_CBVTHRESH = 0.05
DEFAULT_MINEIGENVALUE = 0.1
DEFAULT_SOLVER = "ols"
DEFAULT_MFDEXPONENT = 1.1
DEFAULT_MAXSPEED = 200.0
DEFAULT_STEPSIZE = 0.5
DEFAULT_MAXSTEPS = 2000
DEFAULT_MINLENGTH = 5.0
DEFAULT_SEEDSTRIDE = 2
DEFAULT_MINTERRITORYSIZE = 20


def _get_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for the corrflow command line tool.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser with all required and optional arguments.
    """
    parser = argparse.ArgumentParser(
        prog="corrflow",
        description=(
            "Infer a blood flow map from the 4D similarity function movie by 3D "
            "Lucas-Kanade optical flow, without ever picking a correlation peak."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "corrfile",
        type=lambda x: pf.is_valid_file(parser, x),
        help=(
            "The name of the rapidtide similarity function file, normally "
            "XXX_desc-corrout_info.nii.gz.  The fourth dimension is the lag axis; "
            "its step and origin are read from the header pixdim and toffset."
        ),
    )
    parser.add_argument("outputroot", type=str, help="The root name of the output files.")

    parser.add_argument(
        "--maskfile",
        dest="maskfile",
        type=lambda x: pf.is_valid_file(parser, x),
        metavar="FILE",
        help=(
            "Restrict the analysis to this mask.  Defaults to "
            "XXX_desc-corrfit_mask.nii.gz alongside the input if it exists, otherwise "
            "to voxels where the correlation function is not identically zero."
        ),
        default=None,
    )
    parser.add_argument(
        "--delayfile",
        dest="delayfile",
        type=lambda x: pf.is_valid_file(parser, x),
        metavar="FILE",
        help=(
            "A delay map to use for the routing, depression filling and territory "
            "labeling steps, which need a scalar field that optical flow does not "
            "produce.  Defaults to XXX_desc-maxtime_map.nii.gz if present, otherwise "
            "a peak lag map is derived from the movie by parabolic refinement of the "
            "argmax.  This map is used only to establish an ordering - it never "
            "enters the velocity estimate."
        ),
        default=None,
    )
    parser.add_argument(
        "--cbvthresh",
        dest="cbvthresh",
        type=float,
        metavar="VALUE",
        help=(
            f"Exclude voxels whose CBV proxy is below this value.  The CBV proxy here "
            f"is the peak to peak amplitude of the correlation function, normalized to "
            f"its 98th percentile.  Default is {DEFAULT_CBVTHRESH}."
        ),
        default=DEFAULT_CBVTHRESH,
    )
    parser.add_argument(
        "--fitradius",
        dest="fitradius",
        type=float,
        metavar="RADIUS",
        help=(
            f"The radius, in mm, of the spatial neighborhood over which the "
            f"Lucas-Kanade structure tensor is accumulated.  Larger values are more "
            f"robust but blur the flow field.  Default is {DEFAULT_FITRADIUS}."
        ),
        default=DEFAULT_FITRADIUS,
    )
    parser.add_argument(
        "--derivsigma",
        dest="derivsigma",
        type=float,
        metavar="SIGMA",
        help=(
            f"The sigma, in voxels, of the gaussian derivative kernel used for the "
            f"spatial derivatives of the movie.  Default is {DEFAULT_DERIVSIGMA}."
        ),
        default=DEFAULT_DERIVSIGMA,
    )
    parser.add_argument(
        "--lagsmooth",
        dest="lagsmooth",
        type=float,
        metavar="SIGMA",
        help=(
            f"The sigma, in lag samples, of the gaussian derivative kernel used along "
            f"the lag axis.  Default is {DEFAULT_LAGSMOOTH}."
        ),
        default=DEFAULT_LAGSMOOTH,
    )
    parser.add_argument(
        "--lagoversamp",
        dest="lagoversamp",
        type=int,
        metavar="N",
        help=(
            f"Interpolate the similarity function onto a lag axis N times finer "
            f"before estimating the flow.  This is exact rather than approximate - "
            f"the correlation of bandlimited data is bandlimited, so rapidtide "
            f"already samples the lag axis far above its Nyquist rate.  It lets the "
            f"temporal derivative kernel be narrowed to match the spatial one, which "
            f"is what removes the speed bias; see --derivsigma.  Costs memory "
            f"proportional to N.  The default is {DEFAULT_LAGOVERSAMP} (off) because "
            f"imbalance in EITHER direction biases the speed - oversampling without "
            f"also reducing --derivsigma simply moves the imbalance to the other side "
            f"and biases the speed high - so there is no globally safe factor.  Set it "
            f"together with --derivsigma using the balance condition "
            f"lagsmooth*lagstep == derivsigma*voxdim/speed."
        ),
        default=DEFAULT_LAGOVERSAMP,
    )
    parser.add_argument(
        "--noweightbycorr",
        dest="weightbycorr",
        action="store_false",
        help=(
            "Do not weight each lag sample by its correlation value.  By default lags "
            "are weighted by the positive part of the correlation function, which "
            "concentrates the fit where the wavefront actually is."
        ),
        default=True,
    )
    parser.add_argument(
        "--mineigenvalue",
        dest="mineigenvalue",
        type=float,
        metavar="VALUE",
        help=(
            f"Eigenvalues of the structure tensor below this fraction of the largest "
            f"are dropped from the pseudoinverse.  Do NOT lower this to try to recover "
            f"more of the velocity: for a coherent travelling wavefront the tensor is "
            f"genuinely rank one, and admitting the near null directions just divides "
            f"noise by a small number.  On a noiseless synthetic radial wave, 1e-4 "
            f"gives 13 degrees of direction error and 0.1 gives 1.6.  Default is "
            f"{DEFAULT_MINEIGENVALUE}."
        ),
        default=DEFAULT_MINEIGENVALUE,
    )
    parser.add_argument(
        "--solver",
        dest="solver",
        type=str,
        choices=["tls", "ols"],
        help=(
            f'The estimator for the brightness constancy system.  "ols" is ordinary '
            f"least squares, which assumes the spatial gradient is noise free; because "
            f"it is not, the recovered speed is biased toward zero as noise rises "
            f'(errors-in-variables attenuation).  "tls" is total least squares, which '
            f"spreads the error over all four derivative components and largely removes "
            f"that bias.  Default is {DEFAULT_SOLVER}."
        ),
        default=DEFAULT_SOLVER,
    )
    parser.add_argument(
        "--maxspeed",
        dest="maxspeed",
        type=float,
        metavar="MMPERSEC",
        help=(
            f"Clip the recovered speed at this value, in mm/s.  Unlike delayflow there "
            f"is no clean analytic ceiling here, so this is a blunt sanity limit.  "
            f"Default is {DEFAULT_MAXSPEED}."
        ),
        default=DEFAULT_MAXSPEED,
    )
    parser.add_argument(
        "--mfdexponent",
        dest="mfdexponent",
        type=float,
        metavar="P",
        help=f"Slope exponent for the flow routing.  Default is {DEFAULT_MFDEXPONENT}.",
        default=DEFAULT_MFDEXPONENT,
    )
    parser.add_argument(
        "--minterritorysize",
        dest="minterritorysize",
        type=int,
        metavar="NVOXELS",
        help=(
            f"Territories smaller than this many voxels are set to zero.  Default is "
            f"{DEFAULT_MINTERRITORYSIZE}."
        ),
        default=DEFAULT_MINTERRITORYSIZE,
    )
    parser.add_argument(
        "--nofill",
        dest="dofill",
        action="store_false",
        help="Skip the depression filling step.",
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
        help=f"Seed streamlines from every Nth voxel.  Default is {DEFAULT_SEEDSTRIDE}.",
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
        help=f"Maximum steps per streamline.  Default is {DEFAULT_MAXSTEPS}.",
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
        help="Will disable showing progress bars.",
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


def getlagaxis(theheader: Any, numlags: int) -> NDArray:
    """
    Reconstruct the lag axis of the similarity function movie from its header.

    rapidtide writes the similarity function with the lag step in pixdim[4] and the
    first lag value in toffset, so the axis comes straight back out of the header.

    Parameters
    ----------
    theheader : nifti header
        The header of the similarity function file.
    numlags : int
        The number of lag samples.

    Returns
    -------
    NDArray
        The lag values, in seconds.
    """
    thelagstep = float(theheader["pixdim"][4])
    thelagstart = float(theheader["toffset"])
    if thelagstep <= 0.0:
        raise ValueError(
            "the similarity function header has a nonpositive lag step in pixdim[4] - "
            "this does not look like a rapidtide corrout file"
        )
    return thelagstart + np.arange(numlags, dtype=np.float64) * thelagstep


def oversamplelagaxis(corrdata: NDArray, lagaxis: NDArray, factor: int) -> Tuple[NDArray, NDArray]:
    """
    Interpolate the similarity function onto a finer lag axis.

    This is free information, not invention.  The cross correlation of two
    bandlimited signals is bandlimited to the same band - R(f) = X*(f) Y(f) - so
    with fMRI data lowpass filtered to the LFO band the correlation function has a
    Nyquist lag interval of several seconds.  rapidtide already samples it far
    above that, so interpolating the lag axis recovers the underlying continuous
    function rather than inventing detail.  There is no need to pay for a higher
    --oversampfac in rapidtide itself, with the memory and runtime that costs.

    Why bother: see the note on kernel balance in computeopticalflow.  The temporal
    derivative kernel cannot be narrower than one lag sample, and on natively
    sampled data it is often far too wide relative to the spatial kernel, which
    biases the speed downward.  Interpolating lets it be narrowed to match.

    Parameters
    ----------
    corrdata : NDArray
        The 4D similarity function, with lag along the last axis.
    lagaxis : NDArray
        The lag values, in seconds.
    factor : int
        The interpolation factor.  1 leaves the data untouched.

    Returns
    -------
    corrdata : NDArray
        The interpolated similarity function.
    lagaxis : NDArray
        The finer lag axis.
    """
    if factor <= 1:
        return corrdata, lagaxis
    newlagaxis = np.linspace(
        lagaxis[0], lagaxis[-1], (len(lagaxis) - 1) * factor + 1, endpoint=True
    )
    thespline = CubicSpline(lagaxis, corrdata, axis=3)
    return thespline(newlagaxis), newlagaxis


def peaklag(corrdata: NDArray, lagaxis: NDArray, themask: NDArray) -> NDArray:
    """
    Find the lag of the correlation peak, refined by parabolic interpolation.

    This exists only to give the routing and territory labeling steps the scalar
    ordering they require.  It is deliberately crude - a three point parabolic
    refinement of the argmax - because nothing here feeds the velocity estimate,
    and the whole point of this program is to avoid depending on peak quality.

    Parameters
    ----------
    corrdata : NDArray
        The 4D similarity function, with lag along the last axis.
    lagaxis : NDArray
        The lag values, in seconds.
    themask : NDArray
        The 3D mask of valid voxels.

    Returns
    -------
    NDArray
        The 3D peak lag map, in seconds.
    """
    thelagstep = lagaxis[1] - lagaxis[0]
    numlags = len(lagaxis)
    thepeakindex = np.argmax(corrdata, axis=-1)

    # parabolic refinement, skipping the end points where there is no parabola
    theinterior = (thepeakindex > 0) & (thepeakindex < numlags - 1)
    thesafeindex = np.clip(thepeakindex, 1, numlags - 2)
    theindices = np.indices(thepeakindex.shape)
    theleft = corrdata[theindices[0], theindices[1], theindices[2], thesafeindex - 1]
    thecenter = corrdata[theindices[0], theindices[1], theindices[2], thesafeindex]
    theright = corrdata[theindices[0], theindices[1], theindices[2], thesafeindex + 1]

    thedenominator = theleft - 2.0 * thecenter + theright
    theshift = np.where(
        np.abs(thedenominator) > 1.0e-12,
        0.5
        * (theleft - theright)
        / np.where(np.abs(thedenominator) > 1.0e-12, thedenominator, 1.0),
        0.0,
    )
    theshift = np.clip(np.where(theinterior, theshift, 0.0), -0.5, 0.5)

    return np.where(themask > 0, lagaxis[thepeakindex] + theshift * thelagstep, 0.0)


def computeopticalflow(
    corrdata: NDArray,
    themask: NDArray,
    voxdims: Tuple[float, float, float],
    lagaxis: NDArray,
    fitradius: float = DEFAULT_FITRADIUS,
    derivsigma: float = DEFAULT_DERIVSIGMA,
    lagsmooth: float = DEFAULT_LAGSMOOTH,
    weightbycorr: bool = True,
    mineigenvalue: float = DEFAULT_MINEIGENVALUE,
    solver: str = DEFAULT_SOLVER,
    showprogressbar: bool = True,
    debug: bool = False,
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Estimate a static velocity field from the similarity function movie.

    Three dimensional Lucas-Kanade, accumulating the structure tensor over both a
    spatial neighborhood and the whole lag axis.  See the module docstring for why
    accumulating over lag beats solving per lag frame and averaging.

    The structure tensor is inverted through its eigendecomposition rather than
    directly, so that aperture limited voxels degrade gracefully: components along
    poorly conditioned eigenvectors are dropped rather than being allowed to blow
    up.  This is the pseudoinverse, and the number of eigenvalues that survive the
    threshold is reported as the rank map.

    Read the rank map as local geometry, not as a quality score.  For a clean
    propagating wavefront the tensor is rank one and that is the correct, healthy
    answer - see the module docstring.  Ranks 2 and 3 mean the neighborhood holds
    structure in more than one direction, which is real at bifurcations and
    confluences and is also what noise looks like.

    Parameters
    ----------
    corrdata : NDArray
        The 4D similarity function, with lag along the last axis.
    themask : NDArray
        The 3D mask of valid voxels.
    voxdims : tuple of float
        The voxel dimensions in mm.
    lagaxis : NDArray
        The lag values, in seconds.
    fitradius : float, optional
        The radius of the spatial accumulation neighborhood, in mm.
    derivsigma : float, optional
        Sigma of the spatial gaussian derivative kernel, in voxels.
    lagsmooth : float, optional
        Sigma of the lag axis gaussian derivative kernel, in lag samples.
    weightbycorr : bool, optional
        Weight each lag sample by the positive part of the correlation function.
    mineigenvalue : float, optional
        Relative eigenvalue threshold for the pseudoinverse.
    solver : str, optional
        "tls" for total least squares, which corrects the errors-in-variables
        attenuation of the speed, or "ols" for ordinary least squares.
    showprogressbar : bool, optional
        Show a progress bar.
    debug : bool, optional
        Enable debug output.

    Returns
    -------
    velocity : NDArray
        The 4D velocity field, in mm/s.
    therank : NDArray
        The number of eigenvalues surviving the threshold at each voxel, 0 to 3.
        Rank 1 is the expected regime for a coherent wavefront.
    thecoherence : NDArray
        The ratio of smallest to largest eigenvalue, a conditioning measure.
    theresidual : NDArray
        The rms brightness constancy residual, a goodness of fit measure.
    """
    theshape = corrdata.shape[:3]
    numlags = corrdata.shape[3]
    thelagstep = float(lagaxis[1] - lagaxis[0])

    # Extend outside the mask by nearest in mask value before differentiating.  The
    # movie is zero outside the brain, and differencing against those zeros creates
    # an enormous spurious gradient all around the cortical surface, which would
    # otherwise dominate the structure tensor at exactly the voxels we care about.
    theextensionindices = maskextensionindices(themask)

    # spatial neighborhood weights for the accumulation
    halfwidths = [max(int(np.ceil(fitradius / thedim)), 1) for thedim in voxdims]
    thekernels = []
    for thehalfwidth, thedim in zip(halfwidths, voxdims):
        theoffsets = np.arange(-thehalfwidth, thehalfwidth + 1) * thedim
        thekernel = np.exp(-0.5 * (theoffsets / (fitradius / 2.0)) ** 2)
        thekernel[np.abs(theoffsets) > fitradius] = 0.0
        thekernels.append(thekernel / thekernel.sum())

    def smoothspatially(thevolume: NDArray) -> NDArray:
        for theaxis, thekernel in enumerate(thekernels):
            thevolume = correlate1d(thevolume, thekernel, axis=theaxis, mode="nearest")
        return thevolume

    # the six unique structure tensor entries plus the three rhs entries
    themoments = np.zeros(theshape + (9,), dtype=np.float64)
    theweightsum = np.zeros(theshape, dtype=np.float64)
    theresidualsum = np.zeros(theshape, dtype=np.float64)

    thepairs = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]

    for thelag in tqdm(
        range(numlags),
        desc="Lag frames",
        unit="frames",
        disable=(not showprogressbar),
    ):
        theframe = corrdata[:, :, :, thelag][theextensionindices]

        # Spatial derivatives, per VOXEL rather than per mm, and below the lag
        # derivative is per lag SAMPLE rather than per second.  Working in index
        # units is what makes total least squares valid: with the same kernel width
        # on every axis the derivative noise is then isotropic across the four
        # components of [grad(I), dI/dl].  In physical units it is not - with 2 mm
        # voxels and a 0.4 s lag step the lag component is five times larger, TLS
        # picks the wrong null direction once noise is appreciable, and the answer
        # comes out roughly orthogonal to the truth.  The velocity is converted back
        # to mm/s after the solve.
        thespatialderivs = []
        for theaxis in range(3):
            thespatialderivs.append(
                gaussian_filter1d(theframe, derivsigma, axis=theaxis, order=1, mode="nearest")
            )

        # lag derivative, in per second, by centred gaussian derivative along lag
        thelo = max(thelag - int(np.ceil(3 * lagsmooth)), 0)
        thehi = min(thelag + int(np.ceil(3 * lagsmooth)) + 1, numlags)
        thestack = corrdata[:, :, :, thelo:thehi][theextensionindices]
        thelagderiv = gaussian_filter1d(thestack, lagsmooth, axis=3, order=1, mode="nearest")[
            :, :, :, thelag - thelo
        ]

        if weightbycorr:
            theweight = np.clip(theframe, 0.0, None)
        else:
            theweight = np.ones(theshape, dtype=np.float64)

        for theindex, (i, j) in enumerate(thepairs):
            themoments[:, :, :, theindex] += smoothspatially(
                theweight * thespatialderivs[i] * thespatialderivs[j]
            )
        for i in range(3):
            themoments[:, :, :, 6 + i] += smoothspatially(
                theweight * thespatialderivs[i] * thelagderiv
            )
        theweightsum += smoothspatially(theweight)
        theresidualsum += smoothspatially(theweight * thelagderiv**2)

    # assemble and solve the systems, voxel by voxel, via eigendecomposition
    validvoxels = np.nonzero(themask > 0)
    numvalid = len(validvoxels[0])
    if debug:
        print(f"solving {numvalid} structure tensor systems")

    thetensor = np.zeros((numvalid, 3, 3), dtype=np.float64)
    for theindex, (i, j) in enumerate(thepairs):
        thetensor[:, i, j] = themoments[:, :, :, theindex][validvoxels]
        thetensor[:, j, i] = thetensor[:, i, j]
    therhs = -np.stack([themoments[:, :, :, 6 + i][validvoxels] for i in range(3)], axis=-1)

    # symmetric, so eigh is both correct and fast
    theeigenvalues, theeigenvectors = np.linalg.eigh(thetensor)
    thelargest = np.max(np.abs(theeigenvalues), axis=-1, keepdims=True)
    thelargest = np.where(thelargest > 0.0, thelargest, 1.0)
    thekeep = theeigenvalues > (mineigenvalue * thelargest)

    therankvector = np.sum(thekeep, axis=-1)

    if solver == "ols":
        # pseudoinverse: project the rhs onto the surviving eigenvectors only
        theprojection = np.einsum("nij,nj->ni", np.transpose(theeigenvectors, (0, 2, 1)), therhs)
        theinverted = np.where(
            thekeep, theprojection / np.where(thekeep, theeigenvalues, 1.0), 0.0
        )
        thesolution = np.einsum("nij,nj->ni", theeigenvectors, theinverted)
    else:
        # Total least squares.  Ordinary least squares treats grad(I) as noise free
        # and puts all the error on dI/dl, which is simply false here - the movie is
        # noisy, so both are contaminated.  The consequence is the classic
        # errors-in-variables attenuation: noise inflates the sum of grad(I)**2 in
        # the denominator faster than it inflates the numerator, and the recovered
        # speed is biased toward zero.  On the synthetic movie OLS loses two thirds
        # of the speed at the noisiest level tested.
        #
        # TLS instead finds the direction of least variance of the augmented vector
        # [grad(I), dI/dl], which distributes the error over all four components.
        # The solve is done inside the well conditioned subspace of the spatial
        # tensor, so the aperture problem is still handled: for a rank one voxel this
        # reduces to a 2x2 eigenproblem, which is exactly the normal flow estimate
        # with the attenuation removed.
        #
        # This works, but only above a certain SNR.  TLS trades bias for variance,
        # and once the noise is large enough the smallest eigenvalue of the augmented
        # system is set by noise rather than by the constraint, so the null direction
        # becomes arbitrary.  On the synthetic movie it tracks the truth to within
        # 1.6 degrees up to a noise level of 0.05 and then falls apart, returning
        # directions roughly orthogonal to the truth.  That is why ols is the default:
        # it is biased, but it fails gradually and visibly instead of confidently.
        thecross = np.stack([themoments[:, :, :, 6 + i][validvoxels] for i in range(3)], axis=-1)
        thelagsquared = theresidualsum[validvoxels]
        thesolution = np.zeros((numvalid, 3), dtype=np.float64)

        for therank in (1, 2, 3):
            theselection = np.nonzero(therankvector == therank)[0]
            if len(theselection) == 0:
                continue
            # eigh returns eigenvalues ascending, so the surviving ones are the last
            thebasis = theeigenvectors[theselection][:, :, -therank:]
            thesubtensor = np.einsum(
                "nir,nij,njs->nrs", thebasis, thetensor[theselection], thebasis
            )
            thesubcross = np.einsum("nir,ni->nr", thebasis, thecross[theselection])

            theaugmented = np.zeros((len(theselection), therank + 1, therank + 1))
            theaugmented[:, :therank, :therank] = thesubtensor
            theaugmented[:, :therank, therank] = thesubcross
            theaugmented[:, therank, :therank] = thesubcross
            theaugmented[:, therank, therank] = thelagsquared[theselection]

            # the null direction of the augmented system is the TLS solution
            dummy, theaugvectors = np.linalg.eigh(theaugmented)
            thenull = theaugvectors[:, :, 0]
            thescale = thenull[:, therank]
            # a vanishing last component means the constraint does not determine a
            # finite velocity at all, so leave those voxels at zero
            theusable = np.abs(thescale) > 1.0e-12
            thecoefficients = np.where(
                theusable[:, np.newaxis],
                thenull[:, :therank] / np.where(theusable, thescale, 1.0)[:, np.newaxis],
                0.0,
            )
            thesolution[theselection] = np.einsum("nir,nr->ni", thebasis, thecoefficients)

    # index units (voxels per lag sample) back to physical units (mm/s)
    velocity = np.zeros(theshape + (3,), dtype=np.float64)
    for i in range(3):
        velocity[:, :, :, i][validvoxels] = thesolution[:, i] * voxdims[i] / thelagstep

    therank = np.zeros(theshape, dtype=np.uint16)
    therank[validvoxels] = therankvector

    thecoherence = np.zeros(theshape, dtype=np.float64)
    thecoherence[validvoxels] = np.min(theeigenvalues, axis=-1) / thelargest[:, 0]

    # rms residual of the brightness constancy constraint, normalized by the weight
    thevariance = np.zeros(theshape, dtype=np.float64)
    thevariance[validvoxels] = theresidualsum[validvoxels] / np.where(
        theweightsum[validvoxels] > 0.0, theweightsum[validvoxels], 1.0
    )
    theresidual = np.sqrt(np.clip(thevariance, 0.0, None))

    return velocity, therank, thecoherence, theresidual


def corrflow(args: argparse.Namespace) -> None:
    """
    Infer a blood flow map from the 4D similarity function movie.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command line arguments.

    Returns
    -------
    None
    """
    starttime = time.time()
    therunoptions: Dict[str, Any] = {}
    for thekey, thevalue in vars(args).items():
        if isinstance(thevalue, (str, int, float, bool, type(None))):
            therunoptions[thekey] = thevalue

    # work out the root name so we can find the companion files
    theroot = args.corrfile
    for theending in ["_desc-corrout_info.nii.gz", "_desc-corrout_info.nii"]:
        if theroot.endswith(theending):
            theroot = theroot[: -len(theending)]
            break
    else:
        theroot, _ = tide_io.niftisplitext(theroot)

    ##########################################################################
    # read the similarity function movie
    ##########################################################################
    print(f"reading similarity function from {args.corrfile}")
    (
        corr_img,
        corr_data,
        corr_hdr,
        corr_dims,
        corr_sizes,
    ) = tide_io.readfromnifti(args.corrfile)
    xsize, ysize, numslices, numlags = tide_io.parseniftidims(corr_dims)
    xdim, ydim, slicethickness, thelagstep = tide_io.parseniftisizes(corr_sizes)
    voxdims = (xdim, ydim, slicethickness)
    if numlags < 5:
        raise ValueError(
            f"the similarity function has only {numlags} lag samples - this is not "
            f"enough to estimate optical flow"
        )
    corrdata = np.nan_to_num(corr_data).astype(np.float64)
    lagaxis = getlagaxis(corr_hdr, numlags)
    print(
        f"movie is {xsize}x{ysize}x{numslices}, {numlags} lags from "
        f"{lagaxis[0]:.2f} to {lagaxis[-1]:.2f} s in steps of {thelagstep:.3f} s"
    )
    if args.lagoversamp > 1:
        corrdata, lagaxis = oversamplelagaxis(corrdata, lagaxis, args.lagoversamp)
        thelagstep = float(lagaxis[1] - lagaxis[0])
        print(
            f"interpolated the lag axis {args.lagoversamp}x to {len(lagaxis)} samples, "
            f"effective step {thelagstep:.4f} s"
        )

    ##########################################################################
    # CBV proxy: peak to peak amplitude of the correlation function
    ##########################################################################
    print("computing the CBV proxy from the correlation function peak to peak amplitude")
    cbv = np.ptp(corrdata, axis=-1)

    ##########################################################################
    # mask
    ##########################################################################
    if args.maskfile is None:
        thecandidate = f"{theroot}_desc-corrfit_mask.nii.gz"
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
        if not tide_io.checkspacematch(corr_hdr, mask_hdr):
            raise ValueError("mask dimensions do not match similarity function dimensions")
        themask = np.uint16(mask_data.reshape((xsize, ysize, numslices)) > 0)
    else:
        print("no mask found - masking on nonzero correlation functions")
        themask = np.uint16(cbv > 0.0)
    therunoptions["maskfile"] = args.maskfile

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

    output_hdr = makeoutputheader(corr_hdr, numvols=1)
    vector_hdr = makeoutputheader(corr_hdr, numvols=3)
    tide_io.savetonifti(
        themask, output_hdr, f"{args.outputroot}_desc-flowfit_mask", debug=args.debug
    )
    tide_io.savetonifti(
        cbv.astype(np.float32),
        output_hdr,
        f"{args.outputroot}_desc-cbvproxy_map",
        debug=args.debug,
    )

    ##########################################################################
    # optical flow
    ##########################################################################
    print("estimating optical flow by 3D Lucas-Kanade")
    velocity, therank, thecoherence, theresidual = computeopticalflow(
        corrdata,
        themask,
        voxdims,
        lagaxis,
        fitradius=args.fitradius,
        derivsigma=args.derivsigma,
        lagsmooth=args.lagsmooth,
        weightbycorr=args.weightbycorr,
        mineigenvalue=args.mineigenvalue,
        solver=args.solver,
        showprogressbar=args.showprogressbar,
        debug=args.debug,
    )

    speed = np.linalg.norm(velocity, axis=-1)
    theclipped = int(np.sum(speed > args.maxspeed))
    if theclipped > 0:
        print(f"clipping {theclipped} voxels with speed above {args.maxspeed} mm/s")
        thescale = np.where(
            speed > args.maxspeed, args.maxspeed / np.where(speed > 0, speed, 1.0), 1.0
        )
        velocity *= thescale[:, :, :, np.newaxis]
        speed = np.linalg.norm(velocity, axis=-1)

    direction = np.zeros_like(velocity)
    safespeed = np.where(speed > 0.0, speed, 1.0)
    for i in range(3):
        direction[:, :, :, i] = np.where(themask > 0, velocity[:, :, :, i] / safespeed, 0.0)

    for thecount in range(4):
        thefraction = np.sum((therank == thecount) & (themask > 0)) / numvalid
        print(f"  {100.0 * thefraction:5.1f}% of voxels have structure tensor rank {thecount}")
    print(
        "  rank 1 is the expected regime for a coherent propagating wavefront; rank 0 "
        "means no usable information, and ranks 2-3 mean multidirectional structure "
        "(real at bifurcations, but also what noise looks like)"
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
    tide_io.savetonifti(
        speed.astype(np.float32),
        output_hdr,
        f"{args.outputroot}_desc-speed_map",
        debug=args.debug,
    )
    tide_io.savetonifti(
        therank, output_hdr, f"{args.outputroot}_desc-flowrank_map", debug=args.debug
    )
    tide_io.savetonifti(
        thecoherence.astype(np.float32),
        output_hdr,
        f"{args.outputroot}_desc-flowcoherence_map",
        debug=args.debug,
    )
    tide_io.savetonifti(
        theresidual.astype(np.float32),
        output_hdr,
        f"{args.outputroot}_desc-flowresidual_map",
        debug=args.debug,
    )

    ##########################################################################
    # divergence of flux diagnostic
    ##########################################################################
    if args.dodivergence:
        print("computing div(cbv * v) as a conservation diagnostic")
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
    # the scalar field the topology steps need
    ##########################################################################
    if args.delayfile is None:
        thecandidate = f"{theroot}_desc-maxtime_map.nii.gz"
        if os.path.isfile(thecandidate):
            args.delayfile = thecandidate
    if args.delayfile is not None:
        print(f"reading the ordering field from {args.delayfile}")
        (
            delay_img,
            delay_data,
            delay_hdr,
            delay_dims,
            delay_sizes,
        ) = tide_io.readfromnifti(args.delayfile)
        if not tide_io.checkspacematch(corr_hdr, delay_hdr):
            raise ValueError("delay map dimensions do not match similarity function dimensions")
        tau = np.nan_to_num(delay_data.reshape((xsize, ysize, numslices))).astype(np.float64)
    else:
        print("deriving a peak lag map from the movie for the topology steps only")
        tau = peaklag(corrdata, lagaxis, themask)
    therunoptions["delayfile"] = args.delayfile
    tau = np.where(themask > 0, tau, 0.0)
    tide_io.savetonifti(
        tau.astype(np.float32),
        output_hdr,
        f"{args.outputroot}_desc-orderingfield_map",
        debug=args.debug,
    )

    if args.dofill:
        print("filling depressions")
        taudown = -priorityflood(-tau, themask, showprogressbar=args.showprogressbar)
        tauup = priorityflood(tau, themask, showprogressbar=args.showprogressbar)
    else:
        taudown = tau
        tauup = tau

    ##########################################################################
    # flow accumulation and territories
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
            print("sampling scalars along streamlines")
            datapervertex = samplealongstreamlines(
                thestreamlines,
                {
                    "arrivaltime": tau,
                    "speed": speed,
                    "logflowaccum": logaccumulation,
                    "flowrank": therank.astype(np.float64),
                    "territory": territories.astype(np.float64),
                },
                themask=themask,
                nearestneighbor=["territory", "flowrank"],
            )
            savestreamlines(
                thestreamlines,
                corr_img.affine,
                (xsize, ysize, numslices),
                voxdims,
                f"{args.outputroot}_desc-flow_streamlines.trk",
                datapervertex=datapervertex,
                debug=args.debug,
            )
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
    therunoptions["numlags"] = int(numlags)
    therunoptions["lagstart"] = float(lagaxis[0])
    therunoptions["lagend"] = float(lagaxis[-1])
    therunoptions["lagstep"] = float(thelagstep)
    therunoptions["numterritories"] = len(thesizes)
    therunoptions["fractionrank3"] = float(np.sum((therank == 3) & (themask > 0)) / numvalid)
    therunoptions["runtime"] = time.time() - starttime
    tide_io.writedicttojson(therunoptions, f"{args.outputroot}_desc-runoptions_info.json")
    print(f"done in {time.time() - starttime:.1f} seconds")


def main() -> None:
    try:
        args = _get_parser().parse_args()
    except SystemExit:
        _get_parser().print_help()
        raise
    corrflow(args)


if __name__ == "__main__":
    main()
