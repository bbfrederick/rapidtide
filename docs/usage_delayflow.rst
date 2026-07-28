delayflow
---------

Description:
^^^^^^^^^^^^

Infers a blood flow map from a rapidtide delay map.

The delay map tau(x) is an arrival time field.  A parcel of blood carrying the
sLFO waveform along a trajectory x(t) arrives at x(t) at time t, so
tau(x(t)) = t.  Differentiating along the trajectory gives the eikonal
relation::

    grad(tau) . v = 1

so the component of velocity along the arrival time gradient is 1/|grad(tau)|,
and flow points toward *increasing* delay.  Unlike diffusion tractography this
gives a true direction rather than just an orientation, since the sign of the
gradient distinguishes upstream from downstream.

Note that this is one scalar equation in three unknowns, so any velocity
component tangent to the isochrone surfaces is invisible - this is the aperture
problem, and it is a property of the measurement rather than of the algorithm.
What is recovered is the normal velocity of the arrival time wavefront, which
equals the true blood velocity only where flow is perpendicular to the
isochrones.  The minimum norm solution v = grad(tau)/|grad(tau)|**2 is what is
reported.

Processing proceeds in six stages: robust gradient estimation by CBV weighted
local plane fitting; velocity and speed from the eikonal relation, with an
explicit speed ceiling set by the delay noise level; depression filling of the
delay map by the priority flood algorithm; multiple flow direction routing and
flow accumulation (borrowed from terrain analysis, treating the delay map as an
elevation surface); catchment labeling to give vascular territories; and
streamline tractography of the velocity field.  A divergence of flux diagnostic
is also produced - blood is conserved, so div(cbv * v) should be near zero away
from true inlets and outlets, and large values flag where the flow estimate is
untrustworthy.

Interpretive caveat: a voxel contains arterial, capillary, and venous
compartments with very different delays, and the measured lag is a CBV weighted
blend in which the venous compartment dominates BOLD weighting.  The flow map
produced here is therefore largely a *venous drainage* map in parenchyma, with
arterial structure visible only in and near large arteries.

This is a proof of concept.

Inputs:
^^^^^^^

A rapidtide delay map, normally ``XXX_desc-maxtime_map.nii.gz``, in seconds.

If they exist alongside the delay map, ``XXX_desc-maxcorrsq_map.nii.gz`` is
picked up automatically as a relative CBV proxy and ``XXX_desc-corrfit_mask.nii.gz``
as the analysis mask.  Both can be overridden with ``--cbvfile`` and
``--maskfile``.  Note that maxcorrsq does double duty: it is a rough relative
blood volume proxy, and because it is the fraction of variance explained by the
delay fit it is simultaneously a delay reliability weight.

Outputs:
^^^^^^^^

All outputs are named ``OUTPUTROOT_desc-XXX_map.nii.gz``:

    **tausmoothed**: the mask aware smoothed delay map that was differentiated.

    **gradmag**: |grad(tau)|, in s/mm.

    **speed**: 1/|grad(tau)|, in mm/s, capped at the resolvable ceiling.

    **velocity**: the 3 component velocity vector field, in mm/s.

    **direction**: the unit flow direction vector field.

    **taufilleddown**, **taufilledup**: the depression filled delay maps used
    for downstream routing and upstream labeling respectively.

    **flowaccum**, **logflowaccum**: flow accumulation.  High values identify
    trunk draining vessels.

    **territories**: integer labeled vascular territories, with sizes written to
    ``OUTPUTROOT_desc-territorysizes_info.tsv``.

    **fluxdivergence**: div(cbv * v), the conservation diagnostic.

    **streamlinedensity**: the number of streamlines passing through each voxel.

Plus ``OUTPUTROOT_desc-flowfit_mask.nii.gz`` (the analysis mask),
``OUTPUTROOT_desc-speedresolved_mask.nii.gz`` (voxels whose speed is below the
ceiling and therefore actually measured rather than bounded),
``OUTPUTROOT_desc-flow_streamlines.trk``, and
``OUTPUTROOT_desc-runoptions_info.json``.

Usage:
^^^^^^

.. argparse::
   :ref: rapidtide.workflows.delayflow._get_parser
   :prog: delayflow
   :func: _get_parser

   Debugging options : @skip
      skip debugging options
