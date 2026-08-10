..
   Headings are organized in this manner:
   =====
   -----
   ^^^^^
   """""
   '''''

Resolving delay ambiguity: theory of operation
==============================================

This describes what ``--resolvedelays`` does, why it works, and — as importantly —
what it cannot do.  It assumes familiarity with fMRI and with rapidtide's basic
delay estimation, but not with phase unwrapping or graph labelling.

The problem
-----------

Rapidtide estimates transit delay by cross-correlating each voxel's time series
against a reference regressor and taking the lag of the strongest peak.  The
similarity function usually has *several* local maxima, and taking the strongest is
a guess that is sometimes wrong.

Two things put extra peaks there:

**Regressor autocorrelation.**  The reference regressor is bandlimited to the LFO
band, so it has structure at its own dominant period — typically 10–20 s.  That
structure appears in every cross-correlation as sidelobes spaced one period apart.
Where noise or a baseline shift knocks down the true peak, a sidelobe can end up
taller, and the delay comes out wrong by nearly a full period.

**Low SNR.**  Where the LFO signal is weak the similarity function is shallow and
noisy, and which bump happens to be tallest is close to arbitrary.

Both causes vary slowly in space, so the resulting errors are not isolated specks.
They come in **contiguous patches** where every voxel is displaced by about the same
amount.  That matters for what can fix them.

Why despeckling is not enough
-----------------------------

Rapidtide's despeckler compares each voxel to the median of its neighbourhood and
refits anything that deviates by more than ``--despecklethresh``.  This works well
for isolated outliers.  It works poorly on a patch, because inside a patch the
local median *is* the wrong value — the voxel agrees with its neighbours, and there
is nothing for the despeckler to flag.

Despeckling also has a second property worth naming: it produces a refit lag that
need not correspond to any peak of the similarity function.  It can put the delay
somewhere the data does not actually support.

The reframing
-------------

Delay resolution treats this as a **labelling problem rather than a smoothing
problem**.

For each voxel, the similarity function offers a short menu of candidate delays —
its local maxima.  One of them is presumably right.  The job is to choose one
candidate per voxel such that the resulting map is spatially coherent.

The consequence worth internalising: **the output is always a peak of that voxel's
own similarity function.**  Resolution can move a voxel from one peak to another,
but it cannot invent a delay that the voxel's own data does not support.  A
smoother can, and does.  This is why resolution and despeckling are complementary
rather than redundant, and it bounds how wrong resolution can be.

The algorithm
-------------

Three stages.

Stage 1: candidate extraction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``findcandidatepeaks`` finds every interior local maximum of each voxel's
similarity function, refines each to sub-sample precision by three-point parabolic
interpolation, and returns them sorted by amplitude.  Index 0 is exactly what naive
peak picking would have chosen, which makes "did resolution change this voxel?" a
trivial comparison.

Up to ``maxcandidates`` (default 6) are kept per voxel; empty slots hold NaN.

Stage 2: confidence-ordered region growing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``resolvedelaymap`` is where the real work happens.  It is a quality-guided flood
fill, the same idea used for phase unwrapping in InSAR and in MRI field mapping.

Each voxel gets a **confidence** score: the amplitude gap between its best and
second-best candidate.  A voxel with one dominant peak is unambiguous; a voxel with
two peaks of similar height is a coin flip.

The fill then proceeds:

1. Seed at the most confident unassigned voxel and give it its strongest peak.
2. Push it onto a priority queue ordered by confidence.
3. Pop the most confident queued voxel and look at its unassigned neighbours.
4. For each such neighbour, form a **prediction** from *all* of its already-assigned
   neighbours — not just the voxel that happened to pop — by taking the median of
   what they imply.  Assign the candidate closest to that prediction, and queue it.
5. Repeat until nothing is reachable, then seed a new region.

Ordering by confidence is the key design choice.  Unambiguous tissue is resolved
first and establishes the frame; ambiguous voxels are decided last, when they have
the most already-settled neighbours to lean on.  A coin-flip voxel never gets to
seed a region and drag good tissue after it.

Two refinements guard against the method propagating its own mistakes:

- **Consensus prediction.**  Using the median over every assigned neighbour, rather
  than the single popped one, stops one wrapped voxel from dragging a correct
  neighbour with it.
- **Trusted sources.**  With ``--minconfidence`` above zero, voxels below that
  confidence quantile are still assigned but are not allowed to predict for anyone
  else, so long as some trusted neighbour exists.

Optionally a velocity field can be supplied (from ``corrflow``), in which case the
prediction uses the eikonal relation ``grad(tau) = v / |v|^2`` to propagate an
expected delay *change* across each step rather than assuming neighbours are
similar.  Inside rapidtide this is not used — the flow-free prior is the default —
but it is available in the standalone tool.

Stage 3: ICM refinement
^^^^^^^^^^^^^^^^^^^^^^^

The region grow is deterministic and takes no delay map as input, so running it
twice changes nothing.  Useful iteration means feeding the solution back as a
prior: ``icmrefine`` median-filters the current map and re-snaps every voxel to
whichever of its candidates lies closest to that smoothed field.  This is iterated
conditional modes on a smoothness-regularised labelling problem, and it converges
because the candidate set is finite.

Returns diminish sharply.  On HCP data the wrapped-voxel count runs 1685, 1463,
1354, 1272 over the first three iterations and creeps to about 1000 by iteration
20 — but the count of large *non-periodic* jumps rises slowly over the same span,
which is the signature of the prior starting to invent structure rather than repair
it.  ``--resolvepasses`` defaults to 3 for that reason; running to convergence
trades a little more repair against slowly accumulating damage.

.. note::
   The median filter used to build the prior extends the map outside the brain
   mask with each voxel's nearest in-mask value.  Zero-filling instead — the
   obvious implementation — drags surface voxels toward tau = 0.  On a UK Biobank
   run that moved 1069 assignments by more than half a second, 94% of them within
   one voxel of the mask surface, roughly halving the delay assigned there.
   Anything that filters or interpolates near a mask edge needs an explicit story
   about what lies outside; the default answer, zero, is always wrong and never
   raises an error.

What the prior can and cannot do
--------------------------------

This is the part to understand before trusting the output on a clinical cohort.

Resolution chooses, among candidates, whichever makes the map smoothest.  It
therefore **cannot distinguish a genuinely long delay that varies smoothly from its
surroundings from a fitting error**, because at a discontinuity both look the same.
Vascular pathology produces real delays of tens of seconds, and what marks them as
real is not their size but their spatial coherence — which a smoothness prior will
happily flatten.

This was measured directly.  Long-delay voxels were defined on unrepaired control
runs, split by connected-component size, and tracked into paired resolved runs:

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 15 15

   * - cohort
     - reassigned, coherent
     - reassigned, scattered
     - selectivity
     - coherent surviving
   * - HCP young adult
     - 0.446
     - 0.610
     - 1.30x
     - 0.545
   * - UK Biobank
     - 0.833
     - 0.831
     - 1.03x
     - 0.200

In young healthy subjects resolution is modestly biased toward removing scattered
errors over coherent structure.  In an older cohort that selectivity is gone
entirely, and 80% of coherent long-delay structure is rewritten.

Whether that is correct depends on whether the structure was real — which no
statistic computed from a single run can settle, because coherent error and
coherent physiology have the same shape.  A useful partial check is the fit-quality
cost: reading the similarity function at the old and new lags shows resolution
typically moves to a peak about half as tall (amplitude ratio 0.51 in HCP, 0.57 in
UK Biobank).  That sounds damning but is not — in the ambiguity case the method
targets, the *wrong* peak is usually the taller one, which is precisely why naive
peak picking chose it.  The similar ratios across cohorts say resolution is not
behaving anomalously on the older data; they do not say it is right.

**Practical advice:** on cohorts where long delays may be real, run
``--saveresolvemaps`` and look at what was rewritten before trusting the map.

Using it
--------

Inside rapidtide
^^^^^^^^^^^^^^^^

.. code-block:: bash

   rapidtide input.nii.gz output --resolvedelays --despecklepasses 4

``--resolvedelays``
   Turn it on.  Runs on every pass, immediately before despeckling.  Not gated on
   anything.

``--resolvepasses N``
   ICM refinement iterations (default 3).  1 means region grow only.

``--saveresolvemaps``
   Write the audit trail (see below).  Honoured at every ``--outputlevel``,
   including ``min``.

Resolution runs once per pass and its result feeds regressor refinement, so it
affects the reference regressor for subsequent passes, not just the final map.
The per-pass count of reassigned voxels is recorded in the runoptions as
``resolved_passN``; a voxel counts as reassigned if it moved by more than 0.5 s.

Ordering with despeckling
^^^^^^^^^^^^^^^^^^^^^^^^^

Both run, resolution first, and **the order is not symmetric**:

- *Despeckle then resolve* is worse than resolving alone.  Resolution can only
  select among peaks, so it silently discards despeckle's refit values wherever
  those are not peaks.
- *Resolve then despeckle* beats either alone.  Resolution makes the discrete
  choice of which lobe; despeckling then refits the residual local outliers, which
  resolution cannot do because it cannot produce a non-peak value.

Outputs
^^^^^^^

With ``--saveresolvemaps``:

``_desc-maxtimepreresolve_map``
   The delay map as it stood going into resolution on the final pass, in seconds.

``_desc-resolveshift_map``
   The change resolution applied, in seconds.

``_desc-resolvechanged_mask``
   Voxels moved by more than 0.5 s.

.. warning::
   These describe the **final pass only**.  Resolution runs on every pass and its
   output feeds refinement, so the map entering the last pass has already been
   resolved.  Nothing computed inside one run can show the cumulative effect of
   resolving versus not resolving — that requires a paired run without
   ``--resolvedelays``.

Standalone
^^^^^^^^^^

The same algorithm is available as the ``resolvedelays`` command, operating on a
saved ``_desc-corrout_info.nii.gz`` (which requires ``--savecorrout``).  Useful for
experimenting with parameters without re-running the whole pipeline, and for the
flow-guided prior.  See :doc:`usage_resolvedelays`.

What it costs
-------------

Resolution takes about 28 s on a 145 000 voxel UK Biobank run, roughly 7% of
runtime.  It **reduces total runtime** — adding it took the median run from 8.0 to
6.8 minutes, because despeckling then has far less to do (135 s to 45 s).

The region grow is a sequential flood fill: each voxel's assignment depends on
neighbours assigned earlier in the same loop.  That dependency is the algorithm,
not an implementation artifact, so it is not parallelisable across voxels without
changing the answer.  The ICM stage is already fully vectorised.

Evidence
--------

Paired calibrations, scored as the count of voxels more than ``despecklethresh``
from their 5x5x5 local median.  "Naive" is the argmax of the *undespeckled*
similarity function, so all four conditions come from identical correlation data
and only the repair differs:

.. list-table::
   :header-rows: 1
   :widths: 22 20 20 20 18

   * - cohort
     - naive
     - despeckle
     - resolve
     - both
   * - HCP (16 runs)
     - 198397
     - 70424 (-65%)
     - 44193 (-78%)
     - 29064 (-85%)
   * - UK Biobank (26 runs)
     - 909316
     - 326748 (-64%)
     - 150862 (-83%)
     - 122574 (-87%)

Resolution alone beat despeckling alone in 41 of those 42 runs.  Adding
despeckling on top of resolution helped in 42 of 42 (median -34% in HCP, -18% in
UK Biobank), which is why both run rather than one replacing the other.  Counts of
large adjacent-voxel jumps fall monotonically across the same four conditions, so
the improvement is not a smoother erasing real gradients.

Gating was removed on this evidence.  The original design fired resolution only
where a sidelobe was measurable, on the argument that without one there is no
periodic ambiguity and the method degenerates into smoothing.  The calibration
refuted that: the map improved in every run including those with the lowest
measurable ambiguity and no detectable sidelobe.  The mechanism is also not what
the name suggests — the median change is small and two-sided, not the one-sided
one-period shift a true sidelobe wrap produces.  Only runs with a real measured
sidelobe show the periodic signature.  This is a general delay-map repair that
handles sidelobe wrapping as one special case.
