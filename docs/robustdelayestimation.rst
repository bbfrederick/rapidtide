..
   Headings are organized in this manner:
   =====
   -----
   ^^^^^
   """""
   '''''

Robust delay estimation: handling wrong-peak artifacts
======================================================

Background
----------

Rapidtide estimates the transit delay of the blood-borne low-frequency
oscillation (LFO) signal at each voxel by fitting the peak of a
cross-correlation function between the voxel time series and a global
reference regressor.  The cross-correlation function typically has a single
dominant peak, whose lag is the estimated delay.  However, the reference
regressor has autocorrelation structure — its power is concentrated in the
LFO band (roughly 0.01–0.15 Hz) — and this autocorrelation causes the
cross-correlation function to have sidelobe peaks spaced at multiples of the
dominant autocorrelation period (typically 10–20 seconds, depending on the
dominant LFO frequency in a given dataset).

Under favorable conditions (high SNR, large delay relative to the sidelobe
spacing) the true peak is clearly dominant and the fitter selects it
correctly.  Under less favorable conditions — especially near the boundaries
of brain regions where the signal-to-noise ratio is lower or where the
baseline of the correlation function shifts due to local autocorrelation
structure — the sidelobe peak can exceed the true peak in height.  When this
happens the fitter selects the wrong peak, assigning a lag offset by one or
more sidelobe spacings from the true value.

Because the factors that degrade the true peak (noise, baseline shift) vary
slowly in space, the wrong-peak selection tends to affect contiguous regions.
The result is a spatially coherent "artifact patch" — a zone in the lag map
where all voxels are displaced by approximately the same offset from the
surrounding tissue.


Why artifact patches are hard to correct
-----------------------------------------

The naive correction strategy — detect any spatially coherent region whose
lag differs from its surroundings by more than a threshold, and refit those
voxels using the surrounding lag as the initial estimate — fails in two
important ways.

First, **the true peak need not be the highest peak**.  Near the boundary of
an artifact patch, the true-lag peak in the correlation function is already
lower than the sidelobe peak (that is precisely why the fitter jumped to the
sidelobe).  A test of the form "is the neighbor-derived lag associated with
a higher correlation than the current lag?" will incorrectly confirm the
wrong-peak choice at the boundary, because C(τ\ :sub:`sidelobe`) >
C(τ\ :sub:`true`) there by construction.

Second, **vascular pathology produces delays of similar spatial character**.
The brain's arterial supply is organised into vascular territories.  Within
each territory delays are spatially smooth; at the interface between
territories, sharp discontinuities are physiologically normal.  In cases of
pathology (e.g. arterial stenosis or occlusion) an entire territory may be
delayed by many tens of seconds relative to normal tissue.  Such a territory
is spatially coherent, has a sharp boundary with adjacent normal tissue, and
appears offset from its surroundings — exactly as an artifact patch does.
Any correction algorithm must preserve true pathological delays while
correcting artifacts.

These two difficulties eliminate the candidate discriminators that seem
obvious at first:

* **Magnitude of the offset** cannot separate artifacts from pathology because
  genuine stenotic delays can be arbitrarily large (tens of seconds), while
  artifacts are bounded only by the number of sidelobe multiples that fit
  within the lag search range.

* **Comparison of peak heights** cannot separate artifacts from pathology
  because, at the voxels where the wrong peak was chosen, it IS the highest
  peak locally.

* **Gradient analysis** is necessary but not sufficient: both artifact
  boundaries and true territory boundaries can produce sharp step
  discontinuities.


The key discriminator
---------------------

Even though the true peak may not be the tallest peak in the correlation
function, it is still present — it has merely been reduced relative to the
sidelobe by baseline shift and noise.  For a genuinely delayed vascular
territory, by contrast, there is *no* significant peak at the "normal" lag
because the blood has not arrived by that time.

This observation motivates the operative test:

  **Does a valid peak exist near the spatially extrapolated expected lag?**

* For an artifact-patch voxel: **yes**.  The true-lag peak is present in the
  correlation function, attenuated but findable.
* For a true-territory-boundary voxel: **no**.  The signal arrived at a
  fundamentally different time; no peak exists at the extrapolated expected
  lag.

This test replaces the unreliable "is the expected-lag peak taller?" with the
more robust "does the expected-lag peak exist at all?", and it forms the
foundation of the region-growing algorithm described below.


The ACF sidelobe parameters
----------------------------

Each pass of the rapidtide main workflow calls ``cleanregressor``, which
computes the autocorrelation of the reference regressor and records the lag
and amplitude of the first sidelobe:

* ``optiondict["acsidelobelag" + passsuffix]`` — the lag (in seconds) of the
  first sidelobe of the regressor's autocorrelation function.  This is the
  inter-peak spacing of the cross-correlation function; wrong-peak artifacts
  are shifted by integer multiples of this value.
* ``optiondict["acsidelobeamp" + passsuffix]`` — the amplitude of the first
  sidelobe.  This bounds how much higher the sidelobe peak can be relative to
  the true peak in the absence of noise, setting the scale for how difficult
  detection will be.

These values are used directly by the correction algorithm:

* The peak search window around the extrapolated expected lag is set to
  ±(``search_width`` / 2), where ``search_width`` defaults to 5 seconds and
  is configurable on the command line.  In practice it should be less than
  ``acsidelobelag / 2`` to avoid accidentally accepting a sidelobe peak as
  the true peak.
* The offset of a detected candidate patch is tested against integer multiples
  of ``acsidelobelag``; a close match increases confidence that the patch is
  an artifact rather than a true delayed territory.
* ``acsidelobeamp`` bounds the worst-case height advantage of the sidelobe
  over the true peak, giving a principled lower bound on how attenuated the
  true peak can be before the algorithm fails to find it.


Proposed algorithm: multi-seed region growing
----------------------------------------------

The correct framework is spatial rather than local.  The "right" peak at each
voxel is not the locally tallest one but the one that produces a globally
consistent map.  This is structurally identical to phase unwrapping, where
the true signal is continuous but measurements may have wrapped by the
sidelobe period.  Phase unwrapping is solved not by comparing peak heights but
by path integration from trusted reference points.

The algorithm proceeds in five steps.

Step 1: identify anchor voxels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An anchor voxel is one where the peak selection is almost certainly correct.
A voxel qualifies as an anchor if:

* Its **peak dominance ratio** ``C_max / C_second`` exceeds a configurable
  threshold (default 1.5).  A voxel with two nearly equal peaks is ambiguous
  and must not be an anchor; a voxel with a clearly dominant peak was almost
  certainly fitted correctly.
* Its R² is above a high-confidence threshold (e.g. the 70th percentile of
  all fitted voxels).
* Its fit failure flags are clean.

Critically, anchors are identified on the basis of dominance ratio, *not* on
the basis of lag value.  A voxel in a truly delayed vascular territory, where
the signal arrives tens of seconds late but the correlation peak is unambiguous
and dominant, is a perfectly valid anchor.  An artifact-patch voxel, where the
wrong peak was selected and typically has a competing true peak at a similar
height, will generally fail the dominance test and will not become an anchor.

Step 2: multi-seed simultaneous region growing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Region growing is seeded from all anchor voxels simultaneously, using a
priority queue to process frontier voxels in order of decreasing confidence
(e.g. descending anchor R²).  This is equivalent to constructing a
confidence-weighted Voronoi diagram of the lag space seeded by the anchors.

For each frontier voxel *v* with spatially extrapolated expected lag
τ\ :sub:`expected` (derived by smoothly extrapolating from the already-processed
neighbors):

1. Search the peaks of *v*'s full correlation function (available in the
   ``corrout`` array, which is already resident in memory) within a window of
   ±``search_width`` / 2 around τ\ :sub:`expected`.

2. **If a peak is found** within the window:

   * Assign that peak's lag to *v*, even if it is not the highest peak in the
     correlation function.
   * Mark *v* as processed and add it to the growing frontier.
   * This is the critical step for artifact correction: the true-lag peak is
     chosen because it is spatially consistent, even though it has been
     attenuated below the sidelobe.

3. **If no peak is found** within the window:

   * If *v*'s current fit has high dominance ratio and high R², treat *v* as a
     **territory boundary**: the signal truly arrives at a very different time.
     Accept *v*'s current fitted lag, promote *v* to a new anchor seed, and
     continue growing from this new seed into the adjacent territory.
   * If *v*'s current fit is also of low quality, *v* is uncertain.  Leave its
     lag unchanged (see Step 5 below).

Step 3: territory boundaries emerge naturally
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The growing fronts from anchor seeds in different territories will eventually
collide.  At a true territory boundary:

* The front from the normal side cannot find a peak at the normal lag in the
  delayed territory (the signal has not arrived yet), so it stalls.
* The front from the delayed side cannot find a peak at the delayed lag in the
  normal region (the signal has long since passed), so it also stalls.
* The boundary is preserved as a discontinuity without any explicit detection
  or special-case handling.

An artifact patch, by contrast, has no internal anchors (its voxels fail the
dominance test) and no competing growing front from the wrong-lag side.  The
front from the exterior normal region propagates across the transition zone
(finding attenuated but real true-lag peaks throughout) and floods the entire
interior of the patch, reassigning all voxels to the correct lag.

Step 4: correction by refitting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After region growing, voxels whose assigned lag disagrees with the current
fitted lag are queued for refitting.  The refitting call
``fitcorr(despeckling=True, multipeak=True)`` is used, with the assigned lag
as the initial estimate, reusing the existing infrastructure.

Step 5: uncertain and isolated voxels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Voxels that region growing could not reach — either because no peak was found
near τ\ :sub:`expected` and the current fit is also of poor quality, or because
they reside in a region with no nearby anchors — are left unchanged.  A
diagnostic count of such voxels is recorded in the output so that the user
can judge the severity of the unresolved cases.

The ICM (Iterated Conditional Modes) optimizer implemented in
``notreadyforprimetime/fitSimFuncMap.py`` is a candidate post-pass for
resolving some of these uncertain voxels, using a spatial energy function to
balance data fidelity against neighborhood consistency.


Comparison with existing approaches
-------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Property
     - Current approach
     - Proposed approach
   * - Peak selection criterion
     - Highest peak
     - Spatially consistent peak
   * - Reference for comparison
     - Large-kernel median (contaminated by patch interior)
     - Anchor-based propagation (contamination-free)
   * - Territory boundaries
     - May be erroneously smoothed
     - Preserved naturally
   * - Pathological delays (tens of seconds)
     - May be incorrectly corrected
     - Preserved (high dominance ratio → anchors inside territory)
   * - True peak lower than sidelobe
     - Cannot handle
     - Handles — looks for presence, not height
   * - Boundary transition zone
     - Blind correction
     - Stalls gracefully at true boundaries


Configurable parameters
------------------------

The following parameters are exposed on the command line with the defaults
noted:

``--anchor-dominance-threshold`` (default 1.5)
    Minimum ratio C\ :sub:`max` / C\ :sub:`second` for a voxel to qualify
    as an anchor.  Increasing this value makes anchor selection more
    conservative (fewer, more reliable anchors).

``--anchor-peak-search-width`` (default 5.0 seconds)
    Full width of the search window around τ\ :sub:`expected` when looking
    for the true-lag peak in a frontier voxel's correlation function.
    Should be set to less than ``acsidelobelag`` to avoid accepting a
    sidelobe peak as the true peak.


Limitations and known failure modes
-------------------------------------

* **Severe baseline erosion**: If the baseline shift of the correlation
  function is large enough that the true-lag peak is reduced to noise level,
  the growing front will stall even inside an artifact patch — the algorithm
  cannot distinguish this from a true territory boundary.  The voxels are
  then left unchanged (Step 5).  The ``acsidelobeamp`` value gives an upper
  bound on how large this erosion can be under ideal conditions, but noise
  can increase it arbitrarily.

* **No internal anchors in large patches**: An artifact patch with no
  high-dominance voxels anywhere in its interior will be corrected by
  exterior growing if the true-lag peak remains findable.  If the patch is
  so large that the growing front exhausts all penetrable voxels before
  reaching the interior, a residual core may remain uncorrected.  These
  voxels are counted in the diagnostic output.

* **Non-simply-connected geometry**: In complex 3D geometry there may be
  multiple paths from different anchor sets to the same frontier voxel,
  yielding inconsistent τ\ :sub:`expected` predictions.  This is analogous
  to the residue problem in phase unwrapping.  A priority-queue ordering
  by anchor R² mitigates (but does not eliminate) this.

* **The underlying cause is not addressed**: The baseline shift in the
  correlation function that causes the true peak to erode is a consequence
  of autocorrelation structure in both the reference regressor and the voxel
  time series.  The region-growing algorithm corrects the downstream
  consequences of this shift but does not remove the shift itself.
  Preprocessing the correlation functions to flatten their baselines before
  peak finding would be a more fundamental fix, but is beyond the scope of
  this algorithm.


Implementation notes
---------------------

The following existing components are reused:

* ``fitSimFuncMap._detect_shifted_patches()`` and the patchshift section of
  ``fitSimFuncMap.fitSimFunc()`` remain as an earlier-pass first-line
  correction for the obvious cases that the large-kernel median approach
  handles reliably.
* ``simfuncfit._find_and_try_peaks()`` already enumerates all peaks in a
  voxel's correlation function; this is called at each frontier voxel.
* ``fitcorr(despeckling=True, multipeak=True)`` is the refitting call used
  in Step 4.
* The ``corrout`` array is already in memory during ``fitSimFunc()``.
* ``optiondict["acsidelobelag"]`` and ``optiondict["acsidelobeamp"]`` provide
  the sidelobe parameters without additional computation.
