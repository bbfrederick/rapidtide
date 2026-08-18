resolvedelays
-----------

Description:
^^^^^^^^^^^^

Resolves sidelobe ambiguity in a rapidtide delay map by phase unwrapping.

The similarity function of an LFO signal has sidelobes spaced by the dominant
period of the waveform (rapidtide writes its estimate to
``XXX_autocorr_sidelobetime_passN.txt``).  When a sidelobe exceeds the main lobe
in a voxel, peak picking assigns a delay wrong by nearly one full period.  These
errors are large, and because neighboring voxels see the same waveform they fail
together, in coherent patches rather than isolated speckles - which is what a
median filter based despeckler struggles with, since the neighbors it votes with
are wrong too.

The framing is phase unwrapping: the absolute delay is ambiguous modulo the
sidelobe period, while the local gradient is not.  Each voxel offers candidate
delays (the local maxima of its similarity function), and quality guided region
growing assigns each voxel the candidate closest to the value predicted from its
already assigned neighbors, starting from the least ambiguous voxels.

Before you use this, check two things:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Measure your sidelobe.**  It depends on the LFO spectrum of the particular
acquisition and is not a constant.  rapidtide writes its estimate to
``XXX_autocorr_sidelobetime_passN.txt`` and records ``acsidelobelag_passN`` in the
runoptions, leaving it ``None`` when it finds none; the autocorrelation itself is
saved as ``XXX_desc-autocorr_timeseries``.  Two real examples, measured from that
file within the search range:

======================  ===================  ==============================
dataset                 strongest sidelobe   normalized amplitude
======================  ===================  ==============================
sub-RAPIDTIDETEST       13.25 s              0.026
a CVR study subject     12.00 s              0.001  (i.e. none)
======================  ===================  ==============================

With no sidelobe there is no periodic ambiguity to resolve, and large delay
changes are being driven by something else - usually noise repicking the peak
more or less at random inside the search range.  This program will still
"repair" those, but it is then smoothing, not unwrapping.  The run prints a
diagnostic: a genuine alias moves delays in ONE direction by close to one period
(the test dataset gives 0% positive changes, median 14.01 s, IQR 5.24 s), while
noise repicking moves them BOTH ways over a broad range (the CVR subject gives
13% positive, median 12.06 s, IQR 5.91 s, and triggers a warning).

**Long delays are not automatically wrong.**  Vascular pathology can produce
genuine delays of tens of seconds.  What separates real long transit from a
repicking error is not magnitude but spatial behaviour - real delay varies
smoothly from surrounding tissue, an error jumps.  The smoothness prior used here
cannot tell the difference at a discontinuity, so on a patient with focal delay
pathology this program may erase the finding of interest.  Inspect the
``resolvechanged`` mask before trusting the output.

Measured performance:
^^^^^^^^^^^^^^^^^^^^^

On the test dataset, processed without ``--despecklepasses``, scored against the
despeckled run:

===================================================  ====================
method                                               sidelobe errors
===================================================  ====================
rapidtide raw peak picking                           380 (1.45%)
resolvedelays                                          89 (0.34%)
===================================================  ====================

a 77% reduction, in under a second.  Under controlled injection - corrupting the
similarity function so the sidelobe outranks the main lobe - it recovers about
97% of injected errors even when they are placed in the lowest ``maxcorrsq``
quartile, where real errors actually live.

Error propagation, and what fixed it:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Region growing propagates its own mistakes - a voxel assigned a wrapped value
drags its correct neighbors to match - so the method creates new wrapped voxels
while fixing old ones.  Scoring only the errors present in the input hides this
and flatters any aggressive method.

Measured on HCP data with a strong measured sidelobe (13.2 s, amplitude 0.25),
using an alias-specific detector (a voxel is wrapped if it sits one sidelobe
period from its local median, which does not penalise genuine smooth long delays):

=================================  =======  =======  ==========  ===========
variant                            fixed    left     new wrap    net total
=================================  =======  =======  ==========  ===========
raw peak picking                   -        -        -           9591
rapidtide despeckle                7578     2013     651         2664
unwrap, single neighbor predict    8712     879      1696        2575
**unwrap, consensus predict**      **8858** **733**  **952**     **1685**
=================================  =======  =======  ==========  ===========

Predicting from the median over every already assigned neighbor, rather than from
whichever single neighbor popped off the heap, cut new wraps by 44% and moved the
method from a tie with despeckling to 37% fewer residual wrapped voxels.  It also
raised cross-stream consistency from 98.93% to 99.37%.

``--minconfidence`` was tried for the same purpose and does not work: net totals
of 1685, 2449, 1982, 1706, 1685 for floors of 0.0, 0.25, 0.5, 0.75, 0.9.  Every
nonzero setting is neutral or worse, because excluding sources shrinks the
consensus and makes the median less robust.  It defaults to off and is retained
only to document the negative result.

Multiple passes:
^^^^^^^^^^^^^^^^

Simply rerunning the region grow does nothing - it is deterministic and takes no
delay map as input, so a literal second pass is a verified no-op.  Iteration only
means something as feedback: smooth the current solution, then re-snap every voxel
to whichever candidate lies nearest that smoothed field.  That is iterated
conditional modes on a smoothness regularised labelling problem, controlled by
``--numpasses`` (default 3).

=================  ==========  ==================
passes             wrapped     non-alias jumps
=================  ==========  ==================
raw                9591        27052
1 (region grow)    1685        24949
2                  1463        24914
3                  1272        24941
5                  1184        25013
9                  1129        25013
21                 1000        25227
=================  ==========  ==================

Returns diminish sharply after two or three passes, and the count of large
*non*-periodic jumps starts creeping up - the signature of a smoothness prior
beginning to invent structure rather than repair it.  Three is a reasonable
stopping point; running to convergence trades a little more wrap reduction against
slowly accumulating damage elsewhere.

A note on ``--despeckle-patch-detection``:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

rapidtide used to carry a ``--despeckle-patch-detection`` option aimed at the same
failure mode as this program: large connected patches of shifted delay that a small
median filter cannot see.  It was off by default and was never enabled in any of
the comparisons here.  It has since been removed.

What killed it was the constrained flood fill that grew each confirmed patch
inward.  The growth tolerance collapsed to half the despeckle threshold, about
2.5 s, which is comparable to the entire interquartile spread of delays in a
typical brain, and growth never re-checked that a recruited voxel was still an
outlier - it only asked whether the voxel resembled the patch median.  Any patch
whose interior median happened to sit near the bulk of the delay distribution
therefore flooded most of the brain.

Measured on the 26 run UK Biobank calibration set, at the shipped defaults
(threshold 5 s, reference kernel 9, minimum size 10, consistency 0.5):

========================  ==============  ==============  ==================
arm                       detected        after the fill  ended above 50%
========================  ==============  ==============  ==================
despeckle alone           15.8% of mask   33.2% of mask   8 of 26 runs
resolution alone           5.6% of mask   62.7% of mask   18 of 26 runs
resolution + despeckle     5.1% of mask   80.2% of mask   17 of 26 runs
========================  ==============  ==============  ==================

Medians, and the run to run spread is enormous - the expansion factor ranges from
0.00 to 51.7.  Enabling the option would sometimes have done nothing and sometimes
have refit four fifths of the brain with the fit window pinned to +/-2.5 s around a
patch exterior median.

Note also that resolution had already removed 61% of what the detector flags
(519944 voxels down to 200596 across the set), so the residual the option was aimed
at had shrunk substantially before the flood fill was even considered.

On the choice of prior:
^^^^^^^^^^^^^^^^^^^^^^^

``--prior smooth`` (the default) predicts that a voxel resembles its neighbors.
``--prior flow`` predicts using grad(tau) = v/|v|**2 from the ``corrflow`` optical
flow field.

The flow prior was the original motivation, on the argument that a sidelobe of a
travelling wave moves at the same velocity as the main lobe, so optical flow is
immune to the ambiguity that defeats peak picking.  That argument is correct, but
it turns out not to matter much here: the sidelobe period is about 13.8 s while
the median delay difference between adjacent voxels is 0.24 s, a ratio of nearly
60, so the ambiguity is enormous compared to the local gradient and almost any
sensible local prior picks the right candidate.  Measured head to head, flow gave
84-100 residual errors against 89-99 for smooth across several injection
conditions - no reliable advantage, at the cost of computing the flow field.

Use ``--prior flow`` if your sidelobe spacing is small relative to your delay
gradients, which is where it should begin to pay.

What it actually does:
^^^^^^^^^^^^^^^^^^^^^^

The name describes a special case.  A 16 run paired calibration on HCP data, each
run processed with and without unwrapping and spanning the ambiguity range into
its bottom quartile, improved the delay map in **16 of 16 runs** (median 57 percent
fewer outlier voxels, Wilcoxon p=3e-5):

==========  ===  ==================  ============  ========  ==================
band         n    ambiguity           median gain   better    non-alias jumps
==========  ===  ==================  ============  ========  ==================
A            4   0.0152-0.0160        +43.7%        4/4       -13.4%
B            4   0.0301-0.0302        +59.3%        4/4       -22.5%
C            4   0.0455-0.0462        +57.8%        4/4       -21.9%
D            4   0.0652-0.0664        +73.5%        4/4       -21.4%
==========  ===  ==================  ============  ========  ==================

Band A was chosen to be the negative control - bottom quartile of the ambiguity
distribution, no measurable sidelobe - and still gained 44 percent.  Non-periodic
discontinuities fell in every run, so the gain is not smoothing.

The mechanism is not periodic unwrapping on most runs.  Median change was 1.3 s
with only 36 percent of changes positive, i.e. two sided and small.  A genuine
sidelobe wrap is one sided at close to one full period, which is what the two
runs with a measurable sidelobe showed (15-21 percent positive, ~6 s median).
Elsewhere the method is repairing ordinary delay fitting errors by picking a
better candidate peak.

Consequently there is **no sidelobe gate by default**.  The benefit does scale
with ambiguity (Spearman +0.568, p=0.022), so the ambiguity measure predicts how
much you gain, not whether you gain.

Use inside rapidtide:
^^^^^^^^^^^^^^^^^^^^^

The same algorithm is available inside the main rapidtide pipeline via
``--resolvedelays``.  It runs once per pass, on every pass, immediately before
despeckling.  It is not gated on anything: gating on the measured sidelobe
amplitude was the original design, and paired calibrations on 16 HCP and 26 UK
Biobank runs refuted the assumption behind it, so the gate was removed.

Progress is logged and the per pass count of reassigned voxels is recorded in the
runoptions as ``resolved_passN``.  Use ``--saveresolvemaps`` to write out what the
final pass changed: the map going in, the shift applied, and the voxels affected.

Ordering matters, and it is not symmetric.  Unwrapping runs BEFORE despeckling and
both run.  Despeckle first then unwrap is worse than unwrapping alone, because
unwrap can only select among local maxima of the similarity function and so
silently discards despeckle's refit values wherever those are not peaks.  Unwrap
first then despeckle is better than either alone: unwrap makes the discrete lobe
choice, then despeckle refits the residual local outliers, which unwrap cannot do
because it can never produce a non-peak value.

Measured on five HCP runs with a real sidelobe (wrapped voxels):

==================  =======  ===========  ========  ==================  ======
run                  naive    despeckle    unwrap    unwrap+despeckle    gain
==================  =======  ===========  ========  ==================  ======
102614 REST1_RL       14917         3334       661                 489   +26.0%
100307 REST2_RL       13050         2664       490                 380   +22.4%
102513 REST2_RL        8027          844       194                 166   +14.4%
100307 REST1_RL       15444         2034       447                 337   +24.6%
134627 REST1_LR        9424         2295      1110                 925   +16.7%
==================  =======  ===========  ========  ==================  ======

Note that the despeckle step in that table is emulated from the derivatives - the
raw HCP data was not available locally to rerun the pipeline.  The emulation
reproduced rapidtide's real despeckled map to 94 percent within 2 s, and was
slightly WEAKER than the real thing (2725 versus 2295 wrapped on 134627), so the
composed figures are likely conservative.

Multi-subject evidence:
^^^^^^^^^^^^^^^^^^^^^^^

Across 40 HCP runs (10 subjects x REST1/REST2 x LR/RL), with the sidelobe measured
per run from ``desc-autocorr_timeseries``:

=========================  ===  =======  ===========  ========  ============
sidelobe amplitude          n    naive    despeckle    unwrap    unwrap/desp
=========================  ===  =======  ===========  ========  ============
strong, >0.15                4    13666         2132       388           0.21
moderate, 0.05-0.15          6     7290         2264       598           0.30
weak, 0.02-0.05              2     5662         1635       314           0.19
none, <=0.02                28     3696         1196       750            n/a
=========================  ===  =======  ===========  ========  ============

Median wrapped voxels.  Within the 10 in-domain runs unwrapping won in 10 of 10
(Wilcoxon p=0.002), a median 77 percent reduction against despeckling.

An independent check, free of any smoothness based criterion: taking pairs of
repeat acquisitions of the same subject and looking only at voxels where exactly
one run wrapped, despeckling brought 0.486 of them into agreement and unwrapping
0.593, better in 57 of 60 pairs (Wilcoxon p=2e-11).

Out of domain (28 runs with no usable sidelobe) unwrapping changed only 2.4
percent of voxels and did not add spurious structure, which is what the gate is
there to guarantee.

Two cautions from that survey.  Only about a quarter of runs had a usable sidelobe
at all - rapidtide's own detector fired on 6 of 40 - so this is not a common case.
And sidelobe presence tracked phase encode direction strongly (8 of 20 RL runs
above 0.05 versus 2 of 20 LR), which is worth understanding in its own right.

Use inside rapidtide:
^^^^^^^^^^^^^^^^^^^^^

The same algorithm is available inside the main rapidtide pipeline via
``--resolvedelays``.  It runs once per pass, on every pass, immediately before
despeckling.  It is not gated on anything: gating on the measured sidelobe
amplitude was the original design, and paired calibrations on 16 HCP and 26 UK
Biobank runs refuted the assumption behind it, so the gate was removed.

Progress is logged, and the per pass count of reassigned voxels is recorded in the
runoptions as ``resolved_passN``.  Use ``--saveresolvemaps`` to write out what the
final pass changed: the map going in, the shift applied, and the voxels affected.

Why replace rather than follow despeckling: running despeckle first and unwrapping
afterwards measured worse than unwrapping alone.  Unwrap can only select among
local maxima of the similarity function, so wherever despeckle has produced a
refit value that is not a peak, unwrap silently discards it - the two do not
compose in that order.

Multi-subject evidence:
^^^^^^^^^^^^^^^^^^^^^^^

Across 40 HCP runs (10 subjects x REST1/REST2 x LR/RL), with the sidelobe measured
per run from ``desc-autocorr_timeseries``.  Median wrapped voxels:

=========================  ===  =======  ===========  ========  ============
sidelobe amplitude          n    naive    despeckle    unwrap    unwrap/desp
=========================  ===  =======  ===========  ========  ============
strong, >0.15                4    13666         2132       388           0.21
moderate, 0.05-0.15          6     7290         2264       598           0.30
weak, 0.02-0.05              2     5662         1635       314           0.19
none, <=0.02                28     3696         1196       750            n/a
=========================  ===  =======  ===========  ========  ============

Within the 10 in-domain runs unwrapping won in 10 of 10 (Wilcoxon p=0.002), a
median 77 percent reduction against despeckling.

An independent check, free of any smoothness based criterion: taking pairs of
repeat acquisitions of the same subject and looking only at voxels where exactly
one run wrapped, despeckling brought 0.486 of them into agreement and unwrapping
0.593 - better in 57 of 60 pairs (Wilcoxon p=2e-11).

Out of domain (28 runs with no usable sidelobe) unwrapping changed only 2.4
percent of voxels and did not add spurious structure, which is what the gate
exists to guarantee.

Two cautions from that survey.  Only about a quarter of runs had a usable sidelobe
at all - rapidtide's own detector fired on 6 of 40 - so this is not the common
case.  And sidelobe presence tracked phase encode direction strongly (8 of 20 RL
runs above 0.05 versus 2 of 20 LR), which is worth understanding in its own right.

Inputs:
^^^^^^^

The similarity function, ``XXX_desc-corrout_info.nii.gz``.  The mask defaults to
``XXX_desc-corrfit_mask.nii.gz`` if present.

Outputs:
^^^^^^^^

    **maxtimeresolved**: the corrected delay map.

    **maxtimenaive**: what naive peak picking would have given, for comparison.

    **resolvechanged**: mask of voxels assigned something other than their
    strongest peak.

    **resolveconfidence**: the ambiguity gap between the best and second best
    peak, which is what orders the region growing.

Usage:
^^^^^^

.. argparse::
   :ref: rapidtide.workflows.resolvedelays._get_parser
   :prog: resolvedelays
   :func: _get_parser

   Debugging options : @skip
      skip debugging options
