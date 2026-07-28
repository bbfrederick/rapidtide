unwrapdelay
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
``unwrapchanged`` mask before trusting the output.

Measured performance:
^^^^^^^^^^^^^^^^^^^^^

On the test dataset, processed without ``--despecklepasses`` and without
``--robustdelayfit``, scored against the despeckled run:

===================================================  ====================
method                                               sidelobe errors
===================================================  ====================
rapidtide raw peak picking                           380 (1.45%)
unwrapdelay                                          89 (0.34%)
===================================================  ====================

a 77% reduction, in under a second.  Under controlled injection - corrupting the
similarity function so the sidelobe outranks the main lobe - it recovers about
97% of injected errors even when they are placed in the lowest ``maxcorrsq``
quartile, where real errors actually live.

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

Inputs:
^^^^^^^

The similarity function, ``XXX_desc-corrout_info.nii.gz``.  The mask defaults to
``XXX_desc-corrfit_mask.nii.gz`` if present.

Outputs:
^^^^^^^^

    **maxtimeunwrapped**: the corrected delay map.

    **maxtimenaive**: what naive peak picking would have given, for comparison.

    **unwrapchanged**: mask of voxels assigned something other than their
    strongest peak.

    **unwrapconfidence**: the ambiguity gap between the best and second best
    peak, which is what orders the region growing.

Usage:
^^^^^^

.. argparse::
   :ref: rapidtide.workflows.unwrapdelay._get_parser
   :prog: unwrapdelay
   :func: _get_parser

   Debugging options : @skip
      skip debugging options
