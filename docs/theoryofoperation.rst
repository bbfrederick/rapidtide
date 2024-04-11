..
   Headings are organized in this manner:
   =====
   -----
   ^^^^^
   """""
   '''''

Theory of operation
-------------------
If you're bored enough or misguided enough to be reading this section, you are
my intended audience!

rapidtide
^^^^^^^^^

What is rapidtide trying to do?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Rapidtide attempts to separate an fMRI or NIRS dataset into two components - a
single timecourse that appears throughout the dataset with varying time delays
and intensities in each voxel, and  everything else.  We and others have
observed that a large proportion of the "global mean signal", commonly referred
to as "physiological noise" seen throughout in vivo datasets that quantify time
dependant fluctuations in hemodynamic measures can be well modelled by a single
timecourse with a range of time shifts.  This has been seen in fMRI and NIRS
data recorded througout the brain and body, with time lags generally increasing
at locations farther from the heart along the vasculature.  This appears to be a
signal carried by the blood, as changes in blood oxygenation and/or volume that
propagate with bulk blood flow.  The source of the signal is not known, being
variously attributed to cardiac and respiratory changes over time, changes in
blood CO2, gastric motility, and other sources (for a survey, see [Tong2019]_.)
As biology is complicated, it's probably some mixture of these sources and
others that we may not have considered. No matter what the source of the signal,
this model can be exploited for a number of purposes.

If you're interested in hemodynamics, using  rapidtide to get the time delay in
every voxel gives you a lot of information  that's otherwise hard or impossible
to obtain noninvasively, namely the arrival time of blood in each voxel, and the
fraction of the variance in that voxel  that's accounted for by that moving
signal, which is related to regional CBV (however there's also a factor that's
due to blood oxygenation, so you have  to interpret it carefully).  You can use
this information to understand the blood flow changes arising from vascular
pathology, such as  stroke or moyamoya disease, or to potentially see changes in
blood flow due to a pharmacological intervention. In this case, the moving
signal is not noise - it's the signal of interest.  So the various maps
rapidtide produces can be used to describe hemodynamics.

However, if you are interested in local rather than global hemodynamics,
due to, say, neuronal activation, then this moving signal is rather pernicious
in-band noise.  Global mean regression is often used to remove it, but this is
not optimal - in fact it can generate spurious anticorrelations, which are
not helpful.  Rapidtide will regress out the moving signal, appropriately
delayed in each voxel.  This removes significantly more variance, and also
avoids generating spurious correlations.  For a detailed consideration of this,
look here [Erdogan2016]_.


What is the difference between RIPTiDe and rapidtide?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RIPTiDe (Regressor Interpolation at Progressive Time Delays) is the name of
the technique used for finding and removing time lagged physiological signals
in fMRI data.  In the original RIPTiDe papers, we generated a
set of regressors over a range of different time shifts (starting from a regressor recorded
outside of the brain), and then ran a GLM in FSL using the entire set of regressors.
We realized that this 1) doesn't give you the optimal delay value directly,
which turns out to be a useful thing to know, 2) burns degrees of freedom
unnecessarily, since having one optimally delayed regressor in each voxel gets
you pretty much the same degree of noise removal (this is assuming that in each
voxel there is one and only one pool of delayed blood, which while not true,
is true enough, since almost every voxel is dominated by a single pool of
delayed blood), 3) is slow, since you're doing way more calculation than you
need to, and 4) doesn't necessarily get you the best noise removal, since
the systemic noise signal recorded outside the brain has its own characteristics
and noise mechanisms that may make it diverge somewhat from what is actually
getting into the brain (although on the plus side, it is inarguably non-neuronal, 
so you don't have to have any arguments about slow neuronal waves).

In contrast rapidtide (lets say it means Rapid Time Delay) is the newer faster,
self-contained python program that implements an updated version of the RIPTiDe
algorithm which estimates delay in every voxel and recursively refines an estimate
of the "true" systemic noise signal propagating through the brain by shifting and
merging the voxel timecourses to undo this effect. This refinement procedure is
shown in Figure 5 of Tong, 2019 (reference 6 in the Physiology section below). In recent
years, I've personally become more interested in estimating blood flow in the brain than
denoising resting state data, so a lot of the documentation talks about that, but the
two procedures are tightly coupled, and as the final step, rapidtide does regress
the optimally delayed refined estimate of the systemic noise signal out of the data.
We have found that it works quite well for resting state noise removal while avoiding
the major problems of global signal regression (which we refer to as "static global
signal regression" as opposed to "dynamic global signal regression", which is
what rapidtide does). For a detailed exploration of this topic, see Erdogan, 2016 (also 
in the Physiology section below).


How does rapidtide work?
^^^^^^^^^^^^^^^^^^^^^^^^
In order to perform this task, rapidtide does a number of things:

1. Obtain some initial estimate of the moving signal.
2. Preprocess this signal to emphasize the bloodborne component.
3. Analyze the signal to find and correct, if possible, non-ideal properties
   that may confound the estimation of time delays.
4. Preprocess the incoming dataset to determine which voxels are suitable for
   analysis, and to emphasize the bloodborne component.
5. Determine the time delay in each voxel by finding the time when the voxel
   timecourse has the maximum similarity to the moving signal.
6. Optionally use this time delay information to generate a better estimate of
   the moving signal.
7. Repeat steps 3-7 as needed.
8. Parametrize the similarity between the moving signal and each voxels'
   timecourse, and save these metrics.
9. Optionally regress the voxelwise time delayed moving signal out of the
   original dataset.

Each of these steps has nuances which will be discussed below.


Generation of Masks
"""""""""""""""""""
By default, rapidtide calculates masks dynamically at run time.  There
are 5 masks used: 1) the global mean mask, which determines which voxels
are used to generate the initial global mean regressor, 2) The
correlation mask, which determines which voxels you actually calculate
rapidtide fits in (what you are describing here), 3) the refine mask,
which selects which voxels are used to generate a refined regressor for
the next fitting pass, 4) the offset mask, which determines which voxels are 
used to estimate the "zero" time of the delay distribution, 
and 5) the GLM mask, which determines which
voxels have the rapidtide regressors removed.

Below is a description of how this works currently.  NB: this is not how
I THOUGHT is worked - until I just looked at the code just now.  It
built up over time, and evolved into something that was not quite what I
designed.  I'm going to fix it up, but this what it's doing as of 2.6.1,
which works most of the time, but may not be what you want.

The default behavior is to first calculate the correlation mask using
nilearn.masking.compute_epi_mask with default values.  This is a
complicated function, which I'm using as a bit of a black box.
Documentation for it is here:
https://nilearn.github.io/stable/modules/generated/nilearn.masking.compute_epi_mask.html#nilearn.masking.compute_epi_mask.  
If you have
standard, non-zero-mean fMRI data, it seems to work pretty well, but you
can specify your own mask using --corrmask NAME[:VALSPEC] (include any
non-zero voxels in the file NAME in the mask.  If VALSPEC is provided, only
include voxels with integral values specified by VALSPEC in the mask).
VALSPEC is a comma separated list of integers (1,2,7,12) and/or integer
ranges (2-7,12-15) so you can make masks of complicated combinations of
regions from an atlas.  So for example --corrmask mymask.nii.gz:1,7-9,54
would include any voxels in mymask with values of 1, 7, 8, 9, or 54,
whereas --corrmask mymask.nii.gz would include any non-zero voxels in
mymask.

**For the global mean mask:**
If --globalmeaninclude MASK[:VALSPEC] is specified, include all voxels
selected by MASK[:VALSPEC].  If it is not specified, include all voxels
in the mask.  Then, if --globalmeanexclude MASK[:VALSPEC] is specified,
remove any voxels selected by MASK[:VALSPEC] from the mask.  If it is
not specified, don't change the mask.

**For the refine mean mask:**
If --refineinclude MASK[:VALSPEC] is specified, include all voxels
selected by MASK[:VALSPEC].  If it is not specified, include all voxels
in the correlation mask mask.  Then if --refineexclude MASK[:VALSPEC] is specified,
remove any voxels selected by MASK[:VALSPEC] from the mask.  If it is
not specified, don't change the mask.  Then multiply by corrmask, since
you can't use voxels where rapidtide was not run to do refinement.

**For the offset mask**
If --offsetinclude MASK[:VALSPEC] is specified, include all voxels
selected by MASK[:VALSPEC].  If it is not specified, include all voxels
in the correlation mask.  Then if --offsetexclude MASK[:VALSPEC] is specified,
remove any voxels selected by MASK[:VALSPEC] from the mask.  If it is
not specified, don't change the mask.  Then multiply by corrmask, and use the voxels within
the mask to generate a histogram of delay values.  Calculate the offset of the peak of the delay histogram,
and subtract this value from all delay values within the correlation mask.

**For the GLM mask:**
Include all voxels, unless you are calculating a CVR map, in which caserates other than the TR. Therefore
the first step in moving regressor processing is to resample the moving regressor estimate to match the (oversampled)
data sample rate.

**Temporal filtering:**  By default, all data and moving regressors are temporally bandpass filtered to 0.009-0.15Hz
(our standard definition of the LFO band).  This can be overridden with ``--filterband`` and ``--filterfreqs`` command line
options.

Depending on your data (including pathology), and what you want to accomplish, using the default correlation
mask is not ideal.  For example, if a subject has obvious pathology, you may want to exclude these voxels
from being used to generate the initial global mean signal estimate, or from being used in refinement.


Initial Moving Signal Estimation
""""""""""""""""""""""""""""""""
You can stabilize and improve rapidtide's delay estimation quite a bit by making sure you have a good starting
regressor, estimating the global mean signal from "good" brain regions that don't have wacky delay structures.
While just using the whole brain works well in young, healthy subjects (like the HCP-YA dataset), as people get older,
their delays become weird - my working theory is that over time various routine vascular insults and unhealthy habits
accumulate, leading to increasing heterogeneity between vascular territories (which I like to call "vascular
personality"). So the global mean may be made up of several pools of blood, delayed by up to several seconds
relative to each other, leading to weird autocorrelation in the global mean (essentially, confusing echoes of the
moving signal) that can confuse my delay finding algorithm, because it
invalidates my assumption that the global mean is a good initial estimate of the "true" moving regressor.
One way to combat this is to limit the brain region that you get your initial regressor from, so that you are only
sampling a single "pool" of delays. For example, you
could use a gray matter mask for the global regresor estimation, since white matter has a smaller contribution from
the moving blood signal, and tends to get blood much later than gray matter anyway.  Just add the option
``--globalmeaninclude graymask.nii.gz`` to your rapidtide command line.  If you are using
fmriprep, you can get a gray matter mask using:

::

  fslmaths \
      BIDSHOME/derivatives/fmriprep/sub-XXX/anat/sub-YYY_space-MNI152NLin6Asym_res-2_label-GM_probseg.nii.gz \
      -s 3 \
      -thr 0.25 \
      -bin \
      graymask

If you want to be even more proactive, you could select a more focal brain region that you think has unperturbed circulation.
For an Alzheimer's study that I am currently working on, we ended up starting only from blood in right and
left cerebellar gray matter (freesurfer aparc+aseg regions 8 and 47) on the theory that if circulation in your cerebellum
is too messed up, you're dead, so would not be in the dataset. That made our delay estimates work a lot better.
So we used the freesurfer parcellations from fmriprep, transformed to standard space, to do that
preselection, using the option ``--globalmeaninclude standardspaceaparcasegfilename.nii.gz:8,47``.

fmriprep does not provide a standard space aparc+aseg file - it's in T1 native space at 1mm resolution
(because that's the space freesurfer works in).  Resampling to standard space is easy, BUT you must
remember to use NearestNeighbor
interpolation, or you'll get smeared, averaged boundaries between brain regions, which you REALLY don't want.
This command should get you a ``standardspaceaparcasegfilename.nii.gz`` (you need to have ANTs installed for this):

::

  antsApplyTransforms \
      -d 3 \
      -i BIDSHOME/derivatives/sub-XXX/anat/sub-XXX_desc-aparcaseg_dseg.nii.gz \
      -o BIDSHOME/derivatives/sub-XXX/anat/mymnispace_desc-aparcaseg_dseg.nii.gz \
      -r BIDSHOME/derivatives/sub-XXX/anat/sub-XXX_space-MNI152NLin6Asym_res-2_desc-preproc_T1w.nii.gz \
      --interpolation NearestNeighbor \
      --transform BIDSHOME/derivatives/sub-XXX/anat/sub-XXX_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5


Moving Signal Preprocessing
"""""""""""""""""""""""""""
Before launching into in each pass, we process our moving regressor to make it more amenable to calculations.
This includes the following operations:

**Oversampling:**  In order to simplify delay calculation, rapidtide performs all delay estimation operations
on data with a sample rate of 2Hz or faster.  Since most fMRI is recorded with a TR > 0.5s, this is achieved by
oversampling the data.  The oversampling factor can be specified explicitly 
(using the ``--oversampfac`` command line argument), but if it is
not given, for data with a sample rate of less than 2Hz, all data and regressors
are internally upsampled by the lowest
integral factor that results in a sample rate >= 2Hz.

**Regressor resampling:** In the case where we are using the global mean signal 
as the moving signal, the moving signal estimate and the fMRI data have 
the same sample rate, but if we use external
recordings, such as NIRS or etCO2 timecourses, these will in general have sample
rates other than the TR, and may start before and/or end after the fMRI acquisition.
Therefore the first step in moving regressor processing
is to resample the moving regressor estimate to match the (oversampled)
data sample rate and time range.

**Temporal filtering:** The moving regressor is then filtered to the appropriate frequency range - by default the
LFO band (0.009-0.15Hz).

**Detrending and normalization:** The regressor is detrended to Nth order (N=3 by default), demeaned, and divided
by the standard deviation over time.

Moving Signal Massaging
"""""""""""""""""""""""
Because the moving signal is "noise", we can't select or specify its properties, and sometimes the sLFO signal
you end up with is problematic for one reason or another.  Rapidtide attempts to correct, where possible,
problems in the moving signal that will impair proper delay estimation.  Again, if you're just doing
signal denoising, these are not that important to you.

**Pseudoperiodicity:**  The first potential problem in the sLFO regressor is
pseudoperiodicity.  From time to time, signal energy in the 0.009-0.15 Hz
band will be strongly concentrated in one or more spectral peaks.
This can be completely random, or it can arise due to some pathological or
congenital condition that affects circulation. It seems
for the most part to be purely by chance, as you occasionally see it
when looking at multiple runs in the same subject, where one run is
pseudoperiodic while the rest are not. The effect of this is to cause
the crosscorrelation between the probe signal and voxel timecourses to
have more than one strong correlation peak.  This means that in the
presence of noise, or extreme spectral concentration of the sLFO, the
wrong crosscorrelation peak can appear larger, leading to an incorrect
delay estimation.  This is particularly problematic if the pseudoperiod
is shorter than the reciprocal of the search window (for example, if the
search window for correlation peaks is between -5 and +5 seconds, and
the sLFO has a strong spectral component at 0.1Hz or higher, more than
one correlation peak will occur within the search window).  As the width
of the search range increases, the spectral range of potentially
confounding spectral peaks covers more of the sLFO frequency band.

**Implications of pseudoperiodicity:** The extent to which
pseudoperiodicity is a problem depends on the application.  In the case
of noise removal, where the goal is to remove the global sLFO signal,
and leave the local or networked neuronal signal variance, it turns out
not to be much of a problem at all.  If the sLFO signal in a given voxel
is sufficiently periodic that that the correctly delayed signal is
indistinguishable from the signal one or more periods away, then it
doesn’t matter which signal is removed – the resulting denoised signal
is the same.  As the Host in Westworld asked - "Well if you can't tell, does it matter?"
In this case, no.  Sadly, for those of you care more about hemodynamics than neuronal
activation (raises hand), this is NOT ok, and we have to figure out how to deal with it.

**Mitigation of pseudoperiodicity:** While we continue to work on fully
resolving this issue, we have a number of hackish ways of dealing with this.
First of all, spectral analysis of the sLFO signal allows us to
determine if the signal may be problematic.  Rapidtide checks the
autocorrelation function of the sLFO signal for large sidelobes with
periods within the delay search window and issues a warning when these
signals are present.  Then after delay maps are calculated, they are
processed with an iterative despeckling process analogous to phase
unwrapping.  The delay of each voxel is compared to the median delay of
its neighbors.  If the voxel delay differs by the period of an
identified problematic sidelobe, the delay is switched to the “correct”
value, and refit.  This procedure greatly attenuates, but does not
completely solve, the problem of bad sidelobes.  A more general solution
to the problem of non-uniform spectra will likely improve the
correction.

**Correlation weighting:** Another method I've recently implemented is "regressor weighting" the correlation
function - since we do correlation in the spectral domain, you can normalize the
power spectrum magnitude by the power spectrum of the sLFO regressor - this deemphasizes
spectral peaks.  It helps, but it's not a magic wand.

**Echo cancellation:**  One thing that I keep thinking about is that in the case of pathology causing disparate
delay pools, we are essentially looking at an echo cancellation problem.  We have a driving signal, and it is
corrupted by delayed copies of itself being added in.  This is a problem that Bell Labs solved in the 60s or 70s (well
digitally - I think analog echo cancellation existed long before that).  It seems like I should be able to dust off
some audio library somewhere that would fix this right up, but I haven't found anything yet.  Any bored audio engineers
looking to get involved in a FOSS neuroimaging project :-) ?

Most of the options languishing in the "experimental" group of command line options are partially implemented versions
of various regressor fixes.

Dataset Preprocessing
"""""""""""""""""""""
Prior to processing, I do a few things to the fMRI dataset:

**Spatial filtering:**  While the moving signal can contribute up to 50% of the low frequency variance in gray matter
voxels, it's often MUCH less than that, especially in white matter.  So anything you can do to boost your SNR is a plus.
Spatial filtering works for that - for the most part, the delay time varies quite smoothly over space, since capillary
blood (to which we are most sensitive) moves in a pretty orderly fashion.  Even a small amount of smoothing is
sufficient to boost the quality of the delay maps a lot.  A Gaussian kernel with a radius of ~1/2 the average voxel
dimension in all three axes turns out to be pretty good.  Use ``--spatialfilt SIGMA`` to set the filtering.  Set
SIGMA to -1 to have it set automatically as described above (default), or set SIGMA to the kernel size in mm.
SIGMA=0 turns spatial filtering off.

**Mask, trim to size and reshape:**  Select only the voxels and timpoints that are going to be processed, as
specified by the spatial masks, and the ``--numskip`` and ``--timerange`` options, and reformat the remaining data
into a voxel by time array.  This simplifies all of the subsequent processing.  Spatial filtering (done previously)
and despeckling (managed by mapping lag data back to x, y, z space to check against neighbors)
are the only operations that require us to know the spatial relationship between voxels.

Time delay determination
""""""""""""""""""""""""
This is the core of the program, that actually does the delay determination.  It's currently divided into two parts -
calculation of a time dependant similarity function between the sLFO regressor and each voxel, and then a fitting
step to find the time delay and strength of association between the two.

Signal preparation
``````````````````

Types of similarity function
````````````````````````````
**Crosscorrelation:** The most straightforward way to calculate similarity between two timecourses is crosscorrelation.  It has several
advantages - interpretation is easy - the magnitude of the function ranges from 0 (no similarity) to 1 (timecourses
are identical).  Negative magnitudes mean that the one timecourse is inverted relative to the other.  It is also
extremely fast to calculate in the spectral domain (O(2Nlog2N) rather than O(N2)).  For signals of the length of
typical fMRI scans, calculation in the spectral domain is substantially faster than in the time domain.
However, it does have drawbacks.  First, it
assumes the relationship between the signals is linear.  In practice, this is generally ok for our purposes, but is
not ideal.  More problematic is unpredicatible performance when the SNR is low (as it is in voxels with lower
blood content, such as white matter), which can make the signal harder to
quantify.  Correlation peaks can be very broad due to low pass filtering, autocorrelation and window function choices,
and baseline roll can lead to incorrect peak identification.  This
makes the peak fitting process complicated.  Use ``--similaritymetric correlation`` to select crosscorrelation (default).

**Mutual information:**  Mutual information (MI) is a very different method of quantifying similarity.  It is a measure of
the amount of information you can gain about one signal from the other (yes, I know the definition is about "random
variables", but for our purposes, we mean timecourses).  So, there is no assumption of linearity (or in fact
any assumption whatsoever about the functional form of the relationship).  That's cool, because it really frees you
up in terms of what you can look at (as an aside, I'm not sure why this isn't used more in task based analyses - it
seems like it could get past having to know the exact form of the hemodynamic response function). MI
is especially useful in image registration, for example, lining T2 weighted functional images up with T1 weighted
anatomics.  The cross-MI has some nice properties.

    * It tends to give sharp peaks when signals are aligned, even in cases where the source data is lowpass filtered.
    * As mentioned above, it really doesn't care how signals are related, only that they are.  So you aren't restricted to linear relationships between signals.

So why don't we use it for everything?  A couple of reasons.

    * It's much more computationally expensive than correlation (O(N2) at least).  My implementation of a cross-MI function (which is actually pretty fast) still takes about 10x as long to calculate as crosscorrelation for typical fMRI data.
    * It does not have as straightforward an interpretation as crosscorrelation - while there are "normalized" calculations, "1" does not mean identical, "0" does not mean unrelated, and it's positive definite.  The MI of a signal with itself is the same as the MI of -1 times itself.  For cross-MI, you can really only rely on the fact that you get a maximum when the signals are most aligned.

Use ``--similaritymetric mutualinfo`` to select MI.

**Hybrid similarity:**  I'm kind of proud of this one.  Crosscorrelation is fast and interpretable, but has the
problem of ambiguous time delay values, whereas
cross-MI is very slow and hard to interpret, but quite unambiguous in selecting the best match.  Enter "hybrid similarity" -
Use the crosscorrelation to identify candidate peaks, then calculate the MI only at those peak locations, pick the one
that has the higher MI, and then proceed to the fitting step for full quantification.  This is almost as fast as
straight correlation, but does tend to be more stable. Use ``--similaritymetric hybrid`` to select hybrid similarity.

Peak fitting and quantification
```````````````````````````````


Generating a Better Moving Signal Estimate
""""""""""""""""""""""""""""""""""""""""""
Now that we have an estimate of when the moving regressor arrives at every voxel, we can make a better estimate of the
driving signal.

Voxel selection
```````````````
First we pick the voxels we want to use to generate the new estimate.  We can set the starting mask explicitly using
the ``--refineinclude MASKFILE:VALSPEC`` and ``--refineexclude MASKFILE:VALSPEC`` command line options.  If left unset,
we use all voxels with valid correlation fits.  We can further
tune which voxels are excluded from refinement with the ``--norefinedespeckled``, ``--lagminthresh``,
``--lagmaxthresh``, ``--ampthresh``, and ``--sigmathresh`` options.

Alignment
`````````
In each of the voxels selected for refinement, we first negate the time delay in every voxel and timeshift the
voxel by that amount.
This will have the effect of bringing the portion of the signal in each voxel due to the moving sLFO signal into
alignment.

Prescaling
``````````
We then weight the voxels to use in the fit by first prenormalizing them using their mean, variance, or standard deviation over time,
the inverse of the lag time,
or leave them unweighted.  Selection is via the ``--refineprenorm`` option.  The default is to do no prenormalization.

Timecourse generation
`````````````````````
The new timecourse is then generated from the set of aligned, scaled timecourses using a method specified with ``--refinetype``:

    **(pca - default)** Perform a principal component analysis on the timecourses, reprojecting them onto a reduced set of components (specified by ``--pcacomponents`` - the default is the set explaining >=80% of total variance).  Average the result.

    **(ica)** Perform an independent component analysis on the timecourses, reprojecting them onto a reduced set of components (specified by ``--pcacomponents`` - the default is the set explaining >=80% of total variance).  Average the result.

    **(weighted_average)** Each voxel is scaled with either the correlation strength from the current pass, the square of the correlation strength, or is left unscaled.  This is selected with the ``--refineweighting`` option - the default is "R2".  The timecourses are then averaged.

    **(unweighted average)**  Average the voxels.

Lather, Rinse, Repeat
"""""""""""""""""""""
Now that there is a new starting regressor, repeat the entire process some number of times.  This can be a fixed number
of passes, specified by ``--passes NUMPASSES``.  The default is to do 3 passes.  Alternatively, by specifying
``--convergencethresh THRESH``, the process is repeated until either the MSE between the new sLFO regresssor and the
regressor from the previous pass falls below THRESH, or the number of passes reaches MAX, specified
by ``--maxpasses MAX`` (default is 15).

Save Useful Parameters
""""""""""""""""""""""


Regress Out the Moving Signal
"""""""""""""""""""""""""""""

References
""""""""""

.. [Tong2019] Tong, Y., Hocke, L.M., and Frederick, B.B., Low Frequency
   Systemic Hemodynamic "Noise" in Resting State BOLD fMRI: Characteristics,
   Causes, Implications, Mitigation Strategies, and Applications.
   Front Neurosci, 2019. 13: p. 787.
   \| http://dx.doi.org/10.3389/fnins.2019.00787


.. [Erdogan2016] Erdoğan S, Tong Y, Hocke L, Lindsey K, Frederick B. Correcting
   resting state fMRI-BOLD signals for blood arrival time enhances
   functional connectivity analysis. Front. Hum. Neurosci., 28 June 2016
   \| http://dx.doi.org/10.3389/fnhum.2016.00311
