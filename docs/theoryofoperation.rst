..
   Headings are organized in this manner:
   =====
   -----
   ^^^^^
   """""
   '''''

Theory of operation
===================

If you're bored enough or misguided enough to be reading this section,
you are my intended audience!


What is rapidtide trying to do?
-------------------------------

Rapidtide attempts to separate an fMRI or NIRS dataset into two components -
a single timecourse that appears throughout the dataset with varying time delays and
intensities in each voxel,
and everything else.
We and others have observed that a large proportion of the "global mean signal",
commonly referred to as "physiological noise" seen throughout in vivo datasets that
quantify time dependent fluctuations in hemodynamic measures can be well modelled by
a single timecourse with a range of time shifts.
This has been seen in fMRI and NIRS data recorded throughout the brain and body,
with time lags generally increasing at locations farther from the heart along the vasculature.
This appears to be a signal carried by the blood,
as changes in blood oxygenation and/or volume that propagate with bulk blood flow.
The source of the signal is not known,
being variously attributed to cardiac and respiratory changes over time,
changes in blood CO2, gastric motility, and other sources
(for a survey, see :footcite:t:`tong2019low`).
As biology is complicated, it's probably some mixture of these sources and
others that we may not have considered.
No matter what the source of the signal,
this model can be exploited for a number of purposes.

If you're interested in hemodynamics,
using rapidtide to get the time delay in every voxel gives you a lot of information
that's otherwise hard or impossible to obtain noninvasively,
namely the arrival time of blood in each voxel,
and the fraction of the variance in that voxel that's accounted for by that moving signal,
which is related to regional CBV
(however there's also a factor that's due to blood oxygenation, so you have to interpret it carefully).
You can use this information to understand the blood flow changes arising from vascular pathology,
such as stroke or moyamoya disease,
or to potentially see changes in blood flow due to a pharmacological intervention.
In this case, the moving signal is not noise - it's the signal of interest.
So the various maps rapidtide produces can be used to describe hemodynamics.

However, if you are interested in local rather than global hemodynamics,
due to, say, neuronal activation,
then this moving signal constitutes rather pernicious in-band noise.
Global mean regression is often used to remove it,
but this is not optimal - in fact it can generate spurious anticorrelations,
which are not at all helpful.
Rapidtide will regress out the moving signal, appropriately delayed in each voxel.
This removes significantly more variance, and also avoids generating spurious correlations.
For a detailed consideration of this, see :footcite:t:`erdougan2016correcting`.


What is the difference between RIPTiDe and rapidtide?
-----------------------------------------------------

RIPTiDe (Regressor Interpolation at Progressive Time Delays) is the name of
the technique used for finding and removing time lagged physiological signals in fMRI data.
In the original RIPTiDe papers,
we generated a set of regressors over a range of different time shifts
(starting from a regressor recorded outside of the brain),
and then ran a GLM in FSL using the entire set of regressors.
We realized that this
1) doesn't give you the optimal delay value directly,
which turns out to be a useful thing to know,
2) burns degrees of freedom unnecessarily,
since having one optimally delayed regressor in each voxel gets you pretty much
the same degree of noise removal
(this is assuming that in each voxel there is one and only one pool of delayed blood,
which while not true, is true enough,
since almost every voxel is dominated by a single pool of delayed blood),
3) is slow, since you're doing way more calculation than you need to,
and 4) doesn't necessarily get you the best noise removal,
since the systemic noise signal recorded outside the brain has its own characteristics
and noise mechanisms that may make it diverge somewhat from what is actually
getting into the brain
(although on the plus side, it is inarguably non-neuronal,
so you don't have to have any arguments about slow neuronal waves).

In contrast rapidtide (let's say it means Rapid Time Delay) is the newer, faster,
self-contained Python program that implements an updated version of the RIPTiDe
algorithm which estimates delay in every voxel and recursively refines an estimate
of the "true" systemic noise signal propagating through the brain by shifting and
merging the voxel timecourses to undo this effect.
This refinement procedure is shown in Figure 5 of :footcite:t:`tong2019low`.
In recent years,
I've personally become more interested in estimating blood flow in the brain than
denoising resting state data,
so a lot of the documentation talks about that,
but the two procedures are tightly coupled,
and, as the final step,
rapidtide does regress the optimally delayed refined estimate of the systemic noise signal out of the data.
We have found that it works quite well for resting state noise removal while avoiding
the major problems of global signal regression
(which we refer to as "static global signal regression" as opposed to
"dynamic global signal regression", which is what rapidtide does).
For a detailed exploration of this topic,
we refer you again to :footcite:t:`erdougan2016correcting`.


How does rapidtide work?
------------------------

In order to perform this task, rapidtide does a number of things:

1. Obtain some initial estimate of the moving signal.
2. Preprocess this signal to selectively emphasize the bloodborne component.
3. Analyze the signal to find and correct, if possible, non-ideal properties
   that may confound the estimation of time delays.
4. Preprocess the incoming dataset to determine which voxels are suitable for
   analysis, and to emphasize the bloodborne component.
5. Determine the time delay in each voxel by finding the time when the voxel
   timecourse has the maximum similarity to the moving signal.
6. Optionally use this time delay information to generate a better estimate of the moving signal.
7. Repeat steps 3-7 as needed.
8. Parametrize the similarity between the moving signal and each voxels'
   timecourse, and save these metrics.
9. Optionally regress the voxelwise time delayed moving signal out of the original dataset.

Each of these steps (and substeps) has nuances which will be discussed below.


Generation of Masks
^^^^^^^^^^^^^^^^^^^

By default, rapidtide calculates masks dynamically at run time.
There are 5 masks used:

1. The global mean mask, which determines which voxels
   are used to generate the initial global mean regressor
2. The correlation mask, which determines which voxels you actually calculate
   rapidtide fits in (what you are describing here)
3. The refine mean mask, which selects which voxels are used to generate a refined regressor for
   the next fitting pass
4. The offset mask, which determines which voxels are used to estimate the "zero" time of
   the delay distribution
5. The GLM mask, which determines which voxels have the rapidtide regressors removed

Below is a description of how this works currently.
NB: this is not how I THOUGHT is worked - until I just looked at the code just now.
It built up over time, and evolved into something that was not quite what I designed.
I'm going to fix it up, but this what it's doing as of 2.6.1,
which works most of the time, but may not be what you want.

The default behavior is to first calculate the correlation mask using
nilearn.masking.compute_epi_mask with default values.  This is a
complicated function, which I'm using as a bit of a black box.
Documentation for it is here:
https://nilearn.github.io/stable/modules/generated/nilearn.masking.compute_epi_mask.html#nilearn.masking.compute_epi_mask.
If you have standard, non-zero-mean fMRI data, it seems to work pretty well,
but you can specify your own mask using ``--corrmask NAME[:VALSPEC]``
(include any non-zero voxels in the file NAME in the mask.
If VALSPEC is provided, only include voxels with integral values specified by VALSPEC in the mask).
VALSPEC is a comma separated list of integers (1,2,7,12) and/or integer
ranges (2-7,12-15) so you can make masks of complicated combinations of
regions from an atlas.
So for example ``--corrmask mymask.nii.gz:1,7-9,54`` would include any voxels in mymask
with values of 1, 7, 8, 9, or 54,
whereas ``--corrmask mymask.nii.gz`` would include any non-zero voxels in mymask.

.. tip::

   The ``--brainmask`` argument will automatically set all five of these masks,
   but each can be overridden with the individual mask parameters.

.. tip::

   The ``--graymattermask`` argument will automatically set both the global mean mask and the offset mask,
   but each can be overridden with the individual mask parameters.

**For the global mean mask:**
If ``--globalmeaninclude MASK[:VALSPEC]`` is specified, include all voxels selected by ``MASK[:VALSPEC]``.
If it is not specified, include all voxels in the mask.
Then, if ``--globalmeanexclude MASK[:VALSPEC]`` is specified,
remove any voxels selected by ``MASK[:VALSPEC]`` from the mask.
If it is not specified, don't change the mask.

**For the refine mean mask:**
If ``--refineinclude MASK[:VALSPEC]`` is specified, include all voxels selected by ``MASK[:VALSPEC]``.
If it is not specified, include all voxels in the correlation mask mask.
Then if ``--refineexclude MASK[:VALSPEC]`` is specified,
remove any voxels selected by ``MASK[:VALSPEC]`` from the mask.
If it is not specified, don't change the mask.
Then multiply by the correlation mask,
since you can't use voxels where rapidtide was not run to do refinement.

**For the offset mask**
If ``--offsetinclude MASK[:VALSPEC]`` is specified, include all voxels selected by ``MASK[:VALSPEC]``.
If it is not specified, include all voxels in the correlation mask.
Then if ``--offsetexclude MASK[:VALSPEC]`` is specified,
remove any voxels selected by`` MASK[:VALSPEC]`` from the mask.
If it is not specified, don't change the mask.
Then multiply by the correlation mask,
and use the voxels within the mask to generate a histogram of delay values.
Calculate the offset of the peak of the delay histogram,
and subtract this value from all delay values within the correlation mask.

**For the GLM mask:**
Include all voxels, unless you are calculating a CVR map, in which case rates other than the TR.
Therefore the first step in moving regressor processing is to resample the moving regressor
estimate to match the (oversampled) data sample rate.

**Temporal filtering:**
By default, all data and moving regressors are temporally bandpass filtered to 0.009-0.15 Hz
(our standard definition of the LFO band).
This can be overridden with ``--filterband`` and ``--filterfreqs`` command line options.

Depending on your data (including pathology),
and what you want to accomplish, using the default correlation mask is not ideal.
For example, if a subject has obvious pathology,
you may want to exclude these voxels from being used to generate the initial global mean signal estimate,
or from being used in refinement.


Initial Moving Signal Estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can stabilize and improve rapidtide's delay estimation quite a bit by
making sure you have a good starting regressor,
estimating the global mean signal from "good" brain regions that don't have wacky delay structures.
While just using the whole brain works well in young, healthy subjects (like the HCP-YA dataset),
as people get older, their delays become weird -
my working theory is that over time various routine vascular insults and unhealthy habits accumulate,
leading to increasing heterogeneity between vascular territories
(which I like to call "vascular personality").
So the global mean may be made up of several pools of blood,
delayed by up to several seconds relative to each other,
leading to weird autocorrelation in the global mean
(essentially, confusing echoes of the moving signal)
that can confuse my delay finding algorithm,
because it invalidates my assumption that the global mean is a good initial estimate of the
"true" moving regressor.
One way to combat this is to limit the brain region that you get your initial regressor from,
so that you are only sampling a single "pool" of delays.
For example, you could use a gray matter mask for the global regresor estimation,
since white matter has a smaller contribution from the moving blood signal,
and tends to get blood much later than gray matter anyway.

Just add the option ``--graymattermask graymask.nii.gz`` to your rapidtide command line.
If you are using fMRIPrep, you can use the gray matter probabilistic map directly,
as rapidtide will threshold it automatically:

.. code-block:: bash

    rapidtide \
        ... \
        --brainmask sub-XXX/anat/sub-XXX_space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz \
        --graymattermask sub-XXX/anat/sub-XXX_space-MNI152NLin6Asym_res-2_desc-GM_probseg.nii.gz \
        --whitemattermask sub-XXX/anat/sub-XXX_space-MNI152NLin6Asym_res-2_desc-WM_probseg.nii.gz

If you want to be even more proactive,
you could select a more focal brain region that you think has unperturbed circulation.
For an Alzheimer's study that I am currently working on,
we ended up starting only from blood in right and left cerebellar gray matter
(freesurfer aparc+aseg regions 8 and 47)
on the theory that if circulation in your cerebellum is too messed up,
you're dead, so would not be in the dataset.
That made our delay estimates work a lot better.
So we used the freesurfer parcellations from fMRIPrep, transformed to standard space,
to do that preselection,
using the option ``--globalmeaninclude sub-XXX_space-MNI152NLin6Asym_res-2_desc-aparcaseg_dseg.nii.gz:8,47``.

fMRIPrep does not provide a standard space aparc+aseg file - it's in T1 native space at 1mm resolution
(because that's the space freesurfer works in).
Resampling to standard space is easy, BUT you must remember to use NearestNeighbor interpolation
(or GenericLabel if you're using antsApplyTransforms),
or you'll get smeared, averaged boundaries between brain regions, which you REALLY don't want.
This command should get you a standard-space aparc+aseg file named
``sub-XXX_space-MNI152NLin6Asym_res-2_desc-aparcaseg_dseg.nii.gz``
(you need to have ANTs installed for this):

::

  antsApplyTransforms \
      -d 3 \
      -i BIDSHOME/derivatives/sub-XXX/anat/sub-XXX_desc-aparcaseg_dseg.nii.gz \
      -o BIDSHOME/derivatives/sub-XXX/anat/sub-XXX_space-MNI152NLin6Asym_res-2_desc-aparcaseg_dseg.nii.gz \
      -r BIDSHOME/derivatives/sub-XXX/anat/sub-XXX_space-MNI152NLin6Asym_res-2_desc-preproc_T1w.nii.gz \
      --interpolation GenericLabel \
      --transform BIDSHOME/derivatives/sub-XXX/anat/sub-XXX_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5


Moving Signal Preprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before launching into in each pass,
we process our moving regressor to make it more amenable to calculations.
This includes the following operations:

**Oversampling:**
In order to simplify delay calculation,
rapidtide performs all delay estimation operations on data with a sample rate of 2Hz or faster.
Since most fMRI is recorded with a TR > 0.5s, this is achieved by oversampling the data.
The oversampling factor can be specified explicitly
(using the ``--oversampfac`` command line argument),
but if it is not given, for data with a sample rate of less than 2Hz,
all data and regressors are internally upsampled by the lowest
integral factor that results in a sample rate >= 2Hz.

**Regressor resampling:**
In the case where we are using the global mean signal as the moving signal,
the moving signal estimate and the fMRI data have the same sample rate,
but if we use external recordings, such as NIRS or etCO2 timecourses,
these will in general have sample rates other than the TR,
and may start before and/or end after the fMRI acquisition.
Therefore the first step in moving regressor processing
is to resample the moving regressor estimate to match the (oversampled)
data sample rate and time range.

**Temporal filtering:**
The moving regressor is then filtered to the appropriate frequency range -
by default the LFO band (0.009-0.15Hz).

**Detrending and normalization:**
The regressor is detrended to Nth order (N=3 by default), demeaned,
and divided by the standard deviation over time.

**Windowing:**
We apply a window function to the regressor to improve the correlation properties.
By default, this is a Hamming window,
but you can also select Hann, Blackman-Harris, or None, with the ``--windowfunc`` argument.

**Zero padding:**
The regressor is zero padded on each end to twice its length,
so that we will be doing a linear rather than circular correlation
(you can select circular correlation with ``--corrtype``, but I wouldn't recommend it).


Moving Signal Massaging
^^^^^^^^^^^^^^^^^^^^^^^

Because the moving signal is "noise", we can't select or specify its properties,
and sometimes the sLFO signal you end up with is problematic for one reason or another.
Rapidtide attempts to correct, where possible,
problems in the moving signal that will impair proper delay estimation.
Again, if you're just doing signal denoising, these are not that important to you.

**Pseudoperiodicity:**
The first potential problem in the sLFO regressor is pseudoperiodicity.
From time to time,
signal energy in the 0.009-0.15 Hz band will be strongly concentrated in one or more spectral peaks.
This can be completely random,
or it can arise due to some pathological or congenital condition that affects circulation.
It seems for the most part to be purely by chance,
as you occasionally see it when looking at multiple runs in the same subject,
where one run is pseudoperiodic while the rest are not.
The effect of this is to cause the crosscorrelation between the probe signal and voxel
timecourses to have more than one strong correlation peak.
This means that in the presence of noise, or extreme spectral concentration of the sLFO,
the wrong crosscorrelation peak can appear larger,
leading to an incorrect delay estimation.
This is particularly problematic if the pseudoperiod is shorter than the reciprocal of the search window
(for example, if the search window for correlation peaks is between -5 and +5 seconds,
and the sLFO has a strong spectral component at 0.1Hz or higher,
more than one correlation peak will occur within the search window).
As the width of the search range increases,
the spectral range of potentially confounding spectral peaks covers more of the sLFO frequency band.

**Implications of pseudoperiodicity:**
The extent to which pseudoperiodicity is a problem depends on the application.
In the case of noise removal, where the goal is to remove the global sLFO signal,
and leave the local or networked neuronal signal variance,
it turns out not to be much of a problem at all.
If the sLFO signal in a given voxel is sufficiently periodic that that the correctly delayed signal
is indistinguishable from the signal one or more periods away,
then it doesn't matter which signal is removed -
the resulting denoised signal is the same.
As the Host in Westworld asked - "Well if you can't tell, does it matter?"
In this case, no.
Sadly, for those of you care more about hemodynamics than neuronal activation (raises hand),
this is NOT ok, and we have to figure out how to deal with it.

**Mitigation of pseudoperiodicity:**
While we continue to work on fully resolving this issue,
we have a number of hackish ways of dealing with this.
First of all, spectral analysis of the sLFO signal allows us to
determine if the signal may be problematic.
Rapidtide checks the autocorrelation function of the sLFO signal for large sidelobes
with periods within the delay search window and issues a warning when these signals are present.
Then after delay maps are calculated,
they are processed with an iterative despeckling process analogous to phase unwrapping.
The delay of each voxel is compared to the median delay of its neighbors.
If the voxel delay differs by the period of an identified problematic sidelobe,
the delay is switched to the “correct” value, and refit.
This procedure greatly attenuates, but does not completely solve, the problem of bad sidelobes.
A more general solution to the problem of non-uniform spectra will likely improve the correction.

**Correlation weighting:**
Another method I've recently implemented is "regressor weighting" the correlation function -
since we do correlation in the spectral domain,
you can normalize the power spectrum magnitude by the power spectrum of the sLFO regressor -
this deemphasizes spectral peaks.
It helps, but it's not a magic wand.

**Echo cancellation:**
One thing that I keep thinking about is that in the case of pathology causing disparate delay pools,
we are essentially looking at an echo cancellation problem.
We have a driving signal, and it is corrupted by delayed copies of itself being added in.
This is a problem that Bell Labs solved in the '60s or '70s
(well digitally - I think analog echo cancellation existed long before that).
It seems like I should be able to dust off some audio library somewhere that would fix this right up,
but I haven't found anything yet.
Any bored audio engineers looking to get involved in a FOSS neuroimaging project :-) ?

Most of the options languishing in the "experimental" group of command line options are partially
implemented versions of various regressor fixes.


Dataset Preprocessing
^^^^^^^^^^^^^^^^^^^^^

Prior to processing, I do a few things to the fMRI dataset:

**Spatial filtering:**
While the moving signal can contribute up to 50% of the low frequency variance in gray matter voxels,
it's often MUCH less than that, especially in white matter.
So anything you can do to boost your SNR is a plus.
Spatial filtering works for that -
for the most part, the delay time varies quite smoothly over space,
since capillary blood (to which we are most sensitive) moves in a pretty orderly fashion.
Even a small amount of smoothing is sufficient to boost the quality of the delay maps a lot.
A Gaussian kernel with a radius of ~1/2 the average voxel
dimension in all three axes turns out to be pretty good.
Use ``--spatialfilt SIGMA`` to set the filtering.
Set SIGMA to -1 to have it set automatically as described above (default),
or set SIGMA to the kernel size in mm.
SIGMA=0 turns spatial filtering off.

**Mask, trim to size and reshape:**
Select only the voxels and timpoints that are going to be processed,
as specified by the spatial masks, and the ``--numskip`` and ``--timerange`` options,
and reformat the remaining data into a voxel by time array.
This simplifies all of the subsequent processing.
Spatial filtering (done previously) and despeckling
(managed by mapping lag data back to x, y, z space to check against neighbors)
are the only operations that require us to know the spatial relationship between voxels.


Significance threshold estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This step is placed where it is done in the processing stream,
but involves procedures described below.

Estimating the significance threshold for the fitted crosscorrelation measurements done below is not
straightforward.
While there is a standard relationship to convert correlation coefficient R to p for
a given timecourse length,
this assumes that you performing a Pearsonn correlation of truly random signals
(i.e. Gaussian random signals with white noise power spectra).
But the sLFO signals are severely band limited, so if you use these formulae,
you will dramatically overestimate the significance of your correlations.
Moreover, we are selecting the peak of a crosscorrelation over a range of delays,
which will further inflate the values.
There are analytical ways of adjusting for this, but they are tedious -
Monte Carlo simulation by performing and fitting a set of crosscorrelations of the sLFO
regressor with scrambled,
filtered versions of itself are more straightforward
(this is described in :footcite:t:`hocke2016comparison`).
Prior to each pass, we do NREPS of these sham correlations
(NREPS=10000 by default - adjust with ``--numnull NREPS``.
Set to 0 to disable significance estimation).
The p<0.05, p<0.01, and p<0.005 significance thresholds are estimated
by fitting the set of null correlations to a Johnson SB distribution
(the functional form which we empirically found best fits the data).


Time delay determination
^^^^^^^^^^^^^^^^^^^^^^^^

This is the core of the program, that actually does the delay determination.
It's currently divided into two parts -
calculation of a time dependent similarity function between the sLFO regressor and each voxel
(currently using one of three methods),
and then a fitting step to find the peak time delay and strength of association between the two signals.


Signal preparation
""""""""""""""""""

Prior to processing, each timecourse is processed in the same way as the moving regressor
(oversampling, filtering, detrending, applying the same window function used on the reference regressor,
and zeropadding the ends).


Types of similarity function
""""""""""""""""""""""""""""

**Crosscorrelation:**
The most straightforward way to calculate similarity between two timecourses is crosscorrelation.
It has several advantages - interpretation is easy - the magnitude of the function ranges from 0
(no similarity) to 1 (timecourses are identical).
Negative magnitudes mean that the one timecourse is inverted relative to the other.
It is also extremely fast to calculate in the spectral domain
(O(2Nlog2N) rather than O(N2)).
For signals of the length of typical fMRI scans,
calculation in the spectral domain is substantially faster than in the time domain.
However, it does have drawbacks.
First, it assumes the relationship between the signals is linear.
In practice, this is generally ok for our purposes, but is not ideal.
More problematic is unpredictable behavior when the SNR is low
(as it is in voxels with lower blood content, such as white matter),
which can make the signal harder to quantify (more below).

Use ``--similaritymetric correlation`` to select crosscorrelation (default).

**Mutual information:**
Mutual information (MI) is a very different method of quantifying similarity.
It is a measure of the amount of information you can gain about one signal from the other
(yes, I know the definition is about "random variables", but for our purposes, we mean timecourses).
So, there is no assumption of linearity
(or in fact any assumption whatsoever about the functional form of the relationship).
That's cool, because it really frees you up in terms of what you can look at
(as an aside, I'm not sure why this isn't used more in task based analyses -
it seems like it could get past having to know the exact form of the hemodynamic response function).
MI is especially useful in image registration, for example,
lining T2 weighted functional images up with T1 weighted anatomics.
The cross-MI has some nice properties.

   *  It tends to give sharp peaks when signals are aligned,
      even in cases where the source data is lowpass filtered.
   *  As mentioned above, it really doesn't care how signals are related, only that they are.
      So you aren't restricted to linear relationships between signals.

So why don't we use it for everything?  A couple of reasons.

   *  It's much more computationally expensive than correlation (O(N2) at least).
      My implementation of a cross-MI function (which is actually pretty fast)
      still takes about 10x as long to calculate as crosscorrelation for typical fMRI data.
   *  It does not have as straightforward an interpretation as crosscorrelation -
      while there are "normalized" calculations, "1" does not mean identical,
      "0" does not mean unrelated, and it's positive definite.
      The MI of a signal with itself is the same as the MI of -1 times itself.
      For cross-MI, you can really only rely on the fact that you get a maximum when the
      signals are most aligned.

Use ``--similaritymetric mutualinfo`` to select MI.

**Hybrid similarity:**
I'm kind of proud of this one.
Crosscorrelation is fast and interpretable,
but has the problem of ambiguous time delay values,
whereas cross-MI is very slow and hard to interpret,
but quite unambiguous in selecting the best match.
Enter "hybrid similarity" -
Use the crosscorrelation to identify candidate peaks,
then calculate the MI only at those peak locations,
pick the one that has the higher MI,
and then proceed to the fitting step for full quantification.
This is almost as fast as straight correlation,
but does tend to be more stable.

Use ``--similaritymetric hybrid`` to select hybrid similarity.


Peak fitting and quantification
"""""""""""""""""""""""""""""""

The second part of this process is peak fitting and quantification.  For most of this discussion,
I'll refer to crosscorrelation, since its what I usually use.

To first approximation, fitting isn't necessary.  The crosscorrelation function will always have a
maximum somewhere, and if you've chosen your search range to cover the range of time lags that
blood will have, it will fall within that range.  However, that's not a great way to do things.
If you do this, your delay values will be quantized, either to TR, or in our case, TR divided by the
oversampling factor (which is why we oversampled to begin with).  The delay range in healthy young
adults runs from about -2 to +4 seconds, and is strongly peaked near 0.  Using our default
oversampling, which makes the effective TR 0.5 seconds, that gives you at most 13 possible
delay values, with most of them in a more restricted range of 5 or so values.  While somewhat
useful, this is throwing away a lot of information unnecessarily.

Remember that the sLFO signal is bandlimited to 0.009 to 0.15Hz, which means the highest
frequency component in the data has a period of about 6.67 seconds.  So at a minimum, the
correlation peak will be several seconds across, so in addition to the peak location, there will
be several points on either side that carry information about the peak location, height, and
width.  If you fit all the points around the peak, you'll get a much better estimate of the true
delay and correlation value.

Correlation peaks can be a little messy; low pass filtering, weird autocorrelation properties due to
nonuniform power spectra, window function choices,
and baseline roll can lead to incorrect peak identification.
This makes the peak fitting process complicated.


Despeckling
"""""""""""

As mentioned above, your correlation function may be pseudoperiodic due to an unfortunate power spectrum.
At this point, the delay maps are subjected to a multipass despeckling operation,
where voxels that look like they may have had incorrect fits are refit to be more consistent with
their neighbors.


Generating a better moving signal estimate (refinement)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Now that we have an estimate of when the moving regressor arrives at every voxel,
we can make a better estimate of the driving signal.


Voxel selection
"""""""""""""""

First we pick the voxels we want to use to generate the new estimate.
We can set the starting mask explicitly using the ``--refineinclude MASKFILE:VALSPEC`` and
``--refineexclude MASKFILE:VALSPEC`` command line options.
If left unset, we use all voxels with valid correlation fits.
We can further tune which voxels are excluded from refinement with the
``--norefinedespeckled``, ``--lagminthresh``, ``--lagmaxthresh``, and ``--sigmathresh`` options.
By default, we also exclude voxels with correlation strengths less than the p<0.05 threshold
found using the significance threshold estimation step above,
or we can override this threshold using ``--ampthresh``.


Timecourse alignment
""""""""""""""""""""

In each of the voxels selected for refinement,
we first negate the time delay in every voxel and timeshift the voxel by that amount.
This will have the effect of bringing the portion of the signal in each voxel due to the
moving sLFO signal into alignment.


Prescaling
""""""""""

We then prenormalize the voxels to use in the fit using their mean, variance,
or standard deviation over time, the inverse of the lag time, or leave them unscaled.
Selection is via the ``--refineprenorm`` option.
The default is to do no prenormalization.


New timecourse generation
"""""""""""""""""""""""""

The new timecourse is then generated from the set of aligned,
scaled timecourses using a method specified with ``--refinetype``:

-  **pca (default):** Perform a principal component analysis on the timecourses,
   reprojecting them onto a reduced set of components
   (specified by ``--pcacomponents`` - the default is the set explaining >=80% of total variance).
   Average the result.
-  **ica:** Perform an independent component analysis on the timecourses,
   reprojecting them onto a reduced set of components
   (specified by ``--pcacomponents`` - the default is the set explaining >=80% of total variance).
   Average the result.
-  **weighted_average:** Each voxel is scaled with either the correlation strength from the current pass,
   the square of the correlation strength, or is left unscaled.
   This is selected with the ``--refineweighting`` option - the default is "R2".
   The timecourses are then averaged.
-  **unweighted_average:**  Average the voxels.


Lather, Rinse, Repeat
^^^^^^^^^^^^^^^^^^^^^

Now that there is a new starting regressor, repeat the entire process some number of times.
This can be a fixed number of passes, specified by ``--passes NUMPASSES``.
The default is to do 3 passes.
Alternatively, by specifying ``--convergencethresh THRESH``,
the process is repeated until either the MSE between the new sLFO regressor and the
regressor from the previous pass falls below THRESH,
or the number of passes reaches MAX,
specified by ``--maxpasses MAX`` (default is 15).

.. tip::

   As a general rule, the more passes you do, the better the final result will be.
   However, this is a matter of diminishing returns,
   and I have found that 3 passes work well for most data.
   If you are not concerned about memory usage or processing time,
   you can set the number of passes to a higher value.

   The same logic applies to ``--despecklepasses``.


Regress Out the Moving Signal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now that we have optimized the moving blood signal and have final estimates of blood
arrival time at each voxel,
we can do the final regression to (intelligently) remove the sLFO signal from the data.
By default, this is done on the original, unmodified data -
i.e. none of the spatial or temporal filtering, masking, confound regression,
or anything else has been done.
The reason for this is that some of the operations may be needed to get a good sLFO regressor estimate,
or a good delay map,
but they could interfere with whatever further analysis you might want to do after sLFO removal.
You can always do them later if you want.
Also, if you really want to keep all those manipulations,
you can choose to by selecting ``--preservefiltering``.
But don't.

Alternately, instead of loading the original file, you can load a _different_ file,
and denoise that instead.
Why would you want to do that?
This is here for a very particular reason.
HCP data uses FIX, a really spiffy ICA noise removal tool that cleans things up quite a bit.
However, as mentioned above in the rapidtide usage section,
it does tend to remove a lot of hemodynamic signal in some regions,
particularly around the superior sagittal sinus.
That makes rapidtide's sLFO estimation and refinement process a lot less stable.
So you really want to do that estimation on non-FIX'ed data (the "minimally processed" data).
Ideally, you would then run FIX on the rapidtide cleaned data,
but that's a lot of computation that you don't necessarily want to do.
So a cheat is to regress the voxel specific noise regressors out of the FIX cleaned data.
Since the operations are linear, the order shouldn't matter
(waves hands to distract from the fact that FIX has probably generated some spurious negative
correlations by regressing out hemodynamic signal at the wrong time delay).
Anyway, while it's not perfect, it's better than not doing it this way.

Finally, if you don't want to do glm filtering at all
(i.e. you only care about time delays, and want to minimize storage space),
you can shut off the glm filtering with ``--noglm``.


Delay refinement
^^^^^^^^^^^^^^^^

This is new to rapidtide 3.0.  I've added a new method for refining the delay time estimate in
every voxel based on the filtering step.  To the best of my knowledge, this is something I came up
with (well not entirely, but this application).

As we remember from freshman physics, you can extrapolate a signal using a Taylor series approximation.
Which is to say, if you know the value of a function at a time t, and the value of the derivative,
and the second
derivative, and so on, you can calculate the signal at another point t + delta t by using a
weighted sum of those  values.  Neat!  Even more neat is that for sufficiently small values of
delta t, you can get a pretty good approximation using just the function and it's first derivative.

As always, there are some complications:

    * The mapping between fit coefficient ratio and time delay depends on the function, so it needs
      to be determined for each regressor.  It's linear for very small delay value, and then the
      mapping diverges (in a regressor specific way) as the delay increases.

    * As I mentioned, this only works for "small" delay times.  What is small?  For LFO signals
      in the 0.01 to 0.15 Hz band, this is only really good for about +/-3-5 seconds of offset
      (and the linear region is only about +/-0.75 seconds (which is why we can't use this method
      for the initial delay estimation, only for tuning).  The mapping function ends up being
      sigmoid - you can't really calculate the delay from the ratio when the slope gets close to zero.
      When that happens depends on the specific regressor, but you can pretty much always do the
      mapping out to about +/-3.5 seconds.

What is this good for?  Well, one thing I have found is that rapidtide gets much better fits if you
use a fairly strong spatial smoothing filter (5mm gaussian kernel).  That's great for getting rid
of a lot of the annoying speckling in the delay maps, but the result is that you lose a lot of fine
detail in the delay map (which is obvious when you think about it). BUT - we know that delay varies
relatively smoothly in real brains, so the smoothed delay values, while maybe not exactly right in
most voxels, aren't far off.  So the delay in any voxel will be within +/-3 seconds of the smoothed
value in every voxel, so the ratio-of-fit-derivatives method will be able to fit the difference, which
you can then apply as an offset to find the exact delay in every voxel with much higher spatial resolution.
Neat, huh?


References
^^^^^^^^^^

.. footbibliography::
