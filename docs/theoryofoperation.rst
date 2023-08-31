Theory of operation
===================
If you're bored enough or misguided enough to be reading this section, you are
my intended audience!

rapidtide
"""""""""

What is rapidtide trying to do?
-------------------------------
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
delayed in each voxel.  This gives you better noise removal, and also
avoids generating spurious correlations.  For a detailed consideration of this,
look here [Erdogan2016]_.


What is the difference between RIPTiDe and rapidtide?
`````````````````````````````````````````````````````

RIPTiDe (Regressor Interpolation at Progressive Time Delays) is the name of
the technique used for finding and removing time lagged physiological signals
in fMRI data.  In the original RIPTiDe papers, we generated a
set of regressors over a range of different time shifts (starting from a regressor recorded
outside of the brain), and then ran a GLM in FSL using the entire set of regressors.
We realized that this 1) doesn't give you the optimal delay value directly,
which turns out to be a useful thing to know, and 2) burns degrees of freedom
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
algorithm) estimates delay in every voxel and recursively refines an estimate
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
------------------------
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
```````````````````

By default, rapidtide calculates masks dynamically at run time.  There
are 4 masks used 1) the global mean mask, which determines which voxels
are used to generate the initial global mean regressor, 2) The
correlation mask, which determines which voxels you actually calculate
rapidtide fits in (what you are describing here), 3) the refine mask,
which selects which voxels are used to generate a refined regressor for
the next fitting pass, and 4) the GLM mask, which determines which
voxels have the rapidtide regressors removed.

Below is a description of how this works currently.  NB: this is not how
I THOUGHT is worked - until I just looked at the code just now.  It
built up over time, and evolved into something not quite what I
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
non-zero voxels in NAME in the mask.  If VALSPEC is provided, only
include voxels with integral values listed in VALSPEC in the mask).
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
in the mask.  Then if --refineexclude MASK[:VALSPEC] is specified,
remove any voxels selected by MASK[:VALSPEC] from the mask.  If it is
not specified, don't change the mask.  Then multiply by corrmask, since
you can't used voxels rwhere rapidtide was not run to do refinement.

**For the GLM mask:**
Include all voxels, unless you are calculating a CVR map, in which case
only perform the calculation on voxels exceeding 25% of the robust mean
value (this is weird and will change).

Depending on your data (including pathology), and what you want to accomplish, using the default correlation 
mask is not idea.  For example, if a subject has obvious pathology, you may want to exclude these voxels
from being used to generate the intial global mean signal estimate, or from being used in refinement.


Initial Moving Signal Estimation
````````````````````````````````



Moving Signal Preprocessing
```````````````````````````

**Oversampling:**  In order to simplify delay calculation, rapidtide performs all delay estimation operations
on data with a sample rate of 2Hz or faster.  The oversampling factor can be specified explicitly (using the
``--oversampfac`` command line argument), but if it is
not, for data with a sample rate of less than 2Hz, all data and regressors are internally upsampled by the lowest
integral factor that results in a sample rate >= 2Hz.

**Regressor resampling:** In the case where we are using the global mean signal as the moving signal, the moving signal
estimate and the fMRI data have the same sample rate, but if we use external
recordings, such as NIRS or etCO2 timecourses, these will in general have sample rates other than the TR. Therefore
the first step in moving regressor processing is to resample the moving regressor estimate to match the (oversampled)
data sample rate.

**Temporal filtering:**  By default, all data and moving regressors are temporally bandpass filtered to 0.009-0.15Hz
(our standard definition of the LFO band).  This can be overridden with ``--filterband`` and ``--filterfreqs`` command line
options.

**Pseudoperiodicity:**  The first uncontrolled quantity is
pseudoperiodicity.  From time to time, signal energy in the 0.09-0.15 Hz
band will be strongly concentrated in one or more spectral peaks.
Whether this is completely random, or due to some pathological or
congenital condition that affects circulation is not known – it seems
for the most part to be purely by chance, as it is occasionally seen
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
not to be much of a problem at all.  If the sLFO signal in given voxel
is sufficiently periodic that that the correctly delayed signal is
indistinguishable from the signal one or more periods away, then it
doesn’t matter which signal is removed – the resulting denoised signal
is the same.

**Mitigation of pseudoperiodicity:** While we continue to work on fully
resolving this issue, we have a number of ways of dealing with this.
First of all, spectral analysis of the sLFO signal allows us to
determine if the signal may be problematic.  Rapidtide checks the
autocorrelation function of the sLFO signal for large sidelobes with
periods within the delay search window and issues a warning when these
signals are present.  Then after delay maps are calculated, they are
processed with an iterative despeckling process analogous to phase
unwrapping.  The delay of each voxel is compared to the median delay of
its neighbors.  If the voxel delay differs by the period of an
identified problematic sidelobe, the delay is constrained to “correct”
value and refit.  This procedure greatly attenuates, but does not
completely solve, the problem of bad sidelobes.  A more general solution
to the problem of non-uniform spectra will likely improve the
correction.


Moving Signal Massaging
```````````````````````


Dataset Preprocessing
`````````````````````


Time delay determination
````````````````````````


Generating a Better Moving Signal Estimate
``````````````````````````````````````````


Lather, Rinse, Repeat
`````````````````````


Save Useful Parameters
``````````````````````


Regress Out the Moving Signal
`````````````````````````````

.. [Tong2019] Tong, Y., Hocke, L.M., and Frederick, B.B., Low Frequency
   Systemic Hemodynamic "Noise" in Resting State BOLD fMRI: Characteristics,
   Causes, Implications, Mitigation Strategies, and Applications.
   Front Neurosci, 2019. 13: p. 787.
   \| http://dx.doi.org/10.3389/fnins.2019.00787


.. [Erdogan2016] Erdoğan S, Tong Y, Hocke L, Lindsey K, Frederick B. Correcting
   resting state fMRI-BOLD signals for blood arrival time enhances
   functional connectivity analysis. Front. Hum. Neurosci., 28 June 2016
   \| http://dx.doi.org/10.3389/fnhum.2016.00311
