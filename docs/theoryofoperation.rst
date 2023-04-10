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


What is the difference between RIPTiDE and rapidtide?
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

Initial Moving Signal Estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Moving Signal Preprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Moving Signal Massaging
^^^^^^^^^^^^^^^^^^^^^^^

Dataset Preprocessing
^^^^^^^^^^^^^^^^^^^^^

Time delay determination
^^^^^^^^^^^^^^^^^^^^^^^^

Generating a Better Moving Signal Estimate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Lather, Rinse, Repeat
^^^^^^^^^^^^^^^^^^^^^

Save Useful Parameters
^^^^^^^^^^^^^^^^^^^^^^

Regress Out the Moving Signal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. [Tong2019] Tong, Y., Hocke, L.M., and Frederick, B.B., Low Frequency
   Systemic Hemodynamic "Noise" in Resting State BOLD fMRI: Characteristics,
   Causes, Implications, Mitigation Strategies, and Applications.
   Front Neurosci, 2019. 13: p. 787.
   \| http://dx.doi.org/10.3389/fnins.2019.00787


.. [Erdogan2016] Erdoğan S, Tong Y, Hocke L, Lindsey K, Frederick B. Correcting
   resting state fMRI-BOLD signals for blood arrival time enhances
   functional connectivity analysis. Front. Hum. Neurosci., 28 June 2016
   \| http://dx.doi.org/10.3389/fnhum.2016.00311
