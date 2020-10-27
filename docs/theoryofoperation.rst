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
blood CO2, gastric motility, and other sources (for a survey, see [REF].)  As
biology is complicated, it's probably some mixture of these sources and others
that we may not have considered.

No matter what the source of the signal, this model can be exploited for a
number of purposes.  First, if you're interested in hemodynamics, the time
delay in every voxel gives you a lot of information that's otherwise hard
to obtain, namely the arrival time of blood in each voxel, and the fraction
of the variance in that voxel that's accounted for by that moving signal.


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
