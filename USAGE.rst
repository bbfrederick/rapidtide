Usage
=====
Execute any of the commands to run rapidtide on the sample data:

a) Run rapidtide2 to perform dynamic global signal regression (dGSR) on an fMRI file[1]:
::

    rapidtide2 rapidtide/data/examples/src/fmri.nii.gz rapidtide/data/examples/dst/dgsr -L -r -15,15 --passes=3


b) Run rapidtide2 to perform static global signal regression (sGSR) on an fMRI file[1] (this is just global mean regression):
::

    rapidtide2 rapidtide/data/examples/src/fmri.nii.gz rapidtide/data/examples/dst/sgsr -L -Z 0.0


c) Run tidepool to look at all the interesting maps and timecourses from a):
::

    tidepool


(then select the file rapidtide/data/examples/dst/dgsr_lagtimes.nii.gz to load the dataset):

d) Look at the refined regressors produced during dGSR: (the "dgsr_pass1" regressor is the initial global mean before refinement) in their own window.
::

    showtc rapidtide/data/examples/dst/dgsr_reference_fmrires_pass[123].txt
