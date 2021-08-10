Usage
=====
Execute any of the commands to run rapidtide on the sample data:

a) Run rapidtide to perform dynamic global signal regression (dGSR) on an fMRI file[1]:
::

    rapidtide rapidtide/data/examples/src/sub-RAPIDTIDETEST rapidtide/data/examples/dst/sub-RAPIDTIDETEST_dgsr --searchrange -15 15 


b) Run rapidtide to perform static global signal regression (sGSR) on an fMRI file[1] (this is just global mean regression):
::

    rapidtide rapidtide/data/examples/src/sub-RAPIDTIDETEST rapidtide/data/examples/dst/sub-RAPIDTIDETEST_sgsr --fixdelay 0.0 --passes 1


c) Run tidepool to look at all the interesting maps and timecourses from a):
::

    tidepool


(then select the file rapidtide/data/examples/dst/dgsr_lagtimes.nii.gz to load the dataset):

d) Look at the regressors produced and used during dGSR: (the "pass1" regressor is the initial global mean before refinement) in their own window.
::

    showtc rapidtide/data/examples/dst/sub-RAPIDTIDETEST_dgsr_desc-movingregressor_timeseries.json
