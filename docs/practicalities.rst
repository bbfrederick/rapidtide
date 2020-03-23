Practicalities
====================================

I think you mean lagmask.nii

lagmask shows all voxels where the fitting procedure for the correlation values converged, so generally it contains most voxels in the brain.  The fits rarely fail, except if a voxel is particularly noisy or there is very strange circulation in a particular voxel.

Any voxel that is in the p_lt_… masks has to also be in the lagmask, since you can only calculate significance if the fit succeeded.

I would caution against reading too much into the p_lt_0p050 masks - from experience it seems that many, many valid, real correlations are present in voxels that don’t meet the p<0.05 threshold.  Having a significance value that is lower than the p<0.05 threshold does not mean the correlation is necessarily spurious, it just means you can not guarantee that it is not spurious, if that makes sense.  There just happen to be many, true, low correlations.

What rapid tide does is calculate the cross correlation between your “probe regressor” (usually starting with the global mean brain signal then refining it to find the moving blood signal). It then searches for the peak value (the maximum correlation). In order to get the exact delay, it fits a Gaussian to the peak so it can get the delay with better resolution than if it lost picked the largest value, in case the peak is under sampled.)  if the Gaussian fit fails (which it can do if the signal is very noisy and/or the correlation is low, or of the peak is too wide), it will set the lagmask to zero.

If you set the command line option “—nofitfilt”, it will not do this - it will just use the initial guess for the peak parameters before the fit in voxels where the fit fails.

Are you getting a lagmask with a lot of holes in it?  That may indicate some other problem. 

