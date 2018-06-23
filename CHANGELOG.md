# History of changes
## Version 1.5 (6/11/18)
* (documentation) Added description of rapidtide output files.
* (tide_funcs) Fixed a VERY old bug in detrend that added an offset to the detrended timecourse.  The effect of the bug on rapidtide2(x) is probably small, because we almost always use data that has already been detrended.  The effect in some very particular use cases, though, was large enough that I finally noticed it after 3 years.
* (rapidtide2, 2x) Added option to disable progress bars (good when saving output to a file).
* (rapidtide2, 2x) Improved output to memusage file.
* (rapidtide2, 2x) Report fit errors with better granularity.
* (rapidtide2, 2x) Allow specification of external correlation mask.
* (rapidtide2, 2x) Added "MTT" map to hopefully remove the effect of autocorrelation.
* (rapidtide2, 2x) Added some additional diagnostic information to significance estimation.
* (rapidtide2x, tide_funcs) Major changes to peak fitting to try to improve stability using new findmaxlag_gauss_rev function.
* (rapidtide2x, tide_funcs) Moved logmem function to tide_funcs so that it's available for other programs.
* (rapidtide2x) Fixed bug when running despeckling on a single processor.
* (rapidtide2, 2x) Attempting to stabilize the lagsigma measurement with better initial parameter estimates.
* (tide_funcs) Cast timecourse index as a long to avoid an overflow in NIRS timecourses.
* (rapidtide2, 2x, tide_funcs) Added ability to set searchfrac in fits.
* (rapidtide2, 2x) Disable threshold during significance estimation, simplify internal logic for turning on estimation.
* (rapitdide2) Fixed bug in mask generation
* (showtc) Added the ability to select columns to plot, and to read BIDS style json/tsv.gz physiological files
* (showtc, showxy) Updated colormap names for compatibility with matplotlib 2.2+
* (rapidtide2std) Initial support for warping with ANTs (does not work yet)
* (rapidtide2std) Added the ability to align a single file.
* (tidepool) Support for MTT map.
* (ccorrica, showstxcorr) PEP 8 reformatting.
* (testing) Added test for findmaxlag versions.
* (testing) Added test for stxcorr functions.
* (temporaldecomp) Allow 3D masks.
* (atlastool) Changed method for generating 3D files.
* (atlastool) Various bug fixes in 3D atlas generation.
* (resamp1tc) Modernized option selection, Added nodisplay option.
* (showstxcorr) Explicit integer cast of indices.
* (showstxcorr) Removed initial Hamming window.
* (showstxcorr) Added csv output of matrices.
* (linfit) Added to distribution.
* (tide_funcs) Changed value of rcond in leastsq to be compatible over multiple versions of bumpy (least. comprehensible. note. ever.)
* (tide_funcs) Added findexecutable and isexecutable functions
* (tide_funcs) Added a convolution gridding function
* (tide_funcs) Fixed readvec(s) so that is now works properly if a text file ends with an empty line.
* (tide_funcs) Added function to read slice times from a BIDS sidecar file.
* (spatialdecomp, temporaldecomp) Command line is now saved.
* (threeD) Added


## Version 1.4.2 (2/21/18)
* (documentation) Fixed some formatting.
* (showxcorrx) Cleaned up usage statement.


## Version 1.4.0 (2/21/18)
* (rapidtide2, 2x) Added macros to support setting multiple options at once.
* (rapidtide2, 2x) --nirs macro sets a number of parameters to be appropriate for NIRS data processing.
* (rapidtide2, 2x) --venousrefine macro sets refinement parameters to use data only from large draining veins (only works reliably in healthy subjects ATM).
* (rapidtide2, 2x) Now tabulate maximum correlation times without range limit for my own secret, diabolical purposes.
* (rapidtide2, 2x) Fixed a bug that was not shifting all of the timecourses if they were not in the refinement mask.
* (rapidtide2, 2x) Improved memory usage tracking.
* (rapidtide2, 2x) Reduced memory footprint.
* (rapidtide2, 2x) Move large arrays into shared memory for multiprocessor jobs to avoid duplicating RAM.
* (rapidtide2, 2x) You can now set the number of processors used (rather than always using all of them) when multiprocessing.
* (rapidtide2, 2x) Properly shut down worker procedures to free RAM earlier.
* (rapidtide2, 2x) Fixed a bug in the initialization of the dispersion calculation.
* (rapidtide2, 2x) Fixed a maddening bug in output of the refinement mask.
* (rapidtide2, 2x) Fixed the range on the Gaussian filtering progress bar.
* (rapidtide2, 2x) Made some improvements to the despeckling procedure.
* (rapidtide2, 2x) --refinepasses is now deprecated - use --passes instead.
* (rapidtide2, 2x) Added new methods to specify the sample rate (or sample time) of the input data file.
* (rapidtide2, 2x) Revised usage statement to make parameter names better reflect their function.
* (rapidtide2, 2x) A lot of internal code cleanup and dead code removal.
* (rapidtide2, 2x, showxcorrx) Allow specification of the correlation window function (hamming (default), hann, blackmanharris, or None).
* (showtc) Cleaned up some bugs introduced during the last overhaul.
* (tcfrom3col) Added to package (generates a timecourse from an FSL style 3 column regressor file.
* (tidepool) Default to displaying using the valid mask rather than the p<0.05 mask.
* (tidepool) Enabled usage of the refine mask.


## Version 1.3.0 (12/15/17)
* (rapidtide2, 2x) Added new option, '--despeckle', which uses a spatial median filter to find and correct points where the correlation fit picked the wrong autocorrelation lobe.  This dramatically improves the quality of the output maps.  This will probably be turned on by default in the next release.
* (tidepool) FINALLY fixed the click positioning bug.  Worth the update just for this.  That was driving me crazy.
* (tidepool) Formatting improvements.
* (tidepool) Preliminary support for multiple territory atlases and averaging modes in tidepool.
* (tidepool) Atlas averaging is now (mostly) working.
* (rapidtide2, 2x) Now support text format NIRS datasets (2D text files) in addition to NIFTI fMRI files.
* (rapidtide2, 2x) Substantial internal changes to reduce memory footprint, improve speed.
* (rapidtide2x, showxcorrx) Initial support added for Choudry's cepstral analysis method for delay calculation.
* (showtc) Substantial improvements (improved formatting, ability to specify separate subplots, transpose input, line colors, waterfall plots, offsets between lines, titles, etc).
* (rapidtide2std) Internal code cleanup.
* (showxy) Now supports multiple input files.
* Added glmfilt to package to filter 1D or 4D data out of 4D datasets.
* Added spatialdecomp to do spatial PCA decomposition of 4D NIFTI files.
* Added temporaldecomp to do temporal PCA decomposition of 4D NIFTI files.
* Added ccorrica to the distribution to provide cross correlation matrices between all timeseries in a 2D text files.
* Added atlastool to aid in preparation of atlas files for tidepool.

## Version 1.2.0 (6/20/17)
* New release to trigger a Zenodo DOI.
* Fully tested for python 3.6 compatibility.
* Added linfit to the distribution.
* Set a limit of 25 dispersion regressors.
* Reformatted the documentation somewhat.
* Added some recipes to the documentation for common use cases.
* Cleaned up and fixed the resampling code.
* Minor quality and speed improvement to timeshift.
* No longer output "datatoremove" to save space.
* Removed some redundant screen refreshes from tidepool.
* Reorganized and removed dead code.
* Changed default mode for calculating refined regressors to "unweighted_average".
* Synced changes in rapidtide2x to rapidtide2

## Version 1.1.0 (4/3/17)
* I have now synced all of the changes in rapidtide2x back to rapidtide2.
* rapidtide now has multiprocessing support using the --multiproc flag.  This can cause a dramatic increase in processing speed on multicore/processor machines.
* Dispersion calculation is now off by default.
* Tidepool is now compatible with both PyQt4 and 5.
* Reordered some memory allocation and deallocation to keep the RAM footprint down.
* Some additional significant speedups (support for ffftw if present, caching hamming windows).
* Added an optional cross-spectral density filter for adaptive timecourse filtering.  This is a work in progress - not really ready for general use.
* Skeleton of support for Wiener deconvolution to sharpen the correlation function (off by default, not ready for general use).

## Version 1.0.0 (2/13/17)
* I decided to stop hedging and actually commit myself - this is version 1.0.0 - out of beta!
* To promote stability, new features will be put into test versions (the name of the program will have an "x" appended).  This way I can do major internal changes and have them available to users without breaking something they might rely on.  The "x" versions will sync with the "normal" versions after extensive testing.
* Major new feature (rapidtide2x only for now).  Multiprocessing!  Significant speedup when using the --multiproc option on machines with multiple cores.
* showxcorrx has new features and defaults.
* Memory allocation has been reorganized to reduce footprint (rapidtide2x).
* Changed imports for better compatibility when running in the NITRC-CE environment (rapidtide2x).
* rapidtide2std now supports nonlinear alignment.
* histnifti is added to the distribution.
* I've added some additional outputs to rapidtide2 and rapidtide2x during refinement to help figure out if the brain is a dispersive filter.  This doesn't change how rapidtide2 does the refinement - it's just informational at this point.
* Added spatialfit to the distribution.  I use this to analyze delay maps.  More on this later.
* Fully implemented samplerate and sampletime options (rapidtide2)
* Corrected and enhanced the use of alternative correlation weighting functions (PHAT, Liang, and Eckart weighting) (rapidtide).
* Updated all scripts for compatibility with matplotlib 2.0.
* Fixed tidepool for compatibility with the new version of pyqtgraph.
* Significant enhancements to showstxcorr (this is a work in progress).
* Example data is no longer installed in the python directory (this never made sense to do).
* Added code to do matched filtering of regressors with mean PSD and/or cross-spectral density.  It works, but doesn't do much (rapidtide2x).

## Version 0.1.9 (12/19/16)

* Added code to allow runtime memory profiling if memory_profile library is present.
* Extensive casting of variables to lower memory footprint and allow future optimizations.
* Added explicit garbage collection to reduce memory usage.
* Added optional single precision calculation mode to lower memory footprint.
* Added a second script, "rapidtide2x" where I can distribute and test new features without breaking the main code branch.
* Did some speed optimizations in findmaxlag, including faster gaussian fitting and new MUCH faster parabolic fitting (still experimental).
* Minor bug fixes, code reorganization and cleanup.

## Version 0.1.8 (11/30/16)

* Fixed a bug in the GLM filtering code - if spatial filtering was applied in rapidtide2, the smoothed data was filtered rather than the original data.
* Added an option in rapidtide2 ("--glmsourcefile=FILE") to apply GLM filter to a different dataset than the one used to estimate the delays (this is used for HCP data - the "hp2000_clean" data has had LFO signals partially removed and may compromise delay estimation, so that should be done on the un-"FIX"ed data).
* Added the ability to detect autocorrelation properties of the test regressor that may cause delay estimation to fail with the "--accheck" flag.
* Added an option "--acfix" to try to correct for bad test regressor autocorrelation properties.  This is not yet working correctly.
* Added the ability to specify a slicetimes file ("--slicetimes=FILE") if slicetime correction has not yet been applied to the dataset.  Not fully tested.
* (rapidtide2std, tidepool) Added the ability to transform and display functional data to highres anatomic space in addition to MNI152 space.
* Various small bugfixes and format cleanups.

## Version 0.1.7 (11/15/16)

* I think I've resolved the issue of crashes due to numba functioning differently on machines other than mine.
* Fixed a masking bug in tidepool that was due to numbers being very close to, but not exactly, 1.
* Made a number of internal changes to rapidtide2 and tidepool to allow dynamic significance masking (not yet working - actually failing spectacularly for the most part, but it's currently commented out).
* Added showstxcorr to the distribution.
* Added the ability to set the mask threshold for correlation and global mask inclusion (this turns out to be needed if you use non-skull-stripped data.)
* Put in some code to start to figure out how to account for dispersion in the delay function.
* Moved the "start movie" button in tidepool to better align with the numerical spin boxes.
* showtc has gotten a significant upgrade in functionality, adding the ability to display power spectra, phase spectra, and set the sample rate to make the x-axis correct.
* Lots of internal formatting/style fixes, and fixed some formatting in the usage statements and documentation.

## Version 0.1.6 (10/15/16)

* Fixed a critical bug that had been introduced in the last round of changes to findmaxlag.
* Disabled numba for findmaxlag (it seems to cause problems for some users).
* New option --skipsighistfit to omit fitting a Johnson SB function to the significance histogram.
* Fixed the usage statement.
* Fixed a bug that set ampthresh to zero when not doing significance estimation.

## Version 0.1.5 (10/11/16)

* Fixed a bug that made it impossible to specify --regressortstep.
* Added undocumented option --nonumba to turn off just in time compilation if there's a problem with it.
* Print rapidtide version on launch.
* Made pandas import explicit (sklearn requires it).

## Version 0.1.4 (10/10/16)

* Some fixes to usage output.
* Added functions for fitting trapezoids (for risetime calculations).
* Changed argument parsing and option output to avoid conflicts
* Added an option to not zero out bad fits (so as not to skew lag statistics)
* Improved fitting of probability distributions.
* Better handling of failed correlation peak fits.
* Now installations should work properly if not installed using git (fixed _gittag import problem).

## Version 0.1.3 (9/2/16)

* Added a tool (rapidtide2std) to register all output maps to MNI152 coordinates (requires FSL).
* Made a 3mm resolution ASPECTS map for use in tidepool.
* Reference data is now properly installed, and tidepool can find it reliably.
* Redid the version information.  Rapidtide now records both the release version and
  the git hash in in the output data to help with data provenance.
* Reorganized the distribution into what seems to be a more canonical layout.
* Resolved the issues I seem to have introduced with Python 3 compatibility.
* Significantly cleaned up resampling and filtering code and improved reliability.
* Added some unit tests for critical routines.  Strangely, they all fail on Travis-CI, 
  but work on my local machine.  It seems to be a numerical precision issue.  The answers
  are rightish, just not right on Travis.

## Version 0.1.2 (8/5/16)

* Some bug fixes in filtering and resampling code.
* Beginning to add automated tests.
* Biphasic mode is now fully implemented, including two-tailed significance calculation.

## Version 0.1.1 (7/8/16)

* First release



## Version 1.3.0 (12/15/17)
* (rapidtide2, 2x) Added new option, '--despeckle', which uses a spatial median filter to find and correct points where the correlation fit picked the wrong autocorrelation lobe.  This dramatically improves the quality of the output maps.  This will probably be turned on by default in the next release.
* (tidepool) FINALLY fixed the click positioning bug.  Worth the update just for this.  That was driving me crazy.
* (tidepool) Formatting improvements.
* (tidepool) Preliminary support for multiple territory atlases and averaging modes in tidepool.
* (tidepool) Atlas averaging is now (mostly) working.
* (rapidtide2, 2x) Now support text format NIRS datasets (2D text files) in addition to NIFTI fMRI files.
* (rapidtide2, 2x) Substantial internal changes to reduce memory footprint, improve speed.
* (rapidtide2x, showxcorrx) Initial support added for Choudry's cepstral analysis method for delay calculation.
* (showtc) Substantial improvements (improved formatting, ability to specify separate subplots, transpose input, line colors, waterfall plots, offsets between lines, titles, etc).
* (rapidtide2std) Internal code cleanup.
* (showxy) Now supports multiple input files.
* Added glmfilt to package to filter 1D or 4D data out of 4D datasets.
* Added spatialdecomp to do spatial PCA decomposition of 4D NIFTI files.
* Added temporaldecomp to do temporal PCA decomposition of 4D NIFTI files.
* Added ccorrica to the distribution to provide cross correlation matrices between all timeseries in a 2D text files.
* Added atlastool to aid in preparation of atlas files for tidepool.

## Version 1.2.0 (6/20/17)
* New release to trigger a Zenodo DOI.
* Fully tested for python 3.6 compatibility.
* Added linfit to the distribution.
* Set a limit of 25 dispersion regressors.
* Reformatted the documentation somewhat.
* Added some recipes to the documentation for common use cases.
* Cleaned up and fixed the resampling code.
* Minor quality and speed improvement to timeshift.
* No longer output "datatoremove" to save space.
* Removed some redundant screen refreshes from tidepool.
* Reorganized and removed dead code.
* Changed default mode for calculating refined regressors to "unweighted_average".
* Synced changes in rapidtide2x to rapidtide2

## Version 1.1.0 (4/3/17)
* I have now synced all of the changes in rapidtide2x back to rapidtide2.
* rapidtide now has multiprocessing support using the --multiproc flag.  This can cause a dramatic increase in processing speed on multicore/processor machines.
* Dispersion calculation is now off by default.
* Tidepool is now compatible with both PyQt4 and 5.
* Reordered some memory allocation and deallocation to keep the RAM footprint down.
* Some additional significant speedups (support for ffftw if present, caching hamming windows).
* Added an optional cross-spectral density filter for adaptive timecourse filtering.  This is a work in progress - not really ready for general use.
* Skeleton of support for Wiener deconvolution to sharpen the correlation function (off by default, not ready for general use).

## Version 1.0.0 (2/13/17)
* I decided to stop hedging and actually commit myself - this is version 1.0.0 - out of beta!
* To promote stability, new features will be put into test versions (the name of the program will have an "x" appended).  This way I can do major internal changes and have them available to users without breaking something they might rely on.  The "x" versions will sync with the "normal" versions after extensive testing.
* Major new feature (rapidtide2x only for now).  Multiprocessing!  Significant speedup when using the --multiproc option on machines with multiple cores.
* showxcorrx has new features and defaults.
* Memory allocation has been reorganized to reduce footprint (rapidtide2x).
* Changed imports for better compatibility when running in the NITRC-CE environment (rapidtide2x).
* rapidtide2std now supports nonlinear alignment.
* histnifti is added to the distribution.
* I've added some additional outputs to rapidtide2 and rapidtide2x during refinement to help figure out if the brain is a dispersive filter.  This doesn't change how rapidtide2 does the refinement - it's just informational at this point.
* Added spatialfit to the distribution.  I use this to analyze delay maps.  More on this later.
* Fully implemented samplerate and sampletime options (rapidtide2)
* Corrected and enhanced the use of alternative correlation weighting functions (PHAT, Liang, and Eckart weighting) (rapidtide).
* Updated all scripts for compatibility with matplotlib 2.0.
* Fixed tidepool for compatibility with the new version of pyqtgraph.
* Significant enhancements to showstxcorr (this is a work in progress).
* Example data is no longer installed in the python directory (this never made sense to do).
* Added code to do matched filtering of regressors with mean PSD and/or cross-spectral density.  It works, but doesn't do much (rapidtide2x).

## Version 0.1.9 (12/19/16)

* Added code to allow runtime memory profiling if memory_profile library is present.
* Extensive casting of variables to lower memory footprint and allow future optimizations.
* Added explicit garbage collection to reduce memory usage.
* Added optional single precision calculation mode to lower memory footprint.
* Added a second script, "rapidtide2x" where I can distribute and test new features without breaking the main code branch.
* Did some speed optimizations in findmaxlag, including faster gaussian fitting and new MUCH faster parabolic fitting (still experimental).
* Minor bug fixes, code reorganization and cleanup.

## Version 0.1.8 (11/30/16)

* Fixed a bug in the GLM filtering code - if spatial filtering was applied in rapidtide2, the smoothed data was filtered rather than the original data.
* Added an option in rapidtide2 ("--glmsourcefile=FILE") to apply GLM filter to a different dataset than the one used to estimate the delays (this is used for HCP data - the "hp2000_clean" data has had LFO signals partially removed and may compromise delay estimation, so that should be done on the un-"FIX"ed data).
* Added the ability to detect autocorrelation properties of the test regressor that may cause delay estimation to fail with the "--accheck" flag.
* Added an option "--acfix" to try to correct for bad test regressor autocorrelation properties.  This is not yet working correctly.
* Added the ability to specify a slicetimes file ("--slicetimes=FILE") if slicetime correction has not yet been applied to the dataset.  Not fully tested.
* (rapidtide2std, tidepool) Added the ability to transform and display functional data to highres anatomic space in addition to MNI152 space.
* Various small bugfixes and format cleanups.

## Version 0.1.7 (11/15/16)

* I think I've resolved the issue of crashes due to numba functioning differently on machines other than mine.
* Fixed a masking bug in tidepool that was due to numbers being very close to, but not exactly, 1.
* Made a number of internal changes to rapidtide2 and tidepool to allow dynamic significance masking (not yet working - actually failing spectacularly for the most part, but it's currently commented out).
* Added showstxcorr to the distribution.
* Added the ability to set the mask threshold for correlation and global mask inclusion (this turns out to be needed if you use non-skull-stripped data.)
* Put in some code to start to figure out how to account for dispersion in the delay function.
* Moved the "start movie" button in tidepool to better align with the numerical spin boxes.
* showtc has gotten a significant upgrade in functionality, adding the ability to display power spectra, phase spectra, and set the sample rate to make the x-axis correct.
* Lots of internal formatting/style fixes, and fixed some formatting in the usage statements and documentation.

## Version 0.1.6 (10/15/16)

* Fixed a critical bug that had been introduced in the last round of changes to findmaxlag.
* Disabled numba for findmaxlag (it seems to cause problems for some users).
* New option --skipsighistfit to omit fitting a Johnson SB function to the significance histogram.
* Fixed the usage statement.
* Fixed a bug that set ampthresh to zero when not doing significance estimation.

## Version 0.1.5 (10/11/16)

* Fixed a bug that made it impossible to specify --regressortstep.
* Added undocumented option --nonumba to turn off just in time compilation if there's a problem with it.
* Print rapidtide version on launch.
* Made pandas import explicit (sklearn requires it).

## Version 0.1.4 (10/10/16)

* Some fixes to usage output.
* Added functions for fitting trapezoids (for risetime calculations).
* Changed argument parsing and option output to avoid conflicts
* Added an option to not zero out bad fits (so as not to skew lag statistics)
* Improved fitting of probability distributions.
* Better handling of failed correlation peak fits.
* Now installations should work properly if not installed using git (fixed _gittag import problem).

## Version 0.1.3 (9/2/16)

* Added a tool (rapidtide2std) to register all output maps to MNI152 coordinates (requires FSL).
* Made a 3mm resolution ASPECTS map for use in tidepool.
* Reference data is now properly installed, and tidepool can find it reliably.
* Redid the version information.  Rapidtide now records both the release version and
  the git hash in in the output data to help with data provenance.
* Reorganized the distribution into what seems to be a more canonical layout.
* Resolved the issues I seem to have introduced with Python 3 compatibility.
* Significantly cleaned up resampling and filtering code and improved reliability.
* Added some unit tests for critical routines.  Strangely, they all fail on Travis-CI, 
  but work on my local machine.  It seems to be a numerical precision issue.  The answers
  are rightish, just not right on Travis.

## Version 0.1.2 (8/5/16)

* Some bug fixes in filtering and resampling code.
* Beginning to add automated tests.
* Biphasic mode is now fully implemented, including two-tailed significance calculation.

## Version 0.1.1 (7/8/16)

* First release
