# History of changes

## Version 0.1.1 (7/8/16)

* First release

## Version 0.1.2 (8/5/16)

* Some bug fixes in filtering and resampling code.
* Beginning to add automated tests.
* Biphasic mode is now fully implemented, including two-tailed significance calculation.

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

## Version 0.1.4 (10/10/16)

* Some fixes to usage output.
* Added functions for fitting trapezoids (for risetime calculations).
* Changed argument parsing and option output to avoid conflicts
* Added an option to not zero out bad fits (so as not to skew lag statistics)
* Improved fitting of probability distributions.
* Better handling of failed correlation peak fits.
* Now installations should work properly if not installed using git (fixed _gittag import problem).

## Version 0.1.5 (10/11/16)

* Fixed a bug that made it impossible to specify --regressortstep.
* Added undocumented option --nonumba to turn off just in time compilation if there's a problem with it.
* Print rapidtide version on launch.
* Made pandas import explicit (sklearn requires it).

## Version 0.1.6 (10/15/16)

* Fixed a critical bug that had been introduced in the last round of changes to findmaxlag.
* Disabled numba for findmaxlag (it seems to cause problems for some users).
* New option --skipsighistfit to omit fitting a Johnson SB function to the significance histogram.
* Fixed the usage statement.
* Fixed a bug that set ampthresh to zero when not doing significance estimation.

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

## Version 0.1.8 (11/30/16)

* Fixed a bug in the GLM filtering code - if spatial filtering was applied in rapidtide2, the smoothed data was filtered rather than the original data.
* Added an option in rapidtide2 ("--glmsourcefile=FILE") to apply GLM filter to a different dataset than the one used to estimate the delays (this is used for HCP data - the "hp2000_clean" data has had LFO signals partially removed and may compromise delay estimation, so that should be done on the un-"FIX"ed data).
* Added the ability to detect autocorrelation properties of the test regressor that may cause delay estimation to fail with the "--accheck" flag.
* Added an option "--acfix" to try to correct for bad test regressor autocorrelation properties.  This is not yet working correctly.
* Added the ability to specify a slicetimes file ("--slicetimes=FILE") if slicetime correction has not yet been applied to the dataset.  Not fully tested.
* (rapidtide2std, tidepool) Added the ability to transform and display functional data to highres anatomic space in addition to MNI152 space.
* Various small bugfixes and format cleanups.

