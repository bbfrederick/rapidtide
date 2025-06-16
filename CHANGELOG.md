# Release history

## Version 3.0.7.1 (6/16/25)
* (rapidtide) Disabled sLFO filtering by default.  Use ``sLFOfiltmask`` to turn it on.
* (retroregress) Synced with sLFO filtering changes to rapidtide.

## Version 3.0.7 (6/16/25)
* (rapidtide) Save the full autocorrelation function of the probe regressor at each pass.
* (rapidtide) Now save a version of the mean fMRI image prior to filtering
* (rapidtide) Added the ability to disable masking of sLFO filter. Use ``--nosLFOfiltmask`` to disable.
* (happy) Implemented multiprocessing for detrending, phase projection, and phase projection smoothing.
* (happy) Precache convolution gridding kernel outputs to improve repeatability.
* (happy) Early implementation of Wright's method for finding vessels.
* (showarbcorr) Fixed some internal issues.
* (simdata) Many changes to finetune the simulation and make the output more realistic.  Still in progress.
* (Docker) Turned on provenance and sbom in Docker builds.
* (package) Accepted several dependabot PRs.

## Version 3.0.6 (5/21/25)
* (rapidtide) Made a generic, reusable multiprocessing module and moved many functions over to it.
* (ccorrica) Samplerate is now set properly if input is a BIDS tsv file.
* (package) Added several new tests.

## Version 3.0.5 (5/7/25)
* (rapidtide) Loosened the sample rate matching when appending timecourses to files, and read NIFTI tr values using modern header features.  This should help address https://github.com/bbfrederick/rapidtide/issues/205 (thank you to https://github.com/phildeming for finding this).

## Version 3.0.4 (5/5/25)
* (rapidtide, retrotregress, tidepool) Support for maxcorrrefined map.
* (retroregress) Gracefully handles missing mask names in runoptions file.
* (package) Version works again!  Thank you Chris (https://github.com/effigies)!
* (package) Added a few more tests to catch edge cases that showed up in testing fmripost-rapidtide.

## Version 3.0.3 (5/4/25)
* (rapidtide, retroregress) If you supply a brain, GM, WM, or CSF mask, rapidtide extracts the matching timecourse from the mask at startup, and after filtering, if it's done.  retroregress extracts the timecourses after filtering.
* (rapidtide) Fixed mask logic when gray matter mask is provided.
* (rapidtide) Fixed a crash when ``--acfix`` is selected and a bad sidelobe is encountered if the similarity calculation range is not the entire timecourse (THAT was hard to pin down...).
* (rapidtide, tidepool) Fixed a serious cosmetic bug in corrout files introduced in v3.0.2 - teoffset was not being set properly by rapidtide, so the tidepool similarity function display was very wrong.
* (rapidtide) shiftedtcs file is now written during each pass if ``--saveintermediatemaps`` is set and ``--outputlevel`` is ``more`` or ``max`` (in response to a feature request).
* (rapidtide) Added back the ability to have integral powers of the motion regressors.
* (rapidtide) Added the ability to specify a CSF mask.


## Version 3.0.2 (4/28/25)
* (retroregress) Added a new argument to ``--outputlevel``, ``onlyregressors``.  When selected, retroregress generates the voxel specific sLFO regressor(s), saves them, and exits.  This option exists so that you can use the rapidtide voxel specific sLFO regressors in an external processing routine such as XCP-D without extraneous data storage or processing time. 
* (rapidtide) Leaned harder into using the voxelData class, and beefed up its capabilities.
* (happy) Updated to use voxelData.
* (happy) Moved phase projection to an external routine.
* (documentation) Fixed a few more typos.

## Version 3.0.1 (4/23/25)
* (documentation) I missed a bunch of weird errors in the "Theory of Operation" section.  Are you surprised?  I'm not surprised.  You shouldn't be either.  These are now fixed.  This doesn't change any code - mostly affects online documentation.

## Version 3.0 (4/22/25)
* (documentation) Added summary of 3.0 changes, fixed a few typos.

## Version 3.0alpha15 (4/22/25)
* (rapidtide) Fixed a bug in masking logic introduced in last version.
* (rapidtide) Pullled regressor cleaning steps out into a separate routine.
* (rapidtide) Removed unused "noiseregressor" code.  Its functionality is now covered with confound regression.
* (rapidtide) Removed memory profiling code that relied on an unsupported library.
* (rapidtide) Defined a new class to handle the input data.  This simplifies a lot of internal operations.
* (happy) Updated deep learning filter training and inference code, and saved models, for better tensorflow 2 compatibility.
* (docs) Updated installation instructions.
* (package) Updated copyright notices.

## Version 3.0alpha14 (4/8/25)
* (rapidtide) Rationalized handling of zero-mean data - now use the ``--dataiszeromean`` argument to tell rapidtide how to make masks.  ``--globalmaskmethod`` no longer exists.
* (retroregress) Some bugfixes.
* (atlasaverage) 4D file timecourse output is now a BIDS compliant json/tsv.gz pair, with timecourse labels, if desired.
* (delayvar) Secret bugfixes.
* (package) Fixed definition of ``APARC_WHITE`` to add cerebral white matter (oops!).
* (package) Made it easier to run local tests.
* (docs) Expanded theory of operation section.

## Version 3.0alpha13 (3/31/27)
* (rapidtide, retroregress) Preserve the sign of the sLFO filter fit R value (it might be useful for distinguishing arterial blood).
* (rapidtide) Fixed a bug with --dofinalrefine - now the shiftedtcs file IS written.
* (delayvar) New secret program, for secret purposes, to satisfy a secret question I had. More later.
* (package) Clarified that the final sLFO removal filter does NOT use a GLM (the current filter is an OLS filter) by renaming routines and command line options.

## Version 3.0alpha12 (3/19/25)
* (rapidtide) Added option to perform regressor refinement on last pass.
* (rapidtide) Moved regressor refinement out into its own class to make things more modular.
* (refinedelay) Put in some experimental code to do multiderivative delay fits.  Currently doesn't work very well.
* (tidepool) Changed default colormaps for similarity width, MTT.
* (package) Made a script to compile qt templates.
* (package) Added the ability to set value limits to some parser functions.

## Version 3.0alpha11 (3/5/25)
* (tidepool) Substantial speed improvement switching datasets.
* (tidepool) Colormap radio button is now set correctly.
* (tidepool) Mask menu label is now set correctly when restoring a loaded dataset.
* (versioneer) Updated to 0.29.
* (docs) tidepool's documentation has been updated to reflect the current state of the program.

## Version 3.0alpha10 (2/28/25)
* (tidepool) Full support for multiple concurrent datasets.  Dataset loading and selection is now done through the File menu.
* (tidepool) Load additional informational maps, and somewhat more intelligently.
* (tidepool) Fixed pane loading to use all available panes.
* (tidepool) Fixed image saving under pyqt6.
* (tidepool) For the regressor spectrum, only do FFT of the portion of the regressor used for similarity calculation.

## Version 3.0alpha9 (2/26/25)
* (rapidtide) Fixed the calculation of the regressor autocorrelation width (used in estimating MTT).
* (tidepool) First working version that allows loading multiple files. Use ``--dataset DATASET1 DATASET2...``.
* (tidepool) Added new "big" template to let you see more information at once.  Select with ``--uistyle big``.
* (tidepool) Pretty significant code cleanup under the hood.

## Version 3.0alpha8 (2/20/25)
* (rapidtide) Enhanced fit failure reporting.
* (rapidtide) Fixed map output type for non-float maps.
* (tidepool) Clarify how you specify the dataset name in the help text.
* (tidepool) Fixed some typos in outputs.
* (tidepool) Made some internal changes in preparation for allowing loading multiple files.

## Version 3.0alpha7 (2/11/25)
* (happy) Trying out some new approaches for aliased correlation.
* (happy) Put in some initial support for multiecho data.
* (happy) Updated argument parser.
* (package) Various modernizations and cleanups.

## Version 3.0alpha6 (1/28/25)
* (happy) Fixed a major bug in motion filtering (thank you to Sukru Demiral for reporting it!)

## Version 3.0alpha5 (1/25/25)
* (happy) Fixed a typo in argument description.
* (atlasaverage) Added the ability to segment a template atlas based on values of the map being summarized.
* (docs) Significantly beefed up happy section.
* (reference) Constructed a new version of the JHU LVL1 atlas where each territory is split into an early and late subterritory, based on the median delay in the HCP-YA dataset.
* (package) Fully converted to using pyproject.toml for packaging.

## Version 3.0alpha4 (1/21/25)
* (rapidtide, tidepool) Both now only use neglog10p map to dynamically make significance masks.
* (rapidtide, Docker) rapidtide should now properly report its version number in a container even if it is in prerelease.

## Version 3.0alpha3 (1/16/25)
* (rapidtide) Refine delay is now on by default.  Use --norefinedelay to disable it.  --refinedelay still exists for compatibility, but does nothing.
* (rapidtide) Now outputs a map of fit significance (-log10(p)).  In response to https://github.com/bbfrederick/rapidtide/issues/174.
* (showxcorrx) Fixed some issues with axis labelling.

## Version 3.0alpha2 (12/30/24)
* (happy, rapidtide) Now do (correct) checking to see if running in a container that now handles both Docker and Singularity/Apptainer properly. Thank you to Derek Monroe (https://github.com/dcmonroe) for the catch and the fix!
* (package) Now includes codespell checking thanks to Yaroslav O. Halchenko (https://github.com/yarikoptic)!

## Version 3.0alpha1 (12/20/24)
* (rapidtide) The ``--fixdelay`` option has been split into two options.  ``--initialdelay DELAY`` lets you specify either a float that sets the starting delay for every voxel to that value, or a 3D file specifying the initial delay for each voxel.  ``--nodelayfit`` determines whether the delay can be adjusted from its initial value.  Closes https://github.com/bbfrederick/rapidtide/issues/171. KNOWN ISSUE:  If you supply an initial delay map, instead of using the global mean, rapidtide should use the delays to make your first stage regressor.  Currently that is not the case.
* (rapidtide) Reorganized command line options and adjusted the default values.
* (rapidtide) Help output now shows the filter ranges.
* (rapidtide, retroglm) Added delay refinement using the ratio of the fit coefficients of the regressor and its time derivative.
* (rapidtide, retroglm) Fixed a bad shared memory leak.
* (retroglm) Significantly enhanced logging.
* (retroglm) Added canary files.
* (rapidtide) Implemented delay map patching.
* (rapidtide) Write out individual EV timecourses.
* (Docker) Cleaned up some internal variables.
* (Docker) Improved build and testing scripts.
* (io) Added function to compare nifti files with some tolerance.
* (docs) Automated more table generation.
* (package) Merged some dependabot PRs.
* (package) Fixed a fairly big, but not necessarily impactful bug. mlregress returned R2, not R, so anything referring to the R of a fit was actually squared (R was actually R2, R2 was actually R4).  NB: this does NOT affect the calculation of maxcorrsq, which does not use mlregress.

## Version 2.9.9.5 (11/15/24)
* (deployment) New idea - split the README.rst file to remove potentially offending reference stuff.

## Version 2.9.9.4 (11/15/24)
* (deployment) Attempt to fix deployment by essentially giving up.

## Version 2.9.9.3 (11/15/24)
* (deployment) Yet another attempt to fix deployment.

## Version 2.9.9.2 (11/14/24)
* (rapidtide) Fixed https://neurostars.org/t/indexing-error-during-regressor-refinment/30790
* (rapidtide) Addressed https://github.com/bbfrederick/rapidtide/issues/175
* (rapidtide) Added the ability to mask data before spatial filtering.
* (rapidtide) Revised shared memory access to use more modern interface.
* (rapidtide) Clarified option descriptions in help output.
* (Docker) Temporary workaround for https://github.com/bbfrederick/rapidtide/issues/180
* (deployment) Tried to fix pypi deployment.
* (showtc) Fixed legends when using multiple input files.
* (docs) More reference reformatting.

## Version 2.9.9.1 (10/28/24)
* (rapidtide) Properly set the "_nodefault" flag on "passes".  Should resolve https://github.com/bbfrederick/rapidtide/issues/178.
* (rapidtide) Terminate despeckling if the number of voxels to despeckle is not changing.
* (rapidtide,ccorrica) Changed default correlation weighting to "phat".
* (ccorrica) Output filtered timecourses.
* (docs) More documentation enhancements by tsalo.
* (docs) Updated requirements.txt file to reflect recent documentation changes so readthedocs build would complete.
* (docs) Converted all references to bibtex.
* (package) Merged some dependabot PRs.

## Version 2.9.9 (10/22/24)
* (docs) Integrated lots of documentation upgrades from Taylor Salo (tsalo).
* (rapidtide) Changed correlation mask to use brain mask as correlation mask if provided.
* (rapidtide) Lowered the default lagminthresh, lagmaxthresh.
* (rapidtide,happy) Improved creation and deletion of canary files.
* (dlfilter) Simplified to support only newer versions of tensorflow 2.X and fix some deprecation problems.
* (atlastool) Fixed 4D processing.
* (showtc) Allowed display of timecourses with different x axes on the same timeline.
* (package) Defined a new filter band - "lfo_tight" from 0.01-0.1Hz (matches a lot of papers).
* (package) Enhanced filter testing.
* (package) Updated requirements to fix tensorflow installation.
* (package) Merged some dependabot PRs.

## Version 2.9.8.2 (9/24/24)
* (package) Pypi gets mad when you redeploy with the same tag number...

## Version 2.9.8.1 (9/24/24)
* (retrolagtcs) Now only use one mask (corrmask).
* (retrolagtcs) Now only load the fmri file header to save memory.

## Version 2.9.8 (9/24/24)
* (retrolagtcs) Made a new program to generate lagged timecourses from a lagmap, masks, and a lagtc generator (for fmripost_rapidtide).
* (rapidtide) Added coefficient of variation output.
* (tidepool) Now works with PyQt6.
* (package) Some import cleanups.

## Version 2.9.7 (9/16/24)
* (package) Updated requirements for matplotlib and pyqtgraph to make sure tidepool works (thanks to Hui Zheng for finding this - resolves https://github.com/bbfrederick/rapidtide/issues/168).
* (package) Dropped Python 3.8 support (not supported by the version of pyqtgraph I need).
* (filter.py) You can now specify padtype to all filters.
* (reference) Added the newest version of the HCP-YA mean parameter maps.

## Version 2.9.6 (9/5/24)
* (rapidtide) Estimate and save information about sLFO signal growth over the course of a scan.
* (rapidtide) Now you can have both include and exclude tmasks.
* (rapidtide) Fixed some bugs in memory profiling.
* (rapidtide) Removed some extraneous code to save memory.

## Version 2.9.5.2 (8/14/24)
* (package) Deployment fix

## Version 2.9.5.1 (8/14/24)
* (rapidtide) Backed off on using the gray matter mask to limit refine voxels.

## Version 2.9.5 (8/14/24)
* (rapidtide) Now specifying brainmask and/or graymattermask will automatically do smart processing enhancements.  You can specify whitemattermask, but that currently does nothing except pass the map along.
* (rapidtide) pickleft is now the default (you never don't want to do it).  The argument to specify it does nothing, but doesn't throw an error.  You can turn it off with nopickleft (but, seriously, don't).
* (package) In addition to the APARC label macros for specifying atlas values, I've added SSEG_GRAY and SSEG_WHITE to support gray and white mask generation from SynthSeg output.
* (package) Fixed a rarely encountered but serious bug when reading tsv files with missing values - these are now converted to zeros on read rather than propagating NaN's through processing.

## Version 2.9.4.1 (8/7/24)
* (package) Deployment fix

## Version 2.9.4 (8/7/24)
* (rapidtide) Added "--graymattermask" macro to set region for global mean, refinement, and offset calculations.
* (rapidtide) Added whitemattermask input.
* (rapidtide) Calculate normalized motion derivatives when denoising.
* (filttc) Various improvements.
* (docs) Automated calculation of rapidtide data output sizes.
* (package) Multiple buffs and improvements to packaging and installation files.
* (package) Merged many dependabot PRs.

## Version 2.9.3 (6/15/24)
* (rapidtide) Aligned names of histograms with the maps they summarize.
* (rapidtide) Renamed ``--awsmode`` to ``--savecorrout``.
* (docs) Updated file output table.

## Version 2.9.2.2 (6/15/24)
* (rapidtide) Added ``--awsmode`` to force the output of the corrout file independent of output level.
* (rapidtide) When doing confound filtering, output a histogram of filter R2 values.

## Version 2.9.2.1 (6/14/24)
* (rapidtide) Bugfixes, including more graceful handling of null confound regressors.
* (resampletc) Prepping for some additional features.

## Version 2.9.2 (6/12/24)
* (rapidtide-cloud) Made nda-download-data take a dataset name.

## Version 2.9.1 (6/12/24)
* (rapidtide) Made some tweaks to confound regression.
* (rapidtide) Fixed a logging error when doing noise regressor cleaning.
* (rapidtide2std) Also register confoundfilter R2 map.
* (io) Added the ability to read in FSL design.mat file as multicolumn text files.
* (qualitycheck) Added some new metrics.

## Version 2.9.0 (6/3/24)
* (rapidtide) Inband variance maps (Before, After, Change) are now always output if a GLM is performed.
* (rapidtide) Tuned outputlevel configurations.  Added a new level, "less".
* (rapidtide) Fixed globallaghist output.
* (rapidtide2std) Updated for new file names, added option to transform GLM filtered data.
* (fixtr) New utility to change the TR in a nifti header.
* (reference) Made a new lut for JHU LVL1 NoVent to try to highlight hierarchy.
* (package) Found and fixed a lot of weird, rare bugs (mostly typos).

## Version 2.8.9.2 (5/22/24)
* (docs) More updates to output file names, descriptions, and size tables.

## Version 2.8.9.1 (5/21/24)
* (rapidtide) Fixed a very old bug where the loggers were not properly closed, leading to duplicate output messages when rapidtide was run as a workflow.
* (rapidtide) Fixed the selection logic so that shiftedtcs are written if output level is max, passes > 1, but GLM is no longer required.
* (rapidtide) Added "Description" tag to output json files.
* (rapidtide) Rationalized the names of some output files.
* (docs) The output files list now reflects the current output selection logic.
* (docs) Made a semiautomated procedure for updating the output table.
* (package) Merged some dependabot PRs.

## Version 2.8.9 (5/17/24)
* (rapidtide) Simplified specification of output files.  Now use ``--outputlevel`` with values "min", "normal", "more", or "max" to specify what files are output.
* (rapidtide) Improved motion regression and made it more flexible.
* (rapidtide) Merged all bids nifti file writes into a common output routine.
* (retroglm, rapidtide) Factored out all the GLM calculation code so it can be done (or redone with different options) after the fact with "retroglm" as a separate step.
* (retroglm, rapidtide) Allow arbitrary number of derivatives of lagged regressors, and properly save all EVs and fit coefficients.
* (niftidecomp) Now output raw variance explained in addition to variance explained ratio.
* (tidepool) Now show graphically what part of the probe regressor was used for similarity calculation.
* (package) Replaced the "mlregress" routine of unknown provenance with LinearRegression from sklearn.
* (package) Merged some dependabot PRs.
* (docs) Added a section on output data size.

## Version 2.8.8 (4/30/24)
* (rapidtide) Properly initialize "starttime" so that it will use the value in a json file, if present.
* (rapidtide) Added new option to start from a random probe regressor.
* (rapidtide) Moved performance options into their own section.
* (rapidtide) Cleaned up code that reads (or rereads) data prior to GLM.
* (rapidtide) You can specify MKL threads AND multiprocessing - multithreading is disabled and re-enabled automatically as needed.
* (rapidtide) Do refinement on padded data, to infer data past the ends of the imaging window.
* (rapidtide) Save the padded, lagged timecourse generator.
* (rapidtide) Scale voxel timecourses prior to PCA refinement.
* (rapidtide) Change default refinement options.
* (rapidtide) Started to increase the granularity of output file specification.
* (fit) Added routines to do automated component selection from scree tests.
* (package) Updated all calls to np.polyfit to the new Polynomial class.
* (package) Merged some dependabot PRs.
* (package) Made pyfftw an optional dependency.
* (package) Fixed numba crashes (on Intel - not tested on ARM yet) and made it an optional dependency.
* (package) Made compatible with Python 3.12.

## Version 2.8.7 (4/17/24)
* (rapidtide) Spatial filtering with the default kernel (1/2 the average voxel dimension) is now the default.
* (rapidtide) Rewrote the lag time rank order calculation to use scipy.signal.rankdata.
* (rapidtide) Moved some command line options to the "experimental" section, where they should have been to begin with.
* (rapidtide) Moved GLM options to a new "GLM" section.
* (rapidtide) Do not automatically disable windowing for linear correlation.
* (rapidtide, happy) Reorganized, expanded and rationalized confound regression code.
* (tidepool) Replaced the flame colormap option with plasma and jet with turbo (similar, to what they replaced, but perceptually uniform).
* (package) Made some proactive fixes for numpy 2.0.0 compatibility.
* (package) Merged some dependabot PRs.
* (tests) Removed some superfluous metadata from test data files.
* (docs) Further extensive revisions.

## Version 2.8.6 (4/5/24)
* (rapidtide) Tweaked the behavior of the ``--CVR`` flag.
* (rapidtide) Made major improvements to motion regression.
* (rapidtide) Consolidated all glm filtering into a single multiprocessing routine, added some new capabilities to rapidtide GLM filtering.
* (resampletc) Renamed from resamp1tc to make the program names more consistent.
* (package) Made pyfftw an optional dependency, since it seems to be the only thing blocking python 3.12 compatibility.
* (package) Added some new tests.
* (package) Major reorganization of most scripts to make them workflows (this is to make the calling interfaces and documentation consistent.)
* (package) Incorporated dependabot changes.
* (docs) Further cleanup of the program documentation section, especially rapidtide usage.
* (docs) Expanded the Theory of Operation section.
* (docs) Fixed many typos.

## Version 2.8.5.1 (4/1/24)
* (docs) Extensive revision of the rapidtide program usage section.

## Version 2.8.5 (3/30/24)
* (rapidtide) Adjusted some default parameters to reflect the current best picks.
* (rapidtide) Fixed the ``--denoising`` option (it didn't actually do anything!)
* (package) Partially implemented a major reorganization of all the programs in the package to make them workflows with standard interfaces.
* (docs) Started the process of standardizing the program descriptions and properly using autodoc.

## Version 2.8.4 (3/28/24)
* (rapidtide) Output some .json sidecars that I had neglected.
* (glmsim) New program to help develop instructional tools re: how rapidtide works. This is a WIP.
* (docs) Major revisions to the rapidtide usage instructions.
* (package) Accepted several dependabot changes.

## Version 2.8.3 (3/7/24)
* (rapidtide) Fixed the logic for saving lagregressors - they only exist if you do GLM or CVR analysis, so if you set nolimitoutput, check for existence first (thanks to Laura Murray for finding this bug).
* (rapidtide) Changed the name of the file containing the voxel specific EVs that are regressed out by the GLM from "lagregressors_bold" to "lfofilterEVs_bold" (thanks to Tianye Zhai for flagging this).
* (localflow) Added a new program to test a hunch.
* (fit) Gracefully handle singular matrices in mlregress.
* (reference) Corrected the abbreviated name for the MLSR region in the JHU level 1 atlas xml file (oops!).
* (docs) Added description of the lfofilterEVs_bold and shiftedtcs_bold output files to the usage section.

## Version 2.8.2 (2/26/24)
* (rapidtide) Added a lot more internal debugging resources, and fixed a bug where zero range time data that was included due to explicit masks would put NaN's in the maps.
* (rapidtide) Implemented multiprocessing to speed up motion regression.

## Version 2.8.1 (2/19/24)
* (cloud) Now using an s3 working folder.

## Version 2.8.0 (2/18/24)
* (Docker) Set to basecontainer_plus:latest-release to simplify builds.

## Version 2.7.9 (2/18/24)
* (runqualitycheck) Added new tests.
* (reference) Added MNI152NLin2009cAsym versions of the HCP reference maps.
* (cloud) Added support for selecting the dataset  (currently HCPA and ABCD are supported).

## Version 2.7.8 (1/31/24)
* (rapidtide) Added new feature - ``--numtozero NUMPOINTS`` allows you to specify how many points at the beginning of the data set to set to zero prior to processing.  This means that the sLFO fit does not get contaminated by synchronous noise at the beginning of the timecourse (such as T1 decay).  The initial timepoints are filled in from timecourses shifted forward in time.  This means better correlation of timecourses with the sLFO, and better noise removal with the GLM.
* (rapidtide) Fixed a bug in how setting timerange and simcalc range interacted (and in range checking).  It all works now, and simcalc range is specified relative to the restricted time range.
* (runqualitycheck) Fixed some masking bugs.

## Version 2.7.7 (1/31/24)
* (runqualitycheck) Added new tests, and the ability to optionally do tests restricted to gray and/or white matter.
* (package) The makeandsavehistogram routine now saves some useful histogram stats in the json file.
* (package) Added the ability to specify APARC_WHITE and APARC_ALLBUTCSF macros to a mask specification if you have an aparc+aseg file.
* (RapidtideDataset) You can now optionally include gray and white matter masks in the dataset.

## Version 2.7.6 (1/29/24)
* (rapidtide) Added the ability to calculate delays over a limited time range, but still GLM filter the entire timecourse.
* (rapidtide) Fixed a very old bug in null significance distribution estimation.  Multiple worker processes all start with the same random seed (unless you explicitly fix that).  Who knew?
* (rapidtide) Improved significance distribution model fitting for mutualinfo similarity metric.  The distribution is a Gaussian, not a Johnson distribution (as it is for selected correlation).
* (runqualitycheck) Added an automated quality assessment tool.  This will likely evolve quite a bit over time.
* (rapidtide2std) Updated for new maps; also copied over timecourses and options so you can load a rapidtide2std dataset into tidepool.
* (atlasaverage) Set output NIFTI size properly for 3D templates.
* (testing) Parallelized tests on CircleCI for a significant speedup.
* (package) Updated copyright messages, made headers more consistent, removed some old SCCS tags.

## Version 2.7.5 (1/13/24)
* (rapidtide) Moved lagtc generation out of fitcorr into its own module.  This will help with implementation of new, secret evil plans.
* (rapidtide) Altered voxel selection logic for multiproc correlation fitting.  Now singleproc and multiproc outputs are the same.
* (rapidtide) Fixed a multiprocessing bug that's been there since multiprocessing was added - any job with an integral multiple of 50000 tasks would die (don't ask).
* (rapidtide) Fixed a bug that allowed NaNs into the lfoCleanedR2 map.
* (rapidtide) General code cleanup.
* (package) Accepted some dependabot PRs for security updates.

## Version 2.7.4 (1/10/24)
* (rapidtide) Fixed a crash when despeckling is turned off. (thank you to Wesley Richerson for finding this).
* (rapidtide) Adjusted the regreessor frequency setting logic.
* (rapidtide) Adjusted the default absminsigma to 0.05s.
* (rapidtide) Moved motion regression before global mean correction.
* (rapidtide) Properly reinitialize the motion regression output file if you have a previous run.

## Version 2.7.3.3 (12/18/23)
* (rapidtide-cloud) Another bump to improve NDA access.

## Version 2.7.3.2 (12/18/23)
* (rapidtide-cloud) Another bump to improve NDA access.

## Version 2.7.3.1 (12/18/23)
* (rapidtide-cloud) Redoing push to fix a regression due to _somebody_ not testing before deploying (tsk tsk).

## Version 2.7.3 (12/18/23)
* (correlate) Added a new correlation weighting - "regressor", that whitens the correlation spectrum relative to the probe regressor.
* (rapidtide) Add support for the "regressor" correlation weighting.
* (rapidtide) Linear (rather than circular) correlations are now the default.
* (rapidtide) Add infrastructure to support baseline correction of correlation function during lag estimation.
* (rapidtide) Added lag rank map (each voxel is the percentile within the lag time distribution)
* (showarbcorr) Numerous bugfixes and functionality improvements.
* (tidepool) Support new lag rank map.
* (rankimage) Convert lag maps to lag rank maps.
* (rapidtide-cloud) Added tools for NDA download.

## Version 2.7.2 (12/12/23)
* (Docker) Bumped to basecontainer_plus v0.0.3.
* (Docker) Removed push of new containers to ECR.

## Version 2.7.1 (12/12/23)
* (Docker) Fixed deployment of new containers to ECR.

## Version 2.7.0 (12/11/23)
* (Docker) Added caching to build.
* (Docker) Switched to basecontainer_plus to pick up some FSL utilities.

## Version 2.6.9.1 (12/11/23)
* (package) Fixing some mysterious deploy errors.

## Version 2.6.9 (12/11/23)
* (filter) Updated predefined filter bands to include hrv frequency ranges.
* (happy) Tried a new approach for aliased correlation.  Not really done yet.
* (docs) Updated installation instructions.
* (docs) Fixed a build problem.
* (Docker) Update to basecontainer v0.3.0.

## Version 2.6.8 (11/21/23)
* (rapidtide) Rapidtide is now less chatty by default.
* (rapidtide) Put the significance estimation command line options in their own subsection.
* (tidepool) Updated to support the newest pyqtgraph (>=0.13.0).
* (Docker) Update to basecontainer v0.2.9.1.
* (package) Increased test coverage to 49.26%.
* (docs) Fully documented tidepool.

## Version 2.6.7 (10/31/23)
* (Docker) Update to basecontainer v0.2.7.
* (rapidtide) Added the option to delete noise signals from the probe regressor (mostly due to slow breathing).  Currently not working.
* (rapidtide) All outputs from rapidtide are in BIDS derivative format.  The ability to select legacy outputs has been removed.
* (happy) All outputs from happy are in BIDS derivative format.  The ability to select legacy outputs has been removed.
* (rapidtide2x_legacy) The legacy version of rapidtide (rapidtide2x_legacy) has been removed.
* (happy_legacy) The legacy version of happy (happy_legacy) has been removed.
* (showtc) Fixed a very old bug that caused some timecourses to not be properly aligned if they had different start times.
* (showtc) Added ability to normalize all timecourses to make displaying them together more informative.
* (package) Added selfcontained routines to do glm filtering (with or without polynomial expansion, and to align timecourses.
* (package) Added macros for selecting all the gray matter values in an aparc+aseg file.
* (package) Accepted dependabot changes.
* (rapidtide-cloud) Added basecontainer to AWS.
* (rapidtide-cloud) Various tweaks and changes to AWS authentication procedures to deal with NDA.
* (docs) Some updates to theory of operation.

## Version 2.6.6 (10/7/23)
* (adjustoffset) New tool to alter overall delay offset in maxtime maps.
* (Docker, package) Really, truly, actually fixed version reporting.
* (rapidtide) Added debugging option to disable docker memory limit "fix".

## Version 2.6.5 (10/4/23)
* (rapidtide) Report version on startup.  Resolves https://github.com/bbfrederick/rapidtide/issues/91.
* (Docker, package) Fixed version tagging and reporting.  Resolves https://github.com/bbfrederick/rapidtide/issues/96.
* (Docker) Moved some time consuming installations into basecontainer to make building new containers MUCH faster.
* (package) Merged some dependabot security PRs.
* (diffrois) Fixed handling of missing values.

## Version 2.6.4 (9/28/23)
* Mass merge of more dependabot PRs.
* (diffrois) Added a new program to make "vasculomes" - measuring delay differences between ROIs.  This is still in flux.
* (fingerprint, atlasaverage) Implemented a standard masking method with atlas indexed include and exclude masks, and an extra geometric mask.
* (fingerprint) Bug fixes.

## Version 2.6.3 (9/13/23)
* Mass merge of a bunch of dependabot PRs.
* (rapidtide) Fixed return values from findavailablemem() when running in a Docker container with cgroups v1.  Thank you to Jeffrey N Stout for finding this.  Should resolve https://github.com/bbfrederick/rapidtide/issues/122. 
* (Docker) Updated to basecontainer 0.2.3.

## Version 2.6.2 (8/29/23)
* (atlastool) Add ability to use ANTs alignments.
* (atlasaverage) Add ability to restrict statistics to non-zero voxels.
* (documentation) Started beefing up the "Theory of operation" section.
* (Docker) Set memory limits on resource use when running in Docker containers so you don't get silent out of memory failures.

## Version 2.6.1 (8/17/23)
* (rapidtide) Fixed crash when using ``--acfix`` option.  Thanks to Jakub Szewczyk for spotting this.  Should resolve https://github.com/bbfrederick/rapidtide/issues/115.
* (atlasaverage) Added text region summary outputs.
* (atlastool) Enhancing spatial registration options.
* (package) Initial steps to implementing a more flexible way of applying external registration tools to data.
* (package) Moving closer to a single pyproject.toml file with all the packaging information in it.:
* (Docker) Updated to basecontainer 0.2.1 and added new cleanup operations - the container is now ~30% smaller.

## Version 2.6.0 (8/10/23)
* (rapidtide) Added new ``--CVR`` analysis type to generate calibrated CVR maps when given a CO2 regressor as input.  Thanks to Kristina Zvolanek for the suggestion to add it!
* (rapidtide) Fixed calculation and output of variance change after GLM filtering.
* (happy) Moved support functions into a separate file.
* (simdata) Added separate voxel level and volume level noise specification, and a test script.
* (documentation) Added information on CVR mapping outputs, updated funding information.
* (package) Made ArgumentParser initialization uniform to make automatic documentation easier.
* (package) Removed Python 3.7 support (mostly because it doesn't support all the features of f-strings I use.)

## Version 2.5.8 (8/3/23)
* (rapidtide) ``--nofitfilt`` now actually works.  Thank you to https://github.com/poeplau for finding (and fixing) the problem!  Resolves https://github.com/bbfrederick/rapidtide/issues/114

## Version 2.5.7 (5/15/23)
* (glmfilt) Added ability to specify a mask, and to limit output files.

## Version 2.5.6 (5/14/23)
* (niftidecomp) Made some major internal changes to allow processing multiple files at once.
* (gmscalc) New program to do some global mean signal calculations within the package (so we can do them on AWS).

## Version 2.5.5 (5/11/23)
* (Docker) Updated to python 3.11 basecontainer.
* (package) Modernized install procedure.

## Version 2.5.4 (5/10/23)
* (rapidtide) Default to using N processors rather than N-1 when nprocs=-1.  You can toggle old behavior with ``--reservecpu``.
* (rapidtide-cloud) Rapidtide will record the instance type if running on AWS in the options file (AWS_instancetype).

## Version 2.5.3.1 (5/9/23)
* (rapidtide, happy) Fixed a crash when you DIDN'T specify informational tags (SMH).

## Version 2.5.3 (5/9/23)
* (rapidtide, happy) Added the ability to save arbitrary informational tags to the run options (or info) json files using the ``--infotag`` command line argument.

## Version 2.5.2 (5/8/23)
* (rapidtide) Now allow the global mean mask to be completely outside of the correlation mask (the fact this previously wasn't allowed was a bug).  Thank you to Daniele Marinazzo for finding this.
* (rapidtide) Fixed a bug in formatting run timings.
* (filttc) Now allow normalization before or after filtering.
* (showxcorrx) Made fit width limits configurable.
* (calcicc) Moved main calculations into niftistats, made two shells to calculate either icc or tests.
* (package) Disabled numba because of multiple bugs and incompatibility with py3.11 and ARM.
* (package) Made some updates to rapidtide cloud routines to make things a bit more stable.

## Version 2.5.1.2 (4/28/23)
* (package) New release to trigger ECR upload.

## Version 2.5.1.1 (4/28/23)
* (package) New release to trigger ECR upload.

## Version 2.5.1 (4/28/23)
* (package) New release to trigger ECR upload.

## Version 2.5 (4/28/23)
* (package) Fixed and upgraded tests both locally and on CircleCI.
* (package) Fixed coverage calculation and upload and increased coverage to 52%.
* (package) Made some changes to support our new AWS account (dmd).
* (reference) Added XML files for the JHU level 2 arterial atlases.

## Version 2.4.5.1 (4/10/23)
* (docs) Removed duplicate funding source.  Hopefully this will resolve the Pypi upload issue.

## Version 2.4.5 (4/10/23)
* (docs) Added some new sections to theory.
* (package) Completely changed the way I handle and distribute test data.  This makes the package much smaller (~17M), which should fix pypi deployment.  This involved several changes to the Docker and circleCI workflows, which I think are now stable.

## Version 2.4.4 (3/30/23)
* (examples) Separated essential test data from developer test data to make the installation much smaller.
* (package) Major modernization to package build and test files.

## Version 2.4.3 (3/30/23)
* (rapidtide) Some work on phase permutation for null correlation calculation.
* (happy) Put in the skeletons of some new features (upsampling, optical flow calculation).
* (OrthoImageItem.py) Removed the last of the obsolete pyqtgraph calls.
* (package) Attempting to modernize the packaging scripts to avoid deprecated behavior.
* (package) Several changes to fix the build environment.

## Version 2.4.2 (2/8/23)
* (rapidtide) Added ability set a threshold value for "equivalence" of spatial dimensions of NIFTI files (rather than requiring an exact match) using the ``--spatialtolerance`` option.
* (rapidtide) Added "offset masks" to set the region that defines "zero" time offset.
* (fingerprint) Added several new summary statistics for each region.
* (fingerprint) Allow the use of 3D masks with 4D data.
* (tidepool) Resolve a deprecation in Qt.
* (tidepool) Made tidepool less chatty by default (default verbosity is now 0).
* (Docker) Cleaned up the container build.
* (package) Critical bug fix for multiprocessing in python versions 3.10 and above ("10" < "8"!  Who knew?)
* (package) Added ".csv" files as a supported text file type.
* (package) Improved version handling when in a container.
* (reference) Added XML files to make the JHU arterial atlases loadable in FSLeyes.

## Version 2.4.1 (10/12/22)
* (package) Spruced up all the progress bars with tqdm.
* (deployment) Improved the testing structure to cache environment builds.
* (Docker) Build python environment with pip rather than conda now.

## Version 2.4.0 (10/6/22)
* (rapidtide) Added enhanced variance removal assessment.
* (rapidtide) Fixed a rare crashing bug in proctiminglogfile.
* (rapidtide) Output some files indicating run status.
* (package) Fixed a deprecation warning in pearsonr.
* (Docker) Now build amd64 and arm64 containers.

## Version 2.3.1 (9/27/22)
* (Dockerfile) Some tweaks to package versions to try to eliminate error messages.
* (Dockerfile) Add some AWS libraries to facilitate using S3 volumes.
* (Dockerfile) Moved timezone data loading earlier in the file to accommodate the new libraries.
* (reference) Added HCP_negmask_2mm to improve map display.
* (github) Updated codeql actions to v2.

## Version 2.3.0 (9/23/22)
* (rapidtide) Fixed option setting for nirs mode and did some tests on real data.
* (Dockerfile) Rolled back the version of pyfftw in the container to avoid the annoying (and erroneous) warnings about the plan file.
* (package) Made some changes to the reference files and dispatcher to help with upcoming AWS deployment.

## Version 2.2.9 (9/21/22)
* (showarbcorr) Now show the correlation function, fixed a typo
* (reference) Added slicetimes files for some large datasets

## Version 2.2.8.1 (8/29/22)
* (package) Fixed versioneer installation.

## Version 2.2.8 (8/29/22)
* (happy) Some under the hood tweaks to phase analysis to prep for multidimensional phase projection.
* (rapidtide) Exploring the use of complex PCA.
* (util) Added tcfrom2col.
* (rapidtide) Added explicit selection of linear and circular correlations.
* (package) Updated python environment for Docker.
* (package) Updated versioneer.
* (package) Changed some config files to try to fix documentation builds.

## Version 2.2.7.1 (6/30/22)
* (Dockerfile) Updated to a consistent python environment.

## Version 2.2.7 (6/29/22)
* (rapidtide) Fixed GLM noise removal in CIFTI files.
* (rapidtide) Initial support for linear rather than circular correlation.
* (happy) Fixed some pretty broken masking logic when you want to process more (rather than less) voxels than the brain.

## Version 2.2.6 (5/17/22)
* (fingerprint) Various fixes to mask handling.
* (package) Staged some prep work on updating the setup/installation files.

## Version 2.2.5 (4/26/22)
* (rapidtide) Postprocess timing information to make it more useful.
* (rapidtide) Re-enabled numba by default.
* (fingerprint) Fixed handling of 4D atlases, empty regions, and 4D masks.  Added "constant" template, and allow 0th order processing (mean).
* (atlastood) Fixed 4D atlas handling.  Now mask atlas after collapsing to 3D.
* (histnifti) Added ``--transform`` flag to map values to percentiles.

## Version 2.2.4 (4/11/22)
* (fingerprint) Now works properly for 3D input files.
* (tidepool) Turned the default level of verbosity way down, but gave you the ability to crank it back up.
* (RapidtideDataset.py) Fixed the default type of "numberofpasses".

## Version 2.2.3 (4/1/22)
* (rapidtide) Added a new feature, ``--globalmeanselect``, to try to locate a good, uniform, short delay pool of voxels to use for the initial global mean signal.  This is an attempt to fix the "poison regressor" problem - if the initial regressor contains data from multiple, distinct pools of voxels with different delays, the initial global regressor is strongly autocorrelated, and delay fits become ambiguous.  This cannot be corrected by refinement, so better to avoid it altogether. This option selects only voxels with clear, short delays, after a single pass with despeckling disabled.  The result is a mask (XXXdesc-globalmeanpreselect_mask.nii.gz) that can be used with ``--globalmeanincludemask`` for a subsequent run.
* (rapidtide) Fixed a nasty bug that caused offsettime and lagminthresh to interact incorrectly, sometimes leading to almost no voxels for refinement.
* (happy) Moved some code around, changed some internal names, and added secret bits to support future, secret, features.
* (tidepool) Trying to add a little more clarity to the user about image orientation (the image's affine transform is correct, so the mapping between voxel and MNI coordinate is correct, but currently it's not clear if displayed images are radiological or neurological orientation.
* (fingerprint) Added the JHU atlases as options.
* (package) Added slightly modified version of the JHU arterial territorial atlases to the reference section (Paper: https://doi.org/10.1101/2021.05.03.442478, Download: https://www.nitrc.org/projects/arterialatlas).
* (Docker) Fixed a dependency problem for pyfftw (resolves https://github.com/bbfrederick/rapidtide/issues/79)
* (pony) One time offer, today only - every user gets a pony upon request!

## Version 2.2.2 (3/16/22)
* (happy, happy_legacy, simdata) This release corrects a flaw (or perhaps more accurately an ambiguity) in slice time specification.  In FSL slicetime files, slicetimes are specified in fractions of a TR. In .json sidecars, they are specified in seconds. This is now detected on file read, and slicetime files are now converted to seconds.  Until now, happy and simdata assumed all slice times were in seconds.  This will fix behavior when FSL-style (fractional TR) slicetime files are used.  Behavior with .json sidecars is not changed.  Non-json files are assumed to be the FSL style (fractions of a TR) UNLESS the ``--slicetimesareinseconds`` flag is used.

## Version 2.2.1 (3/16/22)
* (rapidtide) Tweaked mask checking logic to address a bug introduced by despeckling changes.
* (histtc, histnifti) Harmonized options between the programs.
* (Docker) Updated Dockerfile to fix a bug that caused automatic build to fail.

## Version 2.2.0 (3/11/22)
* (rapidtide) Major rethink of despeckling.  Despeckling no longer causes negative correlation values when bipolar fitting is not enabled, and voxel parameters are only updated in a despeckled voxel if the correlation fit succeeds.  This results in better fits without mysteriously unfit voxels.
* (showxy) Bland-Altman plots can now use alternative formatting, as per Krouwer, J. S. Why Bland–Altman plots should use X, not (Y+X)/2 when X is a reference method. Stat Med 27, 778–780 (2008).
* (fingerprint) This program is now substantially more useful, working on 4D input files.  Output files are more convenient as well.
* (cleandirs) Cleandirs now keeps cleaning until it runs out of old installations to remove.
 
## Version 2.1.2 (1/10/22)
* (calcicc, calctexticc) Some fixes to indexing.

## Version 2.1.1 (11/4/21)
* (spatialmi, calcicc) Major improvements in performance, stability, and flexibility.
* (showtc) Added support for files with large star time offsets.
* (showxy) Some appearance tweaks.
* (niftidecomp) Improved mask generation.
* (variabilityizer) New program to transform fMRI datasets to variability measures.

## Version 2.1.0 (9/21/21)
* (spatialmi) Added new program to calculate local mutual information between 3D images.
* (calcicc) Tool to calculate ICC(3,1) - quickly - for a set of 3D images.
* (correlate.py) Fixed the reference for mutual_info_2d.
* (package) Simplified and cleaned up release process.

## Version 2.0.9 (8/26/21)
* (rapidtide) Fixed a strange edge case that could lead to "hot pixels" in the maxcorr map.
* (io) Added a "tolerance" for spatial mapping of niftis to account for rounding errors in header creation.

## Version 2.0.8 (8/20/21)
* (rapidtide) Disabled processing of abbreviated arguments.
* (showtc) Suppressed some unnecessary output when not in debug mode.
* (Docker) Added automatic build and push as a github action.

## Version 2.0.7 (8/19/21)
* (reference) Include the new JHU digital arterial territory atlas.
* (Docker) Updated container to Python 3.9.
* (package) General cleanup of imports.

## Version 2.0.6 (8/16/21)
* (package) Merged Taylor Salo's PR that fixes the documentation builds on readthedocs (THANK YOU!) and cleans up and centralizes requirement specifications.

## Version 2.0.5 (8/9/21)
* (package) Further Windows compatibility fixes.
* (documentation) Updated USAGE.rst for the current naming and syntax of rapidtide.

## Version 2.0.4 (7/28/21)
* (package) Fixed a problem where any program using util.py wouldn't run on Windows.
* (roisummarize) New addition to the package.
* (CI) Fixed a bug in the document environment.

## Version 2.0.3 (7/16/21)
* (spatialdecomp, temporaldecomp) Improved the consistency between the programs.
* (showxcorrx) Fixed some command line options.
* (package) Began to clean up and unify text output formatting.
* (package) Addressed some numpy deprecation warnings.
* (all scripts) Corrected file permissions (this may matter on Windows installations).
* (docs) Fixed some typos.
* (showtc) Enhanced debugging output.
* (testing) Tweaked circleci configuration file to update environment prior to installation.

## Version 2.0.2 (6/10/21)
* (rapidtide) Did you know that in python 3.8 and above, the default multiprocessing method is "spawn" rather than "fork"?  Did you know the subtle differences?  Do you know that that breaks rapidtide?  I didn't, now I do, and now it doesn't.
* (rapidtide) Made some tweaks to the timing logger to improve output formatting.
* (rapidtide, happy) Tested on M1.  The tests run more than twice as fast on an M1 mac mini with 8GB of RAM as on a 2017 MBP with a 2.9 GHz Quad-Core Intel Core i7.  Emulated.  Yow.  When I get a native anaconda installation going, watch out.
* (happy, Docker) Now require tensorflow 2.4.0 or above to address a security issue.

## Version 2.0.1 (6/8/21)
* (showxcorrx, plethquality, resamp1tc, simdata, happy, rapidtide) Cleaned up, improved, and unified text file reading and writing.
* (showxcorrx) Various functionality improvements.
* (package) Added options for timecourse normalization and zeropadding correlations.
* (documentation) Further cleanup.
* (Docker) Various fixes to versioning and other internals.

## Version 2.0 (6/2/21)
Much thanks to Taylor Salo for his continuing contributions, with several substantive improvements to code, documentation, and automatic testing, and generally helping devise a sensible release roadmap that made this version possible.

This release is a big one - there are many new programs, new capabilities in existing programs, and workflow breaking syntax changes.  However, this was all with the purpose of making a better package with much more consistent interfaces that allow you to figure out pretty quickly how to get the programs to do exactly what you want.
The biggest change is to rapidtide itself. For several years, there have been two versions of rapidtide; rapidtide2 (the traditional version), and rapidtide2x (the experimental version for testing new features).  When features became stable, I migrated them back to rapidtide2, more and more quickly as time went by, so they became pretty much the same.  I took the 2.0 release as an opportunity to do some cleanup.
As of now, there is only one version of rapidtide, with two parsers.  If you call "rapidtide", you get the spiffy new option parser and much more rational and consistent option naming and specification.  This is a substantial, but simple, change.
For compatibility with old workflows, I preserved the old parser, which is called "rapidtide2x_legacy".  This accepts options just as rapidtide2 and rapidtide2x did in version 1.9.6.
There is only one rapidtide routine.  Once the arguments are all read in and processed, "rapidtide" and "rapidtide2x_legacy" call the same processing workflow.  However, in addition to the new parser, there are completely new options and capabilities in rapidtide, but you can only get to them using the new parser.  This is my way of subtly forcing you to change your workflows if you want the new shiny, without pulling the rug out from under you.  "rapidtide2x_legacy" WILL be going away though, so change over when you can.
Also - all outputs now conform to BIDS naming conventions to improve compatibility with other packages.  Use the "--legacyoutput" flag to get the old output file names.
* (rapidtide2x_legacy): Added deprecation warning.
* (rapidtide): The correlation function has been replaced by a more flexible "similarity function".  There are currently 3 options: "correlation" (the old method), "mutualinfo", which uses a cross mutual information function, and "hybrid", which uses the correlation function, but disambiguates which peak to use by comparing the mutual information for each peak.
* (rapidtide) Pulled a number of default values into global variables so that defaults and help strings will stay in sync.
* (rapidtide) Fixed text file (nirs) processing.
* (rapidtide) Fixed a search range setting error.
* (rapidtide) Fixed the default method for global mean signal generation.
* (rapidtide) Added the '--negativegradient' option in response to https://github.com/bbfrederick/rapidtide/issues/67
* (rapidtide) Added flexibility to regressor input (can use multicolumn and BIDS text files).
* (rapidtide, tidepool) Fixed reading and writing the globalmean mask.
* (rapidtide) Gracefully handle refinement failure.
* (rapidtide) Added a new method for generating global signal using PCA.
* (rapidtide) Did some prep work to implement echo cancellation.
* (rapidtide) Added workaround for occasional MLE PCA component estimation failure (this seems to be an unresolved scikit-learn problem as of 0.23.2)
* (rapidtide) Significant enhancement to PCA refinement options.
* (rapidtide) Rapidtide can now run refinement passes until the change in the probe regressor falls below a specified mean square difference.  Set --convergencethresh to a positive number to invoke this (0.0005 is good).  Rapidtide will refine until the M.S.D. falls below this value, or you hit maxpasses (use --maxpasses NUM to set - default is 15).  This implements the procedure used in Champagne, A. A., et al., NeuroImage 187, 154–165 (2019).
* (rapidtide) The PCA refinement algorithm has been improved to match the method described in Champagne, et al., and is now the default.
* (rapidtide, io) Significant improvement to CIFTI handling - now properly read and write parcellated scalars and time series.
* (rapidtide) Completely revamped CIFTI I/O.  Should now read and write native CIFTI2 files (do not need to convert to NIFTI-2 in workbench).
* (rapidtide) Better handling of motion files.
* (rapidtide) Added coherence calculation.  Not quite working right yet.
* (rapidtide, happy) Switched to using nilearn's mask generator for automatic mask generation, since it's much more sophisticated.  It seems to be a big improvement, and handles data processed by fmriprep and SPM with no fiddling.
* (rapidtide, happy) General improvement of output of floating point numbers.  Limit to 3 decimal places.
* (rapidtide) Use logging module for output.
* (rapidtide, rapidtide_legacy) Options file is now always saved as a json.
* (rapidtide) Added ability to autochoose an appropriate spatial filter by setting --spatialfilt to a negative value.
* (rapidtide, rapidtide2x_legacy) The options file is now always saved in .json format.
* (rapidtide) BIDS format output naming and file structures have been updated to be more compliant with the standard.
* (rapidtide) Fixed a longstanding bug which used an unnecessarily stringent amplitude threshold for selecting voxels to use for refinement.
* (rapidtide) Improvements to processing in "bipolar" mode.
* (rapidtide): The getopt argument parser has been completely rewritten using argparse.  The way you specify many (most?) options has changed.
* (rapidtide): Any option that takes additional values (numbers, file names, etc.) is now specified as '--option VALUE [VALUE [VALUE...]]' rather than as '--option=VALUE[,VALUE[,VALUE...]]'.
* (rapidtide): After a lot of use over the years, I've reset a lot of defaults to reflect typical usage.  You can still do any analysis you were doing before, but it may now require changes to scripts and workflows to get the old default behavior.  For most cases you can get good analyses with a minimum set of command line options now.
* (rapidtide): There are two new macros, --denoise and --delaymapping, which will set defaults to good values for those use cases in subjects without vascular pathology.  Any of the preset values for these macros can be overridden with command line options.
* (rapidtide, rapidtide2x_legacy): Regressor and data filtering has been changed significantly.  While the nominal filter passbands are the same, the transitions to the stopbands have been tightened up quite a bit.  This is most noticeable in the LFO band.  The pasband is still from 0.01-0.15Hz with a trapezoidal rolloff, but the upper stopband now starts at 0.1575Hz instead of 0.20Hz.  The wide transition band was letting in a significant amount of respiratory signal for subjects with low respiratory rates (about half of my subjects seem to breath slower than the nominal adult minimum rate of 12 breaths/minute).
* (rapidtide): The -V, -L, -R and -C filter band specifiers have been retired.  Filter bands are now specified with '--filterband XXX', where XXX is vlf, lfo, lfo_legacy, resp, cardiac, or None.  'lfo' is selected by default (LFO band with sharp transition bands). To skip filtering, use '--filterband None'.  '--filterband lfo_legacy' will filter to the LFO band with the old, wide transition bands.
* (rapidtide): To specify an arbitrary filter, specify the pass freqs with --filterfreqs, and then optionally the stop freqs with --filterstopfreqs (otherwise the stop freqs will be calculated automatically from the pass freqs).
* (rapidtide): The method for specifying the lag search range has changed.  '-r LAGMIN,LAGMAX' has been removed.  You now use '--searchrange LAGMIN LAGMAX'
* (rapidtide): The method for specifying bipolar correlation search has changed.  '-B' is replaced by '--bipolar'.
* (rapidtide): The method for specifying a fixed delay (no correlation lag search) has changed.  '-Z DELAYVAL' is replaced by '--fixdelay DELAYVAL'.
* (rapidtide,rapidtide2x_legacy): The 'timerange' option is now handled properly.  This can be used to restrict processing to a portion of the datafile.  This is useful to get past initial transients if you didn't remove them in preprocessing, or to see if parameters change over the course of a long acquisition.
* (rapidtide): The multiprocessing code path can be forced on, even on a single processor.
* (rapidtide): Multiprocessing can be disabled on a per-routine basis.

Happy also got a new parser and BIDS outputs.  You can call happy with the old interface by calling "happy_legacy".
* (happy) Output files now follow BIDS naming convention.
* (happy) Code cleanup, improved tensorflow selection.
* (happy) Fixed logmem calls to work with new logging structure (and not crash immediately).
* (happy) Fixed a very subtle bug when an externally supplied pleth waveform doesn't start at time 0.0 (fix to issue #59).
* (happy) General formatting improvements.
* (happy) Added new tools for slice time generation.
* (happy) Added support for scans where there is circulating contrast.

General Changes to the entire package:
* (package) Python 2.7 support is now officially ended.  Cleaned out compatibility code.
* (package) Dropped support for python 3.3-3.5 and added 3.9.
* (package) Made pyfftw and numba requirements.
* (package) Significantly increased test coverage by including smoke tests (exercise as many code paths as possible to find crashes in neglected code - this is how the above bugs were found).
* (package) Automated consistent formatting.  black now runs automatically on file updates.
* (package) General cleanup and rationalization of imports.  isort now runs automatically on file updates.
* (package) Fixed a stupid bug that surfaced when reading in all columns of a text file as input.
* (package) Merged tsalo's PR starting transition to new logging output.
* (package) Started to phase out sys.exit() calls in favor of raising exceptions.
* (package) Updated all headers and copyright lines.
* (package) Trimmed the size of the installation bundle to allow deployment on pypi.
* (package) Copied Taylor Salo's improvements to build and deployment from the master branch.
* (package) Renamed some test data for consistency.
* (package) Began effort with T. Salo to address linter errors and generally improve PEP8 conformance - remove dead code, rationalize imports, improve docstrings, convert class names to CamelCase, use snake_case for functions.
* (package) Cleaned up imports and unresolved references
* (package) Addressed many linter issues, updated deprecated numpy and scipy calls.
* (package) readvectorsfromtextfile now handles noncompliant BIDS timecourse files.
* (package) Implemented new, generalized text/tsv/bids text file reader with column selection (readvectorsfromtextfile).
* (package) Significant internal changes to noncausalfilter.
* (package) CircleCI config files changed to keep tests from stepping on each other's caches (thanks to Taylor Salo).

Changes to the Docker container builds:
* (Docker) Fixed some problems with task dispatch and container versioning.
* (Docker) Improvements to version specification, entry point.
* (Docker) Update package versions.
* (Docker) Install python with mamba for ~10x speedup.
* (Docker) Switched to current method for specifying external mounts in the documentation.
* (Docker) Improved build scripts.
* (Docker) Containers are now automatically pushed to dockerhub after build.

Documentation changes:
* (documentation) Fixed errors in documentation files that caused errors building in readthedocs.
* (documentation) New "theory of operation" section for rapidtide.  Still working on it.
* (documentation) The documentation has been expanded and revised to reflect the current state of rapidtide with significant reorganization, reformatting, cleanup and additions

Tidepool:
* (tidepool) Reorganized interface, ability to show dynamic correlation movies, much clearer histogram graphs.
* (tidepool) Tidepool now gracefully handles runs with more than 4 passes.  The timecourses displayed are prefilt, postfilt, pass1, pass2, pass(N-1) and pass(N).
* (tidepool) Fixed to work with Big Sur (macOS 11).
* (tidepool) Corrected sample rate for regressor timecourses.
* (tidepool) Revised to properly handle new BIDS naming conventions for output files.
* (tidepool) You can now turn out-of-range map transparency on and off (it's off by default).
* (tidepool) Internal colortable code is now much cleaner.
* (tidepool): Now properly handles missing timecourses properly.  Some cosmetic fixes.

Miscellaneous changes:
* (aligntcs, applydlfilter, pixelcomp, plethquality, resamp1tc, showstxcorr, showxcorrx) Fixed matplotlib backend initialization to allow headless operation.
* (glmfilt, linfit, temporaldecomp, spatialdecomp): Argument parsers were rewritten in argparse, main routines were moved into workflows.
* (applydlfilter, atlasaverage, ccorrica, filttc, happy2std, histnifti, histtc, pixelcomp, plethquality, rapidtide2std, resamp1tc, showarbcorr, showtc, showxcorrx, simdata, spatialfit) Argument parsers were rewritten in argparse.
* (rapidtide, showxcorrx, showarbcorr, showstxcorr, ccorrica, aligntcs, filttc) Changed argument handling for arbitrary filters.  Specify pass freqs with --filterfreqs, stop freqs with --filterstopfreqs.
* (happy, rapidtide) Now properly handle negative mklthreads specification.
* (physiofreq): New program to get the average frequency of a physiological waveform.
* (showtc): Some cleanup in option specification.
* (showtc) Converted text inputs to standardized code.
* (atlastool) Major fixes to functionality with 4D template files.
* (atlastool) Fixed some import and syntax issues with numpy.
* (showxcorrx) Significant cleanup for maximum flexibility and utility.
* (showxcorr) Renamed to showxcorr_legacy
* (linfit) Renamed to polyfitim to better reflect it's function.
* (histnifti) Major upgrade to functionality.
* (showarbcorr) New program to do crosscorrelations on timecourses with different samplerates.
* (polyfitim) New program to fit a spatial template to a 3D or 4D NIFTI file.
* (endtidalproc) New program to extract end tidal CO2 or O2 waveforms from a gas exhalation recording.
* (dlfilter) Made diagnostics more informative to help get dlfilter enabled.
* (simdata) Complete overhaul - new parser better checks, more flexible input formats.
* (io.py) Vastly improved reading in arbitrarily large text files.
* (stats.py) Fixed a bug in getfracvals when you try to find the maximum value.
* (RapidtideDataset.py) Better handling of different file names.


## Version 2.0alpha29 (6/1/21)
* (Docker) Fixed some problems with task dispatch and container versioning.
* (rapidtide) Pulled a number of default values into global variables so that defaults and help strings will stay in sync.
* (documentation) Reorganization and significant updates.
* (documentation) Fixed errors in documentation files that caused errors building in readthedocs.
* (package) Updated versioneer.

## Version 1.9.6 (6/1/21)
* (Docker) ACTUALLY fixed some problems with task dispatch and container versioning.
* (package) Updated versioneer.

## Version 2.0alpha28 (5/27/21)
* (testing) Synced function calls with some internal changes to Correlator to fix the tests crashing.

## Version 1.9.5 (5/27/21)
* (Docker) Fixed some problems with task dispatch and container versioning.

## Version 2.0alpha27 (5/27/21)
* (rapidtide, showxcorrx, showarbcorr, showstxcorr, ccorrica, aligntcs, filttc) Changed argument handling for arbitrary filters.  Specify pass freqs with --filterfreqs, stop freqs with --filterstopfreqs.
* (happy) Code cleanup, improved tensorflow selection.
* (Docker) Improvements to version specification, entry point.
* (testing) Increased coverage.
* (packaging) Multiple corrections in support files.

## Version 2.0alpha26 (5/5/21)
* (happy, rapidtide) Now properly handle negative mklthreads specification.
* (Dockerfile) Update package versions.
* (Docker container) Added happy test.
* (package) Further increased test coverage.

## Version 2.0alpha25 (5/3/21)
* (rapidtide) Fixed text file (nirs) processing.
* (rapidtide) Fixed a search range setting error.
* (rapidtide) Fixed the default method for global mean signal generation.
* (rapidtide) Fixed a crash when using the mutualinfo similarity metric.
* (rapidtide, io) Significant improvement to CIFTI handling - now properly read and write parcellated scalars and time series.
* (io) Vastly improved reading in arbitrarily large text files.
* (stats) Fixed a bug in getfracvals when you try to find the maximum value.
* (package) Began aggressive implementation of smoke tests (exercise as many code paths as possible to find crashes in neglected code - this is how the above bugs were found).
* (package) More logging refinement.

## Version 2.0alpha24 (4/14/21)
* (rapidtide) Added the '--negativegradient' option in response to https://github.com/bbfrederick/rapidtide/issues/67
* (rapidtide) Added flexibility to regressor input (can use multicolumn and BIDS text files).

## Version 2.0alpha23 (4/14/21)
* (happy) Fixed logmem calls to work with new logging structure (and not crash immediately).

## Version 2.0alpha22 (4/13/21)
* (rapidtide, tidepool) Fixed reading and writing the globalmean mask.
* (package) Fixed a stupid bug that surfaced when reading in all columns of a text file as input (really, this time).

## Version 2.0alpha21 (4/12/21)
* (rapidtide) Gracefully handle refinement failure.
* (happy) Fixed some output timecourse naming.
* (atlastool) Major fixes to functionality with 4D template files.
* (aligntcs, applydlfilter, pixelcomp, plethquality, resamp1tc, showstxcorr, showxcorrx) Fixed matplotlib backend initialization to allow headless operation.
* (package) General cleanup and rationalization of imports.  isort now used by default.
* (package) Dropped support for python 3.3-3.5
* (package) Fixed a stupid bug that surfaced when reading in all columns of a text file as input.
* (package) Merged tsalo's PR starting transition to new logging output.
* (package) Fixed environment for py39 testing.
* (package) Started to phase out sys.exit() calls in favor of raising exceptions.
* (rapidtide) Corrected BIDS naming of intermediate maps.

## Version 2.0alpha20 (3/28/21)
* (package) Python 2.7 support is now officially ended.  Cleaned out compatibility code.
* (package) Made pyfftw and numba requirements.
* (docs) Wrote general description of text input functions, enhanced description of happy, include examples.
* (style) Began effort with T. Salo to address linter errors and generally improve PEP8 conformance - remove dead code, rationalize imports, improve docstrings, convert class names to CamelCase, use snake_case for functions.
* (showtc) Converted text inputs to standardized code.

## Version 2.0alpha19 (3/26/21)
* (showxcorrx) Significant cleanup for maximum flexibility and utility.
* (showxcorr) Renamed to showxcorr_legacy
* (linfit) Renamed to polyfitim to better reflect it's function.
* (histnifti) Major upgrade to functionality.
* (showxcorrx, atlasaverage, happy2std, rapidtide2std, spatialfit, applydlfilter, plethquality, histnifti) Moved to argparse.
* (package) Updated all headers and copyright lines.

## Version 2.0alpha18 (3/17/21)
* (package) Trimmed the size of the installation bundle (really, truly, correctly this time).

## Version 1.9.4 (3/17/21)
* (package) T. Salo made a number of changes to allow pypi deployment.
* (package) Fixed a problem in setup.py that causes installation to fail.
* (rapidtide2x) Backported a critical fix from the dev version so that the refinement threshold is properly set with null correlations (the result is that the refine mask rejects fewer voxels, and gives a better regressor estimate).


## Version 2.0alpha17 (3/17/21)
* (package) Trimmed the size of the installation bundle (maybe even correctly this time).

## Version 2.0alpha16 (3/17/21)
* (package) Trimmed the size of the installation bundle.
* (various) Cleaned up imports and unresolved references

## Version 2.0alpha15 (3/17/21)
* (rapidtide) Added a new method for generating global signal using PCA.
* (all) Further imports from master branch to improve deployment.
* (simdata) Complete overhaul - new parser better checks, more flexible input formats.
* (io.py) Improvements to readvecs to handle very large files.
* (ccorrica, filttc, histtc, pixelcomp, resamp1tc) Facelift, cleanup, new parsers.
* (testing) Removed python 2.7 CI build. 
* (all) Addressed many linter issues, updated deprecated numpy and scipy calls.

## Version 2.0alpha14 (3/1/21)
* (all) readvectorsfromtextfile now handles noncompliant BIDS timecourse files.
* (happy) Fixed a very subtle bug when an externally supplied pleth waveform doesn't start at time 0.0 (fix to issue #59).
* (filttc, histtc, showarbcorr) parser improvements.

## Version 2.0alpha13 (2/22/21)
* (package) Copied Taylor Salo's improvements to build and deployment from the master branch.
* (all) Ran all python files through Black to give consistent formatting (really, truly this time).
* (all) Implemented new, generalized text/tsv/bids text file reader with column selection (readvectorsfromtextfile).
* (atlastool) Fixed some import and syntax issues with numpy.
* (showarbcorr) New program to do crosscorrelations on timecourses with different samplerates.
* (happy) Fixed column selection bug with BIDS files.
* (happy) General formatting improvements.
* (dlfilter) Made diagnostics more informative to help get dlfilter enabled.

## Version 2.0alpha12 (2/5/21)
* (all) Fixed readbidstsv calls.
* (all) Beginning a rethink of a universal text timecourse reader.
* (happy) Added new tools for slice time generation.

## Version 2.0alpha11 (1/5/21)
* (rapidtide) Rolled back default similarity metric to 'correlation' from 'hybrid'.  'hybrid' works very well most of the time, and fails strangely occasionally.  When 'correlation' fails, it does so in more predictable and explicable ways.
* (happy) Restored functionality and options for motion regression that I broke when separating out the command parser.
* (tests) CircleCI config files changed to keep tests from stepping on each other's caches (thanks to Taylor Salo).

## Version 2.0alpha10 (12/21/20)
* (package) Ran all python files through Black to give consistent formatting.
* (rapidtide) Did some prep work to implement echo cancellation.

## Version 2.0alpha9 (12/9/20)
* (rapidtide) Added workaround for occasional MLE PCA component estimation failure (this seems to be an unresolved scikit-learn problem as of 0.23.2)

## Version 2.0alpha8 (12/9/20)
* (rapidtide) Significant enhancement to PCA refinement options.
* (tidepool) Tidepool now gracefully handles runs with more than 4 passes.  The timecourses displayed are prefilt, postfilt, pass1, pass2, pass(N-1) and pass(N).
* (happy) Added support for scans where there is circulating contrast.
* (happy, rapidtide2x, rapidtide) The parsers are now being properly installed during setup.
* (package) Renamed some test data for consistency.

## Version 2.0alpha7 (12/1/20)
* (rapidtide) Rapidtide can now run refinement passes until the change in the probe regressor falls below a specified mean square difference.  Set --convergencethresh to a positive number to invoke this (0.0005 is good).  Rapidtide will refine until the M.S.D. falls below this value, or you hit maxpasses (use --maxpasses NUM to set - default is 15).  This implements the procedure used in Champagne, A. A., et al., NeuroImage 187, 154–165 (2019).
* (rapidtide) The PCA refinement algorithm has been improved to match the method described in Champagne, et al., and is now the default.

## Version 2.0alpha6 (11/30/20)
* (rapidtide) Completely revamped CIFTI I/O.  Should now read and write native CIFTI2 files (do not need to convert to NIFTI-2 in workbench).
* (rapidtide) Better handling of motion files.
* (rapidtide) Added coherence calculation.  Not quite working right yet.
* (happy) Started adding BIDS output.
* (tidepool) Fixed to work with Big Sur (macOS 11).

## Version 2.0alpha5 (10/29/20)
Much thanks to Taylor Salo for his continuing contributions, with several substantive improvements to code, documentation, and automatic testing, and generally helping devise a sensible release roadmap.
* (rapidtide, happy) Switched to using nilearn's mask generator for automatic mask generation, since it's much more sophisticated.  It seems to be a big improvement, and handles data processed by fmriprep and SPM with no fiddling.
* (rapidtide, happy) General improvement of output of floating point numbers.  Limit to 3 decimal places.
* (rapidtide) Use logging module for output.
* (rapidtide, rapidtide_legacy) Options file is now always saved as a json.
* (rapidtide) Added ability to autochoose an appropriate spatial filter by setting --spatialfilt to a negative value.
* (rapidtide_parser) Code cleanup and formatting fixes.
* (documentation) Much reorganization, reformatting and cleanup.
* (documentation) New "theory of operation" section for rapidtide.  Still working on it.

## Version 2.0alpha4 (10/26/20)
* rapidtide2x has been renamed to rapidtide2x_legacy
* (rapidtide, rapidtide2x) The options file is now always saved in .json format.
* (rapidtide) BIDS format output naming and file structures have been updated to be more compliant with the standard.
* (RapidtideDataset.py) Better handling of different file names.
* (documentation) The documentation has been expanded and revised to reflect the current state of rapidtide.
* (all) Code cleanup.

## Version 2.0alpha3 (10/19/20)
* (rapidtide) Fixed sample rate on BIDS regressor outputs.
* (tidepool) Corrected sample rate for regressor timecourses.
* (Docker) Switched to current method for specifying external mounts in the documentation.
* (tests) Fixed test_filter.py to remove bad test.
* (tests) Added test_delayestimation.py to try to get end to end validation on the core of rapidtide.

## Version 2.0alpha2 (10/19/20)
* (all) Significant internal changes to noncausalfilter.
* (rapidtide) Fixed a longstanding bug which used an unnecessarily stringent amplitude threshold for selecting voxels to use for refinement.
* (rapidtide) Improvements to processing in "bipolar" mode.
* (rapidtide) Internal improvements to mutual information normalization.
* (rapidtide) New --bidsoutput option to make all output files BIDS compatible in naming and format.
* (tidepool) Revised to properly handle new naming conventions for output files.
* (tidepool) You can now turn out-of-range map transparency on and off (it's off by default).
* (tidepool) Internal colortable code is now much cleaner.
* (Docker) Improved build scripts.

## Version 2.0alpha1 (8/24/20)
* (all): Python 2.x is no longer supported.  To be fair, I've done nothing to break 2.x compatibility on purpose, so it probably still works, but I'm expending no effort to keep it working.
* (documentation): General updates and cleanups.
* (rapidtide2): rapidtide2 has been eliminated.  If you used it before, you can use rapidtide2x as a dropin replacement (but you really should start moving to using rapidtide, the new version that is actively being developed).
* (rapidtide2x): rapidtide2x has been deprecated and replaced by rapidtide (which is basically rapidtide2x v1.9.3 with a different argument parser and default option values).
* (rapidtide2x): Added deprecation warning.
* (rapidtide): The correlation function has been replaced by a more flexible "similarity function".  There are currently 3 options: "correlation" (the old method), "mutualinfo", which uses a cross mutual information function, and "hybrid", the new default, which uses the correlation function, but disambiguates which peak to use by comparing the mutual information for each peak.
* (rapidtide): Changed the default peak fit type to "fastquad", which does a parabolic fit to the peaks to refine location.
* (rapidtide): The getopt argument parser has been completely rewritten using argparse.  The way you specify many (most?) options has changed.
* (rapidtide): Any option that takes additional values (numbers, file names, etc.) is now specified as '--option VALUE [VALUE [VALUE...]]' rather than as '--option=VALUE[,VALUE[,VALUE...]]'.
* (rapidtide): After a lot of use over the years, I've reset a lot of defaults to reflect typical usage.  You can still do any analysis you were doing before, but it may now require changes to scripts and workflows to get the old default behavior.  For most cases you can get good analyses with a minimum set of command line options now.
* (rapidtide): There are two new macros, --denoise and --delaymapping, which will set defaults to good values for those use cases in subjects without vascular pathology.  Any of the preset values for these macros can be overridden with command line options.
* (rapidtide, rapidtide2x): Regressor and data filtering has been changed significantly.  While the nominal filter passbands are the same, the transitions to the stopbands have been tightened up quite a bit.  This is most noticeable in the LFO band.  The pasband is still from 0.01-0.15Hz with a trapezoidal rolloff, but the upper stopband now starts at 0.1575Hz instead of 0.20Hz.  The wide transition band was letting in a significant amount of respiratory signal for subjects with low respiratory rates (about half of my subjects seem to breath slower than the nominal adult minimum rate of 12 breaths/minute).
* (rapidtide): The -V, -L, -R and -C filter band specifiers have been retired.  Filter bands are now specified with '--filterband XXX', where XXX is vlf, lfo, lfo_legacy, resp, cardiac, or None.  'lfo' is selected by default (LFO band with sharp transition bands). To skip filtering, use '--filterband None'.  '--filterband lfo_legacy' will filter to the LFO band with the old, wide transition bands.
* (rapidtide): To specify an arbitrary filter, use '--filterfreqs LOWERPASS UPPERPASS [LOWERSTOP UPPERSTOP]'.  If you don't specify the stop bands, the stop frequencies are set to 5% below and above LOWERPASS and UPPERPASS, respectively.
* (rapidtide): The method for specifying the lag search range has changed.  '-r LAGMIN,LAGMAX' has been removed.  You now use '--searchrange LAGMIN LAGMAX'
* (rapidtide): The method for specifying bipolar correlation search has changed.  '-B' is replaced by '--bipolar'.
* (rapidtide): The method for specifying a fixed delay (no correlation lag search) has changed.  '-Z DELAYVAL' is replaced by '--fixdelay DELAYVAL'.
* (rapidtide): Options file is saved in json by default now.
* (rapidtide,rapidtide2x): The 'timerange' option is now handled properly.  This can be used to restrict processing to a portion of the datafile.  This is useful to get past initial transients if you didn't remove them in preprocessing, or to see if parameters change over the course of a long acquisition.
* (physiofreq): New program to get the average frequency of a physiological waveform.
* (tidepool): Now properly handles missing timecourses properly.  Some cosmetic fixes.
* (showtc): Converted to argparse, some cleanup in option specification.
* (glmfilt, linfit, temporaldecomp, spatialdecomp): Argument parsers were rewritten, main routines were moved into workflows.
* (docker container): Fixed some build errors, now pushes container to dockerhub.
* (rapidtide): Multiprocessing can be forced on, even on a single processor.
* (rapidtide): Multiprocessing can be disabled on a per-routine basis.

## Version 1.9.3 (7/30/20)
* Bumped version number because I forgot to commit a file

## Version 1.9.2 (7/30/20)
* (all): Changed over to using versioneer to handle version numbers.
* (rapidtide2, rapidtide2x, rapidtide_2x_trans, rapidtideX) Runtimings file now has additional version information.

## Version 1.9.1 (6/17/20)
* (all): Documentation improvements.
* (all): Many internal changes to support future argument specifications.
* (all): Backported bugfixes from the development version.
* (rapidtide2x) Fixed specification of timerange.
* (docker): Fixed an incompatibility in versions between pyfftw and scipy (thank you to Niranjana Shashikumar for reporting the bug and providing the solution!)
* (docker): Improved container labelling.
* (docker): Cleaned up container build.
* (tidepool): Various fixes and improvements backported from the development version.

## Version 1.9 (1/6/20)
* (all): Now compatible with nibabel 3.x
* (all): Significantly expanded test suite.  Code coverage is now at 47%.
* (documentation): Added instructions for installing the deep learning filter code
* (documentation): Numerous tweaks and fixes
* (docker): There is now a containerized version of the rapidtide package, which avoids a lot of installation woes
* (rapidtide2x, showxcorrx): Completely replaced correlation and fitting routines with faster, more robust (and more rigorously tested) versions
* (rapidtide2x, showxcorrx): Enhancements to the permutation methods
* (rapidtide2x, showxcorrx): Revised internals to guarantee xcorr scale matches values
* (rapidtide2x, showxcorrx): Improved fitter performance in edge cases (thin peaks, symmetric around max)
* (rapidtide2x, showxcorrx): Changed limits to avoid crash when peak is at edge of range
* (rapidtide2x, showxcorrx): Fixed some (but apparently not all) dumb errors in calls to null correlation calculations.
* (rapidtide2x): Implemented workaround for unknown crash in GLM filtering when nprocs != 1
* (rapidtide2x): Added experimental respiration processing
* (rapidtide2x): Fixed an uncaught bug in bipolar processing.
* (rapidtide2x): Setting ampthresh to a negative number between 0 and 1 sets the percentile of voxels to use for refinement
* (rapidtide2x): Support for new minimum sigma limit in correlation fit
* (rapidtide2x): Fixed a bug that caused fit fails on very narrow peaks, added diagnostic info
* (rapidtide2x): Putting in scaffolding to support phase randomization for null correlation calculation.
* (rapidtide2x): Properly specify correlation limits in null correlation and accheck
* (rapidtide2x): Slight modifications to pickleft routine to avoid a rare bug
* (rapidtide2x): Masks should now be properly generated for zero mean and non-positive definite data.
* (rapidtide2x): Tweaked the autocorrelation correction
* (rapidtide2x): Added an error check to avoid crashing when no significance tests are nonzero
* (rapidtide2x): Added upper and lower sigma limits to peak fitting uss to match new class
* (showxcorrx): Updated to match rapidtide2x fitting
* (showxcorrx): Enhanced error reporting
* (showxcorrx): Added width range tests
* (showxcorrx): Added norefine option, output more debugging info
* (showxcorrx): Set validity limits for gaussian fit
* (happy, rapidtide2x): Fixed a bug in output nifti header parameters (copy headers by value, not reference!)
* (happy): New multipass architecture allows much better results - initial passes set estimation masks and find potential arterial voxels.
* (happy): Aliased correlation integrated into happy as experimental feature
* (happy): Fixed a conceptual error in how I normalized projected timecourses
* (happy): Added brain cine output
* (happy): If estmask is supplied, us it.  If not, generate a vessel mask and repeat final steps.
* (happy): Added option to detect and invert arterial signals, improved time output.
* (happy): Shorten pulse recon step size to 10 ms
* (happy): Better envelope handling, fixed bug in timecourse phase projection.
* (happy): Some changes to improve performance with long TRs
* (happy): Added happy paper reference
* (happy): Increased gridbins (used in phase projection): default to 2 after testing showed lower noise than 1.5 bins
* (happy): Added ability to pad cyclically rather than reflecting around the ends
* (happy): Added ability to smooth projection in the phase direction (on by default)
* (happy): Significant improvements to GLM processing (spatial and temporal versions, aliased temporal correlation)
* (happy): Added "checkpoint" option to dump more intermediate data for debugging.
* (happy): Added more progress bars, and the ability to turn them off.
* (happy): Print out version info at runtime.
* (tidepool): Major update with new functionality
* (tidepool): The probe regressor, it's spectrum, and how it was filtered are now shown in the main window
* (tidepool): Properly disable atlas buttons when no atlas is loaded, avoiding crashes
* (tidepool): Removed support for pyqt4
* (tidepool): Some UI tweaks
* (tidepool): Added some infrastructure for future support for loading multiple runs
* (tidepool): New atlases to support fmriprep default coordinates
* (tidepool): Numerous bug fixes
* (ccorrica): Added the ability to oversample the data prior to crosscorrelation
* (showtc): Added ability to select a column from a multicolumn file as input.
* (showtc): Can now select a column  from multicolumn input text files for each vector.
* (showtc): Changed the specification of colors and legends. Internal code cleanup.
* (happy2std): New tool to align happy maps
* (happywarp): Improvements in filtering
* (aligntcs): You can now specify single columns out of multicolumn files
* (showxy): Initial support for specifying color names
* (spectrogram): Cleanup, added some configuration options
* (simdata): Some reformatting, updates, and improvements
* (simdata): Put some data in the example directory to use with simdata
* (fingerprint): New addition to the library to decompose delay maps using vascular territory templates
* (fingerprint): Added canonical HCP template maps to the distribution
* (helper_classes): Added freqtrack class
* (correlate.py): Added rudimentary mutual information calculation
* (correlate.py): Multiple aliased correlation methods added, depending on demeaning.
* (io.py): Fixed support for named columns in BIDS tsvs
* (io.py): Relaxed requirements for required fields in BIDS jsons
* (io.py): Added a few convenience routines for dealing with NIFTI files
* (io.py): Fixed import of parser_funcs

## Version 1.8 (5/10/19)
* (fit.py): The following fixes to both variants of findmaxlag_gauss affect rapidtide2, rapidtide2x, showxcorr, showxcorrx, happy, and happyx.
* (fit.py): CRITICAL FIX - edge behavior in both versions of findmaxlag_gauss was very broken.
* (fit.py): Fixed a rare failure in findmaxlag_gauss when the correlation peak is very narrow.
* (fit.py): Constrain initial gaussian fit values to be rational.
* (fit.py): Always return rational (if wrong) values when zerooutbadfit is False.
* (fit.py): Fixed a rare problem where no peaks were found in autocorrcheck, causing crash.
* (fit.py): Fixed a pernicious bug that sometimes caused mayhem when --nofitfilt was set.
* (rapidtide2, rapidtide2x): There is now sanity checking on lagmin and lagmax input using -r.
* (rapidtide2x): Masking logic has been completely redone, with numerous bugfixes, error checks, and new capabilities.
* (rapidtide2x): Added option to refine offset on leftmost lag peak (--pickleft).  This helps a lot with people with vascular pathology.
* (rapidtide2x): Added ability to save options file in json.
* (rapidtide2x): Fixed a bug when timerange was used in conjunction with glm filtering.
* (rapidtide2x): Added fixes to crash where there were bad significance estimates.
* (showtc): Allow multiple linewidths.
* (showtc): Added ability to set x and y axis labels.
* (showtc): Added DPI option to set resolution of file output.
* (showxy): Changed Bland-Altman plot to use open circles.
* (showhist): Add bar histograms.
* (showhist): Added the option to set binsize to makehistogram.
* (happy, happyx): All changes in happyx have now been synced to happy - at this moment, they are identical.  New changes will be tested in happyx.
* (happy, happyx): Fixed starttime and endtime selection.
* (happy, happyx): Lowered maximum heart rate to 140 by default.
* (happy, happyx): Output of info as json is optional, not default.
* (happy, happyx): Save info file as json rather than text.
* (happy, happyx): writedictasjson now supports numpy objects, added readdict function.
* (happy, happyx): Cardiac filter and search range are now specified independently.
* (happy, happyx): Removed lowerpass from cardiac estimation.
* (happy, happyx): Retrained and optimized model after revising happy paper.
* (happy, happyx): Use explicit copies to avoid variable changing out from under me.
* (happy, happyx): Added pleth/filtpleth correlation.
* (happy, happyx): Turn off variance masking by default, correlate raw and filtered waveforms.
* (happy, happyx): Numerous tweaks to resampling to perform better in edge cases.
* (happy, happyx): Fixed problem reading json physio files.
* (happy, happyx): Added ability to force the use of raw cardiac waveform for phase projection.
* (happy, happyx): Allow varmasking by volume or slice, filter prior to cardiac correlation.
* (happy, happyx): Resolved problem with definition of notch filter width.
* (happy, happyx): Corrected a type coercion error for estimation masks.
* (happy, happyx): Reduced verbosity of notch filter.
* (happy, happyx): Altered estimation mask logic.
* (happy, happyx): Added sanity checks to lag range.
* (happy, happyx): Now properly handle uncompressed bids tsv files.
* (happy, happyx): Variance and projection masks are separate, can set variance thresh percent.
* (happy, happyx): Changes in response to paper review.
* (happy, happyx): Filter cardiac waveform to slice samplerate Nyquist frequency when upsampling.
* (happy, happyx): Also added choice of centric or noncentric phase reconstruction..
* (happy, happyx): Fixed implementation of Hilbert transform phase analysis.
* (happy, happyx): Made robust to missing anatomics, started adding ants support.
* (happy, happyx): harmonic notch filter notch pct was not properly scaled.
* (happy, happyx): Now align pleth regressor with cardfromfmri.
* (happy, happyx): Fixed convolution gridding.
* (happy, happyx): Changed default gridding kernel to 3.0 wide Kaiser-Bessel.
* (happy, happyx): Reordered usage to functionally separate flags.
* (happy, happyx): Implemented workaround for strange interaction of tensorflow and MKL.
* (happy, happyx): Lots of little bugs fixed, print statements cleaned up.
* (tidepool): Added support for files in MNI152NLin2009cAsym space (fmriprep output).
* (tidepool): Fixed a crash when no atlas exists.
* (ccorrica): Modernized ccorrica to use new library calls.
* (atlasaverage, filttc, histtc, aligntcs, highresmotion):  Added to the distro.
* (tests): Numerous maintenance fixes.  test_findmaxlag is much more sophisticated now.
* (whole project): Code cleanup, reformatting.

## Version 1.7 (12/5/18)
* (whole project) Stopped pretending happy doesn't exist - adding to the changelog and will start writing docs.
* (whole project) Tried to generate some workflows.
* (whole project) Update issue templates.
* (whole project) Put back some critical information that got lost in the docs reorganization.
* (happyx) Changed default minhr to 40 BPM, cleaned up specification of min/max HR.
* (happyx) Put a lower limit on the envelope function in cleancardiac to limit gain..
* (happyx) Can use weighted masks, calculate envelop normalized cardiac waveforms..
* (happyx) Fixed notch filter to always filter at least one frequency bin.
* (happyx) Added ability to skip trs in fmrifile and motion file.
* (happyx) Mad normalize slice timecourses, refactor code, add some test data.
* (happy, happyx) Moved some routines out of happy(x) into libraries, added trendfilter.
* (happy, happyx, rapidtide2, rapidtide2x) Added motion regressor filtering.
* (happyx, rapidtide2, rapidtide2x) Add high order polynomial detrending.
* (happyx) Added deep learning filter for refining cardiac waveform (off by default).
* (rapidtide2, rapidtide2x) Oversample factor was erroneously set to 0 if TR <=0.5 seconds.
* (showxcorrx) Added file output capability.
* (showxcorrx) Set verbose to False by default.
* (showxcorrx) Trimming extraneous output.
* (tidepool) First crack at fixing atlas averaging.
* (tidepool) Initialize atlasniftiname.
* (showxy) Added Bland-Altman plots with annotations, range specifications, font scaling.
* (showxy) Updated for new matplotlib interface, enabled legends.
* (showtc) Now can specify legend location.
* (showtc) Added fontscalefac option.
* (resample.py) Fixed cutoff frequency on upsample filter.
* (resample.py) Lowpass filter after upsampling.
* (fit.py) Limit peakstart and peakend to stay within legal range.
* (io.py) New additions to readvecs to specify columns.
* (dlfilter.py) added.


## Version 1.6 (9/19/18)
* (whole project) Cleanup and reorganization (tsalo).
* (documentation) Major revisions to clean things up (tsalo).
* (workflows) Initial creation (work in progress) (tsalo).
* (testing) Reorganized and fixed - now it actually works! (tsalo).
* (coverage) Code coverage for testing is now tracked (21% - we can improve significantly with workflows) (tsalo).
* (rapidtide2, 2x, happy) Finally found (and fixed) the reason for a range of random stalls and slowdowns when running on a cluster.  MKL extensions were silently distributing some numpy calculations over all cores (which means running N jobs running on a cluster tried to use N^2 cores - not good at all...).  The maximum number of MKL threads is now settable on the command line, and defaults to 1 (no multiprocessor numpy).  Strangely, this makes everything a little faster in single processor mode, and A LOT faster in multiprocessor mode.
* (tide_funcs.py) tide_funcs.py has been split into filter.py, fit.py, io.py, miscmath.py, resample.py, stats.py, and util.py.  All executables fixed to match.
* (rapidtide2, 2x) Oversample factor is now set automatically by default to make the correlation timestep 0.5 or less.  This dramatically improves fits for longer TRs (> 1.5 seconds).
* (rapidtide2, 2x) Moved the major passes (null correlation, correlation, correlation fit, refine, wiener filter and glm) into separate modules for maintainability and to simplify tinkering.
* (rapidtide2, 2x) Isolated multiprocessing code to make speeding up new routines easier and avoid massive code duplication.
* (rapidtide2, 2x) Fixed some bugs in correlation mask reading and saving include and exclude masks.
* (rapidtide2, 2x) Improved tmask, fixed a bug.
* (rapidtide2, 2x, glmfilt) Made glmpass more general so it could be used in other scripts)
* (resamp1tc, resample.py) Added arbresample, modified dotwostepresample.
* (fit.py) Added Kaiser Bessel window function.
* (io.py) savetonifti now properly sets output data type in header.
* (io.py) Added routine to read in motion timecourses.
* (filter.py) Consolidated doprecalcfftfilt and xfunc into transferfuncfilt.
* (filter.py) Added new "complex" spectrum option.
* (filter.py) Added docstrings, code cleanup and regularization.
* (filter.py) Added new 'spectrum' routine.
* (filter.py) Initial support for precalculated arb filtering.
* (resample.py) Adjusted gridding to be symmetric around output value.
* (util.py) Reimplemented valtoindex to make it faster.
* (showtc) Updated to use new spectrum routine.
* (spectrogram) added to distro.
* (rapidtide2, 2x, resamp1tc showxcorr, showxcorrx, showstxcorr) eliminated all use of Butterworth filters by default.
* (spatialfit) Fixed numpy imports


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
