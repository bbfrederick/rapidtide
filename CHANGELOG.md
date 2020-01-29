# History of changes

## Version 2.0 (1/29/20)
* (all): Python 2.x is no longer supported.  To be fair, I've done nothing to break 2.x compatibility, so it probably still works, but I'm expending no effort to keep it working.
* (documentation): General updates and cleanups.
* (rapidtide2, rapidtide2x): rapidtide2 and rapidtide2x have been deprecated and replaced by rapidtide (which is basically rapidtide2x v1.9.0 with a different argument parser and default option values).
* (rapidtide2x): Added deprecation warning.
* (rapidtide, rapidtide2x): Added a new command line option "--addesskip" to allow you to skip some points at the beginning of the data if needed (and if you didn't trim them out in preprocessing).
* (rapidtide): The getopt argument parser has been completely rewritten using argparse.  The way you specify many (most?) options has changed.
* (rapidtide): Any option that takes additional values (numbers, file names, etc.) is now specified as '--option VALUE [VALUE [VALUE...]]' rather than as '--option=VALUE[,VALUE[,VALUE...]]'.
* (rapidtide): After a lot of use over the years, I've reset a lot of defaults to reflect typical usage.  You can still do any analysis you were doing before, but it may now require changes to scripts and workflows to get the old default behavior.  For most cases you can get good analyses with a minimum set of command line options now.
* (rapidtide): There are two new macros, --denoise and --delaymapping, which will set defaults to good values for those use cases in subjects without vascular pathology.  Any of the preset values for these macros can be overridden with command line options.
* (rapidtide, rapidtide2x): Regressor and data filtering has been changed significantly.  While the nominal filter passbands are the same, the transitions to the stopbands have been tightened up quite a bit.  This is most noticable in the LFO band.  The pasband is still from 0.01-0.15Hz with a trapezoidal rolloff, but the upper stopband now starts at 0.1575Hz instead of 0.20Hz.  The wide transition band was letting in a significant amount of respiratory signal for subjects with low respiratory rates (about half of my subjects seem to breath slower than the nominal adult minimum rate of 12 breaths/minute).
* (rapidtide): The -V, -L, -R and -C filter band specifiers have been retired.  Filter bands are now specified with '--filterband XXX', where XXX is vlf, lfo, lfo_legacy, resp, cardiac, or none.  'lfo' is selected by default (LFO band with sharp transition bands). To skip filtering, use '--filterband none'.  '--filterband lfo_legacy' will filter to the LFO band with the old, wide transition bands.
* (rapidtide): To specify an arbitrary filter, use '--filterfreqs LOWERPASS UPPERPASS [LOWERSTOP UPPERSTOP]'.  If you don't specify the stop bands, the stop frequencies are set to 5% below and above LOWERPASS and UPPERPASS, respectively.
* (rapidtide): The method for specifying the lag search range has changed.  '-r LAGMIN,LAGMAX' has been removed.  You now use '--searchrange LAGMIN LAGMAX'
* (rapidtide): The method for specifying bipolar correlation search has changed.  '-B' is replaced by '--bipolar'.
* (rapidtide): The method for specifying a fixed delay (no correlation lag search) has changed.  '-Z DELAYVAL' is replaced by '--fixdelay DELAYVAL'.
* (rapidtide): Options file is saved in json by default now.
* (physiofreq): New program to get the average frequency of a physiological waveform.
* (tidepool): Now properly handles missing timecourses properly.  Some cosmetic fixes.
* (showtc): Converted to argparse, some cleanup in option specification.
* (glmfilt, linfit, temporaldecomp, spatialdecomp): Argument parsers were rewritten, main routines were moved into workflows.
* (docker container): Fixed some build errors, now pushes container to dockerhub.

# Version 1.9 (1/6/20)
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
* (rapidtide2x, showxcorrx): Fixed some (but apparantly not all) dumb errors in calls to null correlation calculations.
* (rapidtide2x): Implemented workaround for unknown crasher in GLM filtering when nprocs != 1
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
* (tidepool): New atlases to suport fmriprep default coordinates
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
* (rapidtide2, 2x, happy) Finally found (and fixed) the reason for a range of random stalls and slowdowns when running on a cluster.  MKL extensions were silently distributing some numpy calculations over all cores (which means running N jobs running on a cluster tried to use N^2 cores - not good at all...).  The maxiumum number of MKL threads is now settable on the command line, and defaults to 1 (no multiprocessor numpy).  Strangely, this makes everything a little faster in single processor mode, and A LOT faster in multiprocessor mode.
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
