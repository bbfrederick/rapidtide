Usage
=====
For more information about how the rapidtide library can be used, please
see the API page. Common rapidtide workflows can also be called from the
command line.

Run rapidtide2
--------------
This is the full rapidtide workflow. The learn more about this workflow,
check out the workflow documentation:
:py:func:`rapidtide.workflows.rapidtide_workflow`.

.. argparse::
   :ref: rapidtide.workflows.rapidtide2._get_parser
   :prog: rapidtide2
   :func: _get_parser

Run showxcorrx
--------------
This is the showxcorrx workflow.
The learn more about this workflow, check out the workflow
documentation: :py:func:`rapidtide.workflows.showxcorrx_workflow`.

.. argparse::
  :ref: rapidtide.workflows.showxcorrx._get_parser
  :prog: showxcorrx
  :func: _get_parser

Running from the command line
-----------------------------
###rapidtide2

Description:

The central program in this package is rapidtide2.  This is the program that quantifies the correlation strength and time delay of pervasive signals in a BOLD fMRI dataset.

At its core, rapidtide2 is simply performing a full crosscorrelation between a "probe" timecourse and every voxel in an fMRI dataset (by “full” I mean over a range of time lags that account for any delays between the signals, rather than only at zero lag, as in a Pearson correlation).  As with many things, however, the devil is in the details, and so rapidtide2 provides a number of features which make it pretty good at this particular task.  A few highlights:

* There are lots of ways to do something even as simple as a cross-correlation in a nonoptimal way (not windowing, improper normalization, doing it in the time rather than frequency domain, etc.).  I'm pretty sure what rapidtide2 does by default is, if not the best way, at least a very good and very fast way.
* rapidtide2 has been optimized and profiled to speed it up quite a bit; it has an optional dependency on numba – if it’s installed, some of the most heavily used routines will speed up significantly due to judicious use of @jit.
* The sample rate of your probe regressor and the fMRI data do not have to match - rapidtide2 resamples the probe regressor to an integral multiple of the fMRI data rate automatically.
* The probe and data can be temporally prefiltered to the LFO, respiratory, or cardiac frequency band with a command line switch, or you can specify any low, high, or bandpass range you want.
* The data can be spatially smoothed at runtime (so you don't have to keep smoothed versions of big datasets around).  This is quite fast, so no reason not to do it this way.
* rapidtide2 can generate a probe regressor from the global mean of the data itself - no externally recorded timecourse is required.  Optionally you can input both a mask of regions that you want to be included in the mean, and the voxels that you want excluded from the mean (there are situations when you might want to do one or the other or both).
* Determining the significance threshold for filtered correlations where the optimal delay has been selected is nontrivial; using the conventional formulae for the significance of a correlation leads to wildly inflated p values.  rapidtide2 estimates the spurious correlation threshold by calculating the distribution of null correlation values obtained with a shuffling  procedure at the beginning of each run (the default is to use 10000 shuffled correlations), and uses this value to mask the correlation maps it calculates.  As of version 0.1.2 it will also handle two-tailed significance, which you need when using bipolar mode.
* rapidtide2 can do an iterative refinement of the probe regressor by aligning the voxel timecourses in time and regenerating the test regressor.
* rapidtide2 fits the peak of the correlation function, so you can make fine grained distinctions between close lag times. The resolution of the time lag discrimination is set by the length of the timecourse, not the timestep – this is a feature of correlations, not rapidtide2.
* Once the time delay in each voxel has been found, rapidtide2 outputs a 4D file of delayed probe regressors for using as voxel specific confound regressors or to estimate the strength of the probe regressor in each voxel.  This regression is performed by default, but these outputs let you do it yourself if you are so inclined.
* I've put a lot of effort into making the outputs as informative as possible - lots of useful maps, histograms, timecourses, etc.
* There are a lot of tuning parameters you can mess with if you feel the need.  I've tried to make intelligent defaults so things will work well out of the box, but you have the ability to set most of the interesting parameters yourself.
     
#### Inputs:
At a minimum, rapidtide2 needs a data file to work on (space by time), which is generally thought to be a BOLD fMRI data file.  This can be Nifti1 or Nifti2 (for fMRI data, in which case it is time by up to 3 spatial dimensions) or a whitespace separated text file (for NIRS data, each column is a time course, each row a separate channel); I can currently read (probably) but not write Cifti files, so if you want to use grayordinate files you need to convert them to nifti2 in workbench, run rapidtide2, then convert back. As soon as nibabel finishes their Cifti support (EDIT: and I get around to figuring it out), I'll add that.

The file needs one time dimension and at least one spatial dimension.  Internally, the array is flattened to a time by voxel array for simplicity.

The file you input here should be the result of any preprocessing you intend to do.  The expectation is that rapidtide will be run as the last preprocessing step before resting state or task based analysis.  So any slice time correction, motion correction, spike removal, etc. should already have been done.  If you use FSL, this means that if you've run preprocessing, you would use the filtered_func_data.nii.gz file as input.  Temporal and spatial filtering are the two (partial) exceptions here.  Generally rapidtide is most useful for looking at low frequency oscillations, so when you run it, you usually use the "-L" option or some other to limit the analysis to the detection and removal of low frequency systemic physiological oscillations.  So rapidtide will generally apply it's own temporal filtering on top of whatever you do in preprocessing.  Also, you have the option of doing spatial smoothing in rapidtide to boost the SNR of the analysis; the hemodynamic signals rapidtide looks for are often very smooth, so you rather than smooth your functional data excessively, you can do it within rapidtide so that only the hemodynamic data is smoothed at that level.
     
#### Outputs:
Outputs are space or space by time Nifti or text files, depending on what the input data file was, and some text files containing textual information, histograms, or numbers.  Output spatial dimensions and file type match the input dimensions and file type (Nifti1 in, Nifti1 out).  Depending on the file type of map, there can be no time dimension, a time dimension that matches the input file, or something else, such as a time lag dimension for a correlation map.
	
The following files are produced, assuming XXX is the outputname:

    ::

		Informational/diagnostic files
		XXX_commandline.txt                                   - The command line used to run rapidtide
		XXX_formattedcommandline.txt                          - The command line, but formatted nicely
		XXX_memusage.txt                                      - Memory statistics for the program at various timepoints during the run
		XXX_options.txt                                       - A dump of the internal structure with all of the options used during the run.
		XXX_runtimings.txt                                    - The final output showing how long each processing step took

		Pass specific outputs
		XXX_corrdistdata_passN.txt                            - These are all the null correlations produced during the significance estimation 
									step.  These are used to create the significance distribution.
		XXX_nullcorrelationhist_passN_peak.txt                - The location of the peak of the significance distribution histogram.
		XXX_nullcorrelationhist_passN.txt                     - The significance distribution histogram (use showhist to view).
		XXX_referenceautocorr_passN.txt                       - The autocorrelation function of the reference regressor 
									(used for finding sidelobes).
		XXX_reference_fmrires_passN.txt                       - The reference regressor, resampled to the timepoints of the data file.
		XXX_reference_resampres_passN.txt                     - The reference regressor, resampled to the timepoints of the data
									file, with oversampling.
		XXX_refinedregressor_passN.txt                        - The output of the refinement process (to be passed to the next stage).

		Final output maps
		XXX_corrout.nii.gz                                    - The oversampled correlation function over the lag range for each spatial location.
		XXX_gaussout.nii.gz                                   - A fit to the oversampled correlation function over the lag range 
									for each spatial location.
		XXX_lagmask.nii.gz                                    - The mask showing all voxels where correlation values were returned.
		XXX_lagsigma.nii.gz                                   - The width of the largest crosscorrelation peak within the lag range (NB:
									This partially indicates MTT, but in practice it is dominated by the width of 
									the autocorrelation function of the reference regressor, so is less useful than
									it might otherwise be.)
		XXX_lagstrengths.nii.gz                               - The maximum crosscorrelation strength over the lag range (R).
		XXX_lagtimes.nii.gz                                   - The correlation delay with maximum R over the lag range.
		XXX_mean.nii.gz                                       - The mean of the datafile over time for all voxels.
		XXX_p_lt_0pPPP_mask.nii.gz                            - The mask showing all voxels with R meeting the p<PPP significance threshold.
		XXX_R2.nii.gz                                         - The squared maximum correlation coefficient at every voxel.
		XXX_refinemask.nii.gz                                 - The voxels used for refinement in the last refinement pass (only 
									present if refinement is performed).
		XXX_lagregressors.nii.gz                              - The delayed sLFO regressor for every voxel (not scaled).
		XXX_laghist_peak.txt
		XXX_laghist.txt

		GLM filter results
		XXX_filtereddata.nii.gz                               - The input data with the voxel specific delayed LFO waveform regressed out.
		XXX_datatoremove.nii.gz                               - The scaled, delayed regressor that was removed from every voxel.
		XXX_fitcoff.nii.gz                                    - Map of the fit amplitude for the delayed LFO waveform.
		XXX_fitNorm.nii.gz                                    - I have no idea.
		XXX_fitR2.nii.gz                                      - Map of the squared R value for the fit (multiply by 100 to get the percent of the        
									variance explained by the LFO regressor).
		XXX_fitR.nii.gz                                       - Map of the R value for the fit.

		Final output
		XXX_p_lt_0pPPP_thresh.txt
		XXX_sigfit.txt

		XXX_reference_origres_prefilt.txt
		XXX_reference_origres.txt
		XXX_Rhist_peak.txt
		XXX_Rhist.txt
		XXX_strengthhist_peak.txt
		XXX_strengthhist.txt
		XXX_widthhist_peak.txt
		XXX_widthhist.txt

		Other
		XXX_dispersioncalcfreqs_passN.txt                     - These files are produced for me for secret reasons. Well, not secret, but
		XXX_dispersioncalcspecmag_passN.txt                     only partially thought out reasons.  If they come to anything, I'll say
		XXX_dispersioncalcspecphase_passN.txt                   what they are supposed to mean.
		XXX_dispersioncalcvecs_passN.txt
		XXX_globallaghist_passN_peak.txt
		XXX_globallaghist_passN.txt

    
#### Usage:

	::



		usage:  rapidtide2  datafilename outputname 
		[-r LAGMIN,LAGMAX] [-s SIGMALIMIT] [-a] [--nowindow] [--phat] [--liang] [--eckart] [-f GAUSSSIGMA] [-O oversampfac] [-t TSTEP] [--datatstep=TSTEP] [--datafreq=FREQ] [-d] [-b] [-V] [-L] [-R] [-C] [-F LOWERFREQ,UPPERFREQ[,LOWERSTOP,UPPERSTOP]] [-o OFFSETTIME] [-T] [-p] [-P] [-A ORDER] [-B] [-h HISTLEN] [-i INTERPTYPE] [-I] [-Z DELAYTIME] [-N NREPS] [--numskip=SKIP] [--refineweighting=TYPE] [--refineprenorm=TYPE] [--passes=PASSES] [--refinepasses=PASSES] [--excludemask=MASK] [--includemask=MASK] [--lagminthresh=MIN] [--lagmaxthresh=MAX] [--ampthresh=AMP] [--sigmathresh=SIGMA] [--corrmaskthresh=PCT] [--refineoffset] [--pca] [--ica] [--weightedavg] [--avg] [--psdfilter] [--despecklethresh=VAL] [--despecklepasses=PASSES] [--dispersioncalc] [--refineupperlag] [--refinelowerlag] [--nosharedmem] [--tmask=MASKFILE] [--limitoutput] [--timerange=START,END] [--skipsighistfit] [--accheck] [--acfix][--numskip=SKIP] [--slicetimes=FILE] [--glmsourcefile=FILE] [--regressorfreq=FREQ] [--regressortstep=TSTEP][--regressor=FILENAME] [--regressorstart=STARTTIME] [--usesp] [--maxfittype=FITTYPE] [--multiproc] [--nprocs=NPROCS] [--nirs] [--venousrefine]

		Required arguments:
		    datafilename               - The input data file (BOLD fmri file or NIRS)
		    outputname                 - The root name for the output files

		Optional arguments:
		    Arguments are processed in order of appearance.  Later options can override ones earlier on the command line

		Macros:
		    --venousrefine             - This is a macro that sets --lagminthresh=2.5, --lagmaxthresh=6.0,
						 --ampthresh=0.5, and --refineupperlag to bias refinement towards voxels
						 in the draining vasculature for an fMRI scan.
		    --nirs                     - This is a NIRS analysis - this is a macro that sets --nothresh, --preservefiltering,
						 --refinenorm=var, --ampthresh=0.7, and --lagminthresh=0.1.

		Preprocessing options:
		    -t TSTEP,                  - Set the timestep of the data file to TSTEP (or 1/FREQ)
		      --datatstep=TSTEP,         This will override the TR in an fMRI file.
		      --datafreq=FREQ            NOTE: if using data from a text file, for example with
						 NIRS data, using one of these options is mandatory.
		    -a                         - Disable antialiasing filter
		    --nodetrend                - Disable linear trend removal
		    -I                         - Invert the sign of the regressor before processing
		    -i                         - Use specified interpolation type (options are 'cubic',
						 'quadratic', and 'univariate (default)')
		    -o                         - Apply an offset OFFSETTIME to the lag regressors
		    -b                         - Use butterworth filter for band splitting instead of
						 trapezoidal FFT filter
		    -F                         - Filter data and regressors from LOWERFREQ to UPPERFREQ.
						 LOWERSTOP and UPPERSTOP can be specified, or will be
						 calculated automatically
		    -V                         - Filter data and regressors to VLF band
		    -L                         - Filter data and regressors to LFO band
		    -R                         - Filter data and regressors to respiratory band
		    -C                         - Filter data and regressors to cardiac band
		    -N                         - Estimate significance threshold by running NREPS null 
						 correlations (default is 10000, set to 0 to disable)
		    --skipsighistfit           - Do not fit significance histogram with a Johnson SB function
		    --windowfunc=FUNC          - Use FUNC window funcion prior to correlation.  Options are
						 hamming (default), hann, blackmanharris, and None
		    --nowindow                 - Disable precorrelation windowing
		    -f GAUSSSIGMA              - Spatially filter fMRI data prior to analysis using 
						 GAUSSSIGMA in mm
		    -M                         - Generate a global mean regressor and use that as the 
						 reference regressor
		    -m                         - Mean scale regressors during global mean estimation
		    --slicetimes=FILE          - Apply offset times from FILE to each slice in the dataset
		    --numskip=SKIP             - SKIP tr's were previously deleted during preprocessing
						 (default is 0)
		    --nothresh                 - Disable voxel intensity threshold (especially useful
						 for NIRS data)

		Correlation options:
		    -O OVERSAMPFAC             - Oversample the fMRI data by the following integral 
						 factor (default is 2)
		    --regressor=FILENAME       - Read probe regressor from file FILENAME (if none 
						 specified, generate and use global regressor)
		    --regressorfreq=FREQ       - Probe regressor in file has sample frequency FREQ 
						 (default is 1/tr) NB: --regressorfreq and --regressortstep
						 are two ways to specify the same thing
		    --regressortstep=TSTEP     - Probe regressor in file has sample time step TSTEP 
						 (default is tr) NB: --regressorfreq and --regressortstep
						 are two ways to specify the same thing
		    --regressorstart=START     - The time delay in seconds into the regressor file, corresponding
						 in the first TR of the fmri file (default is 0.0)
		    --phat                     - Use generalized cross-correlation with phase alignment 
						 transform (PHAT) instead of correlation
		    --liang                    - Use generalized cross-correlation with Liang weighting function
						 (Liang, et al, doi:10.1109/IMCCC.2015.283)
		    --eckart                   - Use generalized cross-correlation with Eckart weighting function
		    --corrmaskthresh=PCT       - Do correlations in voxels where the mean exceeeds this 
						 percentage of the robust max (default is 1.0)
		    --accheck                  - Check for periodic components that corrupt the autocorrelation

		Correlation fitting options:
		    -Z DELAYTIME               - Don't fit the delay time - set it to DELAYTIME seconds 
						 for all voxels
		    -r LAGMIN,LAGMAX           - Limit fit to a range of lags from LAGMIN to LAGMAX
		    -s SIGMALIMIT              - Reject lag fits with linewidth wider than SIGMALIMIT
		    -B                         - Bipolar mode - match peak correlation ignoring sign
		    --nofitfilt                - Do not zero out peak fit values if fit fails
		    --maxfittype=FITTYPE       - Method for fitting the correlation peak (default is 'gauss'). 
						 'quad' uses a quadratic fit.  Faster but not as well tested
		    --despecklepasses=PASSES   - detect and refit suspect correlations to disambiguate peak locations in PASSES passes
		    --despecklethresh=VAL      - refit correlation if median discontinuity magnitude exceeds VAL (default is 5s)

		Regressor refinement options:
		    --refineprenorm=TYPE       - Apply TYPE prenormalization to each timecourse prior 
						 to refinement (valid weightings are 'None', 
						 'mean' (default), 'var', and 'std'
		    --refineweighting=TYPE     - Apply TYPE weighting to each timecourse prior 
						 to refinement (valid weightings are 'None', 
						 'R', 'R2' (default)
		    --passes=PASSES,           - Set the number of processing passes to PASSES 
		     --refinepasses=PASSES       (default is 1 pass - no refinement).
						 NB: refinepasses is the wrong name for this option -
						 --refinepasses is deprecated, use --passes from now on.
		    --includemask=MASK         - Only use voxels in NAME for global regressor 
						 generation and regressor refinement
		    --excludemask=MASK         - Do not use voxels in NAME for global regressor 
						 generation and regressor refinement
		    --lagminthresh=MIN         - For refinement, exclude voxels with delays less 
						 than MIN (default is 0.5s)
		    --lagmaxthresh=MAX         - For refinement, exclude voxels with delays greater 
						 than MAX (default is 5s)
		    --ampthresh=AMP            - For refinement, exclude voxels with correlation 
						 coefficients less than AMP (default is 0.3)
		    --sigmathresh=SIGMA        - For refinement, exclude voxels with widths greater 
						 than SIGMA (default is 100s)
		    --refineoffset             - Adjust offset time during refinement to bring peak 
						 delay to zero
		    --refineupperlag           - Only use positive lags for regressor refinement
		    --refinelowerlag           - Only use negative lags for regressor refinement
		    --pca                      - Use pca to derive refined regressor (default is 
						 unweighted averaging)
		    --ica                      - Use ica to derive refined regressor (default is 
						 unweighted averaging)
		    --weightedavg              - Use weighted average to derive refined regressor 
						 (default is unweighted averaging)
		    --avg                      - Use unweighted average to derive refined regressor 
						 (default)
		    --psdfilter                - Apply a PSD weighted Wiener filter to shifted
						 timecourses prior to refinement

		Output options:
		    --limitoutput              - Don't save some of the large and rarely used files
		    -T                         - Save a table of lagtimes used
		    -h HISTLEN                 - Change the histogram length to HISTLEN (default is
						 100)
		    --timerange=START,END      - Limit analysis to data between timepoints START 
						 and END in the fmri file
		    --glmsourcefile=FILE       - Regress delayed regressors out of FILE instead of the 
						 initial fmri file used to estimate delays
		    --noglm                    - Turn off GLM filtering to remove delayed regressor 
						 from each voxel (disables output of fitNorm)
		    --preservefiltering        - don't reread data prior to GLM

		Miscellaneous options:
		    --wiener                   - Perform Wiener deconvolution to get voxel transfer functions
		    --usesp                    - Use single precision for internal calculations (may
						 be useful when RAM is limited)
		    -c                         - Data file is a converted CIFTI
		    -S                         - Simulate a run - just report command line options
		    -d                         - Display plots of interesting timecourses
		    --nonumba                  - Disable jit compilation with numba
		    --nosharedmem              - Disable use of shared memory for large array storage
		    --memprofile               - Enable memory profiling for debugging - warning:
						 this slows things down a lot.
		    --multiproc                - Enable multiprocessing versions of key subroutines.  This
						 speeds things up dramatically.  Almost certainly will NOT
						 work on Windows (due to different forking behavior).
		    --nprocs=NPROCS            - Use NPROCS worker processes for multiprocessing.  Setting NPROCS
						 less than 1 sets the number of worker processes to
						 n_cpus - 1 (default).  Setting NPROCS enables --multiproc.
		    --debug                    - Enable additional information output

		Experimental options (not fully tested, may not work):
		    --cleanrefined             - perform additional processing on refined regressor to remove spurious components.
		    --dispersioncalc           - Generate extra data during refinement to allow calculation of dispersion.
		    --acfix                    - Perform a secondary correlation to disambiguate peak location
						 (enables --accheck).  Experimental.
		    --tmask=MASKFILE           - Only correlate during epochs specified in 
						 MASKFILE (NB: each line of MASKFILE contains the 
						 time and duration of an epoch to include
		    -p                         - Prewhiten and refit data
		    -P                         - Save prewhitened data (turns prewhitening on)
		    -A, --AR                   - Set AR model order to ORDER (default is 1)

        
	These options are somewhat self-explanatory.  I will be expanding this section of the manual going forward, but I want to put something here to get this out here.
	
#### Examples:
Rapidtide can do many things - as I've found more interesting things to do with time delay processing, it's gained new functions and options to support these new applications.  As a result, it can be a little hard to know what to use for a new experiment.  To help with that, I've decided to add this section to the manual to get you started.  It's broken up by type of data/analysis you might want to do.

##### Removing low frequency physiological noise from resting state data
This is what I thought most people would use rapidtide for - finding and removing the low frequency (LFO) signal from an existing dataset.  This presupposes you have not made a simultaneous physiological recording (well, you may have, but it assumes you aren't using it).  For this, you can use a minimal set of options, since the defaults are mostly right.

The base command you'd use would be:

	::

		rapidtide2 inputfmrifile outputname -L --passes=3 --refineoffset

This will do a fairly simple analysis.  First, the -L option means that rapidtide2 will prefilter the data to the LFO band (0.009-0.15Hz). It will then construct a regressor from the global mean of the signal in inputfmrifile (default behavior if no regressor is specified), and then use crosscorrelation to determine the time delay in each voxel.  The --refinepasses=3 option directs rapidtide to to perform the delay analysis 3 times, each time generating a new estimate of the global noise signal by aligning all of the timecourses in the data to bring the global signal in phase prior to averaging.  The --refineoffset flag recenters the peak of the delay distribution on zero during the refinement process, which should make datasets easier to compare.  After the three passes are complete, it will then use a GLM filter to remove a lagged copy of the final mean regressor that from the data - this denoised data will be in the file "outputname_filtereddata.nii.gz".  There will also a number of maps output with the prefix "outputname_" of delay, correlation strength and so on.

##### Mapping long time delays in response to a gas challenge experiment

Processing this sort of data requires a very different set of options from the previous case.  Instead of the distribution of delays you expect in healthy controls (a slightly skewed, somewhat normal distribution with a tail on the positive side, ranging from about -5 to 5 seconds), in this case, the maximum delay can be extremely long (100-120 seconds is not uncommon in stroke, moyamoya disesase, and atherosclerosis).  To do this, you need to radically change what options you use, not just the delay range, but a number of other options having to do with refinement and statistical measures.

For this type of analysis, a good place to start is the following:

	::

		rapidtide2 inputfmrifile outputname -N 0 -r -10,140 -F 0.0,0.2 --lagmaxthresh=40 --ampthresh=0.2 --noglm --nofitfilt

The first option (-N 0), shuts off the calculation of the null correlation distribution.  This is used to determine the significance threshold, but the method currently implemented in rapidtide2 is a bit simplistic - it assumes that all the time points in the data are exchangable.  This is certainly true for resting state data (see above), but it is very much NOT true for block paradigm gas challenges.  To properly analyze those, I need to consider what time points are 'equivalent', and up to now, I don't, so setting the number of iterations in the Monte Carlo analysis to zero omits this step.

The second option (-r -10,140) is fairly obvious - this extends the detectable delay range out to 140 seconds.  Note that this is somewhat larger than the maximum delays we frequently see, but to find the correlation peak with maximum precision, you need sufficient additional delay values so that the correlation can come to a peak and then come down enough that you can properly fit it. 

The -noglm option disables data filtering.  If you are using rapidtide to estimate and remove low frequency noise from resting state or task fMRI data, the last step is to use a glm filter to remove this circulatory signal, leaving "pure" neuronal activations, which you'll use in further analyses.  That's not relevant here - the signal you'd be removing is the one you care about. So this option skips that step to save time and disk space.

--nofitfilt skips a step after peak estimation.  Estimating the delay and correlation amplitude in each voxel is a two step process. First you make a quick estimate (where is the maximum point of the correlation function, and what is its amplitude?), then you refine it by fitting a Gaussian function to the peak to improve the estimate.  If this step fails, which it can if the peak is too close to the end of the lag range, or strangely shaped, the default behavior is to mark the point as bad and zero out the parameters for the voxel.  The nofitfilt option means that if the fit fails, output the initial estimates rather than all zeros.   This means that you get some information, even if it's not fully refined.  In my experience it does tend to make the maps for the gas challenge experiments a lot cleaner to use this option since the correlation function is pretty well behaved.


##### Denoising NIRS data (NEW)

When we started this whole research effort, I waw originally planning to denoise NIRS data, not fMRI data.  But one thing led to another, and the NIRS got derailed for the fMRI effort.  Now that we have some time to catch our breaths, and more importantly, we have access to some much higher quality NIRS data, this moved back to the front burner.  The majority of the work was already done, I just needed to account for a few qualities that make NIRS data different from fMRI data:

* NIRS data is not generally stored in NIFTI files.  There is not as yet a standard NIRS format.  In the absence of one, you could do worse than a multicolumn text file, with one column per data channel.  That's what I did here - if the file has a '.txt' extension rather than '.nii.', '.nii.gz', or no extension, it will assume all I/O should be done on multicolumn text files.
* NIRS data is often zero mean.  This turned out to mess with a lot of my assumptions about which voxels have significant data, and mask construction.  This has led to some new options for specifying mask threshholds and data averaging.
* NIRS data is in some sense "calibrated" as relative micromolar changes in oxy-, deoxy-, and total hemoglobin concentration, so mean and/or variance normalizing the timecourses may not be right thing to do.  I've added in some new options to mess with normalizations.



happy
----------

Description:
^^^^^^^^^^^^

	happy is a new addition to the rapidtide suite.  It's complementary to rapidtide - it's focussed on fast, cardiac signals in fMRI, rather than the slow, LFO signals we are usually looking at.  It's sort of a Frankenprogram - it has three distinct jobs, which are related, but are very distinct.

	The first thing happy does is try to extract a cardiac waveform from the fMRI data.  This is something I've been thinking about for a long time.  Words go here
	
	The second task is to take this raw estimate of the cardiac waveform, and clean it up using a deep learning filter.  The original signal is useful, but pretty gross, but I figured you should be able to exploit the pseudoperiodic nature of the signal to greatly improve it.  This is also a testbed to work on using neural nets to process time domain signals.  It seemed like a worthwhile project, so it got grafted in.
	
	The final task (which was actually the initial task, and the reason I wrote happy to begin with) is to implement Henning Voss' totally cool hypersampling with analytic phase projection (guess where the name "happy" comes from).  This is fairly straightforward, as Voss describes his method very clearly.  But I have lots of data with no simultaneously recorded cardiac signals, and I was too lazy to go find datasets with pleth data to play with, so that's why I did the cardiac waveform extraction part.

     
Inputs:
^^^^^^^
	Happy needs a 4D BOLD fMRI data file (space by time) as input.  This can be Nifti1 or Nifti2.  If you have a simultaneously recorded cardiac waveform, it will happily use it, otherwise it will try to construct (and refine) one. NOTE: the 4D input dataset needs to be completely unpreprocessed - gradient distortion correction and motion correction can destroy the relationship between slice number and actual acquisition time, and slice time correction does not behave as expected for aliased signals (which the cardiac component in fMRI most certainly is), and in any case we need the slice time offsets to construct our waveform.

     
Outputs:
^^^^^^^^
	Outputs are space or space by time Nifti or text files, depending on what the input data file was, and some text files containing textual information, histograms, or numbers.  Output spatial dimensions and file type match the input dimensions and file type (Nifti1 in, Nifti1 out).  Depending on the file type of map, there can be no time dimension, a time dimension that matches the input file, or something else, such as a time lag dimension for a correlation map.
	
The following files are produced, assuming XXX is the outputname:

    ::

		Informational/diagnostic files
		XXX_commandline.txt                                   - The command line used to run happy
		XXX_info.txt (or XXX_info.json)                       - Various useful internal variables
		XXX_memusage.csv                                      - Memory statistics for the program at various
		XXX_runtimings.txt                                    - Detailed timing information


		Waveforms
		XXX_cardfromfmri_sliceres.txt                         - The estimated cardiac waveform.  Sample rate is (1/TR) * number of slices / multiband factor.
		XXX_cardfromfmri_sliceres_badpts.txt                  - Points in the above waveform that are probably bad due to motion.
		XXX_cardfromfmri_sliceres_censored.txt                - The estimated waveform with the bad points zeroed out.
		XXX_cardfromfmri_25.0Hz.txt                           - The estimated cardiac waveform resampled to 25.0 Hz
		XXX_cardfromfmri_dlfiltered_25.0Hz.txt                - The above, after passing through the deep learning filter.
		XXX_cardfromfmri_dlfiltered_sliceres.txt              - The above, resample back to sliceres.

		XXX_cardfromfmrienv_25.0Hz.txt                        - The envelope function of the estimated cardiac waveform.
		XXX_normcardfromfmri_25.0Hz.txt                       - Estimated cardiac waveform divided by the envelope function.
		XXX_normcardfromfmri_dlfiltered_25.0Hz.txt
		XXX_cardfromfmri_25.0Hz_badpts.txt
		XXX_overall_sliceres_badpts.txt
		XXX_cardiacfundamental.txt
		XXX_ampenv.txt
		XXX_instphase_unwrapped.txt
		XXX_filtered_instphase_unwrapped.txt
		XXX_orthogonalizedmotion.txt
		XXX_interpinstphase.txt
		
		Histograms
		XXX_histogram_peak.txt
		XXX_histogram.txt

                Images
		XXX_app.nii.gz                                        - The cardiac waveform over one cycle in each voxel. 
		XXX_rawapp.nii.gz
		XXX_mask.nii.gz
		XXX_maskedapp.nii.gz
		XXX_vesselmask.nii.gz
		XXX_minphase.nii.gz
		XXX_maxphase.nii.gz
		XXX_arteries.nii.gz
		XXX_veins.nii.gz
		XXX_vesselmap.nii.gz

    
Usage:
^^^^^^

	::

		happy - Hypersampling by Analytic Phase Projection - Yay!

		usage:  happy  fmrifile slicetimefile outputroot

		required arguments:
		    fmrifile:                      - NIFTI file containing BOLD fmri data
		    slicetimefile:                 - Text file containing the offset time in seconds of each slice relative
						     to the start of the TR, one value per line, OR the BIDS sidecar JSON file
						     for the fmrifile (contains the SliceTiming field
		    outputroot:                    - Base name for all output files

		optional arguments:

		Processing steps:
		    --cardcalconly                 - Stop after all cardiac regressor calculation steps (before phase projection).
		    --dodlfilter                   - Refine cardiac waveform from the fMRI data using a deep learning filter.
						     NOTE: this will only work if you have a working Keras installation;
						     if not, this option is ignored.
						     OTHER NOTE: Some versions of tensorflow seem to have some weird conflict
						     with MKL which I can't seem to be able to fix.  If the dl filter bombs
						     complaining about multiple openmp libraries, try rerunning with the
						     secret and inadvisable '--usesuperdangerousworkaround' flag.  Good luck!
		    --model=MODELNAME              - Use model MODELNAME for dl filter (default is model_revised - from the revised NeuroImage paper.)

		Performance:
		    --mklthreads=NTHREADS          - Use NTHREADS MKL threads to accelerate processing (defaults to 1 - more
						     threads up to the number of cores can accelerate processing a lot, but
						     can really kill you on clusters unless you're very careful.  Use at your
						     own risk.)

		Preprocessing:
		    --numskip=SKIP                 - Skip SKIP tr's at the beginning of the fmri file (default is 0).
		    --motskip=SKIP                 - Skip SKIP tr's at the beginning of the motion regressor file (default is 0).
		    --motionfile=MOTFILE[:COLSPEC] - Read 6 columns of motion regressors out of MOTFILE text file.
						     (with timepoints rows) and regress them, their derivatives, 
						     and delayed derivatives out of the data prior to analysis.
						     If COLSPEC is present, use the comma separated list of ranges to
						     specify X, Y, Z, RotX, RotY, and RotZ, in that order.  For
						     example, :3-5,7,0,9 would use columns 3, 4, 5, 7, 0 and 9
						     for X, Y, Z, RotX, RotY, RotZ, respectively
		    --motionhp=HPFREQ              - Highpass filter motion regressors to HPFREQ Hz prior to regression
		    --motionlp=LPFREQ              - Lowpass filter motion regressors to HPFREQ Hz prior to regression

		Cardiac estimation tuning:
		    --varmaskthreshpct=PCT         - Only include voxels with MAD over time in the PCTth percentile and higher in
						     the generation of the cardiac waveform (default is no variance masking.)
		    --estmask=MASKNAME             - Generation of cardiac waveform from data will be restricted to
						     voxels in MASKNAME and weighted by the mask intensity (overrides
						     normal variance mask.)
		    --minhr=MINHR                  - Limit lower cardiac frequency search range to MINHR BPM (default is 40)
		    --maxhr=MAXHR                  - Limit upper cardiac frequency search range to MAXHR BPM (default is 140)
		    --minhrfilt=MINHR              - Highpass filter cardiac waveform estimate to MINHR BPM (default is 40)
		    --maxhrfilt=MAXHR              - Lowpass filter cardiac waveform estimate to MAXHR BPM (default is 1000)
		    --envcutoff=CUTOFF             - Lowpass filter cardiac normalization envelope to CUTOFF Hz (default is 0.4)
		    --notchwidth=WIDTH             - Set the width of the notch filter, in percent of the notch frequency
						     (default is 1.5)

		External cardiac waveform options:
		    --cardiacfile=FILE[:COL]       - Read the cardiac waveform from file FILE.  If COL is an integer,
						     format json file, use column named COL (if no file is specified 
						     is specified, estimate cardiac signal from data)
		    --cardiacfreq=FREQ             - Cardiac waveform in cardiacfile has sample frequency FREQ 
						     (default is 32Hz). NB: --cardiacfreq and --cardiactstep
						     are two ways to specify the same thing
		    --cardiactstep=TSTEP           - Cardiac waveform in file has sample time step TSTEP 
						     (default is 0.03125s) NB: --cardiacfreq and --cardiactstep
						     are two ways to specify the same thing
		    --cardiacstart=START           - The time delay in seconds into the cardiac file, corresponding
						     in the first TR of the fmri file (default is 0.0)
		    --stdfreq=FREQ                 - Frequency to which the cardiac signals are resampled for output.
						     Default is 25.
		    --forcehr=BPM                  - Force heart rate fundamental detector to be centered at BPM
						     (overrides peak frequencies found from spectrum).  Useful
						     if there is structured noise that confuses the peak finder.

		Phase projection tuning:
		    --outputbins=BINS              - Number of output phase bins (default is 32)
		    --gridbins=BINS                - Width of the gridding kernel in output phase bins (default is 3.0)
		    --gridkernel=KERNEL            - Convolution gridding kernel.  Options are 'old', 'gauss', and 'kaiser'
						     (default is 'kaiser')
		    --projmask=MASKNAME            - Phase projection will be restricted to voxels in MASKNAME
						     (overrides normal intensity mask.)
		    --projectwithraw               - Use fmri derived cardiac waveform as phase source for projection, even
						     if a plethysmogram is supplied

		Debugging arguments (probably not of interest to users):
		    --debug                        - Turn on debugging information
		    --nodetrend                    - Disable data detrending
		    --noorthog                     - Disable orthogonalization of motion confound regressors
		    --normalize                    - Normalize fmri data
		    --nodemean                     - Do not demean fmri data
		    --disablenotch                 - Disable subharmonic notch filter
		    --nomask                       - Disable data masking for calculating cardiac waveform
		    --nocensor                     - Bad points will not be excluded from analytic phase projection
		    --noappsmooth                  - Disable smoothing app file in the phase direction
		    --nophasefilt                  - Disable the phase trend filter (probably not a good idea)
		    --nocardiacalign               - Disable alignment of pleth signal to fmri derived cardiac signal.
						     to blood vessels
		    --saveinfoasjson               - Save the info file in json format rather than text.  Will eventually
		    --trimcorrelations             - Some physiological timecourses don't cover the entire length of the
						     fMRI experiment.  Use this option to trim other waveforms to match 
						     when calculating correlations.


		        
	These options are somewhat self-explanatory.  I will be expanding this section of the manual going forward, but I want to put something here to get this out here.
	
Examples:
^^^^^^^^^

Just getting the cardiac waveform from resting state data
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The base command you'd use would be:

	::

		happy inputfmrifile slicetimefile outputroot --cardcalconly --dodlfilter
		
This won't get you the best cardiac waveform however.  You really should use a vessel mask to do the averaging only over "important" voxels.  Fortunately, you can get this from happy!  So a better way to do this is to run:

        ::
	
	        happy inputfmrifile slicetimefile firstpassoutput --dodlfilter
		happy inputfmrifile slicetimefile secondpassoutput --cardcalconly --dodlfilter --estmask=firstpassoutput_vesselmask.nii.gz
		
This uses the vessel mask produced by the first pass to limit the cardiac waveform calculation to vessel voxels in the second pass, giving a better initial cardiac estimate, which in turn gives a better filtered output.  The 25Hz plethysmogram will be found in secondpassoutput_cardfromfmri_dlfiletered_25.0Hz.txt


rapidtide2std
-------------

Description:
^^^^^^^^^^^^

	This is a utility for registering rapidtide output maps
	to standard coordinates.  It's usually much faster to run rapidtide
	in native space then transform afterwards to MNI152 space.  NB: this 
	will only work if you have a working FSL installation.

Inputs:
^^^^^^^

Outputs:
^^^^^^^^
	New versions of the rapidtide output maps, registered to either MNI152 space or to the hires anatomic images for the subject.  All maps are named with the specified root name with '_std' appended.

Usage:
^^^^^^

	::

		usage: rapidtide2std INPUTFILEROOT OUTPUTDIR FEATDIRECTORY [--all] [--hires]

		required arguments:
		    INPUTFILEROOT      - The base name of the rapidtide maps up to but not including the underscore
		    OUTPUTDIR          - The location for the output files
		    FEADDIRECTORY      - A feat directory (x.feat) where registration to standard space has been performed

		optional arguments:
		    --all              - also transform the corrout file (warning - file may be huge)
		    --hires            - transform to match the high resolution anatomic image rather than the standard
		    --linear           - only do linear transformation, even if warpfile exists


showxcorr
---------

Description:
^^^^^^^^^^^^

	Like rapidtide2, but for single time courses.  Takes two text files as input, calculates and displays 
	the time lagged crosscorrelation between them, fits the maximum time lag, and estimates
	the significance of the correlation.  It has a range of filtering,
	windowing, and correlation options.

Inputs:
^^^^^^^
	showxcorr requires two text files containing timecourses with the same sample rate, one timepoint per line, which are to be correlated, and the sample rate.

Outputs:
^^^^^^^^
	showxcorr outputs everything to standard out, including the Pearson correlation, the maximum cross correlation, the time of maximum cross correlation, and estimates of the significance levels (if specified).  There are no output files.

Usage:
^^^^^^

	::

		usage: showxcorr timecourse1 timecourse2 samplerate [-l LABEL] [-s STARTTIME] [-D DURATION] [-d] [-F LOWERFREQ,UPPERFREQ[,LOWERSTOP,UPPERSTOP]] [-V] [-L] [-R] [-C] [-t] [-w] [-f] [-z FILENAME] [-N TRIALS]

		required arguments:
			timcoursefile1:	text file containing a timeseries
			timcoursefile2:	text file containing a timeseries
			samplerate:	the sample rate of the timecourses, in Hz

		optional arguments:
		    -t            - detrend the data
		    -w            - prewindow the data
		    -l LABEL      - label for the delay value
		    -s STARTTIME  - time of first datapoint to use in seconds in the first file
		    -D DURATION   - amount of data to use in seconds
		    -r RANGE      - restrict peak search range to +/- RANGE seconds (default is 
				    +/-15)
		    -d            - turns off display of graph
		    -F            - filter data and regressors from LOWERFREQ to UPPERFREQ.
				    LOWERSTOP and UPPERSTOP can be specified, or will be 
				    calculated automatically
		    -V            - filter data and regressors to VLF band
		    -L            - filter data and regressors to LFO band
		    -R            - filter data and regressors to respiratory band
		    -C            - filter data and regressors to cardiac band
		    -T            - trim data to match
		    -A            - print data on a single summary line
		    -a            - if summary mode is on, add a header line showing what values 
				    mean
		    -f            - negate (flip) second regressor
		    -z FILENAME   - use the columns of FILENAME as controlling variables and 
				    return the partial correlation
		    -N TRIALS     - estimate significance thresholds by Monte Carlo with TRIALS 
				    repetition


showxcorrx
---------

Description:
^^^^^^^^^^^^

	This is the newest, most avant-garde version of showxcorr.  Because it's an x file, it's more fluid and I don't guarantee that it will keep a stable interface (or even work at any given time).  But every time I add something new, it goes here.  The goal is eventually to make this the "real" version.  Unlike rapidtide2, however, I've let it drift quite a bit without syncing it because some people here actually use showxcorr and I don't want to disrupt workflows...

Inputs:
^^^^^^^
	showxcorrx requires two text files containing timecourses with the same sample rate, one timepoint per line, which are to be correlated, and the sample rate.

Outputs:
^^^^^^^^
	showxcorrx  outputs everything to standard out, including the Pearson correlation, the maximum cross correlation, the time of maximum cross correlation, and estimates of the significance levels (if specified).  There are no output files.

Usage:
^^^^^^

	::

		showxcorrx - calculate and display crosscorrelation between two timeseries

		usage:  showxcorrx  timecourse1 timecourse2 samplerate
		[-l LABEL] [-s STARTTIME] [-D DURATION] [-d] [-F LOWERFREQ,UPPERFREQ[,LOWERSTOP,UPPERSTOP]] [-V] [-L] [-R] [-C] [--nodetrend] [--nowindow] [-f] [-o OUTPUTFILE] [--phat] [--liang] [--eckart] [--savecorr=FILE] [-z FILENAME] [-N TRIALS]

		required arguments:
		    timcoursefile1: text file containing a timeseries
		    timcoursefile2: text file containing a timeseries
		    samplerate:     the sample rate of the timecourses, in Hz

		optional arguments:
		    --nodetrend        - do not detrend the data before correlation
		    --nowindow         - do not prewindow data before corrlation
		    --windowfunc=FUNC  - window function to apply before corrlation (default is hamming)
		    --cepstral         - check time delay using Choudhary's cepstral technique 
		    --phat             - perform phase alignment transform (PHAT) rather than 
					 standard crosscorrelation
		    --liang            - perform phase alignment transform with Liang weighting function rather than 
					 standard crosscorrelation
		    --eckart           - perform phase alignment transform with Eckart weighting function rather than 
					 standard crosscorrelation
		    -l LABEL           - label for the delay value
		    -s STARTTIME       - time of first datapoint to use in seconds in the first file
		    -D DURATION        - amount of data to use in seconds
		    -r RANGE           - restrict peak search range to +/- RANGE seconds (default is 
					 +/-15)
		    -d                 - turns off display of graph
		    -F                 - filter data and regressors from LOWERFREQ to UPPERFREQ.
					 LOWERSTOP and UPPERSTOP can be specified, or will be 
					 calculated automatically
		    -V                 - filter data and regressors to VLF band
		    -L                 - filter data and regressors to LFO band
		    -R                 - filter data and regressors to respiratory band
		    -C                 - filter data and regressors to cardiac band
		    -T                 - trim data to match
		    -A                 - print data on a single summary line
		    -a                 - if summary mode is on, add a header line showing what values 
					 mean
		    -f                 - negate (flip) second regressor
		    -savecorr=FILE     - Save the correlation function to the file FILE in xy format
		    -z FILENAME        - use the columns of FILENAME as controlling variables and 
					 return the partial correlation
		    -N TRIALS          - estimate significance thresholds by Monte Carlo with TRIALS 
					 repetition
		    -o OUTPUTFILE      - Writes summary lines to OUTPUTFILE (sets -A)


showtc
------

Description:
^^^^^^^^^^^^
	A very simple command line utility that takes a text file
	and plots the data in it in a matplotlib window.  That's it.  A
	good tool for quickly seeing what's in a file.  Has some options
	to make the plot prettier.

Inputs:
^^^^^^^
	Text files containing time series data

Outputs:
^^^^^^^^
	None

Usage:
^^^^^^

	::

		showtc - plots the data in text files

		usage: showtc texfilename[:col1,col2...,coln] [textfilename]... [--nolegend] [--pspec] [--phase] [--samplerate=Fs] [--sampletime=Ts]

		required arguments:
		    textfilename	- a text file containing whitespace separated timecourses, one timepoint per line
				       A list of comma separated numbers following the filename and preceded by a colon is used to select columns to plot

		optional arguments:
		    --nolegend               - turn off legend label
		    --pspec                  - show the power spectra magnitudes of the input data instead of the timecourses
		    --phase                  - show the power spectra phases of the input data instead of the timecourses
		    --transpose              - swap rows and columns in the input files
		    --waterfall              - plot multiple timecourses as a waterfall
		    --voffset=VOFFSET        - plot multiple timecourses as with VOFFSET between them (use negative VOFFSET to set automatically)
		    --samplerate=Fs          - the sample rate of the input data is Fs Hz (default is 1Hz)
		    --sampletime=Ts          - the sample time (1/samplerate) of the input data is Ts seconds (default is 1s)
		    --colorlist=C1,C2,..     - cycle through the list of colors specified by CN
		    --linewidth=LW           - set linewidth to LW points (default is 1)
		    --fontscalefac=FAC       - scale all font sizes by FAC (default is 1.0)
		    --legendlist=L1,L2,..    - cycle through the list of legends specified by LN
		    --tofile=FILENAME        - write figure to file FILENAME instead of displaying on the screen
		    --title=TITLE            - use TITLE as the overall title of the graph
		    --separate               - use a separate subplot for each timecourse
		    --separatelinked         - use a separate subplot for each timecourse, but use a common y scaling
		    --noxax                  - don't show x axis
		    --noyax                  - don't show y axis
		    --starttime=START        - start plot at START seconds
		    --endtime=END            - end plot at END seconds
		    --legendloc=LOC          - Integer from 0 to 10 inclusive specifying legend location.  Legal values are:
					       0: best, 1: upper right, 2: upper left, 3: lower left, 4: lower right,
					       5: right, 6: center left, 7: center right, 8: lower center, 9: upper center,
					       10: center.  Default is 2.
		    --debug                  - print debugging information

histnifti
--------

Description:
^^^^^^^^^^^^
	A command line tool to generate a histogram for a nifti file


Inputs:
^^^^^^^
	A nifti file

Outputs:
^^^^^^^^
	A text file containing the histogram information

None

Usage:
^^^^^^

	::

		usage: histnifti inputfile outputroot

		required arguments:
			inputfile	- the name of the input nifti file
			outputroot	- the root of the output nifti names



showhist
--------

Description:
^^^^^^^^^^^^
	Another simple command line utility that displays the histograms generated by rapidtide2.

Inputs:
^^^^^^^
	A textfile generated by rapidtide2 containing histogram information

Outputs:
^^^^^^^^
	None

Usage:
^^^^^^

	::

		usage: showhist textfilename
			plots xy histogram data in text file

		required arguments:
			textfilename	- a text file containing one timepoint per line


resamp1tc
---------

Description:
^^^^^^^^^^^^
	This takes an input text file at some sample rate and outputs a text file resampled to the specified sample rate.


Inputs:
^^^^^^^

Outputs:
^^^^^^^^

Usage:
^^^^^^

	::

		resamp1tc - resample a timeseries file

		usage: resamp1tc infilename insamplerate outputfile outsamplerate [-s]

		required arguments:
			inputfile        - the name of the input text file
			insamplerate     - the sample rate of the input file in Hz
			outputfile       - the name of the output text file
			outsamplerate    - the sample rate of the output file in Hz

		 options:
			-s               - split output data into physiological bands (LFO, respiratory, cardiac)


resamplenifti
-------------

Description:
^^^^^^^^^^^^
	This takes an input nifti file at some TR and outputs a nifti file resampled to the specified TR.
 

Inputs:
^^^^^^^

Outputs:
^^^^^^^^

Usage:
^^^^^^

	::

		usage: resamplenifti inputfile inputtr outputname outputtr [-a]

		required arguments:
			inputfile	- the name of the input nifti file
			inputtr		- the tr of the input file in seconds
			outputfile	- the name of the output nifti file
			outputtr	- the tr of the output file in seconds

		options:
			-a		- disable antialiasing filter (only relevant if you are downsampling in time)


tcfrom3col
------

Description:
^^^^^^^^^^^^
	A  simple command line that takes an FSL style 3 column regressor file and generates a time course (waveform) file.  FSL 3 column files are text files containing one row per "event".  Each row has three columns: start time in seconds, duration in seconds, and waveform value.  The output waveform is zero everywhere that is not covered by an "event" in the file.

Inputs:
^^^^^^^
	A three column text file

Outputs:
^^^^^^^^
	A single column text file containing the waveform

Usage:
^^^^^^

	::

		tcfrom3col - convert a 3 column fsl style regressor into a one column timecourse

		usage: tcfrom3col infile timestep numpoints outfile

		required arguments:
			infile:      a text file containing triplets of start time, duration, and value
			timestep:    the time step of the output time coures in seconds
			numpoints:   the number of output time points
			outfile:     the name of the output time course file


pixelcomp
---------

Description:
^^^^^^^^^^^^
	A program to compare voxel values in two 3D NIFTI files.
        You give pixelcomp two files, each with their own mask.
        Any voxel that has a nonzero mask in both files gets added
        to a list of xy pairs, with the value from the first file being x,
        and the value from the second file being y.  Pixelcomp then:
        1) Makes and displays a 2D histogram of all the xy values. 2) Does a linear
        fit to x and y, and outputs the coefficients (slope and offset)
        to a XXX_linfit.txt file. 3) Writes all the xy pairs to a tab
        separated text file, and 4) Makes a Bland-Altman plot of x vs y

Inputs:
^^^^^^^
	Two 3D NIFTI image files, the accompanying mask files, and the root name for the output files.

Outputs:
^^^^^^^^
	None

Usage:
^^^^^^

	::

		showtc - plots the data in text files

		usage: showtc texfilename[:col1,col2...,coln] [textfilename]... [--nolegend] [--pspec] [--phase] [--samplerate=Fs] [--sampletime=Ts]

		required arguments:
		    textfilename	- a text file containing whitespace separated timecourses, one timepoint per line
				       A list of comma separated numbers following the filename and preceded by a colon is used to select columns to plot

		optional arguments:
		    --nolegend               - turn off legend label
		    --pspec                  - show the power spectra magnitudes of the input data instead of the timecourses
		    --phase                  - show the power spectra phases of the input data instead of the timecourses
		    --transpose              - swap rows and columns in the input files
		    --waterfall              - plot multiple timecourses as a waterfall
		    --voffset=VOFFSET        - plot multiple timecourses as with VOFFSET between them (use negative VOFFSET to set automatically)
		    --samplerate=Fs          - the sample rate of the input data is Fs Hz (default is 1Hz)
		    --sampletime=Ts          - the sample time (1/samplerate) of the input data is Ts seconds (default is 1s)
		    --colorlist=C1,C2,..     - cycle through the list of colors specified by CN
		    --linewidth=LW           - set linewidth to LW points (default is 1)
		    --fontscalefac=FAC       - scale all font sizes by FAC (default is 1.0)
		    --legendlist=L1,L2,..    - cycle through the list of legends specified by LN
		    --tofile=FILENAME        - write figure to file FILENAME instead of displaying on the screen
		    --title=TITLE            - use TITLE as the overall title of the graph
		    --separate               - use a separate subplot for each timecourse
		    --separatelinked         - use a separate subplot for each timecourse, but use a common y scaling
		    --noxax                  - don't show x axis
		    --noyax                  - don't show y axis
		    --starttime=START        - start plot at START seconds
		    --endtime=END            - end plot at END seconds
		    --legendloc=LOC          - Integer from 0 to 10 inclusive specifying legend location.  Legal values are:
					       0: best, 1: upper right, 2: upper left, 3: lower left, 4: lower right,
					       5: right, 6: center left, 7: center right, 8: lower center, 9: upper center,
					       10: center.  Default is 2.
		    --debug                  - print debugging information

glmfilt
---------

Description:
^^^^^^^^^^^^
	Uses a GLM filter to remove timecourses (1D text files or 4D NIFTI files) from 4D NIFTI files.


Inputs:
^^^^^^^

Outputs:
^^^^^^^^

Usage:
^^^^^^

	::

		usage: glmfilt datafile numskip outputroot evfile [evfile_2...evfile_n]
		    Fits and removes the effect of voxel specific and/or global regressors

ccorrica
---------

Description:
^^^^^^^^^^^^
	Find temporal crosscorrelations between all the columns in a text file (for example the timecourse files output by MELODIC.)


Inputs:
^^^^^^^

Outputs:
^^^^^^^^

Usage:
^^^^^^

	::

		ccorrica - find temporal crosscorrelations between ICA components

			usage: ccorrica timecoursefile TR
				timcoursefile:	text file containing multiple timeseries, one per column, whitespace separated
				TR:		the sample period of the timecourse, in seconds



showstxcorr
---------

Description:
^^^^^^^^^^^^
	Calculate and display the short term crosscorrelation between two timeseries (useful for dynamic correlation).


Inputs:
^^^^^^^

Outputs:
^^^^^^^^

Usage:
^^^^^^

	::

		showstxcorr - calculate and display the short term crosscorrelation between two timeseries

		usage: showstxcorr -i timecoursefile1 [-i timecoursefile2] --samplefreq=FREQ -o outputfile [-l LABEL] [-s STARTTIME] [-D DURATION] [-d] [-F LOWERFREQ,UPPERFREQ[,LOWERSTOP,UPPERSTOP]] [-V] [-L] [-R] [-C] [--nodetrend] [-nowindow] [-f] [--phat] [--liang] [--eckart] [-z FILENAME]

		required arguments:
		    -i, --infile= timcoursefile1     - text file containing one or more timeseries
		    [-i, --infile= timcoursefile2]   - text file containing a timeseries
						       NB: if one timecourse file is specified, each column
						       is considered a timecourse, and there must be at least
						       2 columns in the file.  If two filenames are given, each
						       file must have only one column of data.

		    -o, --outfile=OUTNAME:           - the root name of the output files

		    --samplefreq=FREQ                - sample frequency of all timecourses is FREQ 
			   or
		    --sampletime=TSTEP               - time step of all timecourses is TSTEP 
						       NB: --samplefreq and --sampletime are two ways to specify
						       the same thing.

		optional arguments:
		    --nodetrend   - do not detrend the data before correlation
		    --nowindow    - do not prewindow data before corrlation
		    --phat        - perform phase alignment transform (PHAT) rather than 
				    standard crosscorrelation
		    --liang       - perform phase alignment transform with Liang weighting function rather than 
				    standard crosscorrelation
		    --eckart      - perform phase alignment transform with Eckart weighting function rather than 
				    standard crosscorrelation
		    -s STARTTIME  - time of first datapoint to use in seconds in the first file
		    -D DURATION   - amount of data to use in seconds
		    -d            - turns off display of graph
		    -F            - filter data and regressors from LOWERFREQ to UPPERFREQ.
				    LOWERSTOP and UPPERSTOP can be specified, or will be calculated automatically
		    -V            - filter data and regressors to VLF band
		    -L            - filter data and regressors to LFO band
		    -R            - filter data and regressors to respiratory band
		    -C            - filter data and regressors to cardiac band
		    -W WINDOWLEN  - use a window length of WINDOWLEN seconds (default is 50.0s)
		    -S STEPSIZE   - timestep between subsequent measurements (default is 25.0s).  Will be rounded to the nearest sample time
		    -f            - negate second regressor


tidepool
--------

Description:
^^^^^^^^^^^^
	This is a very experimental tool for displaying all of the various maps generated by rapidtide2 in one place, overlayed on an anatomic image.  This makes it a bit easier to see how all the maps are related to one another.  To use it, launch tidepool from the command line, and then select a lag time map - tidpool will figure out the root name and pull in all of the other associated maps.  Works in native or standard space.


Inputs:
^^^^^^^

Outputs:
^^^^^^^^

Usage:
^^^^^^

	::

		usage: tidepool [-h] [-o OFFSETTIME] [-r] [-n] [-t TRVAL] [-d DATAFILEROOT]
					[-a ANATNAME] [-m GEOMASKNAME]

		A program to display the results of a time delay analysis

		optional arguments:
		  -h, --help       show this help message and exit
		  -o OFFSETTIME    Set lag offset
		  -r               enable risetime display
		  -n               enable movie mode
		  -t TRVAL         Set correlation TR
		  -d DATAFILEROOT  Use this dataset (skip initial selection step)
		  -a ANATNAME      Set anatomic mask image
		  -m GEOMASKNAME   Set geometric mask image


tide_funcs.py
-------------

Description:
^^^^^^^^^^^^
	This is the library of the various helper routines that are used by pretty much every program in here for correlation, resampling, filtering, normalization, significance estimation, file I/O, etc.


Inputs:
^^^^^^^

Outputs:
^^^^^^^^

Usage:
^^^^^^

::


OrthoImageItem.py
-----------------

Description:
^^^^^^^^^^^^
	This is a class that implements the orthographic projection module that is used to display all of the maps in tidepool. It uses pyqtgraph to do all the heavy lifting.  None of the built-ins in pyqtgraph did exactly what I wanted in terms of allowing 3D selection, overlays and the like, so I cobbled this together.  It may be generally useful to anybody wanting to display functional data.
        
Inputs:
^^^^^^^

Outputs:
^^^^^^^^

Usage:
^^^^^^

::

