Legacy interface:
^^^^^^^^^^^^^^^^^
For compatibility with old workflows, rapidtide can be called using legacy syntax by using "rapidtide2x_legacy".  Although the underlying code is the same, not all options are settable from the legacy interface.  This interface is deprecated and will be removed in a future version of rapidtide, so please convert existing workflows.


    ::

      usage:  rapidtide2x_legacy  datafilename outputname
      [-r LAGMIN,LAGMAX] [-s SIGMALIMIT] [-a] [--nowindow] [--phat] [--liang] [--eckart] [-f GAUSSSIGMA] [-O oversampfac] [-t TSTEP] [--datatstep=TSTEP] [--datafreq=FREQ] [-d] [-b] [-V] [-L] [-R] [-C] [-F LOWERFREQ,UPPERFREQ[,LOWERSTOP,UPPERSTOP]] [-o OFFSETTIME] [--autosync] [-T] [-p] [-P] [-B] [-h HISTLEN] [-i INTERPTYPE] [-I] [-Z DELAYTIME] [--nofitfilt] [--searchfrac=SEARCHFRAC] [-N NREPS] [--motionfile=MOTFILE] [--pickleft] [--numskip=SKIP] [--refineweighting=TYPE] [--refineprenorm=TYPE] [--passes=PASSES] [--refinepasses=PASSES] [--excluderefine=MASK] [--includerefine=MASK] [--includemean=MASK] [--excludemean=MASK][--lagminthresh=MIN] [--lagmaxthresh=MAX] [--ampthresh=AMP] [--sigmathresh=SIGMA] [--corrmask=MASK] [--corrmaskthresh=PCT] [--refineoffset] [--pca] [--ica] [--weightedavg] [--avg] [--psdfilter] [--noprogressbar] [--despecklethresh=VAL] [--despecklepasses=PASSES] [--dispersioncalc] [--refineupperlag] [--refinelowerlag] [--nosharedmem] [--tmask=MASKFILE] [--limitoutput] [--motionfile=FILENAME[:COLSPEC] [--softlimit] [--timerange=START,END] [--skipsighistfit] [--accheck] [--acfix][--numskip=SKIP] [--slicetimes=FILE] [--glmsourcefile=FILE] [--regressorfreq=FREQ] [--regressortstep=TSTEP][--regressor=FILENAME] [--regressorstart=STARTTIME] [--usesp] [--peakfittype=FITTYPE] [--mklthreads=NTHREADS] [--nprocs=NPROCS] [--nirs] [--venousrefine]

      Required arguments:
          datafilename               - The input data file (BOLD fmri file or NIRS)
          outputname                 - The root name for the output files

      Optional arguments:
          Arguments are processed in order of appearance.  Later options can override ones earlier on
          the command line

      Macros:
          --venousrefine                 - This is a macro that sets --lagminthresh=2.5, --lagmaxthresh=6.0,
                                           --ampthresh=0.5, and --refineupperlag to bias refinement towards
                                           voxels in the draining vasculature for an fMRI scan.
          --nirs                         - This is a NIRS analysis - this is a macro that sets --nothresh,
                                           --preservefiltering, --refinenorm=var, --ampthresh=0.7,
                                           and --lagminthresh=0.1.

      Preprocessing options:
          -t TSTEP,                      - Set the timestep of the data file to TSTEP (or 1/FREQ)
            --datatstep=TSTEP,             This will override the TR in an fMRI file.
            --datafreq=FREQ                NOTE: if using data from a text file, for example with
                                           NIRS data, using one of these options is mandatory.
          -a                             - Disable antialiasing filter
          --detrendorder=ORDER           - Set order of trend removal (0 to disable, default is 1 - linear)
          -I                             - Invert the sign of the regressor before processing
          -i                             - Use specified interpolation type (options are 'cubic',
                                           'quadratic', and 'univariate (default)')
          -o                             - Apply an offset OFFSETTIME to the lag regressors
          --autosync                     - Calculate and apply offset time of an external regressor from
                                           the global crosscorrelation.  Overrides offsettime if specified.
          -b                             - Use butterworth filter for band splitting instead of
                                           trapezoidal FFT filter
          -F  LOWERFREQ,UPPERFREQ[,LOWERSTOP,UPPERSTOP]
                                         - Filter data and regressors from LOWERFREQ to UPPERFREQ.
                                           LOWERSTOP and UPPERSTOP can be specified, or will be
                                           calculated automatically
          -V                             - Filter data and regressors to VLF band
          -L                             - Filter data and regressors to LFO band
          -R                             - Filter data and regressors to respiratory band
          -C                             - Filter data and regressors to cardiac band
          --padseconds=SECONDS           - Set the filter pad time to SECONDS seconds.  Default
                                           is 30.0
          -N NREPS                       - Estimate significance threshold by running NREPS null
                                           correlations (default is 10000, set to 0 to disable).  If you are
                                           running multiple passes, 'ampthresh' will be set to the 0.05 significance.
                                           level unless it is manually specified (see below).
          --permutationmethod=METHOD     - Method for permuting the regressor for significance estimation.  Default
                                           is shuffle
          --skipsighistfit               - Do not fit significance histogram with a Johnson SB function
          --windowfunc=FUNC              - Use FUNC window function prior to correlation.  Options are
                                           hamming (default), hann, blackmanharris, and None
          --nowindow                     - Disable precorrelation windowing
          -f GAUSSSIGMA                  - Spatially filter fMRI data prior to analysis using
                                           GAUSSSIGMA in mm
          -M                             - Generate a global mean regressor and use that as the
                                           reference regressor
          --globalmeaninclude=MASK[:VALSPEC]
                                         - Only use voxels in NAME for global regressor generation (if VALSPEC is
                                           given, only voxels with integral values listed in VALSPEC are used.)
          --globalmeanexclude=MASK[:VALSPEC]
                                         - Do not use voxels in NAME for global regressor generation (if VALSPEC is
                                           given, only voxels with integral values listed in VALSPEC are used.)
          -m                             - Mean scale regressors during global mean estimation
          --slicetimes=FILE              - Apply offset times from FILE to each slice in the dataset
          --numskip=SKIP                 - SKIP tr's were previously deleted during preprocessing (e.g. if you
                                           have done your preprocessing in FSL and set dummypoints to a
                                           nonzero value.) Default is 0.
          --timerange=START,END          - Limit analysis to data between timepoints START
                                           and END in the fmri file. If END is set to -1,
                                           analysis will go to the last timepoint.  Negative values
                                           of START will be set to 0. Default is to use all timepoints.
          --nothresh                     - Disable voxel intensity threshold (especially useful
                                           for NIRS data)
          --motionfile=MOTFILE[:COLSPEC] - Read 6 columns of motion regressors out of MOTFILE text file.
                                           (with timepoints rows) and regress their derivatives
                                           and delayed derivatives out of the data prior to analysis.
                                           If COLSPEC is present, use the comma separated list of ranges to
                                           specify X, Y, Z, RotX, RotY, and RotZ, in that order.  For
                                           example, :3-5,7,0,9 would use columns 3, 4, 5, 7, 0 and 9
                                           for X, Y, Z, RotX, RotY, RotZ, respectively
          --motpos                       - Toggle whether displacement regressors will be used in motion regression.
                                           Default is False.
          --motderiv                     - Toggle whether derivatives will be used in motion regression.
                                           Default is True.
          --motdelayderiv                - Toggle whether delayed derivative  regressors will be used in motion regression.
                                           Default is False.

      Correlation options:
          -O OVERSAMPFAC                 - Oversample the fMRI data by the following integral
                                           factor.  Setting to -1 chooses the factor automatically (default)
          --regressor=FILENAME           - Read probe regressor from file FILENAME (if none
                                           specified, generate and use global regressor)
          --regressorfreq=FREQ           - Probe regressor in file has sample frequency FREQ
                                           (default is 1/tr) NB: --regressorfreq and --regressortstep
                                           are two ways to specify the same thing
          --regressortstep=TSTEP         - Probe regressor in file has sample time step TSTEP
                                           (default is tr) NB: --regressorfreq and --regressortstep
                                           are two ways to specify the same thing
          --regressorstart=START         - The time delay in seconds into the regressor file, corresponding
                                           in the first TR of the fmri file (default is 0.0)
          --phat                         - Use generalized cross-correlation with phase alignment
                                           transform (PHAT) instead of correlation
          --liang                        - Use generalized cross-correlation with Liang weighting function
                                           (Liang, et al, doi:10.1109/IMCCC.2015.283)
          --eckart                       - Use generalized cross-correlation with Eckart weighting function
          --corrmaskthresh=PCT           - Do correlations in voxels where the mean exceeds this
                                           percentage of the robust max (default is 1.0)
          --corrmask=MASK                - Only do correlations in voxels in MASK (if set, corrmaskthresh
                                           is ignored).
          --accheck                      - Check for periodic components that corrupt the autocorrelation

      Correlation fitting options:
          -Z DELAYTIME                   - Don't fit the delay time - set it to DELAYTIME seconds
                                           for all voxels
          -r LAGMIN,LAGMAX               - Limit fit to a range of lags from LAGMIN to LAGMAX
          -s SIGMALIMIT                  - Reject lag fits with linewidth wider than SIGMALIMIT
          -B                             - Bipolar mode - match peak correlation ignoring sign
          --nofitfilt                    - Do not zero out peak fit values if fit fails
          --searchfrac=FRAC              - When peak fitting, include points with amplitude > FRAC * the
                                           maximum amplitude.
                                           (default value is 0.5)
          --peakfittype=FITTYPE          - Method for fitting the peak of the similarity function
                                           (default is 'gauss'). 'quad' uses a quadratic fit. Other options are
                                           'fastgauss' which is faster but not as well tested, and 'None'.
          --despecklepasses=PASSES       - detect and refit suspect correlations to disambiguate peak
                                           locations in PASSES passes
          --despecklethresh=VAL          - refit correlation if median discontinuity magnitude exceeds
                                           VAL (default is 5s)
          --softlimit                    - Allow peaks outside of range if the maximum correlation is
                                           at an edge of the range.

      Regressor refinement options:
          --refineprenorm=TYPE           - Apply TYPE prenormalization to each timecourse prior
                                           to refinement (valid weightings are 'None',
                                           'mean' (default), 'var', and 'std'
          --refineweighting=TYPE         - Apply TYPE weighting to each timecourse prior
                                           to refinement (valid weightings are 'None',
                                           'R', 'R2' (default)
          --passes=PASSES,               - Set the number of processing passes to PASSES
           --refinepasses=PASSES           (default is 1 pass - no refinement).
                                           NB: refinepasses is the wrong name for this option -
                                           --refinepasses is deprecated, use --passes from now on.
          --refineinclude=MASK[:VALSPEC] - Only use nonzero voxels in MASK for regressor refinement (if VALSPEC is
                                           given, only voxels with integral values listed in VALSPEC are used.)
          --refineexclude=MASK[:VALSPEC] - Do not use nonzero voxels in MASK for regressor refinement (if VALSPEC is
                                           given, only voxels with integral values listed in VALSPEC are used.)
          --lagminthresh=MIN             - For refinement, exclude voxels with delays less
                                           than MIN (default is 0.5s)
          --lagmaxthresh=MAX             - For refinement, exclude voxels with delays greater
                                           than MAX (default is 5s)
          --ampthresh=AMP                - For refinement, exclude voxels with correlation
                                           coefficients less than AMP (default is 0.3).  NOTE: ampthresh will
                                           automatically be set to the p<0.05 significance level determined by
                                           the -N option if -N is set greater than 0 and this is not
                                           manually specified.
          --sigmathresh=SIGMA            - For refinement, exclude voxels with widths greater
                                           than SIGMA (default is 100s)
          --refineoffset                 - Adjust offset time during refinement to bring peak
                                           delay to zero
          --pickleft                     - When setting refineoffset, always select the leftmost histogram peak
          --pickleftthresh=THRESH        - Set the threshold value (fraction of maximum) to decide something is a
                                           peak in a histogram.  Default is 0.33.
          --refineupperlag               - Only use positive lags for regressor refinement
          --refinelowerlag               - Only use negative lags for regressor refinement
          --pca                          - Use pca to derive refined regressor (default is
                                           unweighted averaging)
          --ica                          - Use ica to derive refined regressor (default is
                                           unweighted averaging)
          --weightedavg                  - Use weighted average to derive refined regressor
                                           (default is unweighted averaging)
          --avg                          - Use unweighted average to derive refined regressor
                                           (default)
          --psdfilter                    - Apply a PSD weighted Wiener filter to shifted
                                           timecourses prior to refinement

      Output options:
          --limitoutput                  - Don't save some of the large and rarely used files
          -T                             - Save a table of lagtimes used
          -h HISTLEN                     - Change the histogram length to HISTLEN (default is
                                           100)
          --glmsourcefile=FILE           - Regress delayed regressors out of FILE instead of the
                                           initial fmri file used to estimate delays
          --noglm                        - Turn off GLM filtering to remove delayed regressor
                                           from each voxel (disables output of fitNorm)
          --preservefiltering            - don't reread data prior to GLM

      Miscellaneous options:
          --noprogressbar                - Disable progress bars - useful if saving output to files
          --wiener                       - Perform Wiener deconvolution to get voxel transfer functions
          --usesp                        - Use single precision for internal calculations (may
                                           be useful when RAM is limited)
          -c                             - Data file is a converted CIFTI
          -S                             - Simulate a run - just report command line options
          -d                             - Display plots of interesting timecourses
          --nonumba                      - Disable jit compilation with numba
          --nosharedmem                  - Disable use of shared memory for large array storage
          --memprofile                   - Enable memory profiling for debugging - warning:
                                           this slows things down a lot.
          --multiproc                    - Enable multiprocessing versions of key subroutines.  This
                                           speeds things up dramatically.  Almost certainly will NOT
                                           work on Windows (due to different forking behavior).
          --mklthreads=NTHREADS          - Use no more than NTHREADS worker threads in accelerated numpy calls.
          --nprocs=NPROCS                - Use NPROCS worker processes for multiprocessing.  Setting NPROCS
                                           less than 1 sets the number of worker processes to
                                           n_cpus - 1 (default).  Setting NPROCS enables --multiproc.
          --debug                        - Enable additional information output
          --saveoptionsasjson            - Save the options file in json format rather than text.  Will eventually
                                           become the default, but for now I'm just trying it out.

      Experimental options (not fully tested, may not work):
          --cleanrefined                 - perform additional processing on refined regressor to remove spurious
                                           components.
          --dispersioncalc               - Generate extra data during refinement to allow calculation of
                                           dispersion.
          --acfix                        - Perform a secondary correlation to disambiguate peak location
                                           (enables --accheck).  Experimental.
          --tmask=MASKFILE               - Only correlate during epochs specified in
                                           MASKFILE (NB: if file has one column, the length needs to match
                                           the number of TRs used.  TRs with nonzero values will be used
                                           in analysis.  If there are 2 or more columns, each line of MASKFILE
                                           contains the time (first column) and duration (second column) of an
                                           epoch to include.)



These options are somewhat self-explanatory.  I will be expanding this section of the manual going forward, but I want to put something here to get this out here.

When using the legacy interface, file names will be output using the old, non-BIDS names and formats.  rapidtide can be forced to use the old style outputs with the ``--legacyoutput`` flag.

Equivalence between BIDS and legacy outputs:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :header: "BIDS style name", "Legacy name"
   :widths: 20, 20

   "XXX_maxtime_map(.nii.gz, .json)", "XXX_lagtimes.nii.gz"
   "XXX_desc-maxtime_hist(.tsv, .json)", "XXX_laghist.txt"
   "XXX_maxcorr_map(.nii.gz, .json)", "XXX_lagstrengths.nii.gz"
   "XXX_desc-maxcorr_hist(.tsv, .json)", "XXX_strengthhist.txt"
   "XXX_maxcorrsq_map(.nii.gz, .json)", "XXX_R2.nii.gz"
   "XXX_desc-maxcorrsq_hist(.tsv, .json)", "XXX_R2hist.txt"
   "XXX_maxwidth_map(.nii.gz, .json)", "XXX_lagsigma.nii.gz"
   "XXX_desc-maxwidth_hist(.tsv, .json)", "XXX_widthhist.txt"
   "XXX_MTT_map(.nii.gz, .json)", "XXX_MTT.nii.gz"
   "XXX_corrfit_mask.nii.gz", "XXX_fitmask.nii.gz"
   "XXX_corrfitfailreason_map(.nii.gz, .json)", "XXX_failreason.nii.gz"
   "XXX_desc-corrfitwindow_info.nii.gz", "XXX_windowout.nii.gz"
   "XXX_desc-runoptions_info.json", "XXX_options.json"
   "XXX_desc-lfofilterCleaned_bold(.nii.gz, .json)", "XXX_filtereddata.nii.gz"
   "XXX_desc-lfofilterRemoved_bold(.nii.gz, .json)", "XXX_datatoremove.nii.gz"
   "XXX_desc-lfofilterCoeff_map.nii.gz", "XXX_fitcoeff.nii.gz"
   "XXX_desc-lfofilterMean_map.nii.gz", "XXX_meanvalue.nii.gz"
   "XXX_desc-lfofilterNorm_map.nii.gz", "XXX_fitNorm.nii.gz"
   "XXX_desc-lfofilterR2_map.nii.gz", "XXX_r2value.nii.gz"
   "XXX_desc-lfofilterR_map.nii.gz", "XXX_rvalue.nii.gz"
   "XXX_desc-processed_mask.nii.gz", "XXX_corrmask.nii.gz"
   "XXX_desc-globalmean_mask.nii.gz", "XXX_meanmask.nii.gz"
   "XXX_desc-refine_mask.nii.gz", "XXX_refinemask.nii.gz"
   "XXX_desc-despeckle_mask.nii.gz", "XXX_despecklemask.nii.gz"
   "XXX_desc-corrout_info.nii.gz", "XXX_corrout.nii.gz"
   "XXX_desc-gaussout_info.nii.gz", "XXX_gaussout.nii.gz"
   "XXX_desc-autocorr_timeseries(.tsv, .json)", "XXX_referenceautocorr_passN.txt"
   "XXX_desc-corrdistdata_info(.tsv, .json)", "XXX_corrdistdata_passN.txt"
   "XXX_desc-nullsimfunc_hist(.tsv, .json)", "XXX_nullsimfunchist_passN.txt"
   "XXX_desc-plt0p050_mask.nii.gz", "XXX_p_lt_0p050_mask.nii.gz"
   "XXX_desc-plt0p010_mask.nii.gz", "XXX_p_lt_0p010_mask.nii.gz"
   "XXX_desc-plt0p005_mask.nii.gz", "XXX_p_lt_0p005_mask.nii.gz"
   "XXX_desc-plt0p001_mask.nii.gz", "XXX_p_lt_0p001_mask.nii.gz"
   "XXX_desc-globallag_hist(.tsv, .json)", "XXX_globallaghist_passN.txt"
   "XXX_desc-initialmovingregressor_timeseries(.tsv, .json)", "XXX_reference_origres.txt, XXX_reference_origres_prefilt.txt"
   "XXX_desc-movingregressor_timeseries(.tsv, .json)", "XXX_reference_fmrires_passN.txt"
   "XXX_desc-oversampledmovingregressor_timeseries(.tsv, .json)", "XXX_reference_resampres_passN.txt"
   "XXX_desc-refinedmovingregressor_timeseries(.tsv, .json)", "XXX_unfilteredrefinedregressor_passN.txt, XXX_refinedregressor_passN.txt"
   "XXX_commandline.txt", "XXX_commandline.txt"
   "XXX_formattedcommandline.txt", "XXX_formattedcommandline.txt"
   "XXX_memusage.csv", "XXX_memusage.csv"
   "XXX_runtimings.txt", "XXX_runtimings.txt"
..
