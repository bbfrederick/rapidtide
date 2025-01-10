rapidtide
---------

Description:
^^^^^^^^^^^^

The central program in this package is rapidtide.  This is the program that
calculates a similarity function between a "probe" signal and every voxel of
a BOLD fMRI dataset.  It then determines the peak value, time delay, and wi
dth of the similarity function to determine when and how strongly that probe
signal appears in each voxel.

At its core, rapidtide is simply performing a full crosscorrelation between a
"probe" timecourse and every voxel in an fMRI dataset (by “full” I mean over
a range of time lags that account for any delays between the signals, rather
than only at zero lag, as in a Pearson correlation).  As with many things,
however, the devil is in the details, and so rapidtide provides a number of
features which make it pretty good at this particular task.  A few highlights:

* There are lots of ways to do something even as simple as a cross-correlation in a nonoptimal way (not windowing, improper normalization, doing it in the time rather than frequency domain, etc.).  I'm pretty sure what rapidtide does by default is, if not the best way, at least a very good and very fast way.
* rapidtide has been optimized and profiled to speed it up quite a bit; it has an optional dependency on numba – if it’s installed, some of the most heavily used routines will speed up significantly due to judicious use of @jit.
* The sample rate of your probe regressor and the fMRI data do not have to match - rapidtide resamples the probe regressor to an integral multiple of the fMRI data rate automatically.
* The probe and data can be temporally prefiltered to the LFO, respiratory, or cardiac frequency band with a command line switch, or you can specify any low, high, or bandpass range you want.
* The data can be spatially smoothed at runtime (so you don't have to keep smoothed versions of big datasets around). This is quite fast, so no reason not to do it this way.
* rapidtide can generate a probe regressor from the global mean of the data itself - no externally recorded timecourse is required.  Optionally you can input both a mask of regions that you want to be included in the mean, and the voxels that you want excluded from the mean (there are situations when you might want to do one or the other or both).
* Determining the significance threshold for filtered correlations where the optimal delay has been selected is nontrivial; using the conventional formulae for the significance of a correlation leads to wildly inflated p values. rapidtide estimates the spurious correlation threshold by calculating the distribution of null correlation values obtained with a shuffling  procedure at the beginning of each run (the default is to use 10000 shuffled correlations), and uses this value to mask the correlation maps it calculates.  As of version 0.1.2 it will also handle two-tailed significance, which you need when using bipolar mode.
* rapidtide can do an iterative refinement of the probe regressor by aligning the voxel timecourses in time and regenerating the test regressor.
* rapidtide fits the peak of the correlation function, so you can make fine grained distinctions between close lag times. The resolution of the time lag discrimination is set by the length of the timecourse, not the timestep – this is a feature of correlations, not rapidtide.
* Once the time delay in each voxel has been found, rapidtide outputs a 4D file of delayed probe regressors for using as voxel specific confound regressors or to estimate the strength of the probe regressor in each voxel.  This regression is performed by default, but these outputs let you do it yourself if you are so inclined.
* I've put a lot of effort into making the outputs as informative as possible - lots of useful maps, histograms, timecourses, etc.
* There are a lot of tuning parameters you can mess with if you feel the need.  I've tried to make intelligent defaults so things will work well out of the box, but you have the ability to set most of the interesting parameters yourself.

Inputs:
^^^^^^^

At a minimum, rapidtide needs a data file to work on (space by time), which is generally thought to be a BOLD fMRI
data file.  This can be Nifti1 or Nifti2 (for fMRI data, in which case it is time by up to 3 spatial dimensions) or
a whitespace separated text file (for NIRS data, each column is a time course, each row a separate channel); I can
currently read (probably) but not write Cifti files, so if you want to use grayordinate files you need to convert
them to nifti2 in workbench, run rapidtide, then convert back. As soon as nibabel finishes their Cifti support
(EDIT: and I get around to figuring it out), I'll add that.

The file needs one time dimension and at least one spatial dimension.  Internally, the array is flattened to a
time by voxel array for simplicity.

The file you input here should be the result of any preprocessing you intend to do.  The expectation is that
rapidtide will be run as the last preprocessing step before resting state or task based analysis.  So any slice
time correction, motion correction, spike removal, etc. should already have been done.  If you use FSL, this
means that if you've run preprocessing, you would use the filtered_func_data.nii.gz file as input.  Temporal
and spatial filtering are the two (partial) exceptions here.  Generally rapidtide is most useful for looking
at low frequency oscillations, so when you run it, you usually use the ``--filterband lfo`` option or some
other to limit the analysis to the detection and removal of low frequency systemic physiological oscillations.
So rapidtide will generally apply it's own temporal filtering on top of whatever you do in preprocessing.
Also, you have the option of doing spatial smoothing in rapidtide to boost the SNR of the analysis; the
hemodynamic signals rapidtide looks for are often very smooth, so you rather than smooth your functional
data excessively, you can do it within rapidtide so that only the hemodynamic data is smoothed at that level.

Outputs:
^^^^^^^^

Outputs are space or space by time NIFTI or text files, depending on what the input data file was, and some
text files containing textual information, histograms, or numbers.  File formats and naming follow BIDS
conventions for derivative data for fMRI input data.  Output spatial dimensions and file type match the
input dimensions and file type (Nifti1 in, Nifti1 out).  Depending on the file type of map, there can be
no time dimension, a time dimension that matches the input file, or something else, such as a time lag
dimension for a correlation map.

.. _bidsoutputs:

BIDS Outputs:
^^^^^^^^^^^^^

.. csv-table::  Output files produced by rapidtide
   :header: "Name", "Extension(s)", "Content", "When present"
   :widths: 30, 10, 30, 20

    "XXX_DONE", "txt", "Run status file", "Always if rapidtide has finished without error"
    "XXX_ISRUNNING", "txt", "Run status file", "Only present if rapidtide is running, or has failed"
    "XXX_formattedcommandline", "txt", "Command line used to invoke rapidtide, nicely formatted", "Always"
    "XXX_log", "txt", "Diagnostic log file with a lot of informational output", "Always"
    "XXX_memusage", "tsv", "Memory usage statistics for performance tuning", "Always"
    "XXX_commandline", "txt", "Raw command line used to invoke rapidtide", "Always"
    "XXX_desc-autocorr_timeseries", "tsv.gz, json", "Autocorrelation of the probe regressor for each pass", "Always"
    "XXX_desc-cleansimdistdata_info", "tsv.gz, json", "Individual sham correlation datapoints after outlier removal", "Present if numnull > 0"
    "XXX_desc-confoundfilterCleaned_bold", "nii.gz, json", "fMRI data after motion/confound regression", "Present if doing motion/confound regression and outputlevel is max"
    "XXX_desc-confoundfilterR2_hist", "tsv.gz, json", "Histogram of confound filter R2 values", "Present if doing motion/confound regression"
    "XXX_desc-confoundfilterR2_map", "nii.gz, json", "R2 of the motion/confound regression", "Present if doing motion/confound regression"
    "XXX_desc-corrfit_mask", "nii.gz, json", "Voxels where correlation value was fit", "Always"
    "XXX_desc-corrfitfailreason_info", "nii.gz, json", "Result codes for correlation fit", "Always"
    "XXX_desc-corrfitwindow_info", "nii.gz, json", "The search window for the correlation peak fit", "Present if outputlevel is max"
    "XXX_desc-corrout_info", "nii.gz, json", "Correlation function", "Present if outputlevel is ≥ normal"
    "XXX_desc-corrtimes_timeseries", "tsv.gz, json", "Correlation time axis", "Present if outputlevel is max"
    "XXX_desc-CoV_map", "nii.gz, json", "Voxelwise coefficient of variation of fmri data", "Always"
    "XXX_desc-delayoffset_hist", "tsv.gz, json", "Histogram of delay offsets calculated from GLM", "Present if refinedelay is enabled"
    "XXX_desc-delayoffset_map", "nii.gz, json", "Delay offset correction from delay refinement", "Present if refinedelay is enabled"
    "XXX_desc-despeckle_mask", "nii.gz, json", "Voxels that underwent despeckling in the final pass", "Present if despecklepasses > 0 (default) and outputlevel is max"
    "XXX_desc-EV_timeseries", "tsv.gz, json", "GLM regressor set", "Present if GLM is enabled (default)"
    "XXX_desc-expandedconfounds_timeseries", "tsv.gz, json", "The expanded (via derivatives and powers) set of confound regressors used for prefiltering the data", "Present if doing motion/confound regression"
    "XXX_desc-filteredglmderivratios_map", "nii.gz, json", "glmderivratios, with outliers patched using median filtered data", "Present if refinedelay is enabled and outputlevel is ≥ normal"
    "XXX_desc-formattedruntimings_info", "tsv", "No description", "Always"
    "XXX_desc-gaussout_info", "nii.gz, json", "Simulated correlation function", "Present if outputlevel is max"
    "XXX_desc-glmderivratios_map", "nii.gz, json", "Ratio of the first derivative of delayed sLFO to the delayed sLFO", "Present if refinedelay is enabled and outputlevel is ≥ normal"
    "XXX_desc-globallag_hist", "tsv.gz, json", "Histogram of lag times from global lag calculation", "Always"
    "XXX_desc-globalmean_mask", "nii.gz, json", "Voxels used to calculate global mean", "Always"
    "XXX_desc-initialmovingregressor_timeseries", "tsv.gz, json", "The raw and filtered initial probe regressor, at the original sampling resolution", "Always"
    "XXX_desc-lagtcgenerator_timeseries", "tsv.gz, json", "The lagged timecourse generator", "Always"
    "XXX_desc-lfofilterCleaned_bold", "nii.gz, json", "fMRI data with sLFO signal filtered out", "Present if GLM is enabled (default) and outputlevel is ≥ less"
    "XXX_desc-lfofilterCoeff_map", "nii.gz, json", "Fit coefficient", "Present if GLM is enabled (default) and outputlevel is ≥ normal"
    "XXX_desc-lfofilterCoeffDerivN_map", "nii.gz, json", "Fit coefficient for the Nth temporal derivative", "Present if GLM is enabled (default), glmderivs > 0, and outputlevel is max"
    "XXX_desc-lfofilterEV_bold", "nii.gz, json", "Shifted sLFO regressor to filter", "Present if despecklepasses > 0 (default) and outputlevel is max"
    "XXX_desc-lfofilterEVDerivN_bold", "nii.gz, json", "Nth time derivative of shifted sLFO regressor", "Present if GLM is enabled (default), glmderivs > 0, and outputlevel is max"
    "XXX_desc-lfofilterInbandVarianceAfter_map", "nii.gz, json", "Inband variance after filtering", "Present if GLM is enabled (default)"
    "XXX_desc-lfofilterInbandVarianceBefore_map", "nii.gz, json", "Inband variance prior to filtering", "Present if GLM is enabled (default)"
    "XXX_desc-lfofilterInbandVarianceChange_hist", "tsv.gz, json", "Histogram of percent of inband variance removed by GLM filter", "Present if GLM is enabled (default)"
    "XXX_desc-lfofilterInbandVarianceChange_map", "nii.gz, json", "Change in inband variance after filtering, in percent", "Present if GLM is enabled (default)"
    "XXX_desc-lfofilterMean_map", "nii.gz, json", "Intercept from GLM fit", "Present if GLM is enabled (default) and outputlevel is ≥ normal"
    "XXX_desc-lfofilterNoiseRemoved_timeseries", "tsv.gz, json", "Variance over space of data removed by GLM filter at each timepoint", "Present if GLM is enabled (default) and outputlevel is ≥ less"
    "XXX_desc-lfofilterNorm_map", "nii.gz, json", "Normalized fit coefficient", "Present if GLM is enabled (default) and outputlevel is ≥ normal"
    "XXX_desc-lfofilterNormDerivN_map", "nii.gz, json", "Normalized fit coefficient for the Nth temporal derivative", "Present if GLM is enabled (default), glmderivs > 0, and outputlevel is max"
    "XXX_desc-lfofilterR2_hist", "tsv.gz, json", "Histogram of GLM filter R2 values", "Present if GLM is enabled (default)"
    "XXX_desc-lfofilterR2_map", "nii.gz, json", "Squared R value of the GLM fit (proportion of variance explained)", "Present if GLM is enabled (default) and outputlevel is ≥ less"
    "XXX_desc-lfofilterR_map", "nii.gz, json", "R value of the GLM fit", "Present if GLM is enabled (default) and outputlevel is ≥ normal"
    "XXX_desc-lfofilterRemoved_bold", "nii.gz, json", "sLFO signal filtered out of this voxel", "Present if GLM is enabled (default) and outputlevel is ≥ more"
    "XXX_desc-maxcorr_hist", "tsv.gz, json", "Histogram of maximum correlation coefficients", "Always"
    "XXX_desc-maxcorr_map", "nii.gz, json", "Maximum correlation strength", "Always"
    "XXX_desc-maxcorralt_map", "nii.gz, json", "R value of the GLM fit, with sign", "Present if refinedelay is enabled"
    "XXX_desc-maxcorrsq_map", "nii.gz, json", "Squared maximum correlation strength (proportion of variance explained)", "Always"
    "XXX_desc-maxtime_hist", "tsv.gz, json", "Histogram of maximum correlation times", "Always"
    "XXX_desc-maxtime_map", "nii.gz, json", "Lag time in seconds", "Always"
    "XXX_desc-maxtimerefined_map", "nii.gz, json", "Lag time in seconds, refined", "Present if refinedelay is enabled"
    "XXX_desc-maxwidth_hist", "tsv.gz, json", "Histogram of correlation peak widths", "Always"
    "XXX_desc-maxwidth_map", "nii.gz, json", "Width of corrrelation peak", "Always"
    "XXX_desc-mean_map", "nii.gz, json", "Voxelwise mean of fmri data", "Always"
    "XXX_desc-medfiltglmderivratios_map", "nii.gz, json", "Median filtered version of the glmderivratios map", "Present if refinedelay is enabled and outputlevel is ≥ normal"
    "XXX_desc-mitimes_timeseries", "tsv.gz, json", "Cross mutual information time axis", "Present if outputlevel is max"
    "XXX_desc-movingregressor_timeseries", "tsv.gz, json", "The probe regressor used in each pass, at the time resolution of the data", "Always"
    "XXX_desc-MTT_hist", "tsv.gz, json", "Histogram of correlation peak widths", "Always"
    "XXX_desc-MTT_map", "nii.gz, json", "Mean transit time (estimated)", "Always"
    "XXX_desc-nullsimfunc_hist", "tsv.gz, json", "Null correlation histogram", "Present if numnull > 0"
    "XXX_desc-oversampledmovingregressor_timeseries", "tsv.gz, json", "The probe regressor used in each pass, at the time resolution used for calculating the similarity function", "Always"
    "XXX_desc-plt0p001_mask", "nii.gz, json", "Voxels where the maxcorr value exceeds the p < 0.001 significance level", "Present if numnull > 0"
    "XXX_desc-plt0p005_mask", "nii.gz, json", "Voxels where the maxcorr value exceeds the p < 0.005 significance level", "Present if numnull > 0"
    "XXX_desc-plt0p010_mask", "nii.gz, json", "Voxels where the maxcorr value exceeds the p < 0.010 significance level", "Present if numnull > 0"
    "XXX_desc-plt0p050_mask", "nii.gz, json", "Voxels where the maxcorr value exceeds the p < 0.050 significance level", "Present if numnull > 0"
    "XXX_desc-preprocessedconfounds_timeseries", "tsv.gz, json", "The preprocessed (normalized, filtered, orthogonalized) set of expanded confound regressors used for prefiltering the data", "Present if doing motion/confound regression"
    "XXX_desc-processed_mask", "nii.gz", "No description", "Always"
    "XXX_desc-ratiotodelayfunc_timeseries", "tsv.gz, json", "The function mapping derivative ratio to delay", "Present if refinedelay is enabled"
    "XXX_desc-refine_mask", "nii.gz, json", "Voxels used for refinement", "Present if passes > 1"
    "XXX_desc-refinedmovingregressor_timeseries", "tsv.gz, json", "The raw and filtered probe regressor produced by the refinement procedure, at the time resolution of the data", "Present if passes > 1"
    "XXX_desc-runoptions_info", "json", "A detailed dump of all internal variables in the program.  Useful for debugging and data provenance.", "Always"
    "XXX_desc-shiftedtcs_bold", "nii.gz, json", "The filtered input fMRI data, in voxels used for refinement, time shifted by the negated delay in every voxel so that the moving blood component is aligned.", "Present if passes > 1 and outputlevel is max"
    "XXX_desc-simdistdata_info", "tsv.gz, json", "Individual sham correlation datapoints", "Present if numnull > 0"
    "XXX_desc-sLFOamplitude_timeseries", "tsv.gz, json", "Filtered RMS amplitude of the probe regressor, and a linear fit", "Always"
    "XXX_desc-std_map", "nii.gz, json", "Voxelwise standard deviation of fmri data", "Always"
    "XXX_desc-timepercentile_map", "nii.gz, json", "Percentile ranking of this voxels delay", "Always"
    "XXX_desc-trimmedcorrtimes_timeseries", "tsv.gz, json", "Trimmed correlation time axis", "Present if outputlevel is max"
    "XXX_desc-trimmedmitimes_timeseries", "tsv.gz, json", "Trimmed cross mutual information time axis", "Present if outputlevel is max"
..



Output data size:
^^^^^^^^^^^^^^^^^

The amount of data output by rapidtide varies quite a bit, depending on your run options and the output level you select.
What output level you use depends on what you are trying to do.  The vast majority of the runtime of rapidtide is spent
estimating, extracting and refining the sLFO signal, and calculating the voxelwise blood arrival time delay and signal
strength.  This produces a surprisingly small amount of data - the largest output files are the maps of the various
hemodynamic parameters and some masks, each as large a single TR of the input data set.  So at a minimum (as in, you
select ``"--outputlevel min"`` and do not run GLM denoising: ``"--noglm"``), you produce
16 3D maps as NIFTI files, and a number of masks and timecourse files.  For a single resting state run in the HCP-YA
dataset, this is ~13MB of data (compared to the input data file size of about 1GB).  If you want slightly more data
to help you evaluate the fit quality, and make cool movies, you probably want to leave the outputlevel at the default of
``"normal"``.

You can calculate the output data size approximately with the following formulae (to first approximation, assuming
that the image files dominate the size of the output data).

FMRISIZE is the number of TRs in the input fMRI data.

CORRFUNCSIZE is the size of the correlation function in TRs at the oversampled TR.

* The TR oversampling factor is the smallest integer divisor of the fMRI TR that results in an oversampled TR <= 0.5 seconds.
* CORRFUNCSIZE is the search range in seconds divided by the oversampled TR.

The output sizes in TRs (with no motion regression) are as follows:

.. csv-table::  Total image output data size in TRs
   :header: "Output level", "Passes>1?", "Refine delay?", "GLM?", "Number of TRs"
   :widths: 10, 10, 10, 10, 20

    "min", "No", "No",  "No", "13"
    "min", "No", "Yes",  "No", "16"
    "min", "Yes", "No",  "No", "16"
    "min", "Yes", "Yes",  "No", "19"
    "min", "No", "No",  "Yes", "14"
    "min", "Yes", "No",  "Yes", "17"
    "less", "No", "No",  "No", "13"
    "less", "No", "Yes",  "No", "17 + 1*FMRISIZE"
    "less", "Yes", "No",  "No", "16"
    "less", "Yes", "Yes",  "No", "20 + 1*FMRISIZE"
    "normal", "No", "No",  "No", "13 + 1*CORRFUNCSIZE"
    "normal", "No", "Yes",  "No", "21 + 1*CORRFUNCSIZE + 1*FMRISIZE"
    "normal", "Yes", "No",  "No", "19 + 1*CORRFUNCSIZE"
    "normal", "Yes", "Yes",  "No", "27 + 1*CORRFUNCSIZE + 1*FMRISIZE"
    "more", "No", "No",  "No", "13 + 1*CORRFUNCSIZE"
    "more", "No", "Yes",  "No", "21 + 1*CORRFUNCSIZE + 2*FMRISIZE"
    "more", "Yes", "No",  "No", "19 + 1*CORRFUNCSIZE"
    "more", "Yes", "Yes",  "No", "27 + 1*CORRFUNCSIZE + 2*FMRISIZE"
    "max", "No", "No",  "No", "13 + 3*CORRFUNCSIZE"
    "max", "No", "Yes",  "No", "21 + 3*CORRFUNCSIZE + 3*FMRISIZE"
    "max", "Yes", "No",  "No", "19 + 3*CORRFUNCSIZE"
    "max", "Yes", "Yes",  "No", "27 + 3*CORRFUNCSIZE + 3*FMRISIZE"
    "max", "No", "No",  "Yes", "14 + 3*CORRFUNCSIZE + 1*FMRISIZE"
    "max", "Yes", "No",  "Yes", "20 + 3*CORRFUNCSIZE + 1*FMRISIZE"
..

The data size is then this number of TRs times the size of 1 TR worth of data in the input fMRI file, (plus the size
of the various timecourse files and .json sidecars which are much smaller than the image files).


As an example, the following table shows the size of the data produced by running a rapidtide analysis on one HCP-YA
resting state dataset with various output levels, with and without doing GLM noise removal, either directly, or
with the addition of one voxelwise time derivative.  The correlation function fit was calculated from -5 to 10
seconds, resulting in a correlation function length of 41 points at the oversampled TR of 0.36 seconds. NB: motion
regression is independent of GLM calculation, so to find the size of a GLM analysis with motion regression at a
given output level, add the difference between the sizes of the motion regressed and non-motion regressed non-GLM
analysis.


.. csv-table::  Output data size from running rapidtide on one HCP-YA rsfMRI dataset
   :header: "Output level", "Motion regression", "GLM?", "Derivatives", "Size in bytes"
   :widths: 10, 10, 10, 10, 10

    "min", "No", "No", "", "13M"
    "min", "Yes", "No", "", "15M"
    "min", "No", "Yes", "0", "17M"
    "min", "No", "Yes", "1", "17M"
    "less", "No", "No", "", "13M"
    "less", "Yes", "No", "", "15M"
    "less", "No", "Yes", "0", "1.2G"
    "less", "No", "Yes", "1", "1.2G"
    "less", "No", "No", "", "13M"
    "less", "Yes", "No", "", "15M"
    "less", "No", "Yes", "0", "1.2G"
    "less", "No", "Yes", "1", "1.2G"
    "normal", "No", "No", "", "86M"
    "normal", "Yes", "No", "", "88M"
    "normal", "No", "Yes", "0", "1.3G"
    "normal", "No", "Yes", "1", "1.3G"
    "more", "No", "No", "", "724M"
    "more", "Yes", "No", "", "702M"
    "more", "No", "Yes", "0", "3.2G"
    "more", "No", "Yes", "1", "3.2G"
    "max", "No", "No", "", "805M"
    "max", "Yes", "No", "", "2.9G"
    "max", "No", "Yes", "0", "4.5G"
    "max", "No", "Yes", "1", "5.7G"
..


PRO TIP:  Extraction of the sLFO regressor and calculation of the delay and strength maps take the VAST majority of
the computation time, and generates only a small fraction of the data of a full analysis.
If you are doing computation on AWS (where compute is cheap, storage is semi-pricey,and download costs are extortionate), it makes
sense to do everything except GLM filtering on your data, and download or store the outputs of that, only doing the
GLM step at the time when you need to do it.  For example - running rapidtide on all of the HCP-YA resting state data
generates less than 70GB of output data.  That's not too expensive to download, or store on S3, and costs nothing to
upload.  The denoised data, however is huge (bigger than the input dataset), so you don't want to download it or even
pay to store it for too long.  So make it when you need it, use it for whatever,
then throw it away, and make it again if you need it again.
This will save you an enormous amount of money.



Usage:
^^^^^^

.. argparse::
   :ref: rapidtide.workflows.rapidtide_parser._get_parser
   :prog: rapidtide
   :func: _get_parser


Preprocessing for rapidtide
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Rapidtide operates on data which has been subjected to "standard" preprocessing steps, most importantly motion
correction and slice time correction.

**Motion correction** - Motion correction is good since you want to actually be looking at the same voxels in each timepoint.  Definitely
do it.  There may be spin history effects even after motion correction, so if you give rapidtide a motion file
using ``--motionfile FILENAME`` (and various other options to tune how it does the motion regression)
it can regress out residual motion prior to estimating sLFO parameters. In cases of extreme motion, this will
make rapidtide work a lot better.  If you choose to regress out the motion signals yourself, that's fine too -
rapidtide is happy to work on data that's been run through AROMA (not so much FIX - see a further discussion below).

**Slice time correction** - Since rapidtide is looking for subtle time differences in the arrival of the
sLFO signal, slice acquisition time differences will show up as artifactual offsets in the delay maps if you don't
correct them beforehand.  If you are doing noise removal, that's not
such a big deal, but if you're doing delay mapping, you'll get stripes in your delay maps, which tell you about the
fMRI acquisition, but you care about physiology, so best to avoid that.  Unfortunately, Human Connectome Project data
does NOT have slice time correction applied, and unless you want to rerun the entire processing chain to add it in,
you just have to deal with it.  Fortunately the TR is quite short, so the stripes are subtle.  The geometric
distortion correction and alignment steps done in the HCP distort the stripes, but you can certainly see them.  If you
average enough subjects though, they get washed out.

**Spatial filtering** - I generally do NOT apply any spatial filtering
during preprocessing for a variety of reasons.
fMRIPrep doesn't do it, so I feel validated in this choice.
You can always do it later, and rapidtide lets you do spatial smoothing for the purpose of
estimating the delayed regressor using the ``--gausssigma`` parameter.
This turns out to stabilize the fits for rapidtide and is usually a good thing,
however you probably don't want it for other processing (but that's ok - see below).

**Temporal filtering** - Rapidtide does all it's own temporal filtering; highpass filtering at 0.01Hz, common in r
esting state preprocessing,
doesn't affect the frequency ranges rapidtide cares about for sLFOs, so you can do it or not during preprocessing
as you see fit (but if you're doing CVR or gas challenge experiments you probably shouldn't).

NOTE: Astute readers will notice that between spatial filtering, motion regression, and other procedures, rapidtide
does a lot of it's work of estimating sLFOs on potentially heavily filtered data, which is good for improving the
estimation and fitting of the sLFO signal.  However, you may or may not
want this filtering to have been done for whatever your particular subsequent analysis is.  So prior to GLM denoising, rapidtide
rereads the unmodified fMRI input file, and regresses the voxel specific sLFO out of *that* - since the filtering
process is linear, that's cool - the data you get out is the data you put in, just minus the sLFO signal.  If for
some reason you *do* want to use the data that rapidtide has abused, simply use the ``--preservefiltering`` option,
but I'd recommend you don't do that.

Working with standard fMRI packages
"""""""""""""""""""""""""""""""""""
**FSL** - At the time I first developed rapidtide, I was using FSL almost exclusively, so some of the assumptions
the program makes about the data stem from this.  If you want to integrate rapidtide into your FSL workflow, you would
typically use the ``filtered_func_data.nii.gz`` file from your FEAT directory (the result of FSL preprocessing)
as input to rapidtide.  Note that this is typically in native acquisition space.  You can use this, or do the
processing in standard space if you've done that alignment - either is fine, but for conventional EPI acquisitions,
there are typically far fewer voxels at native resolution, so processing will probably be faster.  On the flip side,
having everything in standard space makes it easier to combine runs and subjects.

**fMRIPrep** - If you do preprocessing in fMRIPrep,
the easiest file to use for input to rapidtide would be either
``derivatives/fmriprep/sub-XXX/ses-XXX/func/XXX_desc-preproc_bold.nii.gz`` (native space) or
``derivatives/fmriprep/sub-XXX/ses-XXX/func/XXX_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz``
(standard space - replace ``MNI152NLin6aAsym_res-2`` with whatever space and resolution you used if not the FSL compatible one).
If you do the analysis in standard space, it makes it easier to use freesurfer parcellations and gray/white/csf
segmentations that fMRIPrep provides for further tuning the rapidtide analysis.
See the "Theory of Operation" section for more on this subject.

You can pass the confounds file from fMRIPrep
(``derivatives/fmriprep/sub-XXX/ses-XXX/func/XXX_desc-confounds_timeseries.tsv``)
directly to rapidtide as ``--motionfile``.
However, if you want to use the ``--confoundfile`` parameter,
you need to create a reduced version of the confounds file with only the columns you want to use for confound regression.

You can also load the confounds file to identify non-steady-state volumes to use for the ``--numtozero`` parameter.

fMRIPrep includes the TR in the output NIfTI files' headers, so you don't need to provide ``--datatstep``,
and it usually performs slice timing correction
(unless you don't have slice timing information in your BIDS dataset or choose ``--ignore slicetiming``),
so you don't need to use ``--slicetiming``.

For most non-clinical participants,
we recommend using the tissue type masks provided by fMRIPrep for many of the masks used in rapidtide.
For example:

.. code-block:: bash

    rapidtide \
        sub-XXX/func/sub-XXX_task-rest_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz \
        /path/to/rapidtide/sub-XXX_task-rest_space-MNI152NLin6Asym_res-2 \
        --brainmask sub-XXX/anat/sub-XXX_space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz \
        --graymattermask sub-XXX/anat/sub-XXX_space-MNI152NLin6Asym_res-2_desc-GM_probseg.nii.gz \
        --whitemattermask sub-XXX/anat/sub-XXX_space-MNI152NLin6Asym_res-2_desc-WM_probseg.nii.gz \
        --motionfile sub-XXX/func/sub-XXX_task-rest_desc-confounds_timeseries.tsv


**AFNI** - Here's a case where you have to take some care - as I mentioned above, rapidtide assumes "FSL-like" data by
default.  The most important difference between AFNI and FSL preprocessing (assuming you've put your AFNI data into
NIFTI format) is that AFNI removes the mean from the preprocessed fMRI data
(this is a valid implementation choice - no judgement, but, no, actually - seriously, WTF?  WHY WOULD YOU DO THAT???).
This makes rapidtide sad, because the mean value of the fMRI data is used for all sorts of things like
generating masks.  Fortunately, this can be easily accommodated.  You have a couple of choices here.  You can
supply a mean mask and correlation mask explicitly using ``--globalmeaninclude FILENAME`` and ``--corrmask FILENAME``,
(FILENAME should definitely be a brain mask for ``--corrmask`` - it can be more focussed for ``--globalmeaninclude`` -
for example, a gray matter mask, but a brain mask works fine in most cases) which will get
rapidtide past the places that zero mean data will confuse it.  Alternately, if you don't have a brain mask, you can
use ``--globalmaskmethod variance`` to make a mask based on the variance over time in a voxel rather than than the
mean.  Rapidtide should then work as normal, although the display in ``tidepool`` will be a little weird unless you
specify a background image explicitly.

**SPM** - I have no reason to believe rapidtide won't work fine with data preprocessed in SPM.  That said, I don't use
SPM, so I can't tell you what file to use, or what format to expect the preprocessed data will be in.  If you,
dear reader, have
any insight into this, PLEASE tell me and I'll do what I need to to support SPM data in the code and documentation.


Analysis Examples:
^^^^^^^^^^^^^^^^^^
Rapidtide can do many things - as I've found more interesting things to do with time delay processing, it's gained
new functions and options to support these new applications.  As a result, it can be a little hard to know what to
use for a new experiment.  To help with that, I've decided to add this section to the manual to get you started.
It's broken up by type of data/analysis you might want to do.

NB: To speed up the analysis, adding the argument ``--nprocs XX`` to any of the following commands will parallelize
the analysis to use XX CPUs - set XX to -1 to use all available CPUs.  This can result in a speedup approaching a
factor of the number of CPUs used.

Removing low frequency physiological noise from fMRI data
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
This is what I figure most people will use rapidtide for - finding and removing the low frequency (LFO) signal
from an existing dataset
(including the case where the signal grows over time :footcite:p:`korponay2024nathumbeh`).
This presupposes you have not made a simultaneous physiological recording
(well, you may have, but it assumes you aren't using it).
For this, you can use a minimal set of options, since the defaults are set to be generally optimal for noise removal.

The base command you'd use would be:

    ::

        rapidtide \
            inputfmrifile \
            outputname \
            --denoising

This will do a the default analysis (but each and every particular can be changed by adding command line options).
By default, rapidtide will:

    #. Temporally prefilter the data to the LFO band (0.009-0.15Hz), and spatially filter with a Gaussian kernel of 1/2 the mean voxel dimension in x, y, and z.

    #. Construct a probe regressor from the global mean of the signal in inputfmrifile (default behavior if no regressor or selections masks are specified).

    #. Do three passes through the data.  In each step, rapidtide will:

        #. Perform a crosscorrelation of each voxel with the probe regressor using the "regressor" weighting.

        #. Estimate the location and strength of the correlation peak using the correlation similarity metric within a range of +/-10 seconds around around the modal delay value.

        #. Generate a new estimate of the global noise signal by:

            #. Aligning all of the voxel timecourses to bring the global signal into phase,

            #. Performing a PCA analysis,

            #. Reconstructing each timecourse using the PCA components accounting for 80% of the signal variance in the aligned voxel timecourses,

            #. Averaging the reconstructed timecourses to produce a new probe regressor,

            #. Applying an offset to the recenter the peak of the delay distribution of all voxels to zero, which should make datasets easier to compare.

    #. After the three passes are complete, rapidtide will then use a GLM filter to remove a voxel specific lagged copy of the final probe regressor from the data - this denoised data will be in the file ``outputname_desc-lfofilterCleaned_bold.nii.gz``.  There will also a number of maps output with the prefix ``outputname_`` of delay, correlation strength and so on.  See the BIDS Output table above for specifics.

Please note that rapidtide plays happily with AROMA, so you don't need to do anything special to
process data that's been run through AROMA.  While FIX and AROMA both use spatiotemporal
analysis of independent components to determine what components to remove, AROMA only targets
ICs related to motion, which are quite distinct from the sLFO signal, so they don't interfere
with each other.  In contrast, FIX targets components that are "bad", for multiple definitions
of the term, which includes some purely hemodynamic components near the back of the brain.
As a result, FIX denoising impedes the operation of rapidtide.  See below.

Removing low frequency physiological noise from fMRI data that has been processed with FIX
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
There is a special case if you are working on HCP data, which has both minimally processed and a fully processed
(including FIX denoising) data files.  FIX denoising is a good thing, but it tends to distort the sLFO signals that
rapidtide is looking for, so the selection and refinement of the sLFO can wander off into the thicket if applied to
FIX processed data.  So ideally, you would run rapidtide, and THEN FIX.  However, since reprocessing the HCP data
is kind of a pain, there's a hack that capitalizes on the fact that all of these operations are linear.  You run
rapidtide on the minimmally processed data, to accurately assess the sLFO regressor and time delays in each voxel,
but you apply the final GLM to the FIX processed data, to remove the data that has the other denoising already done.
This works very well!  To do this, you use the ``--glmsourcefile FILE`` option to specify the file you want to
denoise.  The ``outputname_desc-lfofilterCleaned_bold.nii.gz`` file is the FIX file, with rapidtide denoising applied.

    ::

        rapidtide \
            minimallyprocessedinputfmrifile \
            outputname \
            --denoising \
            --glmsourcefile FIXprocessedfile


Mapping long time delays in response to a gas challenge experiment:
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Processing this sort of data requires a very different set of options from the previous case.
Instead of the distribution of delays you expect in healthy controls
(a slightly skewed, somewhat normal distribution with a tail on the positive side,
ranging from about -5 to 5 seconds),
in this case, the maximum delay can be extremely long
(100-120 seconds is not uncommon in stroke, moyamoya disease, and atherosclerosis).
To do this, you need to radically change what options you use, not just the delay range,
but a number of other options having to do with refinement and statistical measures.

For this type of analysis, a good place to start is the following:

    ::

        rapidtide \
            inputfmrifile \
            outputname \
            --numnull 0 \
            --searchrange -10 140 \
            --filterfreqs 0.0 0.01 \
            --ampthresh 0.2 \
            --noglm \
            --nofitfilt

The first option (``--numnull 0``), shuts off the calculation of the null correlation distribution.  This is used to
determine the significance threshold, but the method currently implemented in rapidtide is a bit simplistic - it
assumes that all the time points in the data are exchangeable.  This is certainly true for resting state data (see
above), but it is very much NOT true for block paradigm gas challenges.  To properly analyze those, I need to
consider what time points are 'equivalent', and up to now, I don't, so setting the number of iterations in the
Monte Carlo analysis to zero omits this step.

The second option (``--searchrange -10 140``) is fairly obvious - this extends the detectable delay range out
to 140 seconds.  Note that this is somewhat larger than the maximum delays we frequently see, but to find the
correlation peak with maximum precision, you need sufficient additional delay values so that the correlation
can come to a peak and then come down enough that you can properly fit it.  Obviously adjust this as needed
for your experiment, to fit the particulars of your gas challenge waveform and/or expected pathology.

Setting ``--filterfreqs 0.0 0.01`` is VERY important.  By default, rapidtide assumes you are looking at
endogenous low frequency oscillations, which typically between 0.009 and 0.15 Hz.  However, gas challenge
paradigms are usually MUCH lower frequency (90 seconds off, 90 seconds on corresponds to 1/180s = ~0.006Hz).
So if you use the default frequency settings, you will completely filter out your stimulus, and presumably,
your response.  If you are processing one of these experiments and get no results whatsoever, this is almost
certainly the problem.

The ``--noglm`` option disables data filtering.  If you are using rapidtide to estimate and remove low frequency
noise from resting state or task fMRI data, the last step is to use a glm filter to remove this circulatory signal,
leaving "pure" neuronal signal, which you'll use in further analyses.  That's not relevant here - the signal you'd
be removing is the one you care about. So this option skips that step to save time and disk space.

``--nofitfilt`` skips a step after peak estimation.  Estimating the delay and correlation amplitude in each voxel
is a two step process. First you make a quick estimate (where is the maximum point of the correlation function,
and what is its amplitude?), then you refine it by fitting a Gaussian function to the peak to improve the
estimate.  If this step fails, which it can if the peak is too close to the end of the lag range, or
strangely shaped, the default behavior is to mark the point as bad and zero out the parameters for the
voxel.  The nofitfilt option means that if the fit fails, output the initial estimates rather than all
zeros.   This means that you get some information, even if it's not fully refined.  In my experience it
does tend to make the maps for the gas challenge experiments a lot cleaner to use this option since the
correlation function is pretty well behaved.


CVR mapping:
""""""""""""
This is a slightly different twist on interpreting the strength of the lagged correlation,
validated in :footcite:t:`donahue2016time`.
In this case, you supply an input regressor that corresponds to a measured, calibrated CO2 quantity
(for example, etCO2 in mmHg).
Rapidtide then does a modified analysis -
it still uses the cross-correlation to find when the input regressor is maximally aligned with
the variance in the voxel signal,
but instead of only returning a correlation strength,
it calculates the percentage BOLD change in each voxel in units of the input regressor (e.g. %BOLD/mmHg),
which is the standard in CVR analysis.

    ::

        rapidtide \
            inputfmrifile \
            outputname \
            --regressor regressorfile \
            --CVR

You invoke this with the ``--CVR`` option.  This is a macro that does a lot of things: I disabled refinement, set
``--passes 1``, set ``--filterfreqs 0.0 0.01`` (for the reasons described above regarding gas challenge experiments),
set ``--searchrange -5 20``,
hijacked the GLM filtering routine, and messed with some normalizations.  If you want to refine your regressor
estimate, or filter the sLFO signal out of your data, you need to do a separate analysis.

You also need to supply the regressor using ``--regressor regressorfile``.  If regressorfile is a bids
tsv/json pair, this will have the sample rate and offset specified.  If the regressor file has sample
rate other than the fMRI TR, or a non-zero offset relative to the fMRI data, you will also need to specify
these parameters using ``--regressorfreq FREQ`` or ``--regressortstep TSTEP`` and/or ``--regressorstart START``.


Denoising NIRS data:
""""""""""""""""""""
Fun fact - when we started this whole research effort, I was originally planning to denoise NIRS data, not fMRI data.  But one
thing led to another, and the NIRS got derailed for the fMRI effort.  Now that we have some time to catch our breaths,
and more importantly, we have access to some much higher quality NIRS data, this moved back to the front burner.
The majority of the work was already done, I just needed to account for a few qualities that make NIRS data different from fMRI data:

* NIRS data is not generally stored in NIFTI files.  While there is one now (SNIRF), at the time I started doing this, there was no standard NIRS file format.  In the absence of one, you could do worse than a multicolumn text file, with one column per data channel.  That's what I did here - if the file has a '.txt' extension rather than '.nii.', '.nii.gz', or no extension, it will assume all I/O should be done on multicolumn text files.  However, I'm a firm believer in SNIRF, and will add support for it one of these days.
* NIRS data is often zero mean.  This turned out to mess with a lot of my assumptions about which voxels have significant data, and mask construction.  This has led to some new options for specifying mask thresholds and data averaging.
* NIRS data is in some sense "calibrated" as relative micromolar changes in oxy-, deoxy-, and total hemoglobin concentration, so mean and/or variance normalizing the timecourses may not be right thing to do.  I've added in some new options to mess with normalizations.


References
^^^^^^^^^^

.. footbibliography::
