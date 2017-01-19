.. rapidtide documentation master file, created by
   sphinx-quickstart on Thu Jun 16 15:27:19 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

The pretty html version of this file can be found here: http://rapidtide.readthedocs.io/en/latest/

Rapidtide
==========
Rapidtide is a suite of python programs used to perform time delay analysis on functional imaging data to find time lagged correlations between the voxelwise time series and other time series. 

.. toctree::
   :maxdepth: 2

NOTE
====
This is an evolving code base.  I'm constantly tinkering with it.  That said, now that I'm releasing this to the world, I'm being somewhat more responsible about locking down stable release points.  In between releases, however, I'll be messing with things. **It's very possible I could break something while doing this, so check back for status updates if you download the code in between releases**.  I've finally become a little more modern and started adding automated testing, so as time goes by hopefully the "in between" releases will be somewhat more reliable.  Check back often for exciting new features and bug fixes!

Introduction
============
Why do I want to know about time lagged correlations?
-----------------------------------------------------
This comes out of work by our group (The Opto-Magnetic group at McLean Hospital - http://www.nirs-fmri.net) looking at the correlations between neuroimaging data (fMRI) and NIRS data recorded simultaneously, either in the brain or the periphery.  We found that a large fraction of the "noise" we found at low frequency in fMRI data was due to real, random[*] fluctuations of blood oxygenation and volume (both of which affect the intensity of BOLD fMRI images) in the blood passing through the brain. More interestingly, because these characteristics of blood move with the blood itself, this gives you a way to determine blood arrival time at any location in the brain. This is interesting in and of itself, but also, this gives you a method for optimally modelling (and removing) in band physiological noise from fMRI data (see references below).
 
After working with this for several years we've also found that you don't need to used simultaneous NIRS to find this blood borne signal - you can get it from blood rich BOLD voxels for example in the superior sagittal sinus, or bootstrap it out of the global mean signal in the BOLD data. You can also track exogenously applied waveforms, such as hypercarbic and/or hyperoxic gas challenges to really boost your signal to noise.  So there are lots of times when you might want to do this type of correlation analysis.  This package provides the tools to make that easier.
      
As an aside, some of these tools are just generally useful for looking at correlations between timecourses from other sources – for example doing PPI, or even some seed based analyses.
      
Why are you releasing your code?
--------------------------------
For a number of reasons.
    #.    I want people to use it!  I think if it were easier for people to do time delay analysis, they’d be more likely to do it.  I don’t have enough time or people in my group to do every experiment that I think would be interesting, so I’m hoping other people will, so I can read their papers and learn interesting things.
      
    #.    It’s the right way to do science – I can say lots of things, but if nobody can replicate my results, nobody will believe it (we’ve gotten that a lot, because some of the implications of what we’ve seen in resting state data can be a little uncomfortable).  We’ve reached a stage in fMRI where getting from data to results involves a huge amount of processing, so part of confirming results involves being able to see how the data were processed. If you had to do everything from scratch, you’d never even try to confirm anybody’s results.
      
    #.    In any complicated processing scheme, it’s quite possible (or in my case, likely) to make dumb mistakes, either coding errors or conceptual errors, and I almost certainly have made some (although hopefully the worst ones have been dealt with at this point).  More users and more eyes on the code make it more likely that they will be found.  As much as I’m queasy about somebody potentially finding a mistake in my code, I’d rather that they did so, so I can fix it[‡].
      
    #.    It’s giving back to the community.  I benefit from the generosity of a lot of authors who have made the open source tools I use for work and play, so I figure I can pony up too.
      
A note on coding quality and style:
-----------------------------------
This code has been in active development since June of 2012.  This has two implications.  The first is that it has been tuned and refined quite a bit over the years, with a lot of optimizations and bug fixes - most of the core routines have been tested fairly extensively to get rid of the stupidest bugs.  I find new bugs all the time, but most of the showstoppers seem to be gone.  The second result is that the coding style is all over the place.  When I started writing this, I had just moved over from C, and it was basically a mental port of how I would write it in C, and was extremely unpythonic (I’ve been told by a somewhat reliable source that looking over some of my early python efforts “made his eyes bleed”).  Over the years, as I've gone back and added functions, I periodically get embarassed and upgrade things to a somewhat more modern coding style.  I even put in some classes - that's what the cool kids do, right?  But the pace of that effort has to be balanced with the fact that when I make major architectural changes, I tend to break things.  So be patient with me, and keep in mind that you get what you pay for, and this cost you nothing!  Function before form.

Python version compatibility: 
-----------------------------
This code has been extensively tested in python 2.7.  I dragged my feet somewhat making it python 3 compatible, since a number of the libraries I needed took a long time to get ported to python 3, and I honestly saw no advantage to doing it.  I have since decided that I’m going to have to do it eventually, so why not now?  As far as I know, the code all works fine in python 3.5 now - I’ve switched over to that on my development machine, and have not hit any version related issues in a while now, and according to PyCharm’s code inspection, there are no incompatible constructions.  However that’s no guarantee that there isn’t a problem in some option I haven’t bothered to test yet, so be vigilant, and please let me know if there is some issue with python 3 that I haven’t caught (or any bugs, really).
      
How do I cite this?
-------------------
Good question!  I think the following will work, although I should probably get a DOI for this.
	Frederick, B, rapidtide [Computer Software] (2016).  Retrieved from https://github.com/bbfrederick/rapidtide.

What’s included in this package?
================================
I’ve included a number of tools to get you going – I’ll add in a number of other utilities as I get them closer to the point that I can release them without people laughing at my code.  For the time being, I’m including the following:

rapidtide2
----------

Description:
^^^^^^^^^^^^

	The central program in this package is rapidtide2.  This is the program that quantifies the correlation strength and time delay of pervasive signals in a BOLD fMRI dataset.

	At its core, rapidtide2 is simply performing a full crosscorrelation between a "probe" timecourse and every voxel in an fMRI dataset (by “full” I mean over a range of time lags that account for any delays between the signals, rather than only at zero lag, as in a Pearson correlation).  As with many things, however, the devil is in the details, and so rapidtide2 provides a number of features which make it pretty good at this particular task.  A few highlights:

	#. There are lots of ways to do something even as simple as a cross-correlation in a nonoptimal way (not windowing, improper normalization, doing it in the time rather than frequency domain, etc.).  I'm pretty sure what rapidtide2 does by default is, if not the best way, at least a very good and very fast way.

	#. rapidtide2 has been optimized and profiled to speed it up quite a bit; it has an optional dependency on numba – if it’s installed, some of the most heavily used routines will speed up significantly due to judicious use of @jit.

	#. The sample rate of your probe regressor and the fMRI data do not have to match - rapidtide2 resamples the probe regressor to an integral multiple of the fMRI data rate automatically.

	#. The probe and data can be temporally prefiltered to the LFO, respiratory, or cardiac frequency band with a command line switch, or you can specify any low, high, or bandpass range you want.

	#. The data can be spatially smoothed at runtime (so you don't have to keep smoothed versions of big datasets around).  This is quite fast, so no reason not to do it this way.

	#. rapidtide2 can generate a probe regressor from the global mean of the data itself - no externally recorded timecourse is required.  Optionally you can input both a mask of regions that you want to be included in the mean, and the voxels that you want excluded from the mean (there are situations when you might want to do one or the other or both).
	
	#. Determining the significance threshold for filtered correlations where the optimal delay has been selected is nontrivial; using the conventional formulae for the significance of a correlation leads to wildly inflated p values.  rapidtide2 estimates the spurious correlation threshold by calculating the distribution of null correlation values obtained with a shuffling  procedure at the beginning of each run (the default is to use 10000 shuffled correlations), and uses this value to mask the correlation maps it calculates.  As of version 0.1.2 it will also handle two-tailed significance, which you need when using bipolar mode.

	#. rapidtide2 can do an iterative refinement of the probe regressor by aligning the voxel timecourses in time and regenerating the test regressor.

	#. rapidtide2 fits the peak of the correlation function, so you can make fine grained distinctions between close lag times. The resolution of the time lag discrimination is set by the length of the timecourse, not the timestep – this is a feature of correlations, not rapidtide2.

	#. Once the time delay in each voxel has been found, rapidtide2 outputs a 4D file of delayed probe regressors for using as voxel specific confound regressors or to estimate the strength of the probe regressor in each voxel.  This regression is performed by default, but these outputs let you do it yourself if you are so inclined.

	#. I've put a lot of effort into making the outputs as informative as possible - lots of useful maps, histograms, timecourses, etc.

	#. There are a lot of tuning parameters you can mess with if you feel the need.  I've tried to make intelligent defaults so things will work well out of the box, but you have the ability to set most of the interesting parameters yourself.
     
Inputs:
^^^^^^^
	At a minimum, rapidtide2 needs a Nifti file to work on (space by time), which is generally thought to be a BOLD fMRI data file.  This can be Nifti1 or Nifti2; I can currently read (probably) but not write Cifti files, so if you want to use grayordinate files you need to convert them to nifti in workbench, run rapidtide2, then convert back. As soon as nibabel finishes their Cifti support, I'll add that.

	The file needs one time dimension and at least one spatial dimension.  Internally, the array is flattened to a time by voxel array for simplicity.

	The file you input here should be the result of any preprocessing you intend to do.  The expectation is that rapidtide will be run as the last preprocessing step before resting state or task based analysis.  So any slice time correction, motion correction, spike removal, etc. should already have been done.  If you use FSL, this means that if you've run preprocessing, you would use the filtered_func_data.nii.gz file as input.  Temporal and spatial filtering are the two (partial) exceptions here.  Generally rapidtide is most useful for looking at low frequency oscillations, so when you run it, you usually use the "-L" option or some other to limit the analysis to the detection and removal of low frequency systemic physiological oscillations.  So rapidtide will generally apply it's own temporal filtering on top of whatever you do in preprocessing.  Also, you have the option of doing spatial smoothing in rapidtide to boost the SNR of the analysis; the hemodynamic signals rapidtide looks for are often very smooth, so you rather than smooth your functional data excessively, you can do it within rapidtide so that only the hemodynamic data is smoothed at that level.
     
Outputs:
^^^^^^^^
	Outputs are space or space by time Nifti files (depending on the file), and some text files containing textual information, histograms, or numbers.  Output spatial dimensions and file type match the input dimensions and file type (Nifti1 in, Nifti1 out).  Depending on the file type of map, there can be no time dimension, a time dimension that matches the input file, or something else, such as a time lag dimension for a correlation map.
    
Usage:
^^^^^^

	::

		usage: rapidtide2 fmrifilename outputname 
		[-r LAGMIN,LAGMAX] [-s SIGMALIMIT] [-a] [--nowindow] [--phat] [--liang] [--eckart] [-f GAUSSSIGMA] [-O oversampfac] [-t TRvalue] [-d] [-b] [-V] [-L] [-R] [-C] [-F LOWERFREQ,UPPERFREQ[,LOWERSTOP,UPPERSTOP]] [-o OFFSETTIME] [-T] [-p] [-P] [-A ORDER] [-B] [-h HISTLEN] [-i INTERPTYPE] [-I] [-Z DELAYTIME] [-N NREPS] --numskip=SKIP[--refineweighting=TYPE] [--refinepasses=PASSES] [--excludemask=MASK] [--includemask=MASK] [--lagminthresh=MIN] [--lagmaxthresh=MAX] [--ampthresh=AMP] [--corrmaskthresh=PCT][--sigmathresh=SIGMA] [--refineoffset] [--pca] [--ica] [--nodispersioncalc] [--refineupperlag] [--refinelowerlag][--tmask=MASKFILE] [--limitoutput] [--timerange=START,END] [--skipsighistfit] [--accheck] [--acfix][--numskip=SKIP] [--slicetimes=FILE] [--glmsourcefile=FILE] [--regressorfreq=FREQ] [--regressortstep=TSTEP][--regressor=FILENAME] [--regressorstart=STARTTIME] [--usesp] [--maxfittype=FITTYPE

		Required arguments:
			fmrifilename             - The BOLD fmri file
			outputname               - The root name for the output files

		Preprocessing options:
			-t TRvalue               - Override the TR in the fMRI file with the value 
									   TRvalue
			-a                       - Disable antialiasing filter
			--nodetrend              - Disable linear trend removal
			-I                       - Invert the sign of the regressor before processing
			-i                       - Use specified interpolation type (options are 'cubic',
									   'quadratic', and 'univariate (default)')
			-o                       - Apply an offset OFFSETTIME to the lag regressors
			-b                       - Use butterworth filter for band splitting instead of
									   trapezoidal FFT filter
			-F                       - Filter data and regressors from LOWERFREQ to UPPERFREQ.
									   LOWERSTOP and UPPERSTOP can be specified, or will be
									   calculated automatically
			-V                       - Filter data and regressors to VLF band
			-L                       - Filter data and regressors to LFO band
			-R                       - Filter data and regressors to respiratory band
			-C                       - Filter data and regressors to cardiac band
			-N                       - Estimate significance threshold by running NREPS null 
									   correlations (default is 10000, set to 0 to disable)
			--skipsighistfit         - Do not fit significance histogram with a Johnson SB function
			--nowindow               - Disable precorrelation windowing
			-f GAUSSSIGMA            - Spatially filter fMRI data prior to analysis using 
									   GAUSSSIGMA in mm
			-M                       - Generate a global mean regressor and use that as the 
									   reference regressor
			-m                       - Mean scale regressors during global mean estimation
			--slicetimes=FILE        - Apply offset times from FILE to each slice in the dataset
			--numskip=SKIP           - SKIP tr's were previously deleted during preprocessing
									   (default is 0)

		Correlation options:
			-O OVERSAMPFAC           - Oversample the fMRI data by the following integral 
									   factor (default is 2)
			--regressor=FILENAME     - Read probe regressor from file FILENAME (if none 
									   specified, generate and use global regressor)
			--regressorfreq=FREQ     - Probe regressor in file has sample frequency FREQ 
									   (default is 1/tr) NB: --regressorfreq and --regressortstep
									   are two ways to specify the same thing
			--regressortstep=TSTEP   - Probe regressor in file has sample time step TSTEP 
									   (default is tr) NB: --regressorfreq and --regressortstep
									   are two ways to specify the same thing
			--regressorstart=START   - The time delay in seconds into the regressor file, corresponding
									   in the first TR of the fmri file (default is 0.0)
			--phat                   - Use generalized cross-correlation with phase alignment 
									   transform (PHAT) instead of correlation
			--liang                  - Use generalized cross-correlation with Liang weighting function
									   (Liang, et al, doi:10.1109/IMCCC.2015.283)
			--eckart                 - Use generalized cross-correlation with Eckart weighting function
			--corrmaskthresh=PCT     - Do correlations in voxels where the mean exceeeds this 
									   percentage of the robust max (default is 1.0)
			--accheck                - Check for periodic components that corrupt the autocorrelation

		Correlation fitting options:
			-Z DELAYTIME             - Don't fit the delay time - set it to DELAYTIME seconds 
									   for all voxels
			-r LAGMIN,LAGMAX         - Limit fit to a range of lags from LAGMIN to LAGMAX
			-s SIGMALIMIT            - Reject lag fits with linewidth wider than SIGMALIMIT
			-B                       - Bipolar mode - match peak correlation ignoring sign
			--nofitfilt              - Do not zero out peak fit values if fit fails
			--maxfittype=FITTYPE     - Method for fitting the correlation peak (default is 'gauss'). 
									   'quad' uses a quadratic fit.  Faster but not as well tested

		Regressor refinement options:
			--refineweighting=TYPE   - Apply TYPE weighting to each timecourse prior 
									   to refinement (valid weightings are 'None', 
									   'R', 'R2' (default)
			--refinepasses=PASSES    - Set the number of refinement passes to PASSES 
									   (default is 1)
			--includemask=MASK       - Only use voxels in NAME for global regressor 
									   generation and regressor refinement
			--excludemask=MASK       - Do not use voxels in NAME for global regressor 
									   generation and regressor refinement
			--lagminthresh=MIN       - For refinement, exclude voxels with delays less 
									   than MIN (default is 1.5s)
			--lagmaxthresh=MAX       - For refinement, exclude voxels with delays greater 
									   than MAX (default is 1000s)
			--ampthresh=AMP          - For refinement, exclude voxels with correlation 
									   coefficients less than AMP (default is 0.3)
			--sigmathresh=SIGMA      - For refinement, exclude voxels with widths greater 
									   than SIGMA (default is 100s)
			--refineoffset           - Adjust offset time during refinement to bring peak 
									   delay to zero
			--refineupperlag         - Only use positive lags for regressor refinement
			--refinelowerlag         - Only use negative lags for regressor refinement
			--pca                    - Use pca to derive refined regressor (default is 
									   averaging)
			--ica                    - Use ica to derive refined regressor (default is 
									   averaging)
			--nodispersioncalc       - Disable dispersion calculation during refinement

		Output options:
			--limitoutput            - Don't save some of the large and rarely used files
			-T                       - Save a table of lagtimes used
			-h HISTLEN               - Change the histogram length to HISTLEN (default is
									   100)
			--timerange=START,END    - Limit analysis to data between timepoints START 
									   and END in the fmri file
			--glmsourcefile=FILE     - Regress delayed regressors out of FILE instead of the 
									   initial fmri file used to estimate delays
			--noglm                  - Turn off GLM filtering to remove delayed regressor 
									   from each voxel (disables output of fitNorm)

		Miscellaneous options:
			--usesp                  - Use single precision for internal calculations (may
									   be useful when RAM is limited)
			-c                       - Data file is a converted CIFTI
			-S                       - Simulate a run - just report command line options
			-d                       - Display plots of interesting timecourses
			--nonumba                - Disable jit compilation with numba
			--memprofile             - Enable memory profiling for debugging - warning:
									   this slows things down a lot.

		Experimental options (not fully tested, may not work):
			--acfix                  - Remove periodic components that corrupt the autocorrelation
									   (enables --accheck).  Experimental.
			--tmask=MASKFILE         - Only correlate during epochs specified in 
									   MASKFILE (NB: each line of MASKFILE contains the 
									   time and duration of an epoch to include
			-p                       - Prewhiten and refit data
			-P                       - Save prewhitened data (turns prewhitening on)
			-A, --AR                 - Set AR model order to ORDER (default is 1)


        
	These options are somewhat self-explanatory.  I will be expanding this section of the manual going forward, but I want to put something here to get this out here.

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

		showxcorr - calculate and display crosscorrelation between two timeseries

		usage: showxcorr timecourse1 timecourse2 samplerate 
			[-l LABEL] [-s STARTTIME] [-D DURATION] [-d] [-F LOWERFREQ,UPPERFREQ[,LOWERSTOP,UPPERSTOP]] [-V] [-L] [-R] [-C] [-t] [-w] [-f] [-g] [-z FILENAME] [-N TRIALS]
	
		required arguments:
			timecoursefile1     - text file containing a timeseries
			timecoursefile2     - text file containing a timeseries
			samplerate          - the sample rate of the timecourses, in Hz

		optional arguments:
			-t 	     - detrend the data
			-w 	     - prewindow the data
			-g 	     - perform phase alignment transform (phat) rather than 
							standard crosscorrelation
			-l LABEL	     - label for the delay value
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

		usage: showtc texfilename [textfilename]... [--nolegend] [--pspec] [--phase] [--samplerate] [--sampletime]

		required arguments:
			textfilename	- a text file containing whitespace separated timecourses, one timepoint per line

		optional arguments:
			--nolegend               - turn off legend label
			--pspec                  - show the power spectra magnitudes of the input data instead of the timecourses
			--phase                  - show the power spectra phases of the input data instead of the timecourses
			--samplerate             - the sample rate of the input data (default is 1Hz)
			--sampletime             - the sample time (1/samplerate) of the input data (default is 1s)


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


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

