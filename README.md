Rapidtide
=======================
Rapidtide is a suite of python programs used to perform rapid time delay 
analysis on functional imaging data to find time lagged  correlations 
between the voxelwise time series and other time series.

HTML documentation is here: http://rapidtide.readthedocs.io/en/latest/

NOTE
====
This is an evolving code base.  I'm constantly tinkering with it.  That said,
now that I'm releasing this to the world, I'm being somewhat more responsible
about locking down stable release points.  In between releases, however, I'll
be messing with things. **It's very possible
I could break something while doing this, so check back for status updates
if you download the code in between releases**.  I've finally become a little
more modern and started adding
automated testing, so as time goes by hopefully the "in between" releases will
be somewhat more reliable.  Check back often for exciting new features and bug
fixes!

**UPDATE 9/2/16:  I have worked out the outstanding issues that arose when I added
Python 3 compatiblity (with the help of unit tests - who knew?).  So things seem to be
working ok.**

Why do I want to know about time lagged correlations?
=======================
This comes out of work by our group (The Opto-Magnetic group at McLean
Hospital - http://www.nirs-fmri.net) looking at the correlations between
neuroimaging data (fMRI) and NIRS data recorded simultaneously, either
in the brain or the periphery.  We found that a large fraction of the
"noise" we found at low frequency in fMRI data was due to real, random[*]
fluctuations of blood oxygenation and volume (both of which affect the
intensity of BOLD fMRI images) in the blood passing through the brain. 
More interestingly, because these characteristics of blood move with the
blood itself, this gives you a way to determine blood arrival time at
any location in the brain. This is interesting in and of itself, but
also, this gives you a method for optimally modelling (and removing) in
band physiological noise from fMRI data (see references below).

After working with this for several years we've also found that you don't
need to used simultaneous NIRS to find this blood borne signal - you can
get it from blood rich BOLD voxels for example in the superior sagittal
sinus, or bootstrap it out of the global mean signal in the BOLD data.
You can also track exogenously applied waveforms, such as hypercarbic
and/or hyperoxic gas challenges to really boost your signal to noise.  So
there are lots of times when you might want to do this type of correlation
analysis.

[*] "random" in this context means "determined by something we don't have 
any information about" - maybe EtCO2 variation, or sympathetic nervous 
system activity - so not really random.

Yes, but correlation analysis is easy - why use this package?
=======================
The simple answer is "correlation analysis is easy, but using a prewritten
package that handles file I/O, filtering, resampling, windowing, and the
rest for you is even easier".  A slightly more complex answer is that
while correlation analysis is pretty easy to do, it's hard to do right;
there are lots and lots of ways to do it incorrectly.  Fortunately, I've
made most of those mistakes for you over the last 6 years, and corrected
my code accordingly.  So rather than repeat my boring mistakes, why not 
make new, interesting mistakes?  Explore your own, unique chunk of
wrongspace...

Ok, I'm sold.  What's in here?
=======================
- **rapidtide2** - This is the heart of the package - this is the
	workhorse program that will determine the time lagged
	correlations between all the voxels in a NIFTI file and a
	temporal "probe" regressor (which can come from a number of places,
	including the data itself) - it rapidly determines time delays...
        There are a truly bewildering array of options, and just about 
        everything can be adjusted, however I've tried to pick a good 
        set of default options for the most basic processing to get you 
        going.  At a minimum, it requires a 4D NIFTI file as input,
        and a root name for all of the output files.  It generates a 
        number of 3D NIFTI file maps of various parameters (lag time
        of maximum correlation, maximum correlation value, a mask of which
        voxels have valid fits, etc.) and some text files with useful 
        information (significance threshholds, processing timing 
        information, a list of values of configurable options).

- **showxcorr** - Like rapidtide2, but for single time courses.  Takes two 
        text files as input, calculates and displays the time lagged cross
        correlation between them, fits the maximum time lag, and estimates
        the significance of the correlation.  It has a range of filtering,
        windowing, and correlation options.

- **rapidtide2std** - This is a utility for registering rapidtide output maps
	to standard coordinates.  It's usually much faster to run rapidtide
        in native space then transform afterwards to MNI152 space.  NB: this 
        will only work if you have a working FSL installation.

- **showtc** - A very simple command line utility that takes a text file
        and plots the data in it in a matplotlib window.  That's it.  A
        good tool for quickly seeing what's in a file.  Has some options
        to make the plot prettier.

- **showhist** - Another simple command line utility that displays the 
        histograms generated by rapidtide2.

- **resamp1tc** - takes an input text file at some sample rate and outputs
        a text file resampled to the specified sample rate.

- **resamplenifti** - takes an input nifti file at some TR and outputs
        a nifti file resampled to the specified TR.

- **tidepool** - This is a very experimental tool for displaying all of the
	various maps generated by rapidtide2 in one place, overlayed on
	an anatomic image.  This makes it a bit easier to see how all the
	maps are related to one another.  To use it, launch tidepool from
	the command line, and then select a lag time map - tidpool will 
	figure out the root name and pull in all of the other associated
	maps.  Works in native or standard space.

- **tide_funcs.py** - This is the library of the various helper routines
	that are used by pretty much every program in here for
	correlation, resampling, filtering, normalization, significance
	estimation, file I/O, etc.

- **OrthoImageItem.py** - This is a class that implements the orthographic
        projection module that is used to display all of the maps in tidepool.
        It uses pyqtgraph to do all the heavy lifting.  None of the built-ins
        in pyqtgraph did exactly what I wanted in terms of allowing 3D selection,
        overlays and the like, so I cobbled this together.  It may be generally
        useful to anybody wanting to display functional data.

References
=======================

Links to PDFs of all papers mentioned here can be found on the OMG website: 
http://www.nirs-fmri.net/publications

Multimodal Cerebral circulation Imaging
-----------------------

1) Tong Y, Frederick BD. (2010) Time lag dependent multimodal processing
	of concurrent fMRI and near-infrared spectroscopy (NIRS) data
	suggests a global circulatory origin for low-frequency
	oscillation signals in human brain. Neuroimage, 53(2), 553-64.

2) Tong Y, Hocke L, Frederick BD. (2011) Isolating the sources of
	widespread physiological fluctuations in fNIRS signals. J Biomed
	Opt. 16(10), 106005.

3) Tong Y, Bergethon PR, Frederick BD. (2011c) An improved method for
	mapping cerebrovascular reserve using concurrent fMRI and
	near-infrared spectroscopy with Regressor Interpolation at
	Progressive Time Delays (RIPTiDe). Neuroimage, 56(4), 2047-2057.

4) Tong Y, Frederick BD. (2012) Concurrent fNIRS and fMRI processing
	allows independent visualization of the propagation of pressure
	waves and bulk blood flow in the cerebral vasculature.
	Neuroimage, Jul 16;61(4): 1419-27.
	
5) Tong Y, Hocke LM, Licata SC, Frederick BD. (2012) Low frequency
	oscillations measured in the periphery with near infrared
	spectroscopy (NIRS) are strongly correlated with blood oxygen
	level-dependent functional magnetic resonance imaging (BOLD fMRI)
	signals. J Biomed Opt, 2012;17(10):106004. doi: 
	10.1117/1.JBO.17.10.106004. PubMed PMID: 23224003; PMCID: 3461094.
	
6) Tong Y, Hocke LM, Frederick BD. (2013) Short repetition time
	multiband EPI with simultaneous pulse recording allows dynamic
	imaging of the cardiac pulsation signal. Magn Reson Med
	2014;72(5):1268-76. Epub Nov 22, 2013. doi: 10.1002/mrm.25041.
	PubMed PMID: 24272768.
	
7) Tong Y, Frederick B. (2014) Studying the Spatial Distribution of
	Physiological Effects on BOLD Signals using Ultrafast fMRI. Front
	Hum Neurosci 2014;5(196). doi: doi: 10.3389/fnhum.2014.00196.
	
8) Tong Y, Frederick B. (2014) Tracking cerebral blood flow in BOLD fMRI
	using recursively generated regressors. Hum Brain Mapp.
	2014;35(11):5471-85. doi: 10.1002/hbm.22564. PubMed PMID:
	24954380; PMCID: PMC4206590. 
	
9) Donahue M, Strother M, Lindsey K, Hocke L, Tong Y, Frederick B.
	(2015) Time delay processing of hypercapnic fMRI allows
	quantitative parameterization of cerebrovascular reactivity and
	blood flow delays. Journal of Cerebral Blood Flow & Metabolism.
	2015. PubMed PMID: 26661192. Epub October 19, 2015. doi: 10.1177/
	0271678X15608643.

10) Hocke L, Cayetano K, Tong Y, Frederick B. (2015) An optimized
	multimodal fMRI/NIRS probe for ultra-high resolution mapping.
	Neurophotonics. 2(4), 045004 (Oct-Dec 2015). doi:
	10.1117/1.NPh.2.4.0450004.

11) Tong Y, Hocke LM, Fan X, Janes AC, Frederick B (2015). Can apparent
	resting state connectivity arise from systemic fluctuations?
	Frontiers in human neuroscience. 2015;9. doi:
	10.3389/fnhum.2015.00285.

12) Tong Y, Lindsey KP, Hocke LM, Vitaliano G, Mintzopoulos D, Frederick
	B. (2016) Perfusion information extracted from resting state
	functional magnetic resonance imaging. Journal of cerebral blood
	flow and metabolism : official journal of the International
	Society of Cerebral Blood Flow and Metabolism. 2016. doi:
	10.1177/0271678X16631755. PubMed PMID: 26873885.

Physiological noise identification and removal using time delay methods
-----------------------

1) Tong Y, Lindsey KP, Frederick BD. (2011b) Partitioning of
	physiological noise signals in the brain with concurrent
	near-infrared spectroscopy (NIRS) and fMRI. J Cereb Blood Flow
	Metab. 31(12), 2352-62. 
	
2) Frederick BD, Nickerson LD, Tong Y. (2012) Physiological denoising of
	BOLD fMRI data using Regressor Interpolation at Progressive Time
	Delays (RIPTiDe) processing of concurrent fMRI and near-infrared
	spectroscopy (NIRS). Neuroimage, Apr 15;60(3): 1419-27.
	
3) Tong Y, Hocke LM, Nickerson LD, Licata SC, Lindsey KP, Frederick BB
	(2013) Evaluating the effects of systemic low frequency
	oscillations measured in the periphery on the independent
	component analysis results of resting state networks. NeuroImage.
	2013;76C:202-15. doi: 10.1016/j.neuroimage.2013.03.019. PubMed
	PMID: 23523805; PMCID: PMC3652630.
	
4) Hocke LM, Tong Y, Lindsey KP, Frederick BB (2016). Comparison of
	peripheral near-infrared spectroscopy low-frequency oscillations
	to other denoising methods in resting state functional MRI with
	ultrahigh temporal resolution. Magnetic resonance in medicine :
	official journal of the Society of Magnetic Resonance in Medicine
	/ Society of Magnetic Resonance in Medicine. 2016. doi:
	10.1002/mrm.26038. PubMed PMID: 26854203.
	
5) Erdoğan S, Tong Y, Hocke L, Lindsey K, Frederick B (2016). Correcting
	resting state fMRI-BOLD signals for blood arrival time enhances
	functional connectivity analysis. Front. Hum. Neurosci., 28 June 2016
	| http://dx.doi.org/10.3389/fnhum.2016.00311
