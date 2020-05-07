Rapidtide
=========

What is this package?
---------------------

Rapidtide is a suite of python programs used to perform rapid time delay
analysis on functional imaging data to find time lagged correlations
between the voxelwise time series and other time series, both in the LFO
band (rapditide2) and now in the cardiac band (happy).

Full documentation is at: http://rapidtide.readthedocs.io/en/latest/

|License| |Documentation Status| |Travis-CI| |Coverage| |DOI| |Funded by
NIH|

Why do I want to know about time lagged correlations?
-----------------------------------------------------

This comes out of work by our group (The Opto-Magnetic group at McLean
Hospital - http://www.nirs-fmri.net) looking at the correlations between
neuroimaging data (fMRI) and NIRS data recorded simultaneously, either
in the brain or the periphery. We found that a large fraction of the
“noise” we found at low frequency in fMRI data was due to real,
random[*] fluctuations of blood oxygenation and volume (both of which
affect the intensity of BOLD fMRI images) in the blood passing through
the brain. More interestingly, because these characteristics of blood
move with the blood itself, this gives you a way to determine blood
arrival time at any location in the brain. This is interesting in and of
itself, but also, this gives you a method for optimally modelling (and
removing) in band physiological noise from fMRI data (see references
below).

After working with this for several years we’ve also found that you
don’t need to used simultaneous NIRS to find this blood borne signal -
you can get it from blood rich BOLD voxels for example in the superior
sagittal sinus, or bootstrap it out of the global mean signal in the
BOLD data. You can also track exogenously applied waveforms, such as
hypercarbic and/or hyperoxic gas challenges to really boost your signal
to noise. So there are lots of times when you might want to do this type
of correlation analysis.

As an aside, some of these tools are just generally useful for looking
at correlations between timecourses from other sources – for example
doing PPI, or even some seed based analyses.

[*] “random” in this context means “determined by something we don’t
have any information about” - maybe EtCO2 variation, or sympathetic
nervous system activity - so not really random.

Correlation analysis is easy - why use this package?
----------------------------------------------------

The simple answer is “correlation analysis is easy, but using a
prewritten package that handles file I/O, filtering, resampling,
windowing, and the rest for you is even easier”. A slightly more complex
answer is that while correlation analysis is pretty easy to do, it’s
hard to do right; there are lots and lots of ways to do it incorrectly.
Fortunately, I’ve made most of those mistakes for you over the last 6
years, and corrected my code accordingly. So rather than repeat my
boring mistakes, why not make new, interesting mistakes? Explore your
own, unique chunk of wrongspace…

Happy
=====

More recently, inspired by Henning Voss’ paper on hypersampling of
cardiac signals in fMRI, we developed a method to extract and clean
cardiac waveforms from fMRI data, even when the fMRI TR is far too long
to properly sample cardiac waveforms. This cardiac waveform can then be
used to track the pulsatile cardiac pressure wave through the brain in
somewhat the same way that we track the LFO signals. Among other things,
this allows you to get cardiac waveforms during scans even when either
1) you didn’t use a plethysmograph, or 2) you did, but the recording was
of poor quality, which happens more than you might think.

What does "happy" have to do with any of this?
----------------------------------------------

As to why happy is part of rapidtide, that's partially for practical reasons (the
libraries in rapidtide have an awful lot of code that is reused in happy), and
partially thematically - rapidtide has evolved from a "let's look at low
frequency signals in fMRI data" package to a "let's look at everything in
fMRI data EXCEPT neuronal activation", so happy fits right in...

Why are you releasing your code?
================================

For a number of reasons.

-  I want people to use it! I think if it were easier for people to do
   time delay analysis, they’d be more likely to do it. I don’t have
   enough time or people in my group to do every experiment that I think
   would be interesting, so I’m hoping other people will, so I can read
   their papers and learn interesting things.
-  It’s the right way to do science – I can say lots of things, but if
   nobody can replicate my results, nobody will believe it (we’ve gotten
   that a lot, because some of the implications of what we’ve seen in
   resting state data can be a little uncomfortable). We’ve reached a
   stage in fMRI where getting from data to results involves a huge
   amount of processing, so part of confirming results involves being
   able to see how the data were processed. If you had to do everything
   from scratch, you’d never even try to confirm anybody’s results.
-  In any complicated processing scheme, it’s quite possible (or in my
   case, likely) to make dumb mistakes, either coding errors or
   conceptual errors, and I almost certainly have made some (although
   hopefully the worst ones have been dealt with at this point). More
   users and more eyes on the code make it more likely that they will be
   found. As much as I’m queasy about somebody potentially finding a
   mistake in my code, I’d rather that they did so, so I can fix it[‡].
-  It’s giving back to the community. I benefit from the generosity of a
   lot of authors who have made the open source tools I use for work and
   play, so I figure I can pony up too.

[‡] or better yet, you, empowered user, can fix it, and push back a fix
that benefits everybody…

Stability, etc.
===============
This is an evolving code base. I’m constantly tinkering with it. That
said, now that I’ve sent this off into to the world, I’m being somewhat
more responsible about locking down stable release points. In between
releases, however, I’ll be messing with things, although for the most
part this will be restricted to the “x” version of programs (e.g. the
bleeding edge version of rapidtide2 is rapidtide2x - use the former if
you want stability, the latter if you want the newest and and most
exciting features). **It’s very possible I could break something while
doing this, so check back frequently for status updates if you download the code in
between releases**. I’ve finally become a little more modern and started
adding automated testing, so as time goes by hopefully the “in between”
releases will be somewhat more reliable.  That said, my tests routinely fail, even
when things actually work.  Probably should deal with that. Check back often for exciting
new features and bug fixes!

Python version compatibility
============================
I switched over a while ago to using python 3 as my daily driver, so I know
that everything works there. However, I know that a lot of people can’t
or won’t switch from python 2x, so I’ve made every effort to write code
that works in both, and I have been testing in both.

That said, the writing is on the wall, and since I depend on a number of
packages that have dropped or are dropping 2.x support, I’m going with
the flow and sunsetting 2.x support. This is the last to support 2.x.

Breaking news!  As of version 1.9.0 I'm also releasing the code in a docker
container (fredericklab/rapidtide), which has everything nicely installed in
a fully configured python 3 environment, so there's really no need to do a 2.x
installation anymore

Ok, I’m sold. What’s in here?
=============================

-  **rapidtide2** - This is the heart of the package - this is the
   workhorse program that will determine the time lagged correlations
   between all the voxels in a NIFTI file and a temporal “probe”
   regressor (which can come from a number of places, including the data
   itself) - it rapidly determines time delays… There are a truly
   bewildering array of options, and just about everything can be
   adjusted, however I’ve tried to pick a good set of default options
   for the most basic processing to get you going. At a minimum, it
   requires a 4D NIFTI file as input, and a root name for all of the
   output files. It generates a number of 3D NIFTI file maps of various
   parameters (lag time of maximum correlation, maximum correlation
   value, a mask of which voxels have valid fits, etc.) and some text
   files with useful information (significance threshholds, processing
   timing information, a list of values of configurable options).

-  **rapidtide2x** - This is the most modern and updated version of rapidtide.
   This is where all the development effort is going (new features, reimplementation of core routines,
   interface changes.)  I port any major bugfixes
   back to rapidtide2, but this will become the new rapidtide2 in the 2.0 release.  This 
   will also come with some incompatible calling changes and modifications
   to filter ranges (I'll have a long explanation for why in the 2.0 release notes).
   However, if you are downloading things between releases, this is not guaranteed to be
   that stable...

-  **happy** - This is a companion to rapidtide that focusses on cardiac signals.
   happy does three things - it attempts to determine the cardiac
   waveform over the time course of an fMRI dataset using slice
   selective averaging of fully unprocessed fMRI data. It also 
   cleans up this initial estimate using a deep learning filter to
   infer what the simultaneously recorded plethysmogram would be.
   Finally, it uses either the derived or a supplied plethysmogram
   signal to construct a cardiac pulsation map over a single
   cycle of the cardiac waveform, a la Voss.

-  **showxcorr** - Like rapidtide2, but for single time courses. Takes
   two text files as input, calculates and displays the time lagged
   cross correlation between them, fits the maximum time lag, and
   estimates the significance of the correlation. It has a range of
   filtering, windowing, and correlation options.

-  **showxcorrx** - The bleeding edge version of showxcorr.  This has a lot
   of new options and defaults; it's not really drop in compatible, so I'm
   keeping it separate at the moment to avoid breaking existing workflows.
   With the 2.0 release, this will become showxcorr, and the current showxcorr
   will become showxcorr_legacy.

-  **rapidtide2std** - This is a utility for registering rapidtide
   output maps to standard coordinates. It’s usually much faster to run
   rapidtide in native space then transform afterwards to MNI152 space.
   NB: this will only work if you have a working FSL installation.

-  **happy2std** - Guess.

-  **showtc** - A very simple command line utility that takes timecourses from text
   files and plots the data in it in a matplotlib window. That’s it. A
   good tool for quickly seeing what’s in a file. Has a number of options to
   make the plot prettier.

-  **showxy** - Another simple command line utility that displays the
   the data contained in text files containing whitespace separated x-y pairs.

-  **showhist** - Another simple command line utility that displays the
   histograms generated by rapidtide2.

-  **resamp1tc** - takes an input text file at some sample rate and
   outputs a text file resampled to the specified sample rate.

-  **resamplenifti** - takes an input nifti file at some TR and outputs
   a nifti file resampled to the specified TR.

-  **tidepool** - This is a GUI tool for displaying all of
   the various maps and timecourses generated by rapidtide2 in one place, overlayed on
   an anatomic image. This makes it a bit easier to see how all the maps
   are related to one another, how the probe regressor evolves over the run, and the
   effect of the filtering parameters. To use it, launch tidepool from the
   command line, and then select a lag time map - tidpool will figure
   out the root name and pull in all of the other associated data. Works
   in native or standard space.

-  **correlate.py, corrfit.py, dlfilter.py, filter.py, fit.py, io.py,
   miscmath.py, multiproc.py, refine.py, resample.py, stats.py,
   util.py, helper_classes.py** - These are the libraries of the various helper routines
   that are used by pretty much every program in here for correlation,
   resampling, filtering, normalization, significance estimation, file
   I/O, etc.

-  **OrthoImageItem.py** - This is a class that implements the
   orthographic projection module that is used to display all of the
   maps in tidepool. It uses pyqtgraph to do all the heavy lifting. None
   of the built-ins in pyqtgraph did exactly what I wanted in terms of
   allowing 3D selection, overlays and the like, so I cobbled this
   together. It may be generally useful to anybody wanting to display
   functional data.

Support
=======

This code base is being developed and supported by a grant from the US
NIH (`1R01 NS097512 <http://grantome.com/grant/NIH/R01-NS097512-02>`__).

References
==========

Links to PDFs of all papers mentioned here can be found on the OMG
website: https://www.nirs-fmri.net/home/publications

General overview of systemic low frequency oscillations in fMRI data
--------------------------------------------------------------------

1) Tong Y, Hocke LM, Frederick BB. (2019) Low Frequency Systemic
   Hemodynamic “Noise” in Resting State BOLD fMRI: Characteristics,
   Causes, Implications, Mitigation Strategies, and Applications. Front.
   Neurosci., 14 August 2019 \| https://doi.org/10.3389/fnins.2019.00787

Multimodal Cerebral Circulation Imaging
---------------------------------------

1)  Tong Y, Frederick BD. (2010) Time lag dependent multimodal
    processing of concurrent fMRI and near-infrared spectroscopy (NIRS)
    data suggests a global circulatory origin for low-frequency
    oscillation signals in human brain. Neuroimage, 53(2), 553-64.

2)  Tong Y, Hocke L, Frederick BD. (2011) Isolating the sources of
    widespread physiological fluctuations in fNIRS signals. J Biomed
    Opt. 16(10), 106005.

3)  Tong Y, Bergethon PR, Frederick BD. (2011c) An improved method for
    mapping cerebrovascular reserve using concurrent fMRI and
    near-infrared spectroscopy with Regressor Interpolation at
    Progressive Time Delays (RIPTiDe). Neuroimage, 56(4), 2047-2057.

4)  Tong Y, Frederick BD. (2012) Concurrent fNIRS and fMRI processing
    allows independent visualization of the propagation of pressure
    waves and bulk blood flow in the cerebral vasculature. Neuroimage,
    Jul 16;61(4): 1419-27.

5)  Tong Y, Hocke LM, Licata SC, Frederick BD. (2012) Low frequency
    oscillations measured in the periphery with near infrared
    spectroscopy (NIRS) are strongly correlated with blood oxygen
    level-dependent functional magnetic resonance imaging (BOLD fMRI)
    signals. J Biomed Opt, 2012;17(10):106004. doi:
    10.1117/1.JBO.17.10.106004. PubMed PMID: 23224003; PMCID: 3461094.

6)  Tong Y, Hocke LM, Frederick BD. (2013) Short repetition time
    multiband EPI with simultaneous pulse recording allows dynamic
    imaging of the cardiac pulsation signal. Magn Reson Med
    2014;72(5):1268-76. Epub Nov 22, 2013. doi: 10.1002/mrm.25041.
    PubMed PMID: 24272768.

7)  Tong Y, Frederick B. (2014) Studying the Spatial Distribution of
    Physiological Effects on BOLD Signals using Ultrafast fMRI. Front
    Hum Neurosci 2014;5(196). doi: doi: 10.3389/fnhum.2014.00196.

8)  Tong Y, Frederick B. (2014) Tracking cerebral blood flow in BOLD
    fMRI using recursively generated regressors. Hum Brain Mapp.
    2014;35(11):5471-85. doi: 10.1002/hbm.22564. PubMed PMID: 24954380;
    PMCID: PMC4206590.

9)  Donahue M, Strother M, Lindsey K, Hocke L, Tong Y, Frederick B.
    (2015) Time delay processing of hypercapnic fMRI allows quantitative
    parameterization of cerebrovascular reactivity and blood flow
    delays. Journal of Cerebral Blood Flow & Metabolism.  2015. PubMed 
    PMID: 26661192. Epub October 19, 2015. doi: 10.1177/0271678X15608643.

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
    flow and metabolism : official journal of the International Society
    of Cerebral Blood Flow and Metabolism. 2016. doi:
    10.1177/0271678X16631755. PubMed PMID: 26873885.

Cardiac waveform extraction and refinement
------------------------------------------

1) Aslan S, Hocke L, Schwarz N, Frederick B. (2019) Extraction of the
   cardiac waveform from simultaneous multislice fMRI data using slice
   sorted averaging and a deep learning reconstruction filter.
   NeuroImage 198, 303–316 (2019).

Physiological noise identification and removal using time delay methods
-----------------------------------------------------------------------

1) Tong Y, Lindsey KP, Frederick BD. (2011b) Partitioning of
   physiological noise signals in the brain with concurrent
   near-infrared spectroscopy (NIRS) and fMRI. J Cereb Blood Flow Metab.
   31(12), 2352-62.

2) Frederick BD, Nickerson LD, Tong Y. (2012) Physiological denoising of
   BOLD fMRI data using Regressor Interpolation at Progressive Time
   Delays (RIPTiDe) processing of concurrent fMRI and near-infrared
   spectroscopy (NIRS). Neuroimage, Apr 15;60(3): 1419-27.

3) Tong Y, Hocke LM, Nickerson LD, Licata SC, Lindsey KP, Frederick BB
   (2013) Evaluating the effects of systemic low frequency oscillations
   measured in the periphery on the independent component analysis
   results of resting state networks. NeuroImage. 2013;76C:202-15. doi:
   10.1016/j.neuroimage.2013.03.019. PubMed PMID: 23523805; PMCID:
   PMC3652630.

4) Hocke LM, Tong Y, Lindsey KP, Frederick BB (2016). Comparison of
   peripheral near-infrared spectroscopy low-frequency oscillations to
   other denoising methods in resting state functional MRI with
   ultrahigh temporal resolution. Magnetic resonance in medicine :
   official journal of the Society of Magnetic Resonance in Medicine /
   Society of Magnetic Resonance in Medicine. 2016. doi:
   10.1002/mrm.26038. PubMed PMID: 26854203.

5) Erdoğan S, Tong Y, Hocke L, Lindsey K, Frederick B (2016). Correcting
   resting state fMRI-BOLD signals for blood arrival time enhances
   functional connectivity analysis. Front. Hum. Neurosci., 28 June 2016
   \| http://dx.doi.org/10.3389/fnhum.2016.00311

.. |License| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0
.. |Documentation Status| image:: https://readthedocs.org/projects/rapidtide/badge/?version=latest
   :target: http://rapidtide.readthedocs.io/en/latest/?badge=latest
.. |Travis-CI| image:: https://travis-ci.org/bbfrederick/rapidtide.svg?branch=master
   :target: https://travis-ci.org/bbfrederick/rapidtide
.. |Coverage| image:: https://codecov.io/gh/bbfrederick/rapidtide/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/bbfrederick/rapidtide
.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.814990.svg
   :target: https://doi.org/10.5281/zenodo.814990
.. |Funded by NIH| image:: https://img.shields.io/badge/NIH-R01--NS097512--01A1-yellowgreen.svg
   :target: http://grantome.com/grant/NIH/R01-NS097512-01A1
