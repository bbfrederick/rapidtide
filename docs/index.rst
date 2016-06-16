.. delaytools documentation master file, created by
   sphinx-quickstart on Thu Jun 16 15:27:19 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to delaytools's documentation!
======================================

Contents:

.. toctree::
   :maxdepth: 2
Introduction
============
Why do I want to know about time lagged correlations?
-----------------------------------------------------
This comes out of work by our group (The Opto-Magnetic group at McLean Hospital - http://www.nirs-fmri.net) looking at the correlations between neuroimaging data (fMRI) and NIRS data recorded simultaneously, either in the brain or the periphery.  We found that a large fraction of the "noise" we found at low frequency in fMRI data was due to real, random[*] fluctuations of blood oxygenation and volume (both of which affect the intensity of BOLD fMRI images) in the blood passing through the brain. More interestingly, because these characteristics of blood move with the blood itself, this gives you a way to determine blood arrival time at any location in the brain. This is interesting in and of itself, but also, this gives you a method for optimally modelling (and removing) in band physiological noise from fMRI data (see references below).
 
After working with this for several years we've also found that you don't need to used simultaneous NIRS to find this blood borne signal - you can get it from blood rich BOLD voxels for example in the superior sagittal sinus, or bootstrap it out of the global mean signal in the BOLD data. You can also track exogenously applied waveforms, such as hypercarbic and/or hyperoxic gas challenges to really boost your signal to noise.  So there are lots of times when you might want to do this type of correlation analysis.  This package provides the tools to make that easier.
      
As an aside, some of these tools are just generally useful for looking at correlations between timecourses from other sources – for example doing PPI, or even some seed based analyses.
      
A note on coding quality and style:
-----------------------------------
This code has been in active development since June of 2012.  This has two implications.  The first is that it has been tuned and refined quite a bit over the years, with a lot of optimizations and bug fixes - most of the core routines have been tested fairly extensively to get rid of the stupidest bugs.  I find new bugs all the time, but most of the showstoppers seem to be gone.  The second result is that the coding style is all over the place.  When I started writing this, I had just moved over from C, and it was basically a mental port of how I would write it in C, and was extremely unpythonic (I’ve been told by a somewhat reliable source that looking over some of my early python efforts “made his eyes bleed”).  Over the years, as I've gone back and added functions, I periodically get embarassed and upgrade things to a somewhat more modern coding style.  I even put in some classes - that's what the cool kids do, right?  But the pace of that effort has to be balanced with the fact that when I make major architectural changes, I tend to break things.  So be patient with me, and keep in mind that you get what you pay for, and this cost you nothing!  Function before form.

Python version compatibility: 
----------------------------
This code has been extensively tested in python 2.7.  I dragged my feet somewhat making it python 3 compatible, since a number of the libraries I needed have took a long time to get ported to python 3, and I honestly saw no advantage to doing it.  I since decided that I’m going to have to do it eventually, so why not now?  As far as I know, the code all works fine in python 3.5 now - I’ve switched over to that on my development machine, and have not hit any version related issues in a while now, and according to PyCharm’s code inspection, there are no incompatible constructions.  However that’s no guarantee that there isn’t a problem in some option I haven’t bothered to test yet, so be vigilant, and please let me know if there is some issue with python 3 that I haven’t caught (or any bugs, really).
      
Why are you releasing your code?
--------------------------------
For a number of reasons.
      
    1)    I want people to use it!  I think if it were easier for people to do time delay analysis, they’d be more likely to do it.  I don’t have enough time or people in my group to do every experiment that I think would be interesting, so I’m hoping other people will, so I can read their papers and learn interesting things.
      
    2)    It’s the right way to do science – I can say lots of things, but if nobody can replicate my results, nobody will believe it (we’ve gotten that a lot, because some of the implications of what we’ve seen in resting state data can be a little uncomfortable).  We’ve reached a stage in fMRI where getting from data to results involves a huge amount of processing, so part of confirming results involves being able to see how the data were processed. If you had to do everything from scratch, you’d never even try to confirm anybody’s results.
      
    3)    In any complicated processing scheme, it’s quite possible (or in my case, likely) to make dumb mistakes, either coding errors or conceptual errors, and I almost certainly have made some (although hopefully the worst ones have been dealt with at this point).  More users and more eyes on the code make it more likely that they will be found.  As much as I’m queasy about somebody potentially finding a mistake in my code, I’d rather that they did so, so I can fix it[‡].
      
    4)    It’s giving back to the community.  I benefit from the generosity of a lot of authors who have made the open source tools I use for work and play, so I figure I can pony up too.
      
How do I cite this?
-------------------
Good question!  I think the following will work, although I should probably get a DOI for this.
Frederick, B, delaytools [Computer Software] (2016).  Retrieved from https://github.com/bbfrederick/delaytools.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

