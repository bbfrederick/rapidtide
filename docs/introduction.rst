Introduction
============
Why do I want to know about time lagged correlations?
-----------------------------------------------------
This comes out of work by our group (The Opto-Magnetic group at McLean
Hospital - http://www.nirs-fmri.net) looking at the correlations between
neuroimaging data (fMRI) and NIRS data recorded simultaneously, either in the
brain or the periphery.  We found that a large fraction of the "noise" we found
at low frequency in fMRI data was due to real, random[*] fluctuations of blood
oxygenation and volume (both of which affect the intensity of BOLD fMRI images)
in the blood passing through the brain. More interestingly, because these
characteristics of blood move with the blood itself, this gives you a way to
determine blood arrival time at any location in the brain. This is interesting
in and of itself, but also, this gives you a method for optimally modeling
(and removing) in band physiological noise from fMRI data (see references
below).

After working with this for several years we've also found that you don't need
to used simultaneous NIRS to find this blood borne signal - you can get it from
blood rich BOLD voxels for example in the superior sagittal sinus, or bootstrap
it out of the global mean signal in the BOLD data. You can also track
exogenously applied waveforms, such as hypercarbic and/or hyperoxic gas
challenges to really boost your signal to noise.  So there are lots of times
when you might want to do this type of correlation analysis.  This package
provides the tools to make that easier.

As an aside, some of these tools are just generally useful for looking at
correlations between timecourses from other sources – for example doing PPI, or
even some seed based analyses.

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

Why are you releasing your code?
--------------------------------
For a number of reasons.
    #.    I want people to use it!  I think if it were easier for people to do
    time delay analysis, they’d be more likely to do it.  I don’t have enough
    time or people in my group to do every experiment that I think would be
    interesting, so I’m hoping other people will, so I can read their papers
    and learn interesting things.

    #.    It’s the right way to do science – I can say lots of things, but if
    nobody can replicate my results, nobody will believe it (we’ve gotten that
    a lot, because some of the implications of what we’ve seen in resting state
    data can be a little uncomfortable).  We’ve reached a stage in fMRI where
    getting from data to results involves a huge amount of processing, so part
    of confirming results involves being able to see how the data were
    processed. If you had to do everything from scratch, you’d never even try
    to confirm anybody’s results.

    #.    In any complicated processing scheme, it’s quite possible (or in my
    case, likely) to make dumb mistakes, either coding errors or conceptual
    errors, and I almost certainly have made some (although hopefully the worst
    ones have been dealt with at this point).  More users and more eyes on the
    code make it more likely that they will be found.  As much as I’m queasy
    about somebody potentially finding a mistake in my code, I’d rather that
    they did so, so I can fix it[‡].

    #.    It’s giving back to the community.  I benefit from the generosity of
    a lot of authors who have made the open source tools I use for work and
    play, so I figure I can pony up too.

Python version compatibility
-----------------------------
I've now switched over to using python 3 as my daily driver, so I know that
everything works there.  However, I know that a lot of people can't or won't
switch from python 2x, so I make every effort to write code that works in both,
and I test in both.  I don't expect to switch over to any python 3 only
constructions anytime soon.  As of the latest version, rapidtide2 does finally
seem to run a little faster in python 3 than 2 if that matters to you.

What’s included in this package?
--------------------------------
I’ve included a number of tools to get you going – I’ll add in a number of
other utilities as I get them closer to the point that I can release them
without people laughing at my code.  For the time being, I’m including the
following:
