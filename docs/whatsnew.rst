What\'s new
-----------

Version 3.0 summary:
^^^^^^^^^^^^^^^^^^^^

The 3.0 release has a lot of improvements and bugfixes across almost all of the programs in the package, and the
documentation.  Although I\'ve been trying to get the release out for almost 5 months by squashing bugs and all, I
keep getting distracted adding things and trying new ideas, which leads to new debugging cycles.  One major goal was
to try to get most of the workflow breaking interface changes in by 3.0, then get it stable, then release, but a) I
have almost certainly missed some, either in implementation or documentation, and b) until more people start using
this, there will almost certainly be some bugs that should be obvious to me, but are not.  So I am just going to
say it\'s different enough from what I was doing before to call it a release and be done, and we\'ll go from here.
Here are some of the major changes:

Rapidtide
"""""""""
* Reorganized and renamed some command line options and adjusted the default values - NOTE: some of these are breaking changes - where possible, I\'ve added deprecation warnings.  Some biggies:

    - You now have very granular control over which files are saved using the ``--outputlevel LEVEL`` option.  As a result, ``--nolimitoutput`` and a few other flags are no longer used.
    - The final noise removal filter has not been a glm filter for quite some time, so all the options and files associated with sLFO filtering have been renamed.   Use ``--nodenoise`` to turn sLFO filtering off.
    - ``--pickleft`` is on by default.  Use ``--nopickleft`` to turn it off.

* I\'ve implemented a new delay refinement step using the ratio of linear fit coefficients.  This seems to be very robust and powerful to improve delay estimation, so this will trickle into other new functions over time.
* Rationalized handling of zero-mean data - now use the ``--dataiszeromean`` argument to tell rapidtide how to make masks.  ``--globalmaskmethod`` no longer exists.
* Mask setting has been streamlined and made somewhat more robust.  This should give you better fits with less tweaking.
* Enhanced fit failure reporting, to make debugging bad fits a bit easier.
* The ``--fixdelay`` option has been split into two options.  ``--initialdelay DELAY`` lets you specify either a float that sets the starting delay for every voxel to that value, or a 3D file specifying the initial delay for each voxel.  ``--nodelayfit`` determines whether the delay can be adjusted from its initial value.  Closes https://github.com/bbfrederick/rapidtide/issues/171. KNOWN ISSUE:  If you supply an initial delay map, instead of using the global mean, rapidtide should use the delays to make your first stage regressor.  Currently that is not the case.
* Now outputs a map of fit significance (-log10(p)).  In response to https://github.com/bbfrederick/rapidtide/issues/174.
* There has been a lot of modularization and reorganization internally so that I can try out some new workflows that I think will be more robust to weird data.

Retroregress
""""""""""""
* The final noise regression can now be performed (or repeated) after a successful rapidtide run, with different filtering options.  This lets you do the computationally expensive parts of rapidtide (regressor extraction and delay finding) without doing the final regression.  Since denoising the input data can produce huge output files, stopping before the final regression step can produce outputs that are orders of magnitude smaller than doing the full processing that nevertheless store the time consuming results.  You can then do the denoising step at your leisure in a small fraction of the time.

Happy
"""""
* I\'m trying out some new approaches for aliased correlation which might produce better phase projection movies.
* I\'ve put in some initial support for multiecho data.  Currently it seems that processing the first echo on its own gives very good results, but there should be some improvement with smart combinations.
* Fixed a major bug in motion filtering (thank you to Sukru Demiral for reporting it!)
* Updated all the deep learning code and saved models for improved tensorflow 2 compatibility (loading models started to throw weird errors in tensorflow 2.19.0).

Tidepool
""""""""
* You can now load multiple datasets from the command line using  ``--dataset DATASET1 DATASET2...``.  You can load additional datasets and switch between them using the File menu.
* Added a new "big" template to let you see more information at once.  Select with ``--uistyle big``.
* Supports pyqt6.
* Lots of bug fixes, code cleanup, and QOL improvements.

Miscellaneous
"""""""""""""
* (atlasaverage) Added the ability to segment a template atlas based on values of the map being summarized.
* (reference) Constructed a new version of the JHU LVL1 atlas where each territory is split into an early and late subterritory, based on the median delay in the HCP-YA dataset.


Documentation
"""""""""""""
* Expanded the rapidtide theory of operation section a lot.
* Significantly beefed up happy section.
* tidepool\'s documentation has been updated to reflect the current state of the program.
* General cleanup and revision to match the current state of the package.


Package
"""""""
* Lots of code cleanup and reorganization to make functions more modular.
* Lots and lots of bug fixes.


Individual releases:
--------------------

.. include:: ../CHANGELOG.md
   :parser: myst_parser.sphinx_
