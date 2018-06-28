.. rapidtide documentation master file, created by
   sphinx-quickstart on Thu Jun 16 15:27:19 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Rapidtide
==========
Rapidtide is a suite of python programs used to perform time delay analysis on
functional imaging data to find time lagged correlations between the voxelwise
time series and other time series.

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0
.. image:: https://travis-ci.org/bbfrederick/rapidtide.svg?branch=master
   :target: https://travis-ci.org/bbfrederick/rapidtide
.. image:: https://codecov.io/gh/bbfrederick/rapidtide/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/bbfrederick/rapidtide

Citing rapidtide
-------------------
Good question! If you use rapidtide, please cite the following:

  Frederick, B, rapidtide [Computer Software] (2016). Available from
  https://github.com/bbfrederick/rapidtide. doi:`10.5281/zenodo.814990`_

.. _10.5281/zenodo.814990: https://doi.org/10.5281/zenodo.814990

NOTE
----
This is an evolving code base.  I'm constantly tinkering with it.  That said,
now that I'm releasing this to the world, I'm being somewhat more responsible
about locking down stable release points.  In between releases, however, I'll
be messing with things. **It's very possible I could break something while
doing this, so check back for status updates if you download the code in
between releases**.  I've finally become a little more modern and started
adding automated testing, so as time goes by hopefully the "in between"
releases will be somewhat more reliable.  Check back often for exciting new
features and bug fixes!

NB:  I have now started making "x" versions of programs, meaning experimental,
to let me test things out between releases.  If I make functional changes to
rapidtide2, they will be found in rapidtide2x.  When I'm happy enough with the
changes to inflict them on the world, I'll replace rapidtide2 with rapidtide2x
and do a new release.  That way I can still monkey with the code to my heart's
content without continually breaking the version that people use.  Unless I
specifically tell you to use rapidtide2x to try out a new function, you should
probably stick with rapidtide2.

Contents
========
.. toctree::
   :maxdepth: 2

   introduction
   usage
   contributing
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
