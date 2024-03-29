.. rapidtide documentation master file, created by
   sphinx-quickstart on Thu Jun 16 15:27:19 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Rapidtide
==========
Rapidtide is a suite of python programs used to perform rapid time delay
analysis on functional imaging data to find time lagged correlations
between the voxelwise time series and other time series, both in the LFO
band (rapditide2) and now in the cardiac band (happy).

.. image:: https://img.shields.io/pypi/v/rapidtide.svg
   :target: https://pypi.python.org/pypi/rapidtide/
.. image:: https://img.shields.io/pypi/pyversions/rapidtide.svg
   :target: https://pypi.python.org/pypi/rapidtide/
.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0
.. image:: https://readthedocs.org/projects/rapidtide/badge/?version=main
   :target: http://rapidtide.readthedocs.io/en/main/?badge=main
.. image:: https://circleci.com/gh/bbfrederick/rapidtide.svg?branch=main&style=shield
   :target: https://circleci.com/gh/bbfrederick/rapidtide
.. image:: https://codecov.io/gh/bbfrederick/rapidtide/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/bbfrederick/rapidtide
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.814990.svg
   :target: https://doi.org/10.5281/zenodo.814990
.. image:: https://img.shields.io/badge/NIH-R01--NS097512--01A1-yellowgreen.svg
   :target: http://grantome.com/grant/NIH/R01-NS097512-01A1

Citing rapidtide
-------------------

  Frederick, B, rapidtide [Computer Software] (2016-2022). Available from
  https://github.com/bbfrederick/rapidtide. doi:`10.5281/zenodo.814990`_

.. _10.5281/zenodo.814990: https://doi.org/10.5281/zenodo.814990

Contents
========
.. toctree::
   :maxdepth: 3
   :caption: Introduction:

   introduction.rst


.. toctree::
   :maxdepth: 2
   :caption: Installation:

   installation.rst


.. toctree::
   :maxdepth: 2
   :caption: Usage:

   usage_general.rst
   usage_rapidtide.rst
   usage_happy.rst
   usage_rapidtide2std.rst
   usage_showxcorrlegacy.rst
   usage_showxcorrx.rst
   usage_other.rst
   legacy.rst
   api.rst
   contributing.rst


.. toctree::
   :maxdepth: 2
   :caption: Digging deeper:

   theoryofoperation.rst


.. toctree::
   :maxdepth: 2
   :caption: History:

   whats_new.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
