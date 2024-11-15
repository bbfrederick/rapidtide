.. rapidtide documentation master file, created by
   sphinx-quickstart on Thu Jun 16 15:27:19 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*********
Rapidtide
*********
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
----------------

  Frederick, B, rapidtide [Computer Software] (2016-2024). Available from
  https://github.com/bbfrederick/rapidtide. doi:`10.5281/zenodo.814990`_

.. _10.5281/zenodo.814990: https://doi.org/10.5281/zenodo.814990

Contents
--------
.. toctree::
   :maxdepth: 3
   :caption: Introduction:

   introduction.rst


.. toctree::
   :maxdepth: 2
   :caption: Installation:

   installation.rst
   support.rst


.. toctree::
   :maxdepth: 3
   :caption: Usage:

   usage_general.rst

   usage_rapidtide.rst
   usage_rapidtide2std.rst
   usage_tidepool.rst
   usage_adjustoffset.rst

   usage_happy.rst
   usage_happy2std.rst
   usage_proj2flow.rst

   usage_showxcorrlegacy.rst
   usage_showxcorrx.rst

   usage_showtc.rst
   usage_showxy.rst
   usage_showhist.rst
   usage_spectrogram.rst

   usage_glmfilt.rst
   usage_atlasaverage.rst
   usage_ccorrica.rst

   usage_tcfrom2col.rst
   usage_tcfrom3col.rst
   usage_pixelcomp.rst
   usage_atlastool.rst
   usage_applydlfilter.rst
   usage_calctexticc.rst
   usage_diffrois.rst
   usage_endtidalproc.rst
   usage_filtnifti.rst
   usage_filttc.rst
   usage_histtc.rst
   usage_histnifti.rst
   usage_resamplenifti.rst
   usage_resampletc.rst
   usage_aligntcs.rst
   usage_temporaldecomp.rst
   usage_spatialdecomp.rst

   usage_polyfitim.rst
   usage_mergequality.rst
   usage_pairproc.rst
   usage_pairwisemergenifti.rst
   usage_physiofreq.rst
   usage_plethquality.rst

   usage_rankimage.rst
   usage_runqualitycheck.rst
   usage_variabilityizer.rst
   usage_fdica.rst
   usage_gmscalc.rst
   usage_roisummarize.rst

   usage_simdata.rst
   usage_spatialfit.rst
   usage_spatialmi.rst

   usage_localflow.rst
   usage_synthASL.rst

   legacy.rst


   api.rst
   contributing.rst


.. toctree::
   :maxdepth: 3
   :caption: Digging deeper:

   theoryofoperation.rst


.. toctree::
   :maxdepth: 2
   :caption: History:

   whatsnew.rst


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
