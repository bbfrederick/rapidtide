happy
-----

Description:
^^^^^^^^^^^^

happy is a new addition to the rapidtide suite.  It's complementary to rapidtide - it's focussed on fast, cardiac signals in fMRI, rather than the slow, LFO signals we are usually looking at.  It's sort of a Frankenprogram - it has three distinct jobs, which are related, but are very distinct.

The first thing happy does is try to extract a cardiac waveform from the fMRI data.  This is something I've been thinking about for a long time.  Words go here

The second task is to take this raw estimate of the cardiac waveform, and clean it up using a deep learning filter.  The original signal is useful, but pretty gross, but I figured you should be able to exploit the pseudoperiodic nature of the signal to greatly improve it.  This is also a testbed to work on using neural nets to process time domain signals.  It seemed like a worthwhile project, so it got grafted in.

The final task (which was actually the initial task, and the reason I wrote happy to begin with) is to implement Henning Voss' totally cool hypersampling with analytic phase projection (guess where the name "happy" comes from).  This is fairly straightforward, as Voss describes his method very clearly.  But I have lots of data with no simultaneously recorded cardiac signals, and I was too lazy to go find datasets with pleth data to play with, so that's why I did the cardiac waveform extraction part.


Inputs:
^^^^^^^
Happy needs a 4D BOLD fMRI data file (space by time) as input.  This can be Nifti1 or Nifti2.  If you have a simultaneously recorded cardiac waveform, it will happily use it, otherwise it will try to construct (and refine) one. NOTE: the 4D input dataset needs to be completely unpreprocessed - gradient distortion correction and motion correction can destroy the relationship between slice number and actual acquisition time, and slice time correction does not behave as expected for aliased signals (which the cardiac component in fMRI most certainly is), and in any case we need the slice time offsets to construct our waveform.


Outputs:
^^^^^^^^

Outputs are space or space by time Nifti or text files, depending on what the input data file was, and some text files containing textual information, histograms, or numbers.  File formats and naming follow BIDS conventions for derivative data for fMRI input data.  Output spatial dimensions and file type match the input dimensions and file type (Nifti1 in, Nifti1 out).  Depending on the file type of map, there can be no time dimension, a time dimension that matches the input file, or something else, such as a time lag dimension for a correlation map.


BIDS Outputs:
^^^^^^^^^^^^^

.. csv-table::
   :header: "Name", "Extension(s)", "Content", "When present"
   :widths: 15, 15, 30, 10

   "XXX_commandline", ".txt", "The command line used to run happy", "Always"
   "XXX_formattedcommandline", ".txt", "The command line used to run happy, attractively formatted", "Always"
   "XXX_desc-rawapp_info", ".nii.gz", "The analytic phase projection map of the cardiac waveform", "Always"
   "XXX_desc-app_info", ".nii.gz", "The analytic phase projection map of the cardiac waveform, voxelwise minimum subtracted", "Always"
   "XXX_desc-normapp_info", ".nii.gz", "The analytic phase projection map of the cardiac waveform, voxelwise minimum subtracted and normalized", "Always"
   "XXX_desc-apppeaks_hist", ".tsv.gz, .json", "Not sure", "Always"
   "XXX_desc-apppeaks_hist_centerofmass", ".txt", "Not sure", "Always"
   "XXX_desc-apppeaks_hist_peak", ".txt", "Not sure", "Always"
   "XXX_desc-slicerescardfromfmri_timeseries", ".tsv.gz, .json", "Cardiac timeseries at the time resolution of slice acquisition ((1/TR * number of slices / multiband factor", "Always"
   "XXX_desc-stdrescardfromfmri_timeseries", ".tsv.gz, .json", "Cardiac timeseries at standard time resolution (25.O Hz)", "Always"
   "XXX_desc-cardpulsefromfmri_timeseries", ".tsv.gz, .json", "The average (over time from minimum) of the cardiac waveform over all voxels", "Always"
   "XXX_desc-cardiaccyclefromfmri_timeseries", ".tsv.gz, .json", "The average (over a single cardiac cycle) of the cardiac waveform over all voxels", "Always"
   "XXX_desc-cine_info", ".nii.gz", "Average image of the fMRI data over a single cardiac cycle", "Always"
   "XXX_desc-cycleaverage_timeseries", ".tsv.gz, .json", "Not sure", "Always"
   "XXX_desc-maxphase_map", ".nii.gz", "Map of the average phase where the maximum amplitude occurs for each voxel", "Always"
   "XXX_desc-minphase_map", ".nii.gz", "Map of the average phase where the minimum amplitude occurs for each voxel", "Always"
   "XXX_desc-processvoxels_mask", ".nii.gz", "Map of all voxels used for analytic phase projection", "Always"
   "XXX_desc-vessels_map", ".nii.gz", "Amplitude of variance over a cardiac cycle (large values are assumed to be vessels)", "Always"
   "XXX_desc-vessels_mask", ".nii.gz", "Locations of voxels with variance over a cardiac cycle that exceeds a threshold (assumed to be vessels)", "Always"
   "XXX_desc-arteries_map", ".nii.gz", "High variance vessels with early maximum values within the cardiac cycle", "Always"
   "XXX_desc-veins_map", ".nii.gz", "High variance vessels with late maximum values within the cardiac cycle", "Always"
   "XXX_info", ".json", "Run parameters and derived values found during the run (quality metrics, derived thresholds, etc.)", "Always"
   "XXX_memusage", ".csv", "Memory statistics at multiple checkpoints over the course of the run", "Always"
   "XXX_runtimings", ".txt", "Detailed timing information", "Always"
..



Usage:
^^^^^^

.. argparse::
   :ref: rapidtide.workflows.happy_parser._get_parser
   :prog: happy
   :func: _get_parser

   Debugging options : @skip
      skip debugging options




Example:
^^^^^^^^^

Extract the cardiac waveform and generate phase projections
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Case 1: When you don't have a pleth recording
'''''''''''''''''''''''''''''''''''''''''''''
There are substantial improvements to the latest versions of happy.
In the old versions, you actually had to run happy twice -
the first time to estimate the vessel locations, and the second
to actually derive the waveform.  Happy now combines these operations interpolation
a single run with multiple passes - the first
pass locates voxels with high variance, labels them as vessels, then reruns
the derivation, restricting the cardiac estimation to these high variance voxels.
This gives substantially better results.

Using the example data in the example directory, try the following:

  ::

    happy \
        rapidtide/data/examples/src/sub-HAPPYTEST.nii.gz \
        rapidtide/data/examples/src/sub-HAPPYTEST.json \
        rapidtide/data/examples/dst/happytest


This will perform a happy analysis on the example dataset.  To see the extracted
cardiac waveform (original and filtered), you can use showtc (also part of them
rapidtide package):

  ::

    showtc \
        rapidtide/data/examples/src/happytest_desc-slicerescardfromfmri_timeseries.json:cardiacfromfmri,cardiacfromfmri_dlfiltered \
        --format separate


