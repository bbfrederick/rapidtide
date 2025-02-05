happy
-----

Description:
^^^^^^^^^^^^

happy is a new(er) addition to the rapidtide suite (it was added in 2019, so it's hardly new at this point).
It's complementary to rapidtide - it's focussed on fast, cardiac signals in fMRI, rather than the slow,
LFO signals we are usually looking at.  It's sort of a Frankenprogram - it has three distinct jobs,
which are related, but are very distinct.

The first thing happy does is try to extract a cardiac waveform from the fMRI data.  This is
something I've been thinking about for a long time.  It occurred to me that while the TR for
most scans is long compared to the required sample time for recording a cardiac waveform, the
scanner is actually recording data at a much faster rate than that - each slice, or in the case
of multiband data, stack of slices, is acquired somewhere during each TR, so the _effective_
samplerate is TR/(number of acquisitions within a TR).

The second task is to take this raw estimate of the cardiac waveform, and clean it up using a deep
learning filter.  The original signal is useful, but pretty gross, but I figured you should be able to
exploit the pseudoperiodic nature of the signal to greatly improve it.  This was also a testbed to work
on using neural nets to process time domain signals.  It seemed like a worthwhile project, so it got grafted in.

The final task (which was actually the initial task, and the reason I wrote happy to begin with) is
to implement Henning Voss' totally cool hypersampling with analytic phase projection (guess where the
name "happy" comes from).  This is fairly straightforward, as Voss describes his method very clearly.
But I have lots of data with no simultaneously recorded cardiac signals, and I was too lazy to go
find datasets with pleth data to play with, so that's why I did the cardiac waveform extraction part.
In retrospect, that's part is pretty cool in it's own right, if I do say so myself.


Inputs:
^^^^^^^
Happy needs a 4D BOLD fMRI data file (space by time) as input.  This can be Nifti1 or Nifti2.  If you have
a simultaneously recorded cardiac waveform, it will happily (heh heh) use it, otherwise it will try to
construct (and refine) one. NOTE: the 4D input dataset needs to be completely unpreprocessed - gradient
distortion correction and motion correction can destroy the relationship between slice number and actual
acquisition time, and slice time correction does not behave as expected for aliased signals (which the
cardiac component in fMRI most certainly is), and in any case we need the slice time offsets to
construct our waveform.


Outputs:
^^^^^^^^

Outputs are space or space by time Nifti or text files, depending on what the input data file was, and
some text files containing textual information, histograms, or numbers.  File formats and naming follow
BIDS conventions for derivative data for fMRI input data.  Output spatial dimensions and file type match
the input dimensions and file type (Nifti1 in, Nifti1 out).  Depending on the file type of map, there
can be no time dimension, a time dimension that matches the input file, or something else, such as a
time lag dimension for a correlation map.


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



Case 2: When you DO have a pleth recording
''''''''''''''''''''''''''''''''''''''''''
If you do have a pleth recording, then by all mean use it.  You simply tell happy where it is (it can be in
pretty much any sort of texty kind of file - .txt, .csv, .tsv, .tsv.gz, properly BIDS compliant, etc.), and
tell it the sample rate and start time (you don't have to cut up a continuous physio recording).
If the data is BIDS, you don't need to specify samplerate and start time, since they are in the file already.:

  ::

    happy \
      --cardiacfile FILE[:COL] \
      --cardiacfreq FREQ \
      --cardiacstart STARTTIME

Where:
    FILE[:COL] is the regressor text filename - if multicolumn, use :COL to specify the column to use either numerically, with 0 being the first column, or by name if the file is a BIDS compliant .tsv file.

    FREQ is the sample rate of the text file in Hz.  Alternately, you can use --cardiactstep TSTEP to give the sample period - these are two ways to specify the same thing.

    STARTTIME is the time delay into the file, in seconds, where the part of the waveform you want to use begins.  This is useful if you have a long pleth recording and don't want to have to chop it up into multiple files.

NB:  If you have a high quality plethysmogram, then a lot of the constraints on happy go away.  The requirement of a combination of TR, multiband factor and slice number that allows you to successfully extract a cardiac waveform aren't really relevant to analytic phase projection.  That should work regardless of any of those factors, so if you want to make a spiffy movie of pulsatility, and you have a pleth recording, you're good to go.


Performance tuning
''''''''''''''''''
If your data is fairly high quality, then happy should Just Work without any tweaking.  However, if you
have bad motion, low SNR, longish TR or lowish multiband factors, it may need a little help finding the signal,
so you can play with the many command line parameters that limit the heartrate search range, etc.  to see a list of
options, along with fairly self explanatory descriptions, type:

  ::

    happy --help


Multiecho data
''''''''''''''
Does this work on multiecho data?  That's a very good question, and I'm glad you asked me that!

Yes, it does.  There are a couple of things to consider though.  The first is that as far as I can tell,
the majority of the cardiac signal that you see in fMRI data is a non-BOLD signal, which is to say that
one of the cool things about multiecho data, the fact that you can separate BOLD and non-BOLD data doesn't
really help you, since if you use tedana or ME-ICA on your data, you're going to strip out the signal
you're looking for, which would be bad.  But in fact, that's not really an issue, since you want to work
on the fully unprocessed data anyway, so happy works on data long before you've even gotten to the
multiecho processing stage.

Which brings us to the next question - there's additional timing information from the fact that
each echo is recorded at a slightly different time - how do we smartly combine the data from all
the echoes?  The answer for now is - you don't, because I'm not that smart.  What I can confirm
through, my limited testing, is that in a 60 slice, TR=1.33s MB=4 4 echo acquisition (a trendy
set of parameters in the circles I travel in) you can extract the cardiac waveform from _any_
of the individual echoes, although the first echo has the highest SNR - it seems that the short echo time
enhances the cardiac variation.  The noise in the
echoes seems largely uncorrelated (other than motion effects), so combining echoes is almost
certainly a good idea.  That said, averaging doesn't make things look noticeably better.
Maybe use PCA to extract the most important timecourse from the set of derived
cardiac regressors?  Or maybe actually using the timing information of the individual echoes
to enhance the time resolution?  I await your PR implementing this with bated breath,
dear reader...
