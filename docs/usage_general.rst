For more information about how the rapidtide library can be used, please
see the API page. Common rapidtide workflows can also be called from the
command line.

..
   Headings are organized in this manner:
   =====
   -----
   ^^^^^
   """""
   '''''

General points
--------------
Before talking about the individual programs, in the 2.0 release and going
forward, I've tried to adhere to some common principals, across all program,
to make them easier to understand and maintain, and more interoperable
with other programs, and to simplify using the outputs.

NB: All commands are shown using backslashes as line continuation characters for clarity to make the commands easier to read.  These aren't needed - you can just put all the options on the same line, in any order.

Standardization of interfaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Wherever possible, I've tried to harmonize the argument parsers between all of the programs in the package.
That means I have some standard argument groups, such as ones pertaining to filtering, correlation, correlation fitting,
sample rate
specification, etc., so if you know how to use one of the programs, that experience should carry over - argument
names should be the same, arguments should interact in the same way.  Given the wide range of years I've been working
on this package, that's not always fully implemented, but I do try to keep all the interfaces up to date as much
as possible.

This also means that I have tried to have pretty standard ways of reading and writing data files in terms of formats
and naming - every program should read a wide variety of text input types using the same method.  All NIFTI file reads
and writes are through a standard nibabel interface and should follow pretty standard conventions.

BIDS Outputs
^^^^^^^^^^^^
By default, all inputs and outputs are in BIDS compatible formats (this is most true
for rapidtide and happy, which get the majority of the work, but the goal
is to eventually make all the programs in the package conform to this).  The
two major ramifications of this are that I have tried to follow BIDS naming
conventions for NIFTI, json, and text files containing time series.  Also,
all text files are by default BIDS continuous timeseries files - data is
in compressed, tab separated column format (.tsv.gz), with the column names,
sample rate, and start time, in the accompanying .json sidecar file.

Text Inputs
^^^^^^^^^^^
A side effect of moving to BIDS is that I've now made a standardized interface
for reading text data into programs in the package to handle many different
types of file.  In general, now if you
are asked for a timeseries, you can supply it in any of the following ways:

A plain text file with one or more columns.
"""""""""""""""""""""""""""""""""""""""""""
You can specify any subset of
columns in any order by adding ":colspec" to the end of the filename.  "colspec"
is a column specification consisting of one or more comma separated "column
ranges".  A "column range" is either a single column number or a hyphen
separated minimum and maximum column number.  The first column in a file is
column 0.

For example specifying, "mytextfile.txt:5-6,2,0,10-12"

would return an array containing all the timepoints from columns 5, 6, 2, 0, 10, 11, and 12
from mytextfile.txt, in that order.  Not specifying ":colspec" returns all
the columns in the file, in order.

If the program in question requires the actual sample rate, this can be specified
using the ``--samplerate`` or ``--sampletime`` flags.  Otherwise 1.0Hz is assumed.

A BIDS continuous file with one or more columns.
""""""""""""""""""""""""""""""""""""""""""""""""
BIDS files have names for each column, so these are used in column specification.
For these files, "colspec" is a comma separated list of one or more column
names:

"thefile_desc-interestingtimeseries_physio.json:cardiac,respiration"

would return the two named columns "cardiac" and "respiration" from the
accompanying .tsv.gz file.
Not specifying ":colspec" returns all the columns in the file, in order.

Because BIDS continuous files require sample rate and start time to be specified
in the sidecar file, these quantities will now already be set.  Using the
``--samplerate``, ``--sampletime`` or ``--starttime`` flags will override any header
values, if specified.

Visualizing files
^^^^^^^^^^^^^^^^^
Any output NIFTI file can be visualized in your favorite NIFTI viewer.  I like
FSLeyes, part of FSL.  It's flexible and fast, and has lots of options for
displaying 3 and 4D NIFTI files.

While there may be nice, general graphing tools for BIDS timeseries files, I
wrote "showtc" many years ago, a matplotlib based file viewer with lots of
nice tweaks to make pretty and informative graphs of various rapidtide input
and output time series files.  It's part of rapidtide, and pretty easy to learn.  Just
type ``showtc --help`` to get the options.

As an example, after running happy, if you want to see the derived cardiac
waveform, you'd run:

::

  showtc \
      happytest_desc-slicerescardfromfmri_timeseries.json:cardiacfromfmri,cardiacfromfmri_dlfiltered \
      --format separate

There are some companion programs - ``showxy`` works on 2D (x, y) data; ``showhist`` is specifically for viewing
histograms (``_hist`` files) generated by several programs in the package, ``spectrogram`` generates and displays spectrograms of
time series data.  Each of these is separately documented below.


