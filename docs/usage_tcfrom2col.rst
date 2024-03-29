tcfrom2col
----------

Description:
^^^^^^^^^^^^
	A  simple command line that takes an FSL style 2 column regressor file and generates a time course (waveform) file.  FSL 3 column files are text files containing one row per "event".  Each row has three columns: start time in seconds, duration in seconds, and waveform value.  The output waveform is zero everywhere that is not covered by an "event" in the file.

Inputs:
^^^^^^^
	An FSL style two column text file (start time, duration)

Outputs:
^^^^^^^^
	A single column text file containing the waveform

Usage:
^^^^^^

.. argparse::
   :ref: rapidtide.workflows.tcfrom2col._get_parser
   :prog: tcfrom2col
   :func: _get_parser

   Debugging options : @skip
      skip debugging options



