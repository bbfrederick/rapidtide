tcfrom3col
----------

Description:
^^^^^^^^^^^^
	A  simple command line that takes an FSL style 3 column regressor file and generates a time course (waveform) file.  FSL 3 column files are text files containing one row per "event".  Each row has three columns: start time in seconds, duration in seconds, and waveform value.  The output waveform is zero everywhere that is not covered by an "event" in the file.

Inputs:
^^^^^^^
	An FSL style three column text file (start time, duration, value)

Outputs:
^^^^^^^^
	A single column text file containing the waveform

Usage:
^^^^^^

.. argparse::
   :ref: rapidtide.workflows.tcfrom3col._get_parser
   :prog: tcfrom3col
   :func: _get_parser

   Debugging options : @skip
      skip debugging options



