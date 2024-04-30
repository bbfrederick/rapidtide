resampletc
----------

Description:
^^^^^^^^^^^^

   Takes an input text file at some sample rate and outputs a text file resampled to the specified sample rate.  If downsampling, antialiasing is applied unless disabled.

Inputs:
^^^^^^^
    A text file with one or more columns.  If there are more than column, only the specified column is resampled.

Outputs:
^^^^^^^^
    A text file, resampled to the specified sample rate.

Usage:
^^^^^^

.. argparse::
   :ref: rapidtide.workflows.resampletc._get_parser
   :prog: resampletc
   :func: _get_parser

   Debugging options : @skip
      skip debugging options


