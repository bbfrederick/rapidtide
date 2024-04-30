aligntcs
--------

Description:
^^^^^^^^^^^^
    Given two timecourses:
        * Resample the second timecourse to the samplerate of the first timecourse
        * Use crosscorrelation to determine the time delay between the sequences
        * Apply a time shift to align the second timecourse with the first timecourse

Inputs:
^^^^^^^
    Two text files containing timecourses

Outputs:
^^^^^^^^
    A text file containing the second input timecourse, resampled and time shifted to match the first timecourse
Usage:
^^^^^^

.. argparse::
   :ref: rapidtide.workflows.aligntcs._get_parser
   :prog: aligntcs
   :func: _get_parser

   Debugging options : @skip
      skip debugging options


