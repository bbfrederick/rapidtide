applydlfilter
-------------

Description:
^^^^^^^^^^^^
    Apply happy's deep learning filter to a noisy cardiac timecourse to get a high quality synthetic plethysmogram

Inputs:
^^^^^^^
    A text file containing a noisy cardiac timecourse

Outputs:
^^^^^^^^
    A cleaned cardiac timecourse text file

Usage:
^^^^^^

.. argparse::
   :ref: rapidtide.workflows.applydlfilter._get_parser
   :prog: applydlfilter
   :func: _get_parser

   Debugging options : @skip
      skip debugging options


