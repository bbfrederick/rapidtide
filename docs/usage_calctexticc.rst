calctexticc
-----------

Description:
^^^^^^^^^^^^
    Calculate per-column ICC(3,1) on a set of text files.

Inputs:
^^^^^^^
    One or more two dimensional text files, with quantities in the columns, and subjects in the rows

Outputs:
^^^^^^^^
    One dimensional text files with ICC, r, and e for each input quantity

Usage:
^^^^^^

.. argparse::
   :ref: rapidtide.workflows.calctexticc._get_parser
   :prog: calctexticc
   :func: _get_parser

   Debugging options : @skip
      skip debugging options


