ccorrica
--------

Description:
^^^^^^^^^^^^
	Find temporal crosscorrelations between the columns of an input file

Inputs:
^^^^^^^
	A text file containing one or more timeseries columns. Use [:COLSPEC] to select which column(s) to use, where COLSPEC is an integer, a column separated list of ranges, or a comma separated set of column names (if input file is BIDS). Default is to use all columns.

Outputs:
^^^^^^^^
	Various NIFTI and text files with the correlation information.

Usage:
^^^^^^

.. argparse::
   :ref: rapidtide.workflows.ccorrica._get_parser
   :prog: ccorrica
   :func: _get_parser

   Debugging options : @skip
      skip debugging options



