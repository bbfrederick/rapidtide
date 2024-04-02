histnifti
---------

Description:
^^^^^^^^^^^^
	A command line tool to generate a histogram for a nifti file


Inputs:
^^^^^^^
	A nifti file

Outputs:
^^^^^^^^
	A text file containing the histogram information

None

Usage:
^^^^^^

.. argparse::
   :ref: rapidtide.workflows.histnifti._get_parser
   :prog: histnifti
   :func: _get_parser

   Debugging options : @skip
      skip debugging options


