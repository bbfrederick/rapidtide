atlasaverage
------------

Description:
^^^^^^^^^^^^
	Average data within atlas regions.

Inputs:
^^^^^^^
	datafile - The name of the 3 or 4D nifti file with the data to be averaged over atlas regions.
	templatefile - The name of the atlas region NIFTI file
	outputroot - The root name of the output files.

Outputs:
^^^^^^^^
	A csv file containing the summary metrics for each region in the atlas.

Usage:
^^^^^^

.. argparse::
   :ref: rapidtide.workflows.atlasaverage._get_parser
   :prog: atlasaverage
   :func: _get_parser

   Debugging options : @skip
      skip debugging options



