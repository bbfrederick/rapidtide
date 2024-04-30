resamplenifti
-------------

Description:
^^^^^^^^^^^^
	Takes an input NIFTI file at some TR and outputs a NIFTI file resampled to the specified TR.  Downsampling is antialiased unless disabled.


Inputs:
^^^^^^^
    The 4D NIFTI file to resample.

Outputs:
^^^^^^^^
    A 4D NIFTI file resampled to the new TR.
Usage:
^^^^^^

.. argparse::
   :ref: rapidtide.workflows.resamplenifti._get_parser
   :prog: resamplenifti
   :func: _get_parser

   Debugging options : @skip
      skip debugging options


