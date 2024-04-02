atlastool
---------

Description:
^^^^^^^^^^^^
    A utility to manipulate nifti atlas files

Inputs:
^^^^^^^
    A template file which is either a 3D NIFTI with different regions specified by integer values, or a 4D NIFTI with nonzero values indicating that a voxel is in the region indicated by the 4th dimension.

Outputs:
^^^^^^^^
    A new NIFTI template file which has been processed using the options specified

Usage:
^^^^^^

.. argparse::
   :ref: rapidtide.workflows.atlastool._get_parser
   :prog: atlastool
   :func: _get_parser

   Debugging options : @skip
      skip debugging options


