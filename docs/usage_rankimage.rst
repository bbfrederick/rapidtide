rankimage
---------

Description:
^^^^^^^^^^^^
    Convert a 3D or 4D NIFTI file into a map of each voxel's rank in the map values.

Inputs:
^^^^^^^
    A 3D or 4D NIFTI file.

Outputs:
^^^^^^^^
    A 3D or 4D NIFTI file of voxel ranks scaled from 0-100.  If 4D, each timepoint is considered separately.

Usage:
^^^^^^

.. argparse::
   :ref: rapidtide.workflows.rankimage._get_parser
   :prog: rankimage
   :func: _get_parser

   Debugging options : @skip
      skip debugging options


