rapidtide2std
-------------

Description:
^^^^^^^^^^^^

	This is a utility for registering rapidtide output maps
	to standard coordinates.  It's usually much faster to run rapidtide
	in native space then transform afterwards to MNI152 space.  NB: this
	will only work if you have a working FSL installation.

Inputs:
^^^^^^^

Outputs:
^^^^^^^^
	New versions of the rapidtide output maps, registered to either MNI152 space or to the hires anatomic images for the subject.  
	All maps are named with the specified root name with '_std' appended.

Usage:
^^^^^^

.. argparse::
   :ref: rapidtide.workflows.rapidtide2std._get_parser
   :prog: rapidtide2std
   :func: _get_parser

   Debugging options : @skip
      skip debugging options



