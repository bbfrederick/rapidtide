happy2std
---------

Description:
^^^^^^^^^^^^

	This is a utility for registering happy output maps
	to standard coordinates.  NB: this
	will only work if you have a working FSL installation.

Inputs:
^^^^^^^

	inputfileroot - the root name of the input NIFTI files (up and including the ‘desc’ but not the underscore).
        
	outputdir - The location for the output files

	featdirectory - Either a feat-like directory (x.feat or x.ica) or an fmriprep derivativesanat directory where the information needed for registration to standard space can be found


Outputs:
^^^^^^^^

	The happy maps, transformed to MNI152 space

Usage:
^^^^^^

.. argparse::
    :ref: rapidtide.workflows.happy2std._get_parser
    :prog: happy2std
    :func: _get_parser

    Debugging options : @skip
        skip debugging options

