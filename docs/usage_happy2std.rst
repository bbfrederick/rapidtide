happy2std
---------

Description:
^^^^^^^^^^^^

	This is a utility for registering happy output maps
	to standard coordinates.  NB: this
	will only work if you have a working FSL or ANTs installation.

Inputs:
^^^^^^^
	Text files containing time series data

Outputs:
^^^^^^^^
	None

Usage:
^^^^^^

.. argparse::
    :ref: rapidtide.workflows.happy2std._get_parser
    :prog: happy2std
    :func: _get_parser

    Debugging options : @skip
        skip debugging options

