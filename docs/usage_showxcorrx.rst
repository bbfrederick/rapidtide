showxcorrx
----------

Description:
^^^^^^^^^^^^

	This is the newest, most avant-garde version of showxcorr.  Because
	it's an x file, it's more fluid and I don't guarantee that it will
	keep a stable interface (or even work at any given time).  But every
	time I add something new, it goes here.  The goal is eventually to
	make this the "real" version. Unlike rapidtide, however, I've let it
	drift quite a bit without syncing it because some people here
	actually use showxcorr and I don't want to disrupt workflows...

Inputs:
^^^^^^^

	showxcorrx requires two text files containing timecourses with the same
	sample rate, one timepoint per line, which are to be correlated, and the
	sample rate.

Outputs:
^^^^^^^^

	showxcorrx  outputs everything to standard out, including the
	Pearson correlation, the maximum cross correlation, the time of
	maximum cross correlation, and estimates of the significance levels
	(if specified).  There are no output files.

Usage:
^^^^^^

.. argparse::
   :ref: rapidtide.workflows.showxcorrx._get_parser
   :prog: showxcorrx
   :func: _get_parser

