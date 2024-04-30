showtc
------

Description:
^^^^^^^^^^^^
	A very simple command line utility that takes a text file
	and plots the data in it in a matplotlib window.  That's it.  A
	good tool for quickly seeing what's in a file.  Has some options
	to make the plot prettier.

Inputs:
^^^^^^^
	Text files containing time series data

Outputs:
^^^^^^^^
	None

Usage:
^^^^^^

.. argparse::
   :ref: rapidtide.workflows.showtc._get_parser
   :prog: showtc
   :func: _get_parser

   Debugging options : @skip
      skip debugging options



