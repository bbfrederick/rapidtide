mergequality
------------

Description:
^^^^^^^^^^^^
    Merge rapidtide quality check data from several runs.

Inputs:
^^^^^^^
    One or more json files containing rapidtide dataset quality metrics (outputs of ``runqualitycheck`` or other tools).

Outputs:
^^^^^^^^
    A csv file with one row per input file, and one column per quality metric, and one histogram file per metric showing the distribution of values over all subjects.

Usage:
^^^^^^

.. argparse::
   :ref: rapidtide.workflows.mergequality._get_parser
   :prog: mergequality
   :func: _get_parser

   Debugging options : @skip
      skip debugging options


