polyfitim
---------

Description:
^^^^^^^^^^^^
    Fit a spatial template to a 3D or 4D NIFTI file.

Inputs:
^^^^^^^
    A 3D or 4D NIFTI file, with mask, to be fit
    A 3D or 4D template file, with mask, to fit to the data file (a spatial EV)

Outputs:
^^^^^^^^
    NIFTI files of the polynomial fit of the template and residuals to the data to the specified order
    Text files giving the coefficients of the fit, and the R values

Usage:
^^^^^^

.. argparse::
   :ref: rapidtide.workflows.polyfitim._get_parser
   :prog: polyfitim
   :func: _get_parser

   Debugging options : @skip
      skip debugging options


