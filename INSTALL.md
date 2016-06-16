Prerequisites
-------------

The processing programs in delaytools require the following to be installed first:

* Python 2.x or 3.x
* numpy
* scipy
* sklearn
* nibabel

The following optional dependencies will be used if present
* numba for faster performance

If you want to use tidepool for image display, you will alto need to install the following:
* pyqt4
* pyqtgraph

Installation
------------

Once you have installed the prerequisites, add the delaytools installation directory to your path and execute any of the commands:

```bash
# a) run rapidtide2 to perform dynamic global mean regression on an fMRI file:
rapidtide2 inputfmrifile.nii.gz outputrootname -L --refinepasses=

# b) run tidepool to look at the results from a):
tidepool
(then select the file outputrootname_lagtimes.nii.gz to load the maps)

# c) look at the refined regressors produced during dGSR:
showtc outputrootname_reference_fmrires_pass[123].txt
```

Testing
-------

To test your setup, run the included unit tests and optionally the benchmark:

```bash
cd testdata
# Run tests
under construction
```
