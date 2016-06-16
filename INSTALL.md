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

If you want to use tidepool for image display, you will also need to install the following:
* pyqt4
* pyqtgraph

Installation
------------

Once you have installed the prerequisites, add the delaytools installation directory to your path and execute any of the commands:

```bash
# a) run rapidtide2 to perform dynamic global signal regression on an fMRI file[1]:
rapidtide2 inputfmrifile.nii.gz outputrootname -L --refinepasses=3

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

References
----------
1) Erdoğan S, Tong Y, Hocke L, Lindsey K, Frederick B (2016). Correcting resting state fMRI-BOLD signals for blood arrival time enhances functional connectivity analysis. Front Hum Neurosci. 2016, In Press.
