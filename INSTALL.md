Prerequisites
-------------

The processing programs in rapidtide require the following to be installed first:

* Python 2.x or 3.x
* numpy
* scipy
* scikit-learn
* nibabel
* matplotlib
* statsmodels

The following optional dependencies will be used if present
* numba for faster performance

If you want to use tidepool for image display, you will also need to install the following:
* pyqt4
* pyqtgraph

Installation
------------

Once you have installed the prerequisites, cd into the package directory, and type the following:
```bash
python setup.py install
```
to install all of the tools in the package.  You should be able to run them from the command line
then (after rehashing).  NOTE: I haven't figured out how to install pyqt4 automatically yet, so you
have install it and pyqtgraph manually.  If you do that, tidepool should work, otherwise it won't.

Updating
--------

If you've previously installed rapidtide and want to update, cd into the package directory and do a git pull first:
```bash
git pull
python setup.py install
```


Usage
------------

Execute any of the commands to run rapidtide on the sample data:

```bash
# run rapidtide2 to perform dynamic global signal regression (dGSR) on an fMRI file[1]:
rapidtide2 examples/src/fmri.nii.gz examples/dst/dgsr -L -r -15,15 --refinepasses=3

# run rapidtide2 to perform static global signal regression (sGSR) on an fMRI file[1] 
# (this is just global mean regression):
rapidtide2 fmri.nii.gz sgsr -L -Z 0.0

# b) run tidepool to look at the results from a):
tidepool
(then select the file examples/dst/dgsr_lagtimes.nii.gz to load the maps)

# c) look at the refined regressors produced during dGSR:
# (the "dgsr_pass1" regressor is the initial global mean before refinement)
showtc examples/dst/dgsr_reference_fmrires_pass[123].txt
```

References
----------
1) Erdoğan S, Tong Y, Hocke L, Lindsey K, Frederick B (2016). Correcting
	resting state fMRI-BOLD signals for blood arrival time enhances
	functional connectivity analysis. Front. Hum. Neurosci., 28 June 2016
	| http://dx.doi.org/10.3389/fnhum.2016.00311
