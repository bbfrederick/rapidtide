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
* pyfftw, also for faster performance

If you want to use tidepool for image display, you will also need to install the following:
* pyqt4 or pyqt5
* pyqtgraph

The simplest way BY FAR to get this all done is to use Anaconda python from Continuum Analytics.  It's a free, curated scientific Python distribution that is easy to maintain and takes a lot of headaches out of maintaining a distribution.  You can get it here: https://www.continuum.io.  Rapidtide works with Python 2 or 3.  If you are new to Python, you should probably just start at 3.

After installing Anaconda python, you just do the following:
Add the condaforge channel (this is an externally maintained channel that is somewhat ahead of Continuum's distribution and has a number of packages that Continuum does not include):
```bash
conda config --add channels conda-forge
```

Then install the dependencies (including some good optional ones:
```bash
conda install nibabel pyqtgraph pyfftw
```

Done.


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
rapidtide2 rapidtide/data/examples/examples/src/fmri.nii.gz rapidtide/data/examples/examples/dst/dgsr -L -r -15,15 --refinepasses=3

# run rapidtide2 to perform static global signal regression (sGSR) on an fMRI file[1] 
# (this is just global mean regression):
rapidtide2 rapidtide/data/examples/srd/fmri.nii.gz rapidtide/data/examples/dst/sgsr -L -Z 0.0

# b) run tidepool to look at the results from a):
tidepool
(then select the file rapidtide/data/examples/dst/dgsr_lagtimes.nii.gz to load the maps)

# c) look at the refined regressors produced during dGSR:
# (the "dgsr_pass1" regressor is the initial global mean before refinement)
showtc rapidtide/data/examples/dst/dgsr_reference_fmrires_pass[123].txt
```

References
----------
1) Erdoğan S, Tong Y, Hocke L, Lindsey K, Frederick B (2016). Correcting
	resting state fMRI-BOLD signals for blood arrival time enhances
	functional connectivity analysis. Front. Hum. Neurosci., 28 June 2016
	| http://dx.doi.org/10.3389/fnhum.2016.00311
