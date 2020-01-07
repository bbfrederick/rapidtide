Installation
============

Required dependencies
---------------------

The processing programs in rapidtide require the following to be
installed first:

-  Python 3.x (I no longer support or test running in 2.x, but it does
   still work for the time being. But I use dependencies like nibabel
   that have dropped 2.x support, so this is not going to last long.)
-  numpy>=1.16
-  scipy
-  pandas
-  scikit-learn
-  nibabel
-  matplotlib
-  statsmodels

If you want to use tidepool for image display, you will also need to
install the following:

-  pyqt5
-  pyqtgraph

Optional dependencies
---------------------

The following optional dependencies will be used if present:

-  numba (for faster performance)
-  pyfftw (also for faster performance)
-  mkl and mkl-service (again, faster performance)

If you want to use the deep learning filter in happy, you’ll need Keras
and some sort of backend. If you want to be able to train filters,
you’ll probably want GPU support. This is currently an annoying,
non-trivial thing to set up, especially on a Mac, which is where I do
things, because Apple and Nvidia aren’t friends at the moment. If you
are on a linux box (or maybe Windows - haven’t tested), WITH an Nvidia
GPU, install:

-  keras
-  tensorflow-gpu (This assumes you have all the necessary CUDA
   libraries. Making this all work together properly is a version
   dependent moving target. Ask The Google the best way to do it this
   week - anything I say here will probably be obsolete by the time you
   read this.)

If you are on linux (or Windows) WITHOUT an Nvidia GPU, install: \*
keras \* tensorflow (and make sure it doesn’t sneakily try to install
the GPU version - that won’t work)

If you are on a Mac, you almost certainly have a non-Nvidia GPU, so you
should install

-  plaidml-keras (it installs Keras and uses PlaidML as the backend
   rather than tensorflow). You will have to run a configuration step in
   plaidML to tell it what GPU to use and how. I use the “metal” option
   with the AMD GPU in my laptop - that seems to be the most stable.
   Currently, I think you have you have to do this from pypi - I haven’t
   seen a conda version of this.

Installing Python
-----------------

The simplest way BY FAR to get this all done is to use Anaconda python
from Continuum Analytics. It’s a free, curated scientific Python
distribution that is easy to maintain and takes a lot of headaches out
of maintaining a distribution. It also already comes with many of the
dependancies for rapidtide installed by default. You can get it here:
https://www.continuum.io. Rapidtide works with Python 2 or 3. If you are
new to Python, you should probably just start at 3.

After installing Anaconda python, install the remaining dependencies
(including some good optional ones:

::

   conda install nibabel pyqtgraph pyfftw 

For the deep learning filter in happy, also do:

::

   conda install keras tensorflow-gpu

(for Linux or Windows WITH Nvidia GPU)

or:

::

   conda install keras tensorflow

(for Linux or Windows WITHOUT Nvidia GPU)

or

::

   pip install plaidml-keras

(on a Mac)

Done.

Installing rapidtide
--------------------

Once you have installed the prerequisites, cd into the package
directory, and type the following:

::

   python setup.py install

to install all of the tools in the package. You should be able to run
them from the command line then (after rehashing). NOTE: I haven’t
figured out how to install pyqt4 automatically yet, so you have install
it and pyqtgraph manually. If you do that, tidepool should work,
otherwise it won’t.

Updating
--------

If you’ve previously installed rapidtide and want to update, cd into the
package directory and do a git pull first:

::

   git pull
   python setup.py install

Usage
-----
Execute any of the commands to run rapidtide on the sample data:

a) Run rapidtide2 to perform dynamic global signal regression (dGSR) on an fMRI file[1]:
::

    rapidtide2 rapidtide/data/examples/src/fmri.nii.gz rapidtide/data/examples/dst/dgsr -L -r -15,15 --refinepasses=3


b) Run rapidtide2 to perform static global signal regression (sGSR) on an fMRI file[1] (this is just global mean regression):
::

    rapidtide2 rapidtide/data/examples/src/fmri.nii.gz rapidtide/data/examples/dst/sgsr -L -Z 0.0


c) run tidepool to look at the results from a):
::

    tidepool


(then select the file rapidtide/data/examples/dst/dgsr_lagtimes.nii.gz to load all the interesting maps and timecourses):

d) Look at the refined regressors produced during dGSR: (the "dgsr_pass1" regressor is the initial global mean before refinement) in their own window.
::

    showtc rapidtide/data/examples/dst/dgsr_reference_fmrires_pass[123].txt


#References 1) Erdoğan S, Tong Y, Hocke L, Lindsey K, Frederick B
(2016). Correcting resting state fMRI-BOLD signals for blood arrival
time enhances functional connectivity analysis. Front. Hum. Neurosci.,
28 June 2016 \| http://dx.doi.org/10.3389/fnhum.2016.00311
