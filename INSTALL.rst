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
