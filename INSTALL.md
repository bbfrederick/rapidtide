Prerequisites
-------------

delaytools requires the following to be installed first:

* Python 2.x or 3.x
* numpy
* scipy
* sklearn
* pyqtgraph
* pyqt
* nibabel

The following optional dependencies will be used if present
* numba for faster performance

Installation
------------

Once you have installed the prerequisites and downloaded cudamat, switch to the
cudamat directory and run either of the following commands to install it:

```bash
# a) Install for your user:
python setup.py install --user
# b) Install for your user, but with pip:
pip install --user .
# c) Install system-wide:
sudo python setup.py install
# d) Install system-wide, but with pip:
sudo pip install .
```

Testing
-------

To test your setup, run the included unit tests and optionally the benchmark:

```bash
cd example
# Run tests
nosetests
```
