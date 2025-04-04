Bare metal installation
-----------------------
This gives you the maximum flexibility if you want to look at the code and/or modify things.  It may seem a little daunting at first,
but it's not that bad.  And if you want a simpler path, skip down to the Docker installation instructions

Required dependencies
`````````````````````

The processing programs in rapidtide require the following to be
installed:

-  Python >= 3.9
-  numpy
-  scipy
-  pandas
-  pyarrow
-  scikit-image
-  scikit-learn
-  nibabel
-  nilearn
-  matplotlib >= 3.3.0
-  pyqt6-sip
-  pyqt6
-  pyqtgraph >= 0.13.4
-  statsmodels
-  tqdm

Optional dependencies
`````````````````````

The following optional dependencies will be used if present:

-  numba (for faster performance)
-  pyfftw (faster performance)
-  mkl and mkl-service (faster performance on Intel CPUs)
-  memory_profiler (allows for more detailed memory usage stats)

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

If you are on linux (or Windows) WITHOUT an Nvidia GPU, or on a Mac, install:

- keras
- tensorflow (and make sure it doesn’t sneakily try to install the GPU version - that won’t work)


Installing Python
`````````````````

The simplest way BY FAR to get this all done is to use Anaconda python
from Continuum Analytics. It’s a free, curated scientific Python
distribution that is easy to maintain and takes a lot of headaches out
of maintaining a distribution. It also already comes with many of the
dependencies for rapidtide installed by default. You can get it here:
https://www.continuum.io. Rapidtide works with Python 3.9 or greater.

After installing Anaconda python, install the remaining dependencies
(including some good optional ones:

::

   conda install nibabel "pyqtgraph>=0.12.0" pyfftw


For the deep learning filter in happy, also do:

::

   conda install keras tensorflow-gpu


(for Linux or Windows WITH Nvidia GPU)

or:

::

   conda install keras tensorflow


(for a Mac, or Linux or Windows WITHOUT Nvidia GPU)


Done.

Installing the rapidtide package
````````````````````````````````

Once you have installed the prerequisites, cd into the package
directory, and type the following:

::

   python install .


to install all of the tools in the package. You should be able to run
them from the command line then (after rehashing).

Updating
````````

If you’ve previously installed rapidtide and want to update, cd into the
package directory and do a git pull first:

::

   git pull
   python setup.py install


Docker installation
-------------------
As of 1.9.0, there is now a Docker container with a full rapidtide installation.  To use this,
first make sure you have docker installed and properly configured, then run the following:
::

    docker pull fredericklab/rapidtide:latest-release


This will download the Docker container from dockerhub.
It's around a 3GB download, so it may take some time, but it caches the file locally, so you won't have to do this again
unless the container updates.  To use a particular version, replace "latest-release" with the version of the
container you want.

If you like to live on the edge, just use:
::

    docker pull fredericklab/rapidtide:latest


This will use the most recent version on dockerhub, which is built automatically on every git push.
NOTE: I don't advise doing this unless you're helping debug something - 
there's no guarantee that "latest" is functional at any given time.

Now that the file is downloaded, you can run and rapidtide command in the Docker container.  For example, to run a simple
rapidtide analysis, you would use the following command (you can do this all in one step - it will just integrate the
first pull into the run time if the version you request hasn't already been downloaded).

Docker runs completely in its own self-contained environment.  If you want to be able to interact with disks outside of
container, you map the volume to a mount point in the container using the --volume=EXTERNALDIR:MOUNTPOINT[,ANOTHERDIR:ANOTHERMOUNTPOINT]
option to docker.
::

    docker run \
        --mount type=bind,source=INPUTDIRECTORY,destination=/data_in \
        --mount type=bind,source=OUTPUTDIRECTORY,destination=/data_out \
        fredericklab/rapidtide:latest-release \
            rapidtide \
                /data_in/YOURNIFTIFILE.nii.gz \
                /data_out/outputname \
                --filterband lfo \
                --searchrange -15 15 \
                --passes 3

NOTE: If you want to run this on the test data, like the examples above for the bare metal installation, the example data is
in the Docker container in the /src/rapidtide/rapidtide/data/examples/src directory.  So to run the first example, you could just do:
::

    docker run \
        --mount type=bind,source=OUTPUTDIRECTORY,destination=/data_out \
        fredericklab/rapidtide:latest-release \
            rapidtide \
                /src/rapidtide/rapidtide/data/examples/src/sub-RAPIDTIDETEST.nii.gz \
                /data_out/dgsr \
                --filterband lfo \
                --searchrange -15 15 \
                --passes 3


You can replace the ``rapidtide blah blah blah`` command with any program in the package - after the fredericklab/rapidtide:latest-release,
just specify the command and arguments as you usually would.  If you're running a program that displays anything,
you'll have to add a few extra arguments to the docker call.  Docker is a little weird about X forwarding - the easiest thing to
do is find the IP address of the machine you're running on (lets call it MYIPADDRESS), and do the following:

::

    xhost +

This disables X11 security - this is almost certainly not the best thing to do, but I don't have a better solution
at this time, and it works.

If you're on a Mac using Xquartz, prior to this you'll also have to do three more things.

1) In Xquartz, go into the security preferences, and make sure "Allow connections from network hosts" is checked.
2) Tell Xquartz to listen for TCP connections (this is not the default).  Go to a terminal window and type:

::

    defaults write org.macosforge.xquartz.X11 nolisten_tcp 0

3) Log out and log back in again (you only need to do this once - it will stay that way until you change it.)

Then the following command will work (you can replace 'tidepool' with any of the rapidtide commands that put up windows):

::

    docker run \
        --network host\
        --volume=INPUTDIRECTORY:/data_in,OUTPUTDIRECTORY:/data_out \
        -it \
        -e DISPLAY=MYIPADDRESS:0 \
        -u rapidtide \
        fredericklab/rapidtide:latest-release \
            tidepool


Singularity installation
------------------------

Many times you can't use Docker, because of security concerns.  Singularity, from LBL, offers containerized computing
that runs entirely in user space, so the amount of mischief you can get up to is significantly less.  Singularity
containers can be created from Docker containers as follows (stealing from the fMRIprep documentation):
::

    singularity build /my_images/rapidtide.simg docker://fredericklab/rapidtide:latest-release


Running the container is similar to Docker.  The "-B" option is used to bind filesystems to mountpoints in the container.
For example, to run the simple rapidtide2x analysis above, type the following:
::

    singularity run \
        --cleanenv \
        -B INPUTDIRECTORY:/data_in,OUTPUTDIRECTORY:/data_out \
        rapidtide.simg \
            rapidtide \
                /data_in/YOURNIFTIFILE.nii.gz \
                /data_out/outputname \
                --filterband lfo \
                --searchrange -15 15 \
                --passes 3


To run a GUI application, you need to disable X security on your host (see comment about this above):

::

    xhost +

then set the display variable to import to the container:
::

    setenv SINGULARITY_DISPLAY MYIPADDRESS:0   (if you are using csh)

or

::

    export SINGULARITY_DISPLAY="MYIPADDRESS:0" (if you are using sh/bash/zsh)

then just run the gui command with the command given above.


References
----------
1) Erdoğan S, Tong Y, Hocke L, Lindsey K, Frederick B
(2016). Correcting resting state fMRI-BOLD signals for blood arrival
time enhances functional connectivity analysis. Front. Hum. Neurosci.,
28 June 2016 \| http://dx.doi.org/10.3389/fnhum.2016.00311
