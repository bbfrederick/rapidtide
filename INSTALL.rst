Bare metal installation
-----------------------
This gives you the maximum flexibility if you want to look at the code and/or modify things.
If you want a simpler path, skip down to the Docker/singularity installation instructions.

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
-  tensorflow
-  tf-keras

Optional dependencies
`````````````````````

The following optional dependencies will be used if present:

-  numba (for faster performance)
-  pyfftw (faster performance)
-  mkl and mkl-service (faster performance on Intel CPUs)


Installing with pip from pypi
`````````````````````````````

This assumes you have a working python installation, and if you use virtual
environments, you've set one up and activated it.  Once that's all ready, you can simply install
with pip (this will pull everything it needs from pypi.org, the Python Package Index):

::

    pip install rapidtide

This will install rapidtide and all dependencies direct from pypi.  You should be
all set.

Done.

Installing from Github
``````````````````````

You can take this route if you want to poke around in the code, and perhaps modify it.
As above, this assumes you have python installed and if you are using a virtual environment,
you've activated it.  Go to the directory where you want to install the rapidtide
source code, and type:

::

   git clone https://github.com/bbfrederick/rapidtide.git
   cd rapidtide
   refresh


to install all of the tools in the package. You should be able to run
them from the command line then, and all the code is in your current directory (and
subdirectories).

If you’ve made edits to the code, or want to sync up with the current version on Github,
cd into the
package directory and type ``refresh``:

::

   refresh


This will uninstall the current version, sync up to github, and reinstall
the package (assuming you don't have any merge conflicts with the version on Github,
which you have to resolve).  This will keep you on the bleeding edge of development,
if that's your thing.


Docker installation
-------------------
As of 1.9.0, there is now a Docker container with a full rapidtide installation.  To use this,
first make sure you have docker installed and properly configured, then run the following:
::

    docker pull fredericklab/rapidtide:latest-release


This will download the Docker container from dockerhub.
It's around a 3GB download, so it may take some time, but it caches the file locally,
so you won't have to do this again
unless the container updates.  To use a particular version,
replace "latest-release" with the version of the
container you want.  Please note that I generate amd64 and arm64 versions of the container,
so it runs natively (the amd64 container will run on an Apple Silicon Mac in Docker, but,
Docker will pull the arm64 version unless you tell it not to).

If you like to live on the edge, just use:
::

    docker pull fredericklab/rapidtide:latest


This will use the most recent version on dockerhub, which is built automatically on every git push.
NOTE: I don't advise doing this unless you're helping debug something - 
there's no guarantee that "latest" is functional at any given time.

Now that the file is downloaded, you can run any rapidtide command in the Docker container.  For example, to run a simple
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
