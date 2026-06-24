#!/bin/bash

DATADIR=/Users/frederic/code/rapidtide

MYIPADDRESS=$(ifconfig en0 | grep 'inet ' | awk '{print $2}')
VERSION=latest

# allow network connections in Xquartz Security settings
xhost +

# make sure the test data is installed
#pushd ${DATADIR}/rapidtide/data/examples/src;./installtestdatahere;popd

#for TESTVERSION in v3.1.4 v3.1.5 v3.1.6 v3.1.7 v3.1.8 v3.1.9
for TESTVERSION in v3.1.8 v3.1.9
do
    docker pull fredericklab/rapidtide:${TESTVERSION}
    docker run \
        --rm \
        --ipc host \
        --mount type=bind,source=${DATADIR}/rapidtide/data/examples,destination=/data \
        -it \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -u rapidtide fredericklab/rapidtide:${TESTVERSION} \
        /cloud/mount-and-run rapidtide \
            /data/src/sub-RAPIDTIDETEST.nii.gz \
            /data/dst/sub-RAPIDTIDETEST_${TESTVERSION} \
            --nprocs -1 \
            --outputlevel max
done
