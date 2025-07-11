#!/bin/bash

DATADIR=/Users/frederic/code/rapidtide

MYIPADDRESS=$(ifconfig en0 | grep 'inet ' | awk '{print $2}')
VERSION=latest-release

# allow network connections in Xquartz Security settings
xhost +

# make sure the test data is installed
#pushd ${DATADIR}/rapidtide/data/examples/src;./installtestdatahere;popd

docker pull fredericklab/rapidtide:${VERSION}
#docker run \
#    --rm \
#    --ipc host \
#    --mount type=bind,source=${DATADIR}/rapidtide/data/examples,destination=/data \
#    -it \
#    -v /tmp/.X11-unix:/tmp/.X11-unix \
#    -u rapidtide fredericklab/rapidtide:${VERSION} \
#    /cloud/mount-and-run rapidtide \
#        /data/src/sub-RAPIDTIDETEST.nii.gz \
#        /data/dst/sub-RAPIDTIDETEST \
#        --passes 3 \
#        --nprocs 4 \
#        --nodenoise


#docker run \
#    --rm \
#    --ipc host \
#    --mount type=bind,source=${DATADIR}/rapidtide/data/examples,destination=/data \
#    -it \
#    -v /tmp/.X11-unix:/tmp/.X11-unix \
#    -u rapidtide fredericklab/rapidtide:${VERSION} \
#    rapidtide \
#        /data/src/sub-RAPIDTIDETEST.nii.gz \
#        /data/dst/sub-RAPIDTIDETEST \
#        --passes 3 \
#        --nprocs -1 \
#        --nodenoise


#docker run \
    #--rm \
    #--ipc host \
    #--mount type=bind,source=${DATADIR}/rapidtide/data/examples,destination=/data \
    #-it \
    #-v /tmp/.X11-unix:/tmp/.X11-unix \
    #-u rapidtide fredericklab/rapidtide:${VERSION} \
    #happy \
        #/data/src/sub-HAPPYTEST.nii.gz \
        #/data/src/sub-HAPPYTEST.json \
        #/data/dst/sub-HAPPYTEST \
        #--model model_revised_tf2 \
        #--mklthreads -1 
        #--nprocs -1 


docker run \
    --network host \
    --mount type=bind,source=${DATADIR}/rapidtide/data/examples,destination=/data \
    -it \
    -e DISPLAY=${MYIPADDRESS}:0 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -u rapidtide fredericklab/rapidtide:${VERSION} \
    tidepool --dataset /data/dst/sub-RAPIDTIDETEST_
