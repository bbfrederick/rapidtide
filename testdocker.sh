#!/bin/bash

MYIPADDRESS=`ifconfig en0 | grep 'inet ' | awk '{print $2}'`
VERSION=latest

# allow network connections in Xquartz Security settings
xhost +

docker pull fredericklab/rapidtide:${VERSION}
docker run \
    --rm \
    --ipc host \
    --mount type=bind,source=/Users/frederic/code/rapidtide/rapidtide/data/examples,destination=/data \
    -it \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -u rapidtide fredericklab/rapidtide:${VERSION} \
    /cloud/mount-and-run rapidtide \
        /data/src/sub-RAPIDTIDETEST.nii.gz \
        /data/dst/sub-RAPIDTIDETEST \
        --passes 3 \
        --nprocs 4 \
        --nodenoise

#docker run \
#    --rm \
#    --ipc host \
#    --mount type=bind,source=/Users/frederic/code/rapidtide/rapidtide/data/examples,destination=/data \
#    -it \
#    -v /tmp/.X11-unix:/tmp/.X11-unix \
#    -u rapidtide fredericklab/rapidtide:${VERSION} \
#    rapidtide \
#        /data/src/sub-RAPIDTIDETEST.nii.gz \
#        /data/dst/sub-RAPIDTIDETEST_disabledockermemfix \
#        --disabledockermemfix \
#        --passes 3 \
#        --nprocs 4 \
#        --nodenoise


#docker run \
#    --rm \
#    --ipc host \
#    --mount type=bind,source=/Users/frederic/code/rapidtide/rapidtide/data/examples,destination=/data \
#    -it \
#    -v /tmp/.X11-unix:/tmp/.X11-unix \
#    -u rapidtide fredericklab/rapidtide:${VERSION} \
#    happy \
#        /data/src/sub-HAPPYTEST.nii.gz \
#        /data/src/sub-HAPPYTEST.json \
#        /data/dst/sub-HAPPYTEST \
#        --model model_revised \
#        --mklthreads -1 

#docker run \
#    --rm \
#    --ipc host \
#    --mount type=bind,source=/Users/frederic/code/rapidtide/rapidtide/data/examples,destination=/data \
#    -it \
#    -v /tmp/.X11-unix:/tmp/.X11-unix \
#    -u rapidtide fredericklab/rapidtide:${VERSION} \
#    gmscalc \
#        /data/src/sub-RAPIDTIDETEST.nii.gz \
#        /data/dst/sub-RAPIDTIDETEST_GMS \
#        --dmask /data/src/sub-RAPIDTIDETEST_brainmask.nii.gz

docker run \
    --network host\
    --mount type=bind,source=/Users/frederic,destination=/data \
    -it \
    -e DISPLAY=${MYIPADDRESS}:0 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -u rapidtide fredericklab/rapidtide:${VERSION} \
    tidepool --dataset /data/code/rapidtide/rapidtide/data/examples/dst/sub-RAPIDTIDETEST_
