#!/bin/bash

MYIPADDRESS=`ifconfig en0 | grep 'inet ' | awk '{print $2}'`

# allow network connections in Xquartz Security settings
xhost +

docker run \
    --rm \
    --ipc host \
    --mount type=bind,source=/Users/frederic/code/rapidtide/rapidtide/data/examples,destination=/data \
    -it \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -u rapidtide fredericklab/rapidtide:latest \
    rapidtide \
        /data/src/sub-RAPIDTIDETEST.nii.gz \
        /data/dst/sub-RAPIDTIDETEST \
        --passes 3 \
        --nprocs 4 \
        --noglm

docker run \
    --rm \
    --ipc host \
    --mount type=bind,source=/Users/frederic/code/rapidtide/rapidtide/data/examples,destination=/data \
    -it \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -u rapidtide fredericklab/rapidtide:latest \
    happy \
        /data/src/sub-HAPPYTEST.nii.gz \
        /data/src/sub-HAPPYTEST.json \
        /data/dst/sub-HAPPYTEST \
        --model model_revised \
        --mklthreads -1 

docker run \
    --rm \
    --ipc host \
    --network host\
    --mount type=bind,source=/Users/frederic/code/rapidtide/rapidtide/data/examples,destination=/data \
    -it \
    -e DISPLAY=${MYIPADDRESS}:0 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -u rapidtide fredericklab/rapidtide:latest \
    tidepool --dataset /data/dst/sub-RAPIDTIDETEST_
