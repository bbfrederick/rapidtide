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
    -u rapidtide fredericklab/rapidtide_dev:latest \
    happy \
        /data/src/sub-HAPPYTEST.nii.gz \
        /data/src/sub-HAPPYTEST.json \
        /data/dst/sub-HAPPYTEST \
        --model model_revised \
        --mklthreads -1 
