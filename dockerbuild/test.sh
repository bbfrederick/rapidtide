#!/bin/bash

MYIPADDRESS=`ifconfig en0 | grep 'inet ' | awk '{print $2}'`

# allow network connections in Xquartz Security settings
xhost +

docker run \
    --network host\
    --mount type=bind,source=/Users/frederic/code/rapidtide/rapidtide/data/examples,destination=/data \
    -it \
    -e DISPLAY=${MYIPADDRESS}:0 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -u rapidtide fredericklab/rapidtide_dev:v2.0alpha2 \
    tidepool
