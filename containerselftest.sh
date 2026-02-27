#!/bin/bash

DATADIR=/Users/frederic/code/rapidtide

MYIPADDRESS=$(ifconfig en0 | grep 'inet ' | awk '{print $2}')
VERSION=localtest

# allow network connections in Xquartz Security settings
xhost +

docker pull fredericklab/rapidtide:${VERSION}

docker run \
    --network host \
    --mount type=bind,source=/Users/frederic/Downloads,destination=/Users/frederic/Downloads \
    -it \
    -e DISPLAY=${MYIPADDRESS}:0 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -u rapidtide fredericklab/rapidtide:${VERSION} \
    /src/rapidtide/rapidtide/tests/runcontainertest
