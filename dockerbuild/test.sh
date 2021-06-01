#!/bin/bash

VERSION=`cat VERSION | sed 's/+/ /g' | sed 's/v//g' | awk '{print $1}'`
MYIPADDRESS=`ifconfig en0 | grep 'inet ' | awk '{print $2}'`

# allow network connections in Xquartz Security settings
xhost +

docker run \
    --network host\
    --volume=/Users/frederic:/data \
    -it \
    -e DISPLAY=${MYIPADDRESS}:0 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -u rapidtide fredericklab/rapidtide:${VERSION} \
    tidepool
