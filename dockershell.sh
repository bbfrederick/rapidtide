#!/bin/bash

DATADIR=/Users/frederic/code/rapidtide/rapidtide/data/examples

VERSION=latest

docker pull fredericklab/rapidtide:${VERSION}
docker run \
    -it fredericklab/rapidtide:${VERSION} \
    bash
