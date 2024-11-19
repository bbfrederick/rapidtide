#!/bin/bash

VERSION=latest

docker pull fredericklab/rapidtide:${VERSION}
docker run -it fredericklab/rapidtide:${VERSION} bash
