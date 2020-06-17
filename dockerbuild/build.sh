#!/bin/bash

# ensure we're up to date
git pull

# bump version
VERSION=`cat VERSION`
echo "version: ${VERSION}"


docker build . -t fredericklab/rapidtide:${VERSION}-dev \
    --build-arg VERSION=${VERSION}-dev \
    --build-arg BUILD_DATE=`date +"%Y%m%dT%k%M%S"` \
    --build-arg VCS_REF=`git rev-parse HEAD`
docker push fredericklab/rapidtide:${VERSION}-dev
docker pull fredericklab/rapidtide:${VERSION}-dev
