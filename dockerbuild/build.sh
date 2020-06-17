#!/bin/bash

# ensure we're up to date
git pull

# bump version
VERSION=`cat VERSION`
echo "version: ${VERSION}"


docker build . -t fredericklab/rapidtide:${VERSION} \
    --build-arg VERSION=$version \
    --build-arg BUILD_DATE=`date +"%Y%m%dT%k%M%S"` \
    --build-arg VCS_REF=`git rev-parse HEAD`
docker build . -t fredericklab/rapidtide:latest \
    --build-arg VERSION=$version \
    --build-arg BUILD_DATE=`date +"%Y%m%dT%k%M%S"` \
    --build-arg VCS_REF=`git rev-parse HEAD`
docker push fredericklab/rapidtide:${VERSION}
docker push fredericklab/rapidtide:latest
docker pull fredericklab/rapidtide:${VERSION}
docker pull fredericklab/rapidtide:latest
