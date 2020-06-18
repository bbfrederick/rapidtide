#!/bin/bash

# ensure we're up to date
git pull

# bump version
VERSION=`cat VERSION`
echo "version: ${VERSION}"


docker build \
    --build-arg VERSION=${VERSION} \
    --build-arg BUILD_DATE=`date -u +"%Y-%m-%dT%H:%M:%SZ"` \
    --build-arg VCS_REF=`git rev-parse HEAD` \
    . -t fredericklab/rapidtide:${VERSION} 
docker build \
    --build-arg VERSION=${VERSION} \
    --build-arg BUILD_DATE=`date -u +"%Y-%m-%dT%H:%M:%SZ"` \
    --build-arg VCS_REF=`git rev-parse HEAD` \
    . -t fredericklab/rapidtide:latest
docker push fredericklab/rapidtide:${VERSION}
docker push fredericklab/rapidtide:latest
docker pull fredericklab/rapidtide:${VERSION}
docker pull fredericklab/rapidtide:latest
