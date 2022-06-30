#!/bin/bash

# ensure we're up to date
git pull

# bump version
VERSION=`cat VERSION | sed 's/+/ /g' | sed 's/v//g' | awk '{print $1}'`
BUILD_DATE=`cat DATE`
echo "version: ${VERSION}"
echo "date:    ${BUILD_DATE}"


docker build \
    --build-arg VERSION=${VERSION} \
    --build-arg BUILD_DATE=${BUILD_DATE} \
    --build-arg VCS_REF=`git rev-parse HEAD` \
    . -t fredericklab/rapidtide:${VERSION} 
#docker push fredericklab/rapidtide:${VERSION}
#docker pull fredericklab/rapidtide:${VERSION}
