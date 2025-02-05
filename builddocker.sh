#!/bin/bash

set -ex

# SET THE FOLLOWING VARIABLES
# docker hub username
USERNAME=fredericklab
# image name
IMAGE=rapidtide

# ensure we're up to date
git pull

# bump version
version=`cat VERSION | sed 's/+/ /g' | sed 's/v//g' | awk '{print $1}'`
echo "version: $version"

# run build
docker buildx build . \
    --platform linux/arm64,linux/amd64 \
    --tag $USERNAME/$IMAGE:latest --tag $USERNAME/$IMAGE:$version \
    --build-arg VERSION=$version \
    --build-arg BUILD_DATE=`date +"%Y%m%dT%H%M%S"` \
    --build-arg GITVERSION=$GITVERSION \
    --build-arg GITDIRECTVERSION=$GITVERSION \
    --build-arg VCS_REF=`git rev-parse HEAD` --push
