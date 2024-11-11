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
    --platform linux/arm64 \
    -t $IMAGE \
    --tag $USERNAME/$IMAGE:latest --tag $USERNAME/$IMAGE:$version \
    --build-arg VERSION=$version \
    --build-arg BUILD_DATE=`date +"%Y%m%dT%H%M%S"` \
    --build-arg GITVERSION=thegitversion \
    --build-arg VCS_REF=`git rev-parse HEAD`
#--push

# tag it
#docker tag $IMAGE:latest $USERNAME/$IMAGE:latest
#docker tag $IMAGE:latest $USERNAME/$IMAGE:$version

# push it
#docker push $USERNAME/$IMAGE:latest
#docker push $USERNAME/$IMAGE:$version
