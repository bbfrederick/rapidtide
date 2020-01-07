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
version=`cat VERSION`
echo "version: $version"

# run build
docker build . -t rapidtide \
    --build-arg VERSION=$version \
    --build-arg BUILD_DATE=`date +"%Y%m%dT%k%M%S"` \
    --build-arg VCS_REF=`git rev-parse HEAD`

# tag it
#git add -A
#git commit -m "version $version"
#git tag -a "$version" -m "version $version"
#git push
#git push --tags
docker tag $IMAGE:latest $USERNAME/$IMAGE:latest
docker tag $IMAGE:latest $USERNAME/$IMAGE:$version

# push it
docker push $USERNAME/$IMAGE:latest
docker push $USERNAME/$IMAGE:$version
