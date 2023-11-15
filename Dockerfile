# Start from the fredericklab base container
#FROM fredericklab/basecontainer:v0.2.8
FROM fredericklab/basecontainer:latest

# get build arguments
ARG BUILD_TIME
ARG BRANCH
ARG GITVERSION
ARG GITSHA
ARG GITDATE

# set and echo environment variables
ENV BUILD_TIME $BUILD_TIME
ENV BRANCH $BRANCH
ENV GITVERSION=${GITVERSION}
ENV GITSHA=${GITSHA}
ENV GITDATE=${GITDATE}

RUN echo "BRANCH: "$BRANCH
RUN echo "BUILD_TIME: "$BUILD_TIME
RUN echo "GITVERSION: "$GITVERSION
RUN echo "GITSHA: "$GITSHA
RUN echo "GITDATE: "$GITDATE

# Install rapidtide
COPY . /src/rapidtide
RUN echo $GITVERSION > /src/rapidtide/VERSION
RUN cd /src/rapidtide && \
    pip install . && \
    versioneer install --no-vendor && \
    rm -rf /src/rapidtide/build /src/rapidtide/dist
RUN cd /src/rapidtide/rapidtide/data/examples/src && \
    ./installtestdatadocker

# clean up
RUN mamba clean -y --all
RUN pip cache purge

# Create a shared $HOME directory
RUN useradd -m -s /bin/bash -G users rapidtide
WORKDIR /home/rapidtide
ENV HOME="/home/rapidtide"

ENV IS_DOCKER_8395080871=1

RUN ldconfig
WORKDIR /tmp/
RUN ln -s /src/rapidtide/cloud /
ENTRYPOINT ["/cloud/mount-and-run"]

LABEL org.label-schema.build-date=$BUILD_TIME \
      org.label-schema.name="rapidtide" \
      org.label-schema.description="rapidtide - a set of tools for delay processing" \
      org.label-schema.url="http://nirs-fmri.net" \
      org.label-schema.vcs-ref=$GITVERSION \
      org.label-schema.vcs-url="https://github.com/bbfrederick/rapidtide" \
      org.label-schema.version=$GITVERSION
