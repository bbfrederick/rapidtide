# Start from the fredericklab base container
FROM fredericklab/basecontainer:latest-release

# get build arguments
ARG BUILD_TIME
ARG BRANCH
ARG GITVERSION
ARG GITDIRECTVERSION
ARG GITSHA
ARG GITDATE

# set and echo environment variables
ENV BUILD_TIME=$BUILD_TIME
ENV BRANCH=$BRANCH
ENV GITVERSION=${GITVERSION}
ENV GITSHA=${GITSHA}
ENV GITDATE=${GITDATE}
ENV GITDIRECTVERSION=${GITDIRECTVERSION}

RUN echo "BRANCH: "$BRANCH
RUN echo "BUILD_TIME: "$BUILD_TIME
RUN echo "GITVERSION: "$GITVERSION
RUN echo "GITSHA: "$GITSHA
RUN echo "GITDATE: "$GITDATE
RUN echo "GITDIRECTVERSION: "$GITDIRECTVERSION

# security patches
RUN uv pip install "cryptography>=42.0.4" "urllib3>=1.26.17" "certifi>=2023.7.22"

# Copy rapidtide into container
COPY . /src/rapidtide
RUN ln -s /src/rapidtide/cloud /
RUN echo $GITVERSION > /src/rapidtide/VERSION

# init and install rapidtide
RUN uv pip install --upgrade pip
RUN cd /src/rapidtide && \
    uv pip install .
RUN chmod -R a+r /src/rapidtide

# clean up install directories
RUN rm -rf /src/rapidtide/build /src/rapidtide/dist

# install test data
RUN cd /src/rapidtide/rapidtide/data/examples/src && \
    ./installtestdatadocker

# update the paths to libraries
RUN ldconfig

# clean up
RUN pip cache purge

# Create a shared $HOME directory
ENV USER=rapidtide
RUN useradd \
    --create-home \
    --shell /bin/bash \
    --groups users \
    --home /home/$USER \
    $USER
RUN chown -R $USER /src/$USER

WORKDIR /home/$USER
ENV HOME="/home/rapidtide"

# set to non-root user
USER rapidtide

# initialize user mamba
RUN /opt/miniforge3/bin/mamba shell init --shell bash
RUN echo "mamba activate science" >> /home/rapidtide/.bashrc

# set up variable for non-interactive shell
ENV PATH=/opt/miniforge3/envs/science/bin:/opt/miniforge3/condabin:.:/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

ENV RUNNING_IN_CONTAINER=1

WORKDIR /tmp/

ENTRYPOINT ["/cloud/mount-and-run"]

LABEL org.label-schema.build-date=$BUILD_TIME \
      org.label-schema.name="rapidtide" \
      org.label-schema.description="rapidtide - a set of tools for delay processing" \
      org.label-schema.url="http://nirs-fmri.net" \
      org.label-schema.vcs-ref=$GITVERSION \
      org.label-schema.vcs-url="https://github.com/bbfrederick/rapidtide" \
      org.label-schema.version=$GITVERSION
