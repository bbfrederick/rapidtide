# Start from the fredericklab base container
FROM fredericklab/basecontainer:v0.2.3

# Installing additional precomputed python packages
# tensorflow seems to really want to install with pip
RUN mamba install h5py 
RUN mamba install keras 
RUN pip install tensorflow

# hack to get around the super annoying "urllib3 doesn't match" warning
RUN pip install --upgrade --force-reinstall requests "certifi>=2023.7.22"

# Install rapidtide
#COPY . /src/rapidtide
RUN git clone https://github.com/bbfrederick/rapidtide.git /src/rapidtide
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

ARG VERSION
ARG BUILD_DATE
ARG VCS_REF

RUN echo $VERSION
RUN echo $BUILD_DATE
RUN echo $VCS_REF

LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="rapidtide" \
      org.label-schema.description="rapidtide - a set of tools for delay processing" \
      org.label-schema.url="http://nirs-fmri.net" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/bbfrederick/rapidtide" \
      org.label-schema.version=$VERSION
