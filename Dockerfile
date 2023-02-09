# Start from the fredericklab base container
FROM fredericklab/basecontainer:latest

# Installing precomputed python packages
RUN mamba install \
                 statsmodels \
                 nibabel \
                 h5py 
RUN mamba install \
                 keras \
                 tensorflow 
RUN mamba install \
                 numba \
                 pyfftw 
RUN mamba install \
                 versioneer \
                 tqdm

# hack to get around the super annoying "urllib3 doesn't match" warning
#RUN pip install --upgrade --force-reinstall  requests
RUN mamba install -y requests --force-reinstall


# Create a shared $HOME directory
RUN useradd -m -s /bin/bash -G users rapidtide
WORKDIR /home/rapidtide
ENV HOME="/home/rapidtide"


# Install rapidtide
COPY . /src/rapidtide
RUN cd /src/rapidtide && \
    python3 setup.py install && \
    rm -rf /src/rapidtide/build /src/rapidtide/dist


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
