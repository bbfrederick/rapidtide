# Start from the fredericklab base container
FROM fredericklab/basecontainer:v0.0.7

# Prepare environment
#RUN apt-get install -y --no-install-recommends \
#                    s3fs \
#                    awscli \
#                    jq

#RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Pre-cache neurodebian key
#COPY ./dockerbuild/neurodebian.gpg /usr/local/etc/neurodebian.gpg

# Installing Neurodebian packages (FSL)
#RUN curl -sSL "http://neuro.debian.net/lists/$( lsb_release -c | cut -f2 ).us-ca.full" >> /etc/apt/sources.list.d/neurodebian.sources.list && \
    #apt-key add /usr/local/etc/neurodebian.gpg && \
    #(apt-key adv --refresh-keys --keyserver hkp://ha.pool.sks-keyservers.net 0xA5D32F012649A5A9 || true)

# Installing precomputed python packages
RUN mamba install -y statsmodels \
                     scikit-image \
                     scikit-learn \
                     pandas \
                     nilearn
RUN mamba install -y nibabel \
                     h5py
RUN mamba install -y keras \
                     "tensorflow>=2.4.0"
RUN mamba install -y pyqtgraph \
                     "pyfftw=0.13.0=py39h51d1ae8_0" \
                     versioneer \
                     numba
RUN chmod -R a+rX /usr/local/miniconda && \
    chmod +x /usr/local/miniconda/bin/* && \
    mamba update requests && \
    conda-build purge-all
RUN mamba clean -y --all
RUN df -h


# Create a shared $HOME directory
RUN useradd -m -s /bin/bash -G users rapidtide
WORKDIR /home/rapidtide
ENV HOME="/home/rapidtide"


# Installing rapidtide
COPY . /src/rapidtide
RUN cd /src/rapidtide && \
    python setup.py install && \
    rm -rf /src/rapidtide/build /src/rapidtide/dist


ENV IS_DOCKER_8395080871=1
RUN apt-get install -y --reinstall libxcb-xinerama0


RUN ldconfig
WORKDIR /tmp/
ENTRYPOINT ["/usr/local/miniconda/bin/rapidtide_dispatcher"]

# set a non-root user
#USER rapidtide

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
