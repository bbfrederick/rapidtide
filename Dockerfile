# Start from the fredericklab base container
FROM fredericklab/basecontainer:0.1.1

# Installing precomputed python packages
RUN pip install \

                 statsmodels \
                 scikit-image \
                 scikit-learn \
                 pandas \
                 nilearn \
                 nibabel \
                 h5py \
                 keras \
                 tensorflow \
                 pyqtgraph \
                 pyqt5 \
                 versioneer \
                 numba \
                 pyfftw \
                 tqdm

#RUN mamba install -y "pyfftw=0.13.0=py39h51d1ae8_0"; true
#RUN mamba install -y statsmodels \
#                     scikit-image \
#                     scikit-learn \
#                     pandas \
#                     nilearn
#RUN mamba install -y nibabel \
#                     h5py
#RUN mamba install -y keras \
#                     tensorflow
#RUN mamba install -y pyqtgraph \
#                     versioneer \
#                     numba
#RUN mamba install -y "pyfftw=0.13.0=py39h51d1ae8_0"; true
#RUN chmod -R a+rX /usr/local/miniconda 
#RUN chmod +x /usr/local/miniconda/bin/* 
#RUN mamba update requests 
#RUN conda-build purge-all
#RUN mamba clean -y --all
#RUN df -h


# Create a shared $HOME directory
RUN useradd -m -s /bin/bash -G users rapidtide
WORKDIR /home/rapidtide
ENV HOME="/home/rapidtide"


# Installing rapidtide
COPY . /src/rapidtide
RUN cd /src/rapidtide && \
    python3 setup.py install && \
    rm -rf /src/rapidtide/build /src/rapidtide/dist


ENV IS_DOCKER_8395080871=1
RUN apt-get install -y --reinstall libxcb-xinerama0
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /root/.cache


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
