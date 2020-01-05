# Use Ubuntu 16.04 LTS
FROM ubuntu:bionic-20191202

# Pre-cache neurodebian key
COPY ./dockerbuild/neurodebian.gpg /usr/local/etc/neurodebian.gpg

# Prepare environment
RUN df -h
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
                    curl \
                    bzip2 \
                    ca-certificates \
                    xvfb \
                    build-essential \
                    autoconf \
                    libtool \
                    gnupg \
                    pkg-config \
                    libgl1-mesa-glx \
                    libx11-xcb1 \
                    firefox \
                    git
#RUN apt-get install --reinstall libxcb-xinerama0
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

#ENV FSL_DIR="/usr/share/fsl/5.0" \
#    OS="Linux" \
#    FS_OVERRIDE=0 \
#    FIX_VERTEX_AREA="" \
#    FSF_OUTPUT_FORMAT="nii.gz" \
#    FREESURFER_HOME="/opt/freesurfer"

# Installing Neurodebian packages (FSL)
RUN curl -sSL "http://neuro.debian.net/lists/$( lsb_release -c | cut -f2 ).us-ca.full" >> /etc/apt/sources.list.d/neurodebian.sources.list && \
    apt-key add /usr/local/etc/neurodebian.gpg && \
    (apt-key adv --refresh-keys --keyserver hkp://ha.pool.sks-keyservers.net 0xA5D32F012649A5A9 || true)

#RUN apt-get update && \
#    apt-get install -y --no-install-recommends \
#                    fsl-core=5.0.9-5~nd16.04+1 \
#                    fsl-mni152-templates=5.0.7-2 && \
#    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

#ENV FSLDIR="/usr/share/fsl/5.0" \
#    FSLOUTPUTTYPE="NIFTI_GZ" \
#    FSLMULTIFILEQUIT="TRUE" \
#    POSSUMDIR="/usr/share/fsl/5.0" \
#    LD_LIBRARY_PATH="/usr/lib/fsl/5.0:$LD_LIBRARY_PATH" \
#    FSLTCLSH="/usr/bin/tclsh" \
#    FSLWISH="/usr/bin/wish" \
#ENV PATH="/usr/lib/fsl/5.0:$PATH"

## Installing ANTs 2.2.0 (NeuroDocker build)
#ENV ANTSPATH=/usr/lib/ants
#RUN mkdir -p $ANTSPATH && \
#    curl -sSL "https://dl.dropbox.com/s/2f4sui1z6lcgyek/ANTs-Linux-centos5_x86_64-v2.2.0-0740f91.tar.gz" \
#    | tar -xzC $ANTSPATH --strip-components 1
#ENV PATH=$ANTSPATH:$PATH


# Installing and setting up miniconda
RUN curl -sSLO https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh && \
    bash Miniconda3-4.5.11-Linux-x86_64.sh -b -p /usr/local/miniconda && \
    rm Miniconda3-4.5.11-Linux-x86_64.sh


# Set CPATH for packages relying on compiled libs (e.g. indexed_gzip)
ENV PATH="/usr/local/miniconda/bin:$PATH" \
    CPATH="/usr/local/miniconda/include/:$CPATH" \
    LANG="C.UTF-8" \
    LC_ALL="C.UTF-8" \
    PYTHONNOUSERSITE=1

# Installing precomputed python packages
RUN df -h
RUN conda config --add channels conda-forge
RUN df -h
RUN conda clean --all
RUN df -h
RUN conda install -y python=3.7.1 \
                     pip=19.3.1 \
                     scipy=1.4.1 \
                     numpy=1.17.3 \
                     mkl=2019.4 \
                     mkl-service=2.3.0 \
                     matplotlib=3.1.2 \
                     statsmodels=0.10.2 \
                     scikit-image=0.16.2 \
                     scikit-learn=0.22 \
                     nibabel=2.5.1 \
                     keras=2.3.1 \
                     tensorflow=1.13.1 \
                     pyqtgraph=0.10.0 \
                     pyfftw=0.11.1 \
                     pandas=0.25.3 \
                     numba=0.46.0; sync && \
    chmod -R a+rX /usr/local/miniconda; sync && \
    chmod +x /usr/local/miniconda/bin/*; sync && \
    conda build purge-all; sync && \
    conda clean -tipsy && sync
RUN df -h


# Unless otherwise specified each process should only use one thread
# will handle parallelization
#ENV MKL_NUM_THREADS=1 


# Create a shared $HOME directory
RUN useradd -m -s /bin/bash -G users rapidtide
WORKDIR /home/rapidtide
ENV HOME="/home/rapidtide"


# Precaching fonts, set 'Agg' as default backend for matplotlib
RUN python -c "from matplotlib import font_manager" && \
    sed -i 's/\(backend *: \).*$/\1Agg/g' $( python -c "import matplotlib; print(matplotlib.matplotlib_fname())" )

# fix qt
#ENV QT_GRAPHICSSYSTEM="native" 
#RUN apt-get install libxkbcommon-x11-dev

# Installing rapidtide
COPY . /src/rapidtide
RUN cd /src/rapidtide && \
    python setup.py install && \
    rm -rf /src/rapidtide/build /src/rapidtide/dist


ENV IS_DOCKER_8395080871=1

RUN ldconfig
WORKDIR /tmp/
#ENTRYPOINT ["/usr/local/miniconda/bin/rapidtide_dispatcher"]
CMD ["/usr/bin/firefox"]


ARG BUILD_DATE
ARG VCS_REF
ARG VERSION
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="rapidtide" \
      org.label-schema.description="rapidtide - a set of tools for delay processing" \
      org.label-schema.url="http://nirs-fmri.net" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/bbfrederick/rapidtide" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"
