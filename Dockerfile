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

RUN /opt/miniforge3/bin/mamba init
RUN echo "mamba activate science" >> /home/rapidtide/.bashrc
RUN echo "/opt/miniforge3/bin/mamba activate science" >> /home/rapidtide/.bashrc

# Precompile Python code
RUN cd /opt/miniforge3/envs/science/lib/python3.12/site-packages/rapidtide && \
    python -m compileall -b .

# switch to the rapidtide user
USER rapidtide

# run things once
#RUN /opt/miniforge3/envs/science/bin/adjustoffset --help
#RUN /opt/miniforge3/envs/science/bin/aligntcs --help
#RUN /opt/miniforge3/envs/science/bin/applydlfilter --help
#RUN /opt/miniforge3/envs/science/bin/atlasaverage --help
#RUN /opt/miniforge3/envs/science/bin/atlastool --help
#RUN /opt/miniforge3/envs/science/bin/calcicc --help
#RUN /opt/miniforge3/envs/science/bin/calctexticc --help
#RUN /opt/miniforge3/envs/science/bin/calcttest --help
#RUN /opt/miniforge3/envs/science/bin/ccorrica --help
#RUN /opt/miniforge3/envs/science/bin/diffrois --help
#RUN /opt/miniforge3/envs/science/bin/endtidalproc --help
#RUN /opt/miniforge3/envs/science/bin/fdica --help
#RUN /opt/miniforge3/envs/science/bin/filtnifti --help
#RUN /opt/miniforge3/envs/science/bin/filttc --help
#RUN /opt/miniforge3/envs/science/bin/fingerprint --help
#RUN /opt/miniforge3/envs/science/bin/fixtr --help
#RUN /opt/miniforge3/envs/science/bin/glmfilt --help
#RUN /opt/miniforge3/envs/science/bin/gmscalc --help
RUN /opt/miniforge3/envs/science/bin/happy --help
#RUN /opt/miniforge3/envs/science/bin/happy2std --help
#RUN /opt/miniforge3/envs/science/bin/happywarp --help
#RUN /opt/miniforge3/envs/science/bin/histnifti --help
#RUN /opt/miniforge3/envs/science/bin/histtc --help
#RUN /opt/miniforge3/envs/science/bin/localflow --help
#RUN /opt/miniforge3/envs/science/bin/mergequality --help
#RUN /opt/miniforge3/envs/science/bin/pairproc --help
#RUN /opt/miniforge3/envs/science/bin/pairwisemergenifti --help
#RUN /opt/miniforge3/envs/science/bin/physiofreq --help
#RUN /opt/miniforge3/envs/science/bin/pixelcomp --help
#RUN /opt/miniforge3/envs/science/bin/plethquality --help
#RUN /opt/miniforge3/envs/science/bin/polyfitim --help
#RUN /opt/miniforge3/envs/science/bin/proj2flow --help
#RUN /opt/miniforge3/envs/science/bin/rankimage --help
RUN /opt/miniforge3/envs/science/bin/rapidtide --help
#RUN /opt/miniforge3/envs/science/bin/rapidtide2std --help
#RUN /opt/miniforge3/envs/science/bin/resamplenifti --help
#RUN /opt/miniforge3/envs/science/bin/resampletc --help
#RUN /opt/miniforge3/envs/science/bin/retroglm --help
#RUN /opt/miniforge3/envs/science/bin/retrolagtcs --help
#RUN /opt/miniforge3/envs/science/bin/roisummarize --help
#RUN /opt/miniforge3/envs/science/bin/runqualitycheck --help
#RUN /opt/miniforge3/envs/science/bin/showarbcorr --help
#RUN /opt/miniforge3/envs/science/bin/showhist --help
#RUN /opt/miniforge3/envs/science/bin/showstxcorr --help
#RUN /opt/miniforge3/envs/science/bin/showtc --help
#RUN /opt/miniforge3/envs/science/bin/showxcorr_legacy --help
#RUN /opt/miniforge3/envs/science/bin/showxcorrx --help
#RUN /opt/miniforge3/envs/science/bin/showxy --help
#RUN /opt/miniforge3/envs/science/bin/simdata --help
#RUN /opt/miniforge3/envs/science/bin/spatialdecomp --help
#RUN /opt/miniforge3/envs/science/bin/spatialfit --help
#RUN /opt/miniforge3/envs/science/bin/spatialmi --help
#RUN /opt/miniforge3/envs/science/bin/spectrogram --help
#RUN /opt/miniforge3/envs/science/bin/stupidramtricks --help
#RUN /opt/miniforge3/envs/science/bin/synthASL --help
#RUN /opt/miniforge3/envs/science/bin/tcfrom2col --help
#RUN /opt/miniforge3/envs/science/bin/tcfrom3col --help
#RUN /opt/miniforge3/envs/science/bin/temporaldecomp --help
#RUN /opt/miniforge3/envs/science/bin/tidepool --help
#RUN /opt/miniforge3/envs/science/bin/variabilityizer --help

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
