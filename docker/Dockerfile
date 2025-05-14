# Import Ubuntu/CUDA/cuDNN
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Configure shell
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]
RUN apt-get update --fix-missing

# Install Python
ENV LANG C.UTF-8
RUN apt-get install -y python3 python3-dev python3-pip
RUN ln -s $(which python3) /usr/local/bin/python
RUN python -m pip install --no-cache-dir --upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools

# Install dev deps
RUN python -m pip install --no-cache-dir jupyter matplotlib

# Install ffmpeg
RUN apt-get install -y software-properties-common
RUN apt-get update && apt-get install -y ffmpeg

# Install Jukebox
RUN python -m pip install --no-cache-dir torch==1.12.0+cu113 --extra-index-url  https://download.pytorch.org/whl/cu113
RUN python -m pip install --no-cache-dir numba==0.58.1
RUN python -m pip install --no-cache-dir resampy==0.2.2
RUN python -m pip install --no-cache-dir librosa==0.10.1
ENV NUMBA_CACHE_DIR /tmp/numba
RUN apt-get install -y libopenmpi-dev
RUN apt-get install -y openssh-server
RUN python -m pip install --no-cache-dir mpi4py==3.1.5
RUN apt-get install -y git 
RUN python -m pip install --no-cache-dir git+https://github.com/XaryLee/jukebox.git@ddc3577b50a4085be0ef65e17e9863686dadef40
RUN python -m pip install --no-cache-dir requests==2.26.0

# Install pretty_midi
RUN apt-get install -y fluidsynth
RUN python -m pip install --no-cache-dir pyFluidSynth==1.3.0
RUN python -m pip install --no-cache-dir pretty_midi==0.2.9

# Install madmom
RUN python -m pip install --no-cache-dir Cython==0.29.24
RUN python -m pip install --no-cache-dir git+https://github.com/XaryLee/madmom.git

# Create build dir
ARG BUILD_DIR=/tmp/build
WORKDIR $BUILD_DIR

# Remove build dir
WORKDIR /
RUN rm -rf $BUILD_DIR

# Install lilypond
RUN apt-get install -y lilypond

# Install mir_eval
RUN python -m pip install --no-cache-dir mir_eval==0.7

# Install youtube-dl
RUN python -m pip install --no-cache-dir git+https://github.com/yt-dlp/yt-dlp.git@2023.07.06
RUN ln -s $(which yt-dlp) /usr/local/bin/youtube-dl

# Install validators
RUN python -m pip install --no-cache-dir validators==0.20.0

# Install Flask
RUN python -m pip install --no-cache-dir requests==2.26.0
RUN python -m pip install --no-cache-dir Flask==2.0.3
RUN python -m pip install --no-cache-dir Flask-Cors==3.0.10

# Create workdir and cache
ARG WORKDIR=/sheetsage
RUN mkdir -p $WORKDIR
WORKDIR $WORKDIR
RUN mkdir cache
ENV SHEETSAGE_CACHE_DIR $WORKDIR/cache
ENV JUKEBOX_CACHE_DIR $WORKDIR/cache

# Install library placeholder
COPY setup.py setup.py
RUN mkdir sheetsage
RUN python -m pip install --no-cache-dir -e .

# Create placeholder folders
RUN mkdir notebooks

# Create user
RUN useradd -ms /bin/bash sheetsage
RUN chmod -R ugo+rw $WORKDIR
RUN chown -R sheetsage $WORKDIR
USER sheetsage
