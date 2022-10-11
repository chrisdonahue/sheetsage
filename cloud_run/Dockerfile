# Import Ubuntu/CUDA/cuDNN
FROM ubuntu:18.04

# Configure shell
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]
RUN apt-get update --fix-missing

# Install Python
ENV LANG C.UTF-8
RUN apt-get install -y python3 python3-dev python3-pip
RUN ln -s $(which python3) /usr/local/bin/python
RUN python -m pip install --no-cache-dir --upgrade "pip<20.3" setuptools

# Install ffmpeg
RUN \
	apt-get install -y software-properties-common \
	&& add-apt-repository ppa:jonathonf/ffmpeg-4 \
	&& apt-get update \
	&& apt-get install -y ffmpeg

# Install Jukebox
RUN python -m pip install --no-cache-dir torch==1.4.0
RUN python -m pip install --no-cache-dir numba==0.48
RUN python -m pip install --no-cache-dir resampy==0.2.2
RUN python -m pip install --no-cache-dir librosa==0.7.2
ENV NUMBA_CACHE_DIR /tmp/numba
RUN apt-get install -y libopenmpi-dev
RUN apt-get install -y openssh-server
RUN python -m pip install --no-cache-dir mpi4py==3.0.3
RUN apt-get install -y git 
RUN python -m pip install --no-cache-dir git+https://github.com/chrisdonahue/jukebox.git@7e0a38b679ff3f64987d8297d9d0eb5a046880c1
RUN python -m pip install --no-cache-dir requests==2.26.0

# Install pretty_midi
RUN apt-get install -y fluidsynth
RUN python -m pip install --no-cache-dir pyFluidSynth==1.3.0
RUN python -m pip install --no-cache-dir pretty_midi==0.2.9

# Install madmom
RUN python -m pip install --no-cache-dir Cython==0.29.24
RUN python -m pip install --no-cache-dir madmom==0.16.1

# Install other deps
RUN python -m pip install --no-cache-dir scipy==1.5.4
RUN python -m pip install --no-cache-dir Pillow==8.4.0

# Create build dir
ARG BUILD_DIR=/tmp/build
WORKDIR $BUILD_DIR

# Install Melisma
RUN \
	wget https://www.link.cs.cmu.edu/music-analysis/melisma2003.tar.gz \
	&& [ "b4db2ab616dd2a14c8baff64787d3d0f257df6b0159452fb52fc3e29411743ad  melisma2003.tar.gz" = "$(sha256sum melisma2003.tar.gz)" ] \
	&& tar xvfz melisma2003.tar.gz \
	&& cd melisma2003/key \
	&& make \
	&& mv key /usr/local/bin/melisma-key \
	&& cd ../../ \
	&& rm -rf * \
	&& ls -lah

# Remove build dir
WORKDIR /
RUN rm -rf $BUILD_DIR

# Install lilypond
RUN apt-get install -y lilypond

# Install youtube-dl
RUN python -m pip install --no-cache-dir git+https://github.com/yt-dlp/yt-dlp.git@2022.05.18
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

# Install library
COPY setup.py setup.py
COPY sheetsage sheetsage/
RUN python -m pip install --no-cache-dir -e .

# Download assets
RUN python -m sheetsage.assets SHEETSAGE_V02_HANDCRAFTED

ENTRYPOINT python -m sheetsage.serve.backend.main --port $PORT
