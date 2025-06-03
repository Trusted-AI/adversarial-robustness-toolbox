# Get base image
FROM python:3.10.17-slim-bookworm

# Set to install things in non-interactive mode
ENV DEBIAN_FRONTEND noninteractive

# Install system wide software
RUN apt-get update \
     && apt-get install -y \
        build-essential \
        gfortran \
        libblas-dev \
        liblapack-dev \
        libatlas-base-dev \
        libgl1-mesa-glx \
        libx11-xcb1 \
        git \
        gcc \
        mono-mcs \
        libavcodec-extra \
        ffmpeg \
        curl \
        libsndfile-dev \
        libsndfile1 \
        gpg \
        gpg-agent \
     && apt-get clean all \
     && rm -r /var/lib/apt/lists/*

# Install necessary libraries for espresso
RUN pip install --upgrade pip setuptools wheel
RUN pip install --ignore-installed torch==1.11.0 torchaudio==0.11.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install numpy==1.21.6 scipy==1.9.3
RUN pip install pytest-cov~=4.1.0 scikit-learn==1.6.1 protobuf==3.20 kaldiio==2.18.0 tensorflow==2.9
RUN pip install --no-build-isolation fairscale==0.3.6

RUN git clone https://github.com/beat-buesser/espresso
RUN cd espresso && git checkout adv && pip install .
RUN pip install sentencepiece numpy==1.21.6

CMD ["/bin/bash"]
