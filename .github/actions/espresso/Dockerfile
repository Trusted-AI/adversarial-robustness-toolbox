# Get base image
FROM python:3.7.17-slim-bullseye

# Set to install things in non-interactive mode
ENV DEBIAN_FRONTEND noninteractive

# Install system wide software
RUN apt-get update \
     && apt-get install -y \
        build-essential \
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
RUN pip install torch==1.6.0 torchaudio==0.6.0 pytest-cov~=4.1.0 scipy==1.2 scikit-learn==0.24 protobuf==3.20 kaldiio==2.18.0 tensorflow==2.9
RUN pip install --no-build-isolation fairscale==0.3.6

RUN git clone https://github.com/beat-buesser/espresso
RUN cd espresso && git checkout adv && pip install .
RUN pip install sentencepiece numpy==1.21.6

CMD ["/bin/bash"]
