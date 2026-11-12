FROM python:3.10.17-slim-bookworm

# System packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libx11-xcb1 \
    git \
    gcc \
    g++ \
    libavcodec-extra \
    ffmpeg \
    curl \
    libsndfile-dev \
    libsndfile1 \
    wget \
    unzip \
    software-properties-common \
    gnupg2 \
    lsb-release \
    && apt-get clean all \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

#
# TensorFlow Models / Object Detection API
#
RUN mkdir -p /TensorFlow

RUN cd /TensorFlow && \
    git clone https://github.com/tensorflow/models.git

#
# Keep the same protoc version that worked previously
#
RUN cd /TensorFlow && \
    wget https://github.com/protocolbuffers/protobuf/releases/download/v3.17.3/protoc-3.17.3-linux-x86_64.zip

RUN cd /TensorFlow && \
    unzip protoc-3.17.3-linux-x86_64.zip -d protobuf

RUN cd /TensorFlow/models/research && \
    /TensorFlow/protobuf/bin/protoc object_detection/protos/*.proto --python_out=.

#
# Preserve the modifications from the working image
#
RUN cd /TensorFlow/models/research && \
    cp object_detection/packages/tf2/setup.py . && \
    sed -i "/^'tensorflow_io',$/d; /^'keras',$/d" setup.py

#
# Install TensorFlow first
#
RUN pip install \
    tensorflow==2.10.1 \
    keras==2.10.0

#
# Install Object Detection API
#
RUN cd /TensorFlow/models/research && \
    python -m pip install .

#
# ART test dependencies
#
RUN pip install \
    numpy \
    matplotlib \
    tqdm \
    requests \
    llvmlite \
    pytest-cov \
    "pytest<8.0.0" \
    "exceptiongroup<1.1.0"

ENV PYTHONPATH="/TensorFlow/models:/TensorFlow/models/research:/TensorFlow/models/research/slim:${PYTHONPATH}"
