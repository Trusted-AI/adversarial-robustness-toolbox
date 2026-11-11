FROM python:3.10.17-slim-bookworm

# System packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libx11-xcb1 \
    git \
    gcc \
    g++ \
    protobuf-compiler \
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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

#
# TensorFlow stack
#
RUN pip install \
    tensorflow==2.20.0 \
    tf-keras==2.20.1 \
    tensorflow-text==2.20.0

#
# TensorFlow Models / Object Detection API
#
RUN git clone https://github.com/tensorflow/models.git /TensorFlow/models

RUN cd /TensorFlow/models/research && \
    protoc object_detection/protos/*.proto --python_out=.

RUN cd /TensorFlow/models/research && \
    cp object_detection/packages/tf2/setup.py . && \
    python -m pip install .

#
# ART test dependencies
#
RUN pip install \
    numpy \
    matplotlib \
    tqdm \
    requests \
    pytest-cov \
    "pytest<8.0.0" \
    "exceptiongroup<1.1.0"

ENV PYTHONPATH="/TensorFlow/models:/TensorFlow/models/research:/TensorFlow/models/research/slim:${PYTHONPATH}"
