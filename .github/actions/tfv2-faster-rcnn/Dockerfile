# Get base image
FROM python:3.10.17-slim-bookworm

# Install system wide softwares
RUN apt-get update \
     && apt-get install -y \
        libgl1-mesa-glx \
        libx11-xcb1 \
        git \
        gcc \
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
     && rm -r /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel
RUN pip install tensorflow==2.19.0

# Install necessary libraries for tensorflow faster rcnn
RUN mkdir TensorFlow
RUN cd TensorFlow && git clone https://github.com/tensorflow/models
RUN cd TensorFlow && wget https://github.com/protocolbuffers/protobuf/releases/download/v3.17.3/protoc-3.17.3-linux-x86_64.zip
RUN cd TensorFlow && unzip protoc-3.17.3-linux-x86_64.zip -d protobuf
RUN cd TensorFlow/models/research && /TensorFlow/protobuf/bin/protoc object_detection/protos/*.proto --python_out=.
RUN cd TensorFlow/models/research && cp object_detection/packages/tf2/setup.py .
RUN cd TensorFlow/models/research && python -m pip install .

RUN pip install tqdm requests llvmlite numba==0.56.4 pytest-cov
