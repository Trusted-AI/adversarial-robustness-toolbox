# Get base image
FROM python:3.8.20-slim-bookworm

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

# Install necessary libraries for tensorflow faster rcnn
RUN mkdir TensorFlow
RUN cd TensorFlow && git clone https://github.com/tensorflow/models
RUN cd TensorFlow && wget https://github.com/protocolbuffers/protobuf/releases/download/v3.17.3/protoc-3.17.3-linux-x86_64.zip
RUN cd TensorFlow && unzip protoc-3.17.3-linux-x86_64.zip -d protobuf
RUN cd TensorFlow/models/research && /TensorFlow/protobuf/bin/protoc object_detection/protos/*.proto --python_out=.
RUN cd TensorFlow/models/research && cp object_detection/packages/tf2/setup.py . && sed -i "/^'tensorflow_io',$/d; /^'keras',$/d" setup.py
RUN cd TensorFlow/models/research && python -m pip install .

RUN pip install tqdm requests llvmlite numba==0.56.4 pytest-cov "pytest<8.0.0" "exceptiongroup<1.1.0" tensorflow==2.6.0 keras==2.6.0
