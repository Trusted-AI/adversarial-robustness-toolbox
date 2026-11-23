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

# Install necessary libraries for tensorflow faster rcnn
RUN mkdir TensorFlow
RUN cd TensorFlow && git clone https://github.com/tensorflow/models
RUN cd TensorFlow && wget https://github.com/protocolbuffers/protobuf/releases/download/v3.17.3/protoc-3.17.3-linux-x86_64.zip
RUN cd TensorFlow && unzip protoc-3.17.3-linux-x86_64.zip -d protobuf
RUN cd TensorFlow/models/research && /TensorFlow/protobuf/bin/protoc object_detection/protos/*.proto --python_out=.
RUN cd TensorFlow/models/research && cp object_detection/packages/tf2/setup.py . && sed -i "/^'tensorflow_io',$/d; /^'keras',$/d" setup.py
# Constrain dependencies to the TensorFlow 2.9 stack; the unpinned requirements of the research package would otherwise pull in the latest TensorFlow 2.x components
RUN printf '%s\n' \
        'tensorflow==2.9.*' \
        'keras==2.9.*' \
        'tensorflow-text==2.9.*' \
        'tensorflow-io==0.26.*' \
        'tf-models-official==2.9.2' \
        'protobuf<3.20' \
        'numpy<1.24' \
        'opencv-python<4.10' \
        'opencv-python-headless<4.10' \
        > /TensorFlow/constraints.txt
# PyYAML 5.4.1 (required by tf-models-official 2.9.2) has no wheel for Python 3.10 and only builds with Cython<3
RUN pip install "cython<3" && pip install --no-build-isolation "pyyaml==5.4.1"
RUN pip install -c /TensorFlow/constraints.txt tensorflow==2.9.* keras==2.9.* tf-models-official==2.9.2
RUN cd TensorFlow/models/research && python -m pip install -c /TensorFlow/constraints.txt .

RUN pip install -c /TensorFlow/constraints.txt numpy matplotlib tqdm requests llvmlite numba==0.56.4 pytest-cov "pytest<8.0.0" "exceptiongroup<1.1.0" tensorflow==2.9.* keras==2.9.*
