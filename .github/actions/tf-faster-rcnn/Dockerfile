# Get base from a TensorFlow image
FROM python:3.7.17-slim

# Install system wide software
RUN apt-get update \
     && apt-get install -y \
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
        wget \
        unzip \
        gpg \
        gpg-agent \
     && apt-get clean all \
     && rm -r /var/lib/apt/lists/*

# Install necessary libraries for TensorFlow Faster-RCNN
RUN mkdir TensorFlow
RUN cd TensorFlow && git clone https://github.com/tensorflow/models
RUN cd TensorFlow && wget https://github.com/protocolbuffers/protobuf/releases/download/v3.17.3/protoc-3.17.3-linux-x86_64.zip
RUN cd TensorFlow && unzip protoc-3.17.3-linux-x86_64.zip -d protobuf
RUN cd TensorFlow/models/research && /TensorFlow/protobuf/bin/protoc object_detection/protos/*.proto --python_out=.
RUN cd TensorFlow/models/research && cp object_detection/packages/tf1/setup.py .
RUN cd TensorFlow/models/research && python -m pip install .

RUN pip install tensorflow==1.15.5 keras==2.2.5 tqdm==4.66.1 requests~=2.31.0 scikit-learn==1.0.2 pytest-cov~=4.1.0 protobuf==3.20

CMD ["/bin/bash"]
