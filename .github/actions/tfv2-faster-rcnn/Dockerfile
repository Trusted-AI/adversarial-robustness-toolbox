# Get base from a tensorflow image
FROM tensorflow/tensorflow:2.10.1

# Set to install things in non-interactive mode
ENV DEBIAN_FRONTEND noninteractive

# Install system wide softwares
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
     && apt-get clean all \
     && rm -r /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda
RUN /miniconda/bin/conda install --yes \
    astropy \
    matplotlib \
    pandas \
    scikit-learn \
    scikit-image

# Install necessary libraries for tensorflow faster rcnn
RUN mkdir TensorFlow
RUN cd TensorFlow && git clone https://github.com/tensorflow/models
RUN cd TensorFlow && wget https://github.com/protocolbuffers/protobuf/releases/download/v3.17.3/protoc-3.17.3-linux-x86_64.zip
RUN cd TensorFlow && unzip protoc-3.17.3-linux-x86_64.zip -d protobuf
RUN cd TensorFlow/models/research && /TensorFlow/protobuf/bin/protoc object_detection/protos/*.proto --python_out=.
RUN cd TensorFlow/models/research && cp object_detection/packages/tf2/setup.py .
RUN cd TensorFlow/models/research && python -m pip install --use-feature=2020-resolver .

RUN pip install tqdm
RUN pip install requests
RUN pip install sklearn
RUN pip install numba==0.50.0
RUN pip install pytest-cov
