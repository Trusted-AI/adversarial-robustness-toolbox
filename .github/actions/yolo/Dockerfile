# Get base image
FROM python:3.9.19-slim

# Install system wide software
RUN apt-get update \
     && apt-get install -y \
        libgl1-mesa-glx \
        libx11-xcb1 \
        git \
        gcc \
        mono-mcs \
        cmake \
        libavcodec-extra \
        ffmpeg \
        curl \
        wget \
        libhdf5-dev \
        gpg \
        gpg-agent \
     && apt-get clean all \
     && rm -r /var/lib/apt/lists/*

RUN pip install pytorchyolo==1.8.0 tensorflow==2.14.1 scikit-learn==1.4.2 pytest-cov~=4.1.0 torch==1.12.1

RUN cd /tmp/ && git clone https://github.com/eriklindernoren/PyTorch-YOLOv3.git && cd ./PyTorch-YOLOv3/weights && ./download_weights.sh

CMD ["/bin/bash"]
