# Get base from a pytorch image
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# Set to install things in non-interactive mode
ENV DEBIAN_FRONTEND noninteractive

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
     && apt-get clean all \
     && rm -r /var/lib/apt/lists/*

RUN pip install six setuptools tqdm
RUN pip install numpy==1.21.6 scipy==1.8.1 scikit-learn==1.1.1 numba==0.55.1
RUN pip install torch==1.11.0
RUN pip install tensorflow==2.9.1
RUN pip install pytest-cov

# Install necessary libraries for Yolo v3
RUN pip install pytorchyolo==1.6.2

RUN cd /tmp/ && git clone https://github.com/eriklindernoren/PyTorch-YOLOv3.git
RUN cd PyTorch-YOLOv3/weights && ./download_weights.sh
