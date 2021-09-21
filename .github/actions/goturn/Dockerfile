# Get base from a pytorch image
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

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
     && apt-get install -y libsm6 libxext6 \
     && apt-get install -y libxrender-dev \
     && apt-get clean all \
     && rm -r /var/lib/apt/lists/*

RUN /opt/conda/bin/conda install --yes \
    astropy \
    matplotlib \
    pandas \
    scikit-learn \
    scikit-image

# Install necessary libraries for goturn
RUN pip install torch==1.4
RUN pip install tensorflow==2.1.4
RUN pip install torchaudio==0.5.0
RUN pip install pytest
RUN pip install numba
RUN pip install scikit-learn==0.20
RUN pip install gdown

RUN git clone https://github.com/nrupatunga/goturn-pytorch.git /root/.art/data/goturn-pytorch
RUN cd /root/.art/data/goturn-pytorch && pip install -r requirements.txt

RUN pip install numpy==1.20.3

ENV PYTHONPATH "${PYTHONPATH}:/root/.art/data/goturn-pytorch/src"
ENV PYTHONPATH "${PYTHONPATH}:/root/.art/data/goturn-pytorch/src/scripts"

RUN mkdir /root/.art/data/goturn-pytorch/src/goturn/models/checkpoints
RUN cd /root/.art/data/goturn-pytorch/src/goturn/models/checkpoints && gdown https://drive.google.com/uc?id=1GouImhqpcoDtV_eLrD2wra-qr3vkAMY4
