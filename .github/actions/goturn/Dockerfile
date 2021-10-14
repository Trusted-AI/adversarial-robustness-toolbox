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
RUN pip install torch==1.9.1
RUN pip install torchvision==0.10.1
RUN pip install tensorflow==2.6.0
RUN pip install pytest
RUN pip install numba
RUN pip install scikit-learn==0.20
RUN pip install pytest-cov
RUN pip install gdown

RUN git clone https://github.com/nrupatunga/goturn-pytorch.git /tmp/goturn-pytorch
#RUN cd /tmp/goturn-pytorch && pip install -r requirements.txt
RUN pip install loguru==0.5.3
RUN pip install torchsummary==1.5.1
RUN pip install tqdm==4.62.3
RUN pip install pytorch_lightning==0.7.1
RUN pip install imutils==0.5.3
RUN pip install torch_lr_finder==0.2.1
RUN pip install numpy==1.20.3
RUN pip install opencv_python==4.3.0.36
RUN pip install Pillow==8.0.1
RUN pip install visdom==0.1.8.9

RUN pip install numpy==1.20.3

ENV PYTHONPATH "${PYTHONPATH}:/tmp/goturn-pytorch/src"
ENV PYTHONPATH "${PYTHONPATH}:/tmp/goturn-pytorch/src/scripts"

RUN mkdir /tmp/goturn-pytorch/src/goturn/models/checkpoints
RUN cd /tmp/goturn-pytorch/src/goturn/models/checkpoints && gdown https://drive.google.com/uc?id=1GouImhqpcoDtV_eLrD2wra-qr3vkAMY4
