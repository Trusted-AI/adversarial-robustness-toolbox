# Get base image
FROM python:3.10.17-slim-bookworm

ENV MONO_TLS_PROVIDER=legacy

# Install system wide software
RUN apt-get update \
     && apt-get install -y \
        build-essential \
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
        libhdf5-dev \
        gpg \
        gpg-agent \
     && apt-get install -y libsm6 libxext6 \
     && apt-get install -y libxrender-dev \
     && apt-get clean all \
     && rm -r /var/lib/apt/lists/*

# Install necessary libraries for goturn
RUN pip install --ignore-installed torch==1.11.0 torchaudio==0.11.0 torchvision==0.12.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --ignore-installed --no-deps pytorch_lightning==0.7.1 torch_lr_finder==0.2.1 torchsummary==1.5.1
RUN pip install tensorflow==2.9 scikit-learn==1.6.1 pytest~=6.2.5 pytest-cov~=4.1.0

RUN git clone https://github.com/nrupatunga/goturn-pytorch.git /tmp/goturn-pytorch
#RUN cd /tmp/goturn-pytorch && pip install -r requirements.txt
RUN pip install loguru==0.5.3 tqdm==4.62.3 imutils==0.5.3 numpy==1.21.6 Pillow==8.0.1 visdom==0.1.8.9 gdown==5.1.0
RUN pip install opencv_python==4.9.0.80 --no-build-isolation

ENV PYTHONPATH "${PYTHONPATH}:/tmp/goturn-pytorch/src"
ENV PYTHONPATH "${PYTHONPATH}:/tmp/goturn-pytorch/src/scripts"

RUN mkdir /tmp/goturn-pytorch/src/goturn/models/checkpoints
RUN cd /tmp/goturn-pytorch/src/goturn/models/checkpoints && gdown https://drive.google.com/uc?id=1GouImhqpcoDtV_eLrD2wra-qr3vkAMY4

CMD ["/bin/bash"]
