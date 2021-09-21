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
RUN pip install tensorflow
RUN pip install torchaudio==0.5.0
RUN pip install pytest
RUN pip install numpy

RUN git clone https://github.com/nrupatunga/goturn-pytorch.git
RUN ls -lh
RUN cd goturn-pytorch && pip install -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/workspace/goturn-pytorch/src"