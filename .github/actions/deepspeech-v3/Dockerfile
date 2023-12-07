pod# Get base from a pytorch image
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

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
        vim \
     && apt-get clean all \
     && rm -r /var/lib/apt/lists/*

# Install necessary libraries for deepspeech v3
RUN pip install --ignore-installed PyYAML torch==2.1.1 tensorflow==2.14.1 torchaudio==2.1.1 pytorch-lightning==2.1.2 scikit-learn==1.3.2
RUN pip install --no-build-isolation fairscale==0.4.13

RUN git clone https://github.com/SeanNaren/deepspeech.pytorch.git
RUN cd deepspeech.pytorch && sed -i '/^sklearn/d' requirements.txt && pip install -r requirements.txt && pip install -e .

RUN pip install numba==0.56.4 pytest-cov==4.1.0 pydub==0.25.1
RUN pip list
