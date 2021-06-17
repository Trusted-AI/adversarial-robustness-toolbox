# Get base from a tensorflow image
FROM tensorflow/tensorflow:1.15.5-py3

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

#RUN /opt/conda/bin/conda install --yes \
#    astropy \
#    matplotlib \
#    pandas \
#    scikit-learn \
#    scikit-image

# Install necessary libraries for tensorflow faster rcnn
RUN mkdir TensorFlow
RUN cd TensorFlow
RUN git clone https://github.com/tensorflow/models



RUN pip install numba==0.50.0
RUN pip install pytest-cov
