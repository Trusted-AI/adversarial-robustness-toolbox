FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

RUN apt-get update -y
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install -y python3 python3-distutils python3-pip

RUN pip3 install tensorflow==2.9.1 keras==2.9.0 numpy==1.22.4 scipy==1.8.1 matplotlib==3.5.2 scikit-learn==1.1.2 \
                 six==1.15.0 Pillow==9.2.0 pytest-cov==3.0.0 tqdm==4.64.0 statsmodels==0.13.2 pydub==0.25.1 \
                 resampy==0.3.1 ffmpeg-python==0.2.0 cma==3.2.2 pandas==1.4.3 h5py==3.7.0 tensorflow-addons==0.17.1 \
                 mxnet==1.6.0 torch==1.12.0 torchaudio==0.12.0 torchvision==0.13.0 catboost==1.0.6 GPy==1.10.0 \
                 lightgbm==3.3.2 xgboost==1.6.1 kornia==0.6.6 lief==0.12.1 pytest==7.1.2 pytest-pep8==1.0.6 \
                 pytest-mock==3.8.2 requests==2.28.1

RUN apt-get -y install ffmpeg libavcodec-extra vim git

RUN mkdir /project
WORKDIR /project
ADD . /project
RUN pip3 install .

RUN echo "You should think about possibly upgrading these outdated packages"
RUN pip3 list --outdated

# NOTE to contributors: When changing/adding packages, please make sure that the packages are consistent with those
# present within the requirements_test.txt files