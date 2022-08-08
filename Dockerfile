FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

RUN apt-get update -y
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install -y python3 python3-distutils python3-pip

RUN pip3 install adversarial-robustness-toolbox

RUN pip3 install tensorflow==2.9.1
RUN pip3 install keras==2.9.0
RUN pip3 install numpy==1.22.4
RUN pip3 install scipy==1.8.1
RUN pip3 install matplotlib==3.5.2
RUN pip3 install scikit-learn==1.1.2
RUN pip3 install six==1.15.0
RUN pip3 install Pillow==9.2.0
RUN pip3 install pytest-cov==3.0.0
RUN pip3 install tqdm==4.64.0
RUN pip3 install statsmodels==0.13.2
RUN pip3 install pydub==0.25.1
RUN pip3 install resampy==0.3.1
RUN pip3 install ffmpeg-python==0.2.0
RUN pip3 install cma==3.2.2
RUN pip3 install pandas==1.4.3
RUN pip3 install h5py==3.7.0
RUN pip3 install tensorflow-addons==0.17.1
RUN pip3 install mxnet==1.6.0
RUN pip3 install torch==1.12.0
RUN pip3 install torchaudio==0.12.0
RUN pip3 install torchvision==0.13.0
RUN pip3 install catboost==1.0.6
RUN pip3 install GPy==1.10.0
RUN pip3 install lightgbm==3.3.2
RUN pip3 install xgboost==1.6.1
RUN pip3 install kornia==0.6.6
RUN pip3 install lief==0.12.1
RUN pip3 install pytest==7.1.2
RUN pip3 install pytest-pep8==1.0.6
RUN pip3 install pytest-mock==3.8.2
RUN pip3 install codecov==2.1.12
RUN pip3 install requests==2.28.1

RUN apt-get -y -q install ffmpeg libavcodec-extra

RUN echo "You should think about possibly upgrading these outdated packages"
RUN pip3 list --outdated

# NOTE to contributors: When changing/adding packages, please make sure that the packages are consistent with those
# present within the requirements_test.txt files
