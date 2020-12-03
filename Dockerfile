FROM  tensorflow/tensorflow:2.2.0
RUN pip3 install keras==2.3.1
#### NOTE: comment these two lines if you wish to use the tensorflow 1 version of ART instead ####
#FROM tensorflow/tensorflow:1.15.2
#RUN pip3 install keras==2.2.5

RUN pip3 install numpy==1.19.1 scipy==1.4.1 matplotlib==3.3.1 scikit-learn==0.22.2 six==1.15.0 Pillow==7.2.0 pytest-cov==2.10.1
RUN pip3 install tqdm==4.48.2 statsmodels==0.11.1 pydub==0.24.1 resampy==0.2.2 ffmpeg-python==0.2.0 cma==3.0.3 mypy==0.770
RUN pip3 install ffmpeg-python==0.2.0
RUN pip3 install pandas==1.1.1

#TODO check if jupyter notebook works
RUN pip3 install jupyter==1.0.0 && pip3 install jupyterlab==2.1.0
# https://stackoverflow.com/questions/49024624/how-to-dockerize-jupyter-lab

# Lingvo ASR dependencies
# supported versions: (lingvo==0.6.4 with tensorflow-gpu==2.1.0)
# note: due to conflicts with other TF1/2 version supported by ART, the dependencies are not installed by default:
# Replace line 1 with: FROM tensorflow/tensorflow:2.1.0
# Comment other TF related lines and uncomment:
# RUN pip3 install tensorflow-gpu==2.1.0
# RUN pip3 install lingvo==0.6.4

RUN pip3 install h5py==2.10.0
RUN pip3 install tensorflow-addons==0.11.1
RUN pip3 install mxnet==1.6.0
RUN pip3 install torch==1.5.0 torchvision==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install catboost==0.24
RUN pip3 install GPy==1.9.9
RUN pip3 install lightgbm==2.3.1
RUN pip3 install xgboost==1.1.1
RUN pip3 install kornia==0.3.1

RUN pip3 install pytest==5.4.1 pytest-pep8==1.0.6 pytest-mock==3.2.0 codecov==2.1.8 requests==2.24.0

RUN mkdir /project; mkdir /project/TMP
VOLUME /project/TMP
WORKDIR /project

# IMPORTANT: please double check that the dependencies above are up to date with the following requirements file. We currently still run pip install on dependencies within requirements.txt in order to keep dependencies in agreement (in the rare cases were someone updated the requirements.txt file and forgot to update the dockefile)
ADD . /project/
RUN pip3 install --upgrade -r /project/requirements.txt

RUN apt-get update
RUN apt-get -y -q install ffmpeg libavcodec-extra

RUN echo "You should think about possibly upgrading these outdated packages"
RUN pip3 list --outdated

EXPOSE 8888

CMD bash run_tests.sh

#Check the Dockerfile here https://www.fromlatest.io/#/

#NOTE to contributors: When changing/adding packages, please make sure that the packages are consitent with those
# present within the requirements.txt and test_requirements.txt files
