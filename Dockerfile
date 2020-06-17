FROM  tensorflow/tensorflow:2.2.0
RUN pip3 install keras==2.3.1
#### NOTE: comment these two lines if you wish to use the tensorflow 1 version of ART instead ####
#FROM tensorflow/tensorflow:1.15.2
#RUN pip3 install keras==2.2.5

RUN pip3 install numpy==1.18.1 scipy==1.4.1 matplotlib==3.2.1 scikit-learn==0.22.1 six==1.13.0 Pillow==6.2.2 
RUN pip3 install tqdm==4.45.0 statsmodels==0.11.0 pydub==0.24.1 resampy==0.2.2 ffmpeg-python==0.2.0 cma==2.7.0 mypy==0.770

#TODO check if jupyter notebook works
RUN pip3 install jupyter==1.0.0 && pip3 install jupyterlab==2.1.0
# https://stackoverflow.com/questions/49024624/how-to-dockerize-jupyter-lab

RUN pip3 install h5py==2.10.0
RUN pip3 install tensorflow-addons==0.10.0
RUN pip3 install mxnet==1.6.0
RUN pip3 install torch==1.5.0 torchvision==0.6.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install catboost==0.23.2
RUN pip3 install GPy==1.9.9
RUN pip3 install lightgbm==2.3.1
RUN pip3 install xgboost==1.1.1

RUN pip3 install pytest==5.4.1 pytest-pep8==1.0.6 pytest-mock==3.1.0 codecov==2.0.22 requests==2.23.0 

RUN mkdir /project; mkdir /project/TMP
VOLUME /project/TMP
WORKDIR /project

RUN echo "You should think about possibly upgrading these outdated packages"
RUN pip3 list --outdated

EXPOSE 8888

CMD bash run_tests.sh

#Check the Dockerfile here https://www.fromlatest.io/#/

#NOTE to contributors: When changing/adding packages, please make sure that the packages are consitent with those
# present within the requirements.txt and test_requirements.txt files
