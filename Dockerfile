FROM  tensorflow/tensorflow:2.1.0-py3
RUN pip3 install keras==2.3.1

#### NOTE: comment these two lines if you wish to use the tensorflow 1 version of ART instead ####
#FROM tensorflow/tensorflow:1.15.2-py3
#RUN pip3 install keras==2.2.5


RUN pip3 install matplotlib==3.2.1 numpy==1.18.1 scipy==1.4.1 six==1.13.0 Pillow==7.0.0 scikit-learn==0.22.1 lightgbm==2.3.1
RUN pip3 install pytest-pep8==1.0.6 codecov==2.0.22 h5py==2.10.0 requests==2.23.0 statsmodels==0.11.0  cma==2.7.0

#TODO check if jupyter notebook works
RUN pip3 install jupyter==1.0.0 && pip3 install jupyterlab==2.1.0
# https://stackoverflow.com/questions/49024624/how-to-dockerize-jupyter-lab

#RUN pip3 install jupyterlab==2.1.0

RUN pip3 install mxnet==1.6.0
RUN pip3 install xgboost==1.0.0

RUN pip3 install GPy==1.9.9
RUN pip3 install torch==1.4.0 torchvision==0.5.0

RUN mkdir /project; mkdir /project/TMP
VOLUME /project/TMP
WORKDIR /project

RUN mkdir /tmp/.art
COPY ./data_original/ /tmp/.art/
#For some reason the tests within docker are looking for iris.data directly in .art and not in .art/data
COPY ./data_original/data/ /tmp/.art/

#TODO check why I need this data both in /tmp and /root
#RUN mkdir /root/.art
#COPY ./data_original/ /root/.art/
##For some reason the tests within docker are looking for iris.data directly in .art and not in .art/data
#COPY ./data_original/data/ /root/.art/

RUN echo "You should think about possibly upgrading these outdated packages"
RUN pip3 list --outdated

EXPOSE 8888

CMD bash run_tests.sh

#Check the Dockerfile here https://www.fromlatest.io/#/