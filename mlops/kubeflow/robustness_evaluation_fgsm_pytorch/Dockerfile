FROM pytorch/pytorch:latest

RUN pip install adversarial-robustness-toolbox pandas minio flask-cors Pillow torchsummary

ENV APP_HOME /app
COPY src $APP_HOME
WORKDIR $APP_HOME

ENTRYPOINT ["python"]
CMD ["robustness_evaluation_fgsm_pytorch.py"]
