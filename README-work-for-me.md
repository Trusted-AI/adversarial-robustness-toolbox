# ReadMe for Chloe

## getting start
```shell
source p3_8_env/bin/activate
make build # only run once; if you change ART library, you need to build again
make run-jupyter
```

## Open Jyupter Notebook on local

In browser
enter url that shows in the terminal where you run `make run-jupyter`. i.e. http://127.0.0.1:8888/?token=377c9499be5ae269ab26c3cda6f54e01af14fe679718fedd, please note the token changes everytime

## Q&A
Question: Error "No such file or directory: 'git'"
```shell
Collecting git+https://github.com/nottombrown/imagenet_stubs
  Cloning https://github.com/nottombrown/imagenet_stubs to /tmp/pip-req-build-an_vftp0
  Running command git clone -q https://github.com/nottombrown/imagenet_stubs /tmp/pip-req-build-an_vftp0
  ERROR: Error [Errno 2] No such file or directory: 'git': 'git' while executing command git clone -q https://github.com/nottombrown/imagenet_stubs /tmp/pip-req-build-an_vftp0
ERROR: Cannot find command 'git' - do you have 'git' installed and in your PATH?
```
Answer: run command in docker container
```shell
apt-get install git
```
----
Question: Error "AlreadyExistsError: Another metric with the same name already exists."
Abswer: pip install following commands
```shell
# supported versions: (tensorflow==2.2.0 with keras==2.3.1) or (tensorflow==1.15.4 with keras==2.2.5)
pip install tensorflow==2.2.0
pip show tensorflow
pip install keras==2.3.1
pip show keras
```
----
Question: Error "ValueError: The shape of mean and the standard deviation must be identical."


Notes
train the classifier with ART
generate the adversarial test examples
goal: classifier doesn't work on adversarial test examples
Decision based adversarial attack(aka hard-labled blackbox attack)

