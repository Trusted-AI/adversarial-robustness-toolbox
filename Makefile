PROJECT_HOME_DIR := ${CURDIR}


buildAndPush:
	docker build -t project-synthetic .
	docker tag project-synthetic  iribigd01.mul.ie.ibm.com:5000/project-synthetic
	docker push iribigd01.mul.ie.ibm.com:5000/project-synthetic

# make pullResults dir="model_output-2019-Aug-11-14-52-02"
pullResults:
	rsync -a --exclude='*.h5' iribigd01:/projects/data-privacy/killian/TMP/${dir}/ ~/Downloads/${dir}/


# Builds the project environment and installs required dependencies
build:
	python3 -m virtualenv venv
	. venv/bin/activate ; \
	pip3 install -r requirements.txt ; \
#	cat nltk.txt | xargs python3 -m nltk.downloader ; \
	pip3 list --outdated

# Tests the project
test:
	. venv/bin/activate ; \
	pycodestyle --max-line-length=125 stdg && \
	nosetests; \
	pip3 list --outdated


# Run entire pipeline
run_pipeline:
	. venv/bin/activate ; \
	python3 run_project.py


build:
	docker build -t project-art-tensorflow .

run-bash:
	docker  run -it -v ${PWD}:/project/ -p 8888:8888 project-art-tensorflow2  /bin/bash

run-test:
	docker  run --rm  -v ${PWD}:/project/ -p 8888:8888 project-art-tensorflow2

run-pep:
	docker  run --rm -v ${PWD}:/project/ -p 8888:8888 project-art-tensorflow2 py.test --pep8 -m pep8

run-jupyter:
	docker  run --rm  -v ${PWD}:/project/ -p 8888:8888 project-art-tensorflow2 jupyter notebook --ip 0.0.0.0 --no-browser --allow-root

