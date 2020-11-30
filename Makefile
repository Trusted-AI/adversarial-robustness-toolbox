PROJECT_HOME_DIR := ${CURDIR}

build:
    # Builds a TensorFlow 2 ART docker container
    # IMPORTANT ! If you have an existing python env folder make sure to first add it to the `.dockerIgnore` \
    to reduce the size of your the art docker image
	docker build -t project-art-tf2 .

build1:
	# Builds a TensorFlow 1 ART docker container
    # IMPORTANT ! If you have an existing python env folder make sure to first add it to the `.dockerIgnore` \
    to reduce the size of your the art docker image
	docker build -t project-art-tf1 .

run-bash:
	docker  run --rm -it --name project-art-run-bash -v ${PWD}:/project/ -v ~/.art/:/root/.art/ project-art-tf2  /bin/bash

run-test:
	docker  run --rm --name project-art-run-test -v ${PWD}:/project/ -v ~/.art/:/root/.art/  project-art-tf2

run-pep:
	docker  run --rm --name project-art-run-pep -v ${PWD}:/project/ -v ~/.art/:/root/.art/ project-art-tf2 py.test --pep8 -m pep8

run-jupyter:
	docker  run --rm  --name project-art-run-jupyter -v ${PWD}:/project/ -v ~/.art/:/root/.art/ -p 8888:8888 project-art-tf2 jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
