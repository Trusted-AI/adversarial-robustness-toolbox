PROJECT_HOME_DIR := ${CURDIR}

build:
    # If you have an existing python env folder make sure to first add it to the `.dockerIgnore` \
    to reduce the size of your the art docker image
	docker build -t project-art-tensorflow .

run-bash:
	docker  run -it -v ${PWD}:/project/ -v ~/.art/:/root/.art/ -p 8888:8888 project-art-tensorflow  /bin/bash

run-test:
	#Download datasets needed for the unit tests at ` ~/.art/data`
	docker  run --rm  -v ${PWD}:/project/ -v ~/.art/:/root/.art/ -p 8888:8888 project-art-tensorflow python3 download_datasets.py
	docker  run --rm  -v ${PWD}:/project/ -v ~/.art/:/root/.art/ -p 8888:8888 project-art-tensorflow

run-pep:
	docker  run --rm -v ${PWD}:/project/ -v ~/.art/:/root/.art/ -p 8888:8888 project-art-tensorflow py.test --pep8 -m pep8

run-jupyter:
	docker  run --rm  -v ${PWD}:/project/ -v ~/.art/:/root/.art/ -p 8888:8888 project-art-tensorflow jupyter notebook --ip 0.0.0.0 --no-browser --allow-root

