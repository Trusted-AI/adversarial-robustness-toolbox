PROJECT_HOME_DIR := ${CURDIR}

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

