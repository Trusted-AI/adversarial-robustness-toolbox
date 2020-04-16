PROJECT_HOME_DIR := ${CURDIR}

build:
	docker build -t project-art-tensorflow .

run-bash:
	docker  run -it -v ${PWD}:/project/ -p 8888:8888 project-art-tensorflow  /bin/bash

run-test:
	docker  run --rm  -v ${PWD}:/project/ -p 8888:8888 project-art-tensorflow

run-pep:
	docker  run --rm -v ${PWD}:/project/ -p 8888:8888 project-art-tensorflow py.test --pep8 -m pep8

run-jupyter:
	docker  run --rm  -v ${PWD}:/project/ -p 8888:8888 project-art-tensorflow jupyter notebook --ip 0.0.0.0 --no-browser --allow-root

