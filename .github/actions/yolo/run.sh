#!/bin/bash

exit_code=0

pytest --cov-report=xml --cov=art --cov-append -q -vv tests/estimators/object_detection/test_pytorch_yolo.py --framework=pytorch --durations=0
if [[ $? -ne 0 ]]; then exit_code=1; echo "Failed estimators/object_detection/test_pytorch_yolo tests"; fi

pytest --cov-report=xml --cov=art --cov-append -q -vv tests/estimators/object_detection/test_object_seeker_yolo.py --framework=pytorch --durations=0
if [[ $? -ne 0 ]]; then exit_code=1; echo "Failed estimators/object_detection/test_object_seeker_yolo tests"; fi

exit ${exit_code}
