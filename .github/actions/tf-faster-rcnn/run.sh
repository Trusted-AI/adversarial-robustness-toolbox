#!/bin/bash

exit_code=0

pytest --cov-report=xml --cov=art --cov-append -q -vv tests/estimators/object_detection/test_tensorflow_faster_rcnn.py --framework=tensorflow --durations=0
if [[ $? -ne 0 ]]; then exit_code=1; echo "Failed estimators/object_detection/test_tensorflow_faster_rcnn.py tests"; fi

pytest --cov-report=xml --cov=art --cov-append -q -vv tests/attacks/test_shapeshifter.py --framework=tensorflow --durations=0
if [[ $? -ne 0 ]]; then exit_code=1; echo "Failed attacks/test_shapeshifter.py tests"; fi

exit ${exit_code}
