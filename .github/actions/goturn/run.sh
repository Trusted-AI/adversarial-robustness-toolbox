#!/bin/bash

exit_code=0

pytest --cov-report=xml --cov=art --cov-append -q -vv tests/estimators/object_tracking/test_pytorch_goturn.py --framework=pytorch --durations=0
if [[ $? -ne 0 ]]; then exit_code=1; echo "Failed estimators/object_tracking/test_pytorch_goturn tests"; fi

exit ${exit_code}
