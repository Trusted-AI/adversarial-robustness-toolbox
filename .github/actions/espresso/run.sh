#!/bin/bash

exit_code=0

pytest --cov-report=xml --cov=art --cov-append -q -vv tests/estimators/speech_recognition/test_pytorch_espresso.py --framework=pytorch --durations=0
if [[ $? -ne 0 ]]; then exit_code=1; echo "Failed estimators/speech_recognition/test_pytorch_espresso tests"; fi

exit ${exit_code}
