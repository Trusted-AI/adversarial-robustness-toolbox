#!/bin/bash

exit_code=0

pytest --cov-report=xml --cov=art --cov-append -q -vv tests/estimators/object_detection/test_pytorch_yolo.py --framework=pytorch --durations=0
if [[ $? -ne 0 ]]; then exit_code=1; echo "Failed estimators/object_detection/test_pytorch_yolo tests"; fi

pytest --cov-report=xml --cov=art --cov-append -q -vv tests/estimators/object_detection/test_object_seeker_yolo.py --framework=pytorch --durations=0
if [[ $? -ne 0 ]]; then exit_code=1; echo "Failed estimators/object_detection/test_object_seeker_yolo tests"; fi

pytest --cov-report=xml --cov=art --cov-append -q -vv tests/attacks/test_overload_attack.py --framework=pytorch --durations=0
if [[ $? -ne 0 ]]; then exit_code=1; echo "Failed attacks/test_overload_attack tests"; fi

pytest --cov-report=xml --cov=art --cov-append -q -vv tests/attacks/test_steal_now_attack_later.py --framework=pytorch --durations=0
if [[ $? -ne 0 ]]; then exit_code=1; echo "Failed attacks/teest_steal_now_attack_later tests"; fi


exit ${exit_code}
