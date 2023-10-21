#!/usr/bin/env bash

exit_code=0

# Set TensorFlow logging to minimum level ERROR
export TF_CPP_MIN_LOG_LEVEL="3"

# --------------------------------------------------------------------------------------------------------------- TESTS

# NOTE: All the tests should be ran within this loop. All other tests are legacy tests that must be made framework
# independent to be incorporated within this loop
frameworkList=("tensorflow" "keras" "pytorch" "scikitlearn" "mxnet" "kerastf")
framework=$1
legacy_module=$2

if [[ ${framework} != "legacy" ]]
then
    echo "#######################################################################"
    echo "############### Running tests with framework $framework ###############"
    echo "#######################################################################"

    pytest --cov-report=xml --cov=art --cov-append  -q -vv tests/attacks/evasion/test_multimodal_attack.py --framework=$framework --durations=0
    if [[ $? -ne 0 ]]; then exit_code=1; echo "Failed multimodal tests"; fi

fi

exit ${exit_code}
