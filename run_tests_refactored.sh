#!/usr/bin/env bash
exit_code=0

mlFrameworkList=("tensorflow" "keras" "pytorch" "scikitlearn")
for mlFramework in "${mlFrameworkList[@]}"; do
  echo "Running tests with framework $mlFramework"
#  pytest -q tests/attacks/evasion/ --mlFramework=$mlFramework --durations=0
  pytest -q tests/classifiersT/ --mlFramework=$mlFramework --durations=0
done

#pytest -q tests/classifiers/test_tensorflow.py --mlFramework=tensorflow --durations=0

exit ${exit_code}

