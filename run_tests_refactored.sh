#!/usr/bin/env bash
exit_code=0

#mlFrameworkList=("tensorflow" "keras" "pytorch" "scikitlearn")
mlFrameworkList=("tensorflow" "keras" "pytorch")
for mlFramework in "${mlFrameworkList[@]}"; do
  echo "Running tests with framework $mlFramework"
  pytest -q tests/attacks/evasion/ --mlFramework=$mlFramework --durations=0
done

exit ${exit_code}
