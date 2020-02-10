#!/usr/bin/env bash
exit_code=0
tensorflow-keras
mlFrameworkList=("tensorflow" "keras" "pytorch" "scikitlearn")
for mlFramework in "${mlFrameworkList[@]}"; do
  echo "Running tests with framework $mlFramework"
  pytest -q tests/attacks/evasion/ --mlFramework=$mlFramework --durations=0
done

exit ${exit_code}

