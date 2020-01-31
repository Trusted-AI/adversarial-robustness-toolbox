#!/usr/bin/env bash
exit_code=0

mlFrameworkList=("tensorflow" "keras" "pytorch" "scikitlearn")
for mlFramework in "${mlFrameworkList[@]}"; do
  echo "Running tests with framework $mlFramework"
  export mlFramework=$mlFramework
  python -m unittest tests.attacks.evasion.test_fast_gradient
done

exit ${exit_code}
