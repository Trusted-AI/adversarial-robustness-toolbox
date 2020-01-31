#!/usr/bin/env bash
exit_code=0

export mlFramework=keras
python -m unittest tests.attacks.test_fast_gradient

exit ${exit_code}
