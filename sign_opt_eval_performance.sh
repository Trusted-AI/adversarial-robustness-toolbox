#!/bin/sh

source ./venv/bin/activate 

python ./art/attacks/evasion/sign_opt_test_chloe.py 1.5 4000 False 100 0.001

python ./art/attacks/evasion/sign_opt_test_chloe.py 1.5 8000 False 100 0.001

python ./art/attacks/evasion/sign_opt_test_chloe.py 1.5 14000 False 100 0.001