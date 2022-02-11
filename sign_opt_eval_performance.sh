#!/bin/sh

source ./venv/bin/activate 

# e, query_limite, targeted, start_index
python ./art/attacks/evasion/sign_opt_test_chloe.py 1.5 4000 False 0

python ./art/attacks/evasion/sign_opt_test_chloe.py 1.5 8000 False 0

python ./art/attacks/evasion/sign_opt_test_chloe.py 1.5 14000 False 0