#!/bin/sh

source ./venv/bin/activate 

test_file=$1
q1=4000
q2=8000
q3=14000
# echo $test_file
# e, query_limite, targeted, start_index, clipped
python $test_file 1.5 $q1 False 0 True

python $test_file 1.5 $q1 False 0 False

python $test_file 1.5 $q2 False 0 True

python $test_file 1.5 $q2 False 0 False

python $test_file 1.5 $q3 False 0 True

python $test_file 1.5 $q3 False 0 False