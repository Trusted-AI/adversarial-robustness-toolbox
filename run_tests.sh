#!/usr/bin/env bash
exit_code=0
python -m unittest discover tests/attacks -p 'test_[a-d]*.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest discover tests/attacks -p 'test_[e-h]*.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest discover tests/attacks -p 'test_[i-l]*.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest discover tests/attacks -p 'test_[m-p]*.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest discover tests/attacks -p 'test_[q-t]*.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest discover tests/attacks -p 'test_[u-z]*.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest discover tests/classifiers -p 'test_[a-e]*.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest discover tests/classifiers -p 'test_[f-l]*.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest discover tests/classifiers -p 'test_[m-q]*.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest discover tests/classifiers -p 'test_[r-z]*.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest discover tests/defences -p 'test_*.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest discover tests/detection -p 'test_*.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest discover tests/poison_detection -p 'test_*.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest discover tests/wrappers -p 'test_*.py'
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest tests.test_data_generators
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest tests.test_metrics
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest tests.test_utils
if [[ $? -ne 0 ]]; then exit_code=1; fi
python -m unittest tests.test_visualization
if [[ $? -ne 0 ]]; then exit_code=1; fi
exit $exit_code