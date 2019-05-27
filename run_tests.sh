#!/usr/bin/env bash
python -m unittest discover tests/attacks -p 'test_*.py'
python -m unittest discover tests/classifiers -p 'test_*.py'
python -m unittest discover tests/defences -p 'test_*.py'
python -m unittest discover tests/detection -p 'test_*.py'
python -m unittest discover tests/poison_detection -p 'test_*.py'
python -m unittest discover tests/wrappers -p 'test_*.py'
python -m unittest tests.test_data_generators
python -m unittest tests.test_metrics
python -m unittest tests.test_utils
python -m unittest tests.test_visualization
