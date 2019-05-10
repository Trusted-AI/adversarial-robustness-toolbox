#!/usr/bin/env bash
python -m unittest discover art/attacks -p '*_unittest.py'
python -m unittest art/classifiers/*_unittest.py
python -m unittest discover art/defences -p '*_unittest.py'
python -m unittest discover art/detection -p '*_unittest.py'
python -m unittest discover art/poison_detection -p '*_unittest.py'
python -m unittest discover art/wrappers -p '*_unittest.py'
python -m unittest art/*_unittest.py
