#!/bin/sh -l

pytest --cov-report=xml --cov=art --cov-append -q -vv tests/estimators/speech_recognition/test_pytorch_deep_speech.py --framework=pytorch --skip_travis=True --durations=0
pytest --cov-report=xml --cov=art --cov-append -q -vv tests/attacks/evasion/test_imperceptible_asr_pytorch.py --framework=pytorch --skip_travis=True --durations=0
