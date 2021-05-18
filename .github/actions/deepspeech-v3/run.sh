#!/bin/sh -l

exit_code=0

pytest --cov-report=xml --cov=art --cov-append -q -vv tests/estimators/speech_recognition/test_pytorch_deep_speech.py --framework=pytorch --skip_travis=True --durations=0
if [[ $? -ne 0 ]]; then exit_code=1; echo "Failed estimators/speech_recognition/test_pytorch_deep_speech tests"; fi
pytest --cov-report=xml --cov=art --cov-append -q -vv tests/attacks/evasion/test_imperceptible_asr_pytorch.py --framework=pytorch --skip_travis=True --durations=0
if [[ $? -ne 0 ]]; then exit_code=1; echo "Failed attacks/evasion/test_imperceptible_asr_pytorch tests"; fi

exit ${exit_code}
