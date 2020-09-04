# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import pytest

from art.attacks.attack import EvasionAttack
from art.attacks.evasion.imperceptible_asr.imperceptible_asr import ImperceptibleAsr
from art.estimators.speech_recognition.speech_recognizer import SpeechRecognizerMixin
from art.estimators.tensorflow import TensorFlowV2Estimator

logger = logging.getLogger(__name__)


class TensorFlowV2AsrDummy(TensorFlowV2Estimator, SpeechRecognizerMixin):
    def get_activations():
        pass

    def predict(self, x):
        pass

    def loss_gradient(self, x, y, **kwargs):
        return x

    def set_learning_phase():
        pass


class TestImperceptibleAsr:
    """
    Test the ImperceptibleAsr attack.
    """

    def test_is_subclass(self):
        assert issubclass(ImperceptibleAsr, EvasionAttack)

    def test_implements_abstract_methods(self):
        ImperceptibleAsr(estimator=TensorFlowV2AsrDummy())

    # # TODO does only work with TF2 >= 2.2....
    # def test_classifier_type_check_fail(self):
    #    backend_test_classifier_type_check_fail(ImperceptibleAsr, [TensorFlowV2Estimator, SpeechRecognizerMixin])
