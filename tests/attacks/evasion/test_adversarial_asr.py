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
from numpy.testing import assert_array_equal

from art.attacks.attack import EvasionAttack
from art.attacks.evasion.adversarial_asr import CarliniWagnerASR
from art.attacks.evasion.imperceptible_asr.imperceptible_asr import ImperceptibleASR
from art.estimators.estimator import BaseEstimator, LossGradientsMixin, NeuralNetworkMixin
from art.estimators.speech_recognition.speech_recognizer import SpeechRecognizerMixin
from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


class TestImperceptibleASR:
    """
    Test the ImperceptibleASR attack.
    """

    @pytest.mark.framework_agnostic
    def test_is_subclass(self, art_warning):
        try:
            assert issubclass(CarliniWagnerASR, (ImperceptibleASR, EvasionAttack))
        except ARTTestException as e:
            art_warning(e)

    @pytest.mark.skip_framework("tensorflow1", "mxnet", "kerastf", "non_dl_frameworks")
    def test_implements_abstract_methods(self, art_warning, asr_dummy_estimator):
        try:
            CarliniWagnerASR(estimator=asr_dummy_estimator())
        except ARTTestException as e:
            art_warning(e)

    @pytest.mark.framework_agnostic
    def test_classifier_type_check_fail(self, art_warning):
        try:
            backend_test_classifier_type_check_fail(
                CarliniWagnerASR, [NeuralNetworkMixin, LossGradientsMixin, BaseEstimator, SpeechRecognizerMixin]
            )
        except ARTTestException as e:
            art_warning(e)

    @pytest.mark.skip_framework("tensorflow1", "mxnet", "kerastf", "non_dl_frameworks")
    def test_generate_batch(self, art_warning, mocker, asr_dummy_estimator, audio_data):
        try:
            test_input, test_target = audio_data

            # mock _create_adversarial and test if result gets passed unchanged through _create_imperceptible
            mocker.patch.object(CarliniWagnerASR, "_create_adversarial", return_value=test_input)

            carlini_asr = CarliniWagnerASR(estimator=asr_dummy_estimator())
            adversarial = carlini_asr._generate_batch(test_input, test_target)

            carlini_asr._create_adversarial.assert_called()
            for a, t in zip(adversarial, test_input):
                assert_array_equal(a, t)
        except ARTTestException as e:
            art_warning(e)
