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

import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_array_equal

from art.attacks.attack import EvasionAttack
from art.attacks.evasion.imperceptible_asr.imperceptible_asr import ImperceptibleAsr, PsychoacousticMasker
from art.estimators.speech_recognition.speech_recognizer import SpeechRecognizerMixin
from art.estimators.tensorflow import TensorFlowV2Estimator

logger = logging.getLogger(__name__)


@pytest.fixture
def audio_sample():
    """
    Create audio sample.
    """
    sample_rate = 16000
    test_input = np.ones((sample_rate)) * 10e3
    return test_input


@pytest.fixture
def audio_data():
    """
    Create audio fixtures of shape (nb_samples=3,) with elements of variable length.
    """
    sample_rate = 16000
    test_input = np.array(
        [
            np.zeros(sample_rate),
            np.ones(sample_rate * 2) * 2e3,
            np.ones(sample_rate * 3) * 3e3,
            np.ones(sample_rate * 3) * 3e3,
        ],
        dtype=object,
    )
    return test_input


@pytest.fixture
def audio_batch_padded():
    """
    Create audio fixtures of shape (batch_size=2,) with elements of variable length.
    """
    sample_rate = 16000
    test_input = np.zeros((2, sample_rate)) * 2e3
    return test_input


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
        ImperceptibleAsr(estimator=TensorFlowV2AsrDummy(), masker=PsychoacousticMasker())

    def test_create_adversarial_numpy(self, mocker, audio_data):
        test_input = audio_data
        test_target = ["dummy"] * test_input.shape[0]

        mocker.patch.object(TensorFlowV2AsrDummy, "predict", return_value=test_target)

        # learning rate of zero ensures that adversarial example equals test input
        imperceptible_asr = ImperceptibleAsr(
            estimator=TensorFlowV2AsrDummy(), masker=PsychoacousticMasker(), max_iter_1=10, learning_rate_1=0
        )
        adversarial = imperceptible_asr._create_adversarial(test_input, test_target)

        # test shape and adversarial example result
        assert [x.shape for x in test_input] == [a.shape for a in adversarial]
        assert [(a - x).sum() for a, x in zip(adversarial, test_input)] == [0.0] * 4

    # # TODO does only work with TF2 >= 2.2....
    # def test_classifier_type_check_fail(self):
    #    backend_test_classifier_type_check_fail(ImperceptibleAsr, [TensorFlowV2Estimator, SpeechRecognizerMixin])

    @pytest.mark.skipif(tf.__version__ != "2.1.0", reason="requires Tensorflow 2.1.0")
    @pytest.mark.skipMlFramework("pytorch", "mxnet", "kerastf", "non_dl_frameworks")
    def test_loss_gradient_masking_threshold_tf(self, audio_batch_padded):
        import tensorflow.compat.v1 as tf1

        tf1.reset_default_graph()

        test_delta = audio_batch_padded
        test_psd_maxium = np.ones((test_delta.shape[0], 28))
        test_masking_threshold = np.zeros((test_delta.shape[0], 1025, 28))

        # TODO mock masker
        imperceptible_asr = ImperceptibleAsr(
            estimator=TensorFlowV2AsrDummy(), masker=PsychoacousticMasker(), max_iter_1=10, learning_rate_1=0
        )
        feed_dict = {
            imperceptible_asr._delta: test_delta,
            imperceptible_asr._power_spectral_density_maximum_tf: test_psd_maxium,
            imperceptible_asr._masking_threshold_tf: test_masking_threshold,
        }
        with tf1.Session() as sess:
            loss_gradient, loss = sess.run(imperceptible_asr._loss_gradient_masking_threshold_op_tf, feed_dict)

        assert loss_gradient.shape == test_delta.shape
        assert loss.ndim == 1 and loss.shape[0] == test_delta.shape[0]


class TestPsychoacousticMasker:
    """
    Test the PsychoacousticMasker.
    """

    def test_power_spectral_density(self, audio_sample):
        test_input = audio_sample

        masker = PsychoacousticMasker()
        psd_matrix, psd_max = masker.power_spectral_density(test_input)

        assert psd_matrix.shape[0] == masker.window_size // 2 + 1
        assert psd_matrix.shape[1] == psd_max.shape[0]

    def test_find_maskers(self):
        test_psd_vector = np.array([2, 10, 96, 90, 35, 40, 36, 60, 55, 91, 40])

        masker = PsychoacousticMasker()
        maskers, masker_idx = masker.find_maskers(test_psd_vector)

        # test masker_idx shape and first, last maskers
        assert masker_idx.tolist() == [2, 5, 7, 9]
        assert_array_equal(
            maskers[[0, -1]], 10 * np.log10(np.sum(10 ** np.array([[1.0, 9.6, 9.0], [5.5, 9.1, 4.0]]), axis=1))
        )

    def test_filter_maskers(self):
        test_psd_vector = np.array([2, 10, 96, 90, 35, 40, 36, 60, 55, 91, 40])
        test_masker_idx = np.array([2, 5, 7, 9])
        test_maskers = test_psd_vector[test_masker_idx]

        masker = PsychoacousticMasker()
        maskers, masker_idx = masker.filter_maskers(test_maskers, test_masker_idx)

        assert masker_idx.tolist() == [9]
        assert maskers.tolist() == [91]

    def test_calculate_individual_threshold(self, mocker):
        test_masker_idx = np.array([2, 5, 7, 9])
        test_maskers = np.array([96, 40, 60, 9])

        masker = PsychoacousticMasker()
        threshold = masker.calculate_individual_threshold(test_maskers, test_masker_idx)

        assert threshold.shape == test_masker_idx.shape + (masker.window_size // 2 + 1,)

    def test_calculate_global_threshold(self, mocker):
        test_threshold = np.array([[0, 10, 20], [10, 0, 20]])

        mocker.patch(
            "art.attacks.evasion.imperceptible_asr.imperceptible_asr.PsychoacousticMasker.absolute_threshold_hearing",
            new_callable=mocker.PropertyMock,
            return_value=np.zeros(test_threshold.shape[1]),
        )

        masker = PsychoacousticMasker()
        threshold = masker.calculate_global_threshold(test_threshold)

        assert threshold.tolist() == (10 * np.log10([12, 12, 201])).tolist()

    def test_calculate_threshold_and_psd_maximum(self, audio_sample):
        test_input = audio_sample

        masker = PsychoacousticMasker()
        threshold, psd_max = masker.calculate_threshold_and_psd_maximum(test_input)

        assert threshold.shape[1] == psd_max.shape[0]
        assert threshold.shape[0] == masker.window_size // 2 + 1
