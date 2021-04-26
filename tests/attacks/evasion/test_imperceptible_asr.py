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

from art.attacks.attack import EvasionAttack
from art.attacks.evasion.imperceptible_asr.imperceptible_asr import ImperceptibleASR, PsychoacousticMasker
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


class TestImperceptibleASR:
    """
    Test the ImperceptibleASR attack.
    """

    @pytest.mark.skip_framework("tensorflow1", "tensorflow2", "mxnet", "kerastf", "non_dl_frameworks")
    def test_is_subclass(self, art_warning):
        try:
            assert issubclass(ImperceptibleASR, EvasionAttack)
        except ARTTestException as e:
            art_warning(e)

    @pytest.mark.skip_framework("tensorflow1", "tensorflow2", "mxnet", "kerastf", "non_dl_frameworks")
    def test_implements_abstract_methods(self, art_warning, asr_dummy_estimator):
        try:
            ImperceptibleASR(estimator=asr_dummy_estimator(), masker=PsychoacousticMasker())
        except ARTTestException as e:
            art_warning(e)

    @pytest.mark.skip_framework("tensorflow1", "tensorflow2", "mxnet", "kerastf", "non_dl_frameworks")
    def test_generate(self, art_warning, mocker, asr_dummy_estimator, audio_data):
        try:
            test_input, test_target = audio_data

            mocker.patch.object(ImperceptibleASR, "_generate_batch")

            imperceptible_asr = ImperceptibleASR(estimator=asr_dummy_estimator(), masker=PsychoacousticMasker())
            _ = imperceptible_asr.generate(test_input, test_target)

            imperceptible_asr._generate_batch.assert_called()
        except ARTTestException as e:
            art_warning(e)

    @pytest.mark.skip_framework("tensorflow1", "tensorflow2", "mxnet", "kerastf", "non_dl_frameworks")
    def test_generate_batch(self, art_warning, mocker, asr_dummy_estimator, audio_data):
        try:
            test_input, test_target = audio_data

            mocker.patch.object(ImperceptibleASR, "_create_adversarial")
            mocker.patch.object(ImperceptibleASR, "_create_imperceptible")

            imperceptible_asr = ImperceptibleASR(estimator=asr_dummy_estimator(), masker=PsychoacousticMasker())
            _ = imperceptible_asr._generate_batch(test_input, test_target)

            imperceptible_asr._create_adversarial.assert_called()
            imperceptible_asr._create_imperceptible.assert_called()
        except ARTTestException as e:
            art_warning(e)

    @pytest.mark.skip_framework("tensorflow1", "tensorflow2", "mxnet", "kerastf", "non_dl_frameworks")
    def test_create_adversarial(self, art_warning, mocker, asr_dummy_estimator, audio_data):
        try:
            test_input, test_target = audio_data

            estimator = asr_dummy_estimator()
            mocker.patch.object(estimator, "predict", return_value=test_target)
            mocker.patch.object(
                ImperceptibleASR,
                "_loss_gradient_masking_threshold",
                return_value=(np.zeros_like(audio_data), [0] * test_input.shape[0]),
            )

            # learning rate of zero ensures that adversarial example equals test input
            imperceptible_asr = ImperceptibleASR(
                estimator=estimator, masker=PsychoacousticMasker(), max_iter_1=15, learning_rate_1=0.5
            )
            # learning rate of zero ensures that adversarial example equals test input
            imperceptible_asr.learning_rate_1 = 0
            adversarial = imperceptible_asr._create_adversarial(test_input, test_target)

            # test shape and adversarial example result
            assert [x.shape for x in test_input] == [a.shape for a in adversarial]
            assert [(a - x).sum() for a, x in zip(adversarial, test_input)] == [0.0] * test_input.shape[0]
        except ARTTestException as e:
            art_warning(e)

    @pytest.mark.skip_framework("tensorflow1", "tensorflow2", "mxnet", "kerastf", "non_dl_frameworks")
    def test_create_imperceptible(self, art_warning, mocker, asr_dummy_estimator, audio_data):
        try:
            test_input, test_target = audio_data
            test_adversarial = test_input

            estimator = asr_dummy_estimator()
            mocker.patch.object(estimator, "predict", return_value=test_target)
            mocker.patch.object(
                ImperceptibleASR,
                "_loss_gradient_masking_threshold",
                return_value=(np.zeros_like(test_input), [0] * test_input.shape[0]),
            )

            imperceptible_asr = ImperceptibleASR(
                estimator=estimator, masker=PsychoacousticMasker(), max_iter_2=25, learning_rate_2=0.5
            )
            # learning rate of zero ensures that adversarial example equals test input
            imperceptible_asr.learning_rate_2 = 0
            adversarial = imperceptible_asr._create_imperceptible(test_input, test_adversarial, test_target)

            # test shape and adversarial example result
            assert [x.shape for x in test_input] == [a.shape for a in adversarial]
            assert [(x - a).sum() for a, x in zip(adversarial, test_input)] == [0.0] * test_input.shape[0]
        except ARTTestException as e:
            art_warning(e)

    @pytest.mark.skip_framework("tensorflow", "mxnet", "kerastf", "non_dl_frameworks")
    def test_loss_gradient_masking_threshold(self, art_warning, asr_dummy_estimator, audio_data):
        try:
            test_input, _ = audio_data
            test_delta = test_input * 0

            imperceptible_asr = ImperceptibleASR(estimator=asr_dummy_estimator(), masker=PsychoacousticMasker())

            masking_threshold, psd_maximum = imperceptible_asr._stabilized_threshold_and_psd_maximum(test_input)
            loss_gradient, loss = imperceptible_asr._loss_gradient_masking_threshold(
                test_delta, test_input, masking_threshold, psd_maximum
            )

            assert [g.shape for g in loss_gradient] == [d.shape for d in test_delta]
            assert loss.ndim == 1 and loss.shape == test_delta.shape
        except ARTTestException as e:
            art_warning(e)

    @pytest.mark.skip_framework("pytorch", "tensorflow1", "tensorflow2", "mxnet", "kerastf", "non_dl_frameworks")
    def test_loss_gradient_masking_threshold_tf(self, art_warning, asr_dummy_estimator, audio_batch_padded):
        try:
            import tensorflow.compat.v1 as tf1

            tf1.reset_default_graph()

            test_delta = audio_batch_padded
            test_psd_maximum = np.ones((test_delta.shape[0]))
            test_masking_threshold = np.zeros((test_delta.shape[0], 1025, 28))

            imperceptible_asr = ImperceptibleASR(estimator=asr_dummy_estimator(), masker=PsychoacousticMasker())
            feed_dict = {
                imperceptible_asr._delta: test_delta,
                imperceptible_asr._power_spectral_density_maximum_tf: test_psd_maximum,
                imperceptible_asr._masking_threshold_tf: test_masking_threshold,
            }
            with tf1.Session() as sess:
                loss_gradient, loss = sess.run(imperceptible_asr._loss_gradient_masking_threshold_op_tf, feed_dict)

            assert loss_gradient.shape == test_delta.shape
            assert loss.ndim == 1 and loss.shape[0] == test_delta.shape[0]
        except ARTTestException as e:
            art_warning(e)

    @pytest.mark.skip_framework("tensorflow", "mxnet", "kerastf", "non_dl_frameworks")
    def test_loss_gradient_masking_threshold_torch(self, art_warning, asr_dummy_estimator, audio_batch_padded):
        try:
            test_delta = audio_batch_padded
            test_psd_maximum = np.ones((test_delta.shape[0], 1, 1))
            test_masking_threshold = np.zeros((test_delta.shape[0], 1025, 28))

            imperceptible_asr = ImperceptibleASR(estimator=asr_dummy_estimator(), masker=PsychoacousticMasker())
            loss_gradient, loss = imperceptible_asr._loss_gradient_masking_threshold_torch(
                test_delta, test_psd_maximum, test_masking_threshold
            )

            assert loss_gradient.shape == test_delta.shape
            assert loss.ndim == 1 and loss.shape[0] == test_delta.shape[0]
        except ARTTestException as e:
            art_warning(e)

    @pytest.mark.skip_framework("pytorch", "tensorflow1", "tensorflow2", "mxnet", "kerastf", "non_dl_frameworks")
    def test_approximate_power_spectral_density_tf(self, art_warning, asr_dummy_estimator, audio_batch_padded):
        try:
            import tensorflow.compat.v1 as tf1

            tf1.reset_default_graph()

            test_delta = audio_batch_padded
            test_psd_maximum = np.ones((test_delta.shape[0]))

            masker = PsychoacousticMasker()
            imperceptible_asr = ImperceptibleASR(estimator=asr_dummy_estimator(), masker=masker)
            feed_dict = {
                imperceptible_asr._delta: test_delta,
                imperceptible_asr._power_spectral_density_maximum_tf: test_psd_maximum,
            }

            approximate_psd_tf = imperceptible_asr._approximate_power_spectral_density_tf(
                imperceptible_asr._delta, imperceptible_asr._power_spectral_density_maximum_tf
            )
            with tf1.Session() as sess:
                psd_approximated = sess.run(approximate_psd_tf, feed_dict)

            assert psd_approximated.ndim == 3
            assert psd_approximated.shape[0] == test_delta.shape[0]  # batch_size
            assert psd_approximated.shape[1] == masker.window_size // 2 + 1
        except ARTTestException as e:
            art_warning(e)

    @pytest.mark.skip_framework("tensorflow", "mxnet", "kerastf", "non_dl_frameworks")
    def test_approximate_power_spectral_density_torch(self, art_warning, asr_dummy_estimator, audio_batch_padded):
        try:
            import torch

            test_delta = audio_batch_padded
            test_psd_maximum = np.ones((test_delta.shape[0], 1, 1))

            masker = PsychoacousticMasker()
            imperceptible_asr = ImperceptibleASR(estimator=asr_dummy_estimator(), masker=masker)
            approximate_psd_torch = imperceptible_asr._approximate_power_spectral_density_torch(
                torch.from_numpy(test_delta), torch.from_numpy(test_psd_maximum)
            )
            psd_approximated = approximate_psd_torch.numpy()

            assert psd_approximated.ndim == 3
            assert psd_approximated.shape[0] == test_delta.shape[0]  # batch_size
            assert psd_approximated.shape[1] == masker.window_size // 2 + 1
        except ARTTestException as e:
            art_warning(e)


class TestPsychoacousticMasker:
    """
    Test the PsychoacousticMasker.
    """

    @pytest.mark.framework_agnostic
    def test_power_spectral_density(self, art_warning, audio_sample):
        try:
            test_input = audio_sample

            masker = PsychoacousticMasker()
            psd_matrix, psd_max = masker.power_spectral_density(test_input)

            assert psd_matrix.shape == (masker.window_size // 2 + 1, 28)
            assert np.floor(psd_max) == 78.0
        except ARTTestException as e:
            art_warning(e)

    @pytest.mark.framework_agnostic
    def test_find_maskers(self, art_warning):
        try:
            test_psd_vector = np.array([2, 10, 96, 90, 35, 40, 36, 60, 55, 91, 40])

            masker = PsychoacousticMasker()
            maskers, masker_idx = masker.find_maskers(test_psd_vector)

            # test masker_idx shape and first, last maskers
            assert masker_idx.tolist() == [2, 5, 7, 9]
            np.testing.assert_array_equal(
                maskers[[0, -1]], 10 * np.log10(np.sum(10 ** np.array([[1.0, 9.6, 9.0], [5.5, 9.1, 4.0]]), axis=1))
            )
        except ARTTestException as e:
            art_warning(e)

    @pytest.mark.framework_agnostic
    def test_filter_maskers(self, art_warning):
        try:
            test_psd_vector = np.array([2, 10, 96, 90, 35, 40, 36, 60, 55, 91, 40])
            test_masker_idx = np.array([2, 5, 7, 9])
            test_maskers = test_psd_vector[test_masker_idx]

            masker = PsychoacousticMasker()
            maskers, masker_idx = masker.filter_maskers(test_maskers, test_masker_idx)

            assert masker_idx.tolist() == [2]
            assert maskers.tolist() == [96]
        except ARTTestException as e:
            art_warning(e)

    @pytest.mark.framework_agnostic
    def test_calculate_individual_threshold(self, art_warning, mocker):
        try:
            test_masker_idx = np.array([2, 5, 7, 9])
            test_maskers = np.array([96, 40, 60, 9])

            masker = PsychoacousticMasker()
            threshold = masker.calculate_individual_threshold(test_maskers, test_masker_idx)

            assert threshold.shape == test_masker_idx.shape + (masker.window_size // 2 + 1,)
        except ARTTestException as e:
            art_warning(e)

    @pytest.mark.framework_agnostic
    def test_calculate_global_threshold(self, art_warning, mocker):
        try:
            test_threshold = np.array([[0, 10, 20], [10, 0, 20]])

            mocker.patch(
                "art.attacks.evasion.imperceptible_asr.imperceptible_asr."
                "PsychoacousticMasker.absolute_threshold_hearing",
                new_callable=mocker.PropertyMock,
                return_value=np.zeros(test_threshold.shape[1]),
            )

            masker = PsychoacousticMasker()
            threshold = masker.calculate_global_threshold(test_threshold)

            assert threshold.tolist() == (10 * np.log10([12, 12, 201])).tolist()
        except ARTTestException as e:
            art_warning(e)

    @pytest.mark.framework_agnostic
    def test_calculate_threshold_and_psd_maximum(self, art_warning, audio_sample):
        try:
            test_input = audio_sample

            masker = PsychoacousticMasker()
            threshold, psd_max = masker.calculate_threshold_and_psd_maximum(test_input)

            assert threshold.shape == (masker.window_size // 2 + 1, 28)
            assert np.floor(psd_max) == 78.0
        except ARTTestException as e:
            art_warning(e)
