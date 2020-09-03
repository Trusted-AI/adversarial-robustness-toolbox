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
import tensorflow.compat.v1 as tf1
from lingvo.core.hyperparams import Params
from numpy.testing import assert_allclose, assert_array_equal

from art.estimators.sequence.sequence import SequenceNetworkMixin
from art.estimators.sequence.tensorflow import LingvoAsr
from art.estimators.tensorflow import TensorFlowV2Estimator

logger = logging.getLogger(__name__)


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
        ]
    )
    return test_input


@pytest.fixture
def audio_batch_padded():
    """
    Create audio fixtures of shape (batch_size=2,) with elements of variable length.
    """
    sample_rate = 16000
    frequency_length = (sample_rate // 2 + 1) // 240 * 3
    test_input = np.zeros((2, sample_rate))
    test_mask_frequency = np.ones((2, frequency_length, 80))
    return test_input, test_mask_frequency


@pytest.mark.skipif(tf1.__version__ != "2.1.0", reason="requires Tensorflow 2.1.0")
@pytest.mark.skipMlFramework("pytorch", "mxnet", "kerastf", "non_dl_frameworks")
class TestLingvoAsr:
    """
    Test the LingvoAsr estimator.
    """

    def test_is_subclass(self):
        assert issubclass(LingvoAsr, (SequenceNetworkMixin, TensorFlowV2Estimator))

    def test_implements_abstract_methods(self):
        tf1.reset_default_graph()
        LingvoAsr()

    def test_check_and_download_params(self):
        tf1.reset_default_graph()

        lingvo = LingvoAsr()
        assert lingvo._check_and_download_params()

    def test_check_and_download_model(self):
        tf1.reset_default_graph()

        lingvo = LingvoAsr()
        assert lingvo._check_and_download_model()

    def test_check_and_download_vocab(self):
        tf1.reset_default_graph()

        lingvo = LingvoAsr()
        assert lingvo._check_and_download_vocab()

    def test_load_model(self):
        tf1.reset_default_graph()

        LingvoAsr()
        graph = tf1.get_default_graph()
        assert graph.get_operations()

    def test_create_decoder_input(self, audio_batch_padded):
        tf1.reset_default_graph()

        test_input, test_mask_frequency = audio_batch_padded
        test_target_dummy = np.array(["DUMMY"] * test_input.shape[0])

        lingvo = LingvoAsr()
        decoder_input_tf = lingvo._create_decoder_input(lingvo._x_padded, lingvo._y_target, lingvo._mask_frequency)

        decoder_input = lingvo._sess.run(
            decoder_input_tf,
            {
                lingvo._x_padded: test_input,
                lingvo._y_target: test_target_dummy,
                lingvo._mask_frequency: test_mask_frequency,
            },
        )
        assert set(decoder_input.keys()).issuperset({"src", "tgt", "sample_ids"})
        assert set(decoder_input.src.keys()).issuperset({"src_inputs", "paddings"})
        assert set(decoder_input.tgt.keys()).issuperset({"ids", "labels", "paddings", "weights"})

    def test_create_log_mel_features(self, audio_batch_padded):
        tf1.reset_default_graph()

        test_input, _ = audio_batch_padded
        lingvo = LingvoAsr()
        features_tf = lingvo._create_log_mel_features(lingvo._x_padded)

        features = lingvo._sess.run(features_tf, {lingvo._x_padded: test_input})
        assert features.shape[2] == 80
        assert len(features.shape) == 4

    def test_pad_audio_input(self):
        tf1.reset_default_graph()

        test_input = np.array([np.array([1]), np.array([2] * 480)])
        test_mask = np.array([[True] + [False] * 479, [True] * 480])
        test_output = np.array([[1] + [0] * 479, [2] * 480])

        lingvo = LingvoAsr()
        output, mask, mask_freq = lingvo._pad_audio_input(test_input)
        assert_array_equal(test_output, output)
        assert_array_equal(test_mask, mask)

    def test_predict_batch(self, audio_batch_padded):
        tf1.reset_default_graph()

        test_input, test_mask_frequency = audio_batch_padded
        test_target_dummy = np.array(["DUMMY"] * test_input.shape[0])

        lingvo = LingvoAsr()
        feed_dict = {
            lingvo._x_padded: test_input,
            lingvo._y_target: test_target_dummy,
            lingvo._mask_frequency: test_mask_frequency,
        }
        predictions = lingvo._sess.run(lingvo._predict_batch_op, feed_dict)
        assert set(predictions.keys()).issuperset(
            {
                "target_ids",
                "target_labels",
                "target_weights",
                "target_paddings",
                "transcripts",
                "topk_decoded",
                "topk_ids",
                "topk_lens",
                "topk_scores",
                "norm_wer_errors",
                "norm_wer_words",
                "utt_id",
            }
        )

    def test_predict(self, audio_data):
        tf1.reset_default_graph()

        test_input = audio_data

        lingvo = LingvoAsr()
        predictions = lingvo.predict(test_input, batch_size=2)
        assert predictions.shape[0] == test_input.shape[0]
        assert isinstance(predictions[0], np.str_)

    @pytest.mark.skip(reason="requires patched Lingvo install")
    def test_loss_gradient_tensor(self, audio_batch_padded):
        tf1.reset_default_graph()

        test_input, test_mask_frequency = audio_batch_padded
        test_target_dummy = np.array(["DUMMY"] * test_input.shape[0])

        lingvo = LingvoAsr()
        feed_dict = {
            lingvo._x_padded: test_input,
            lingvo._y_target: test_target_dummy,
            lingvo._mask_frequency: test_mask_frequency,
        }
        loss_gradient = lingvo._sess.run(lingvo._loss_gradient_op, feed_dict)
        assert test_input.shape == loss_gradient.shape
        assert loss_gradient.sum() == 0.0

    @pytest.mark.skip(reason="requires patched Lingvo install")
    def test_loss_gradient_per_batch(self, audio_data):
        tf1.reset_default_graph()

        test_input = audio_data
        test_target = np.array(["This", "is", "a dummy", "a dummy"])

        lingvo = LingvoAsr()

        gradients = lingvo._loss_gradient_per_batch(test_input, test_target)
        gradients_abs_sum = np.array([np.abs(g).sum() for g in gradients])

        # test shape, equal inputs have equal gradients, non-zero inputs have non-zero gradient sums
        assert test_input.shape == gradients.shape
        assert_allclose(np.abs(gradients[2]).sum(), np.abs(gradients[3]).sum(), rtol=1e-01)
        assert_array_equal(gradients_abs_sum > 0, [False, True, True, True])

    @pytest.mark.skip(reason="requires patched Lingvo install")
    def test_loss_gradient_per_sequence(self, audio_data):
        tf1.reset_default_graph()

        test_input = audio_data
        test_target = np.array(["This", "is", "a dummy", "a dummy"])

        lingvo = LingvoAsr()

        gradients = lingvo._loss_gradient_per_sequence(test_input, test_target)
        gradients_abs_sum = np.array([np.abs(g).sum() for g in gradients])

        # test shape, equal inputs have equal gradients, non-zero inputs have non-zero gradient sums
        assert test_input.shape == gradients.shape
        assert_allclose(np.abs(gradients[2]).sum(), np.abs(gradients[3]).sum(), rtol=1e-01)
        assert_array_equal(gradients_abs_sum > 0, [False, True, True, True])


if __name__ == "__main__":
    pytest.cmdline.main("-q -s {} --mlFramework=tensorflow --durations=0".format(__file__).split(" "))
