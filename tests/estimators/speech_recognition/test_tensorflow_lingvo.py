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
from numpy.testing import assert_allclose, assert_array_equal
from scipy.io.wavfile import read

from art.estimators.speech_recognition.speech_recognizer import SpeechRecognizerMixin
from art.estimators.speech_recognition.tensorflow_lingvo import TensorFlowLingvoASR
from art.estimators.tensorflow import TensorFlowV2Estimator
from art.utils import get_file
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


class TestTensorFlowLingvoASR:
    """
    Test the TensorFlowLingvoASR estimator.
    """

    @pytest.mark.skip_module("lingvo")
    @pytest.mark.skip_framework("pytorch", "tensorflow1", "tensorflow2", "mxnet", "kerastf", "non_dl_frameworks")
    def test_is_subclass(self, art_warning):
        try:
            assert issubclass(TensorFlowLingvoASR, (SpeechRecognizerMixin, TensorFlowV2Estimator))
        except ARTTestException as e:
            art_warning(e)

    @pytest.mark.skip_module("lingvo")
    @pytest.mark.skip_framework("pytorch", "tensorflow1", "tensorflow2", "mxnet", "kerastf", "non_dl_frameworks")
    def test_implements_abstract_methods(self, art_warning):
        try:
            TensorFlowLingvoASR()
        except ARTTestException as e:
            art_warning(e)

    @pytest.mark.skip_module("lingvo")
    @pytest.mark.skip_framework("pytorch", "tensorflow1", "tensorflow2", "mxnet", "kerastf", "non_dl_frameworks")
    def test_load_model(self, art_warning):
        try:
            import tensorflow.compat.v1 as tf1

            TensorFlowLingvoASR()
            graph = tf1.get_default_graph()
            assert graph.get_operations()
        except ARTTestException as e:
            art_warning(e)

    @pytest.mark.skip_module("lingvo")
    @pytest.mark.skip_framework("pytorch", "tensorflow1", "tensorflow2", "mxnet", "kerastf", "non_dl_frameworks")
    def test_create_decoder_input(self, art_warning, audio_batch_padded):
        try:
            test_input, test_mask_frequency = audio_batch_padded
            test_target_dummy = np.array(["DUMMY"] * test_input.shape[0])

            lingvo = TensorFlowLingvoASR()
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
        except ARTTestException as e:
            art_warning(e)

    @pytest.mark.skip_module("lingvo")
    @pytest.mark.skip_framework("pytorch", "tensorflow1", "tensorflow2", "mxnet", "kerastf", "non_dl_frameworks")
    def test_create_log_mel_features(self, art_warning, audio_batch_padded):
        try:
            test_input, _ = audio_batch_padded
            lingvo = TensorFlowLingvoASR()
            features_tf = lingvo._create_log_mel_features(lingvo._x_padded)

            features = lingvo._sess.run(features_tf, {lingvo._x_padded: test_input})
            assert features.shape[2] == 80
            assert len(features.shape) == 4
        except ARTTestException as e:
            art_warning(e)

    @pytest.mark.skip_module("lingvo")
    @pytest.mark.skip_framework("pytorch", "tensorflow1", "tensorflow2", "mxnet", "kerastf", "non_dl_frameworks")
    def test_pad_audio_input(self, art_warning):
        try:
            test_input = np.array([np.array([1]), np.array([2] * 480)], dtype=object)
            test_mask = np.array([[True] + [False] * 479, [True] * 480])
            test_output = np.array([[1] + [0] * 479, [2] * 480])

            lingvo = TensorFlowLingvoASR()
            output, mask, mask_freq = lingvo._pad_audio_input(test_input)
            assert_array_equal(test_output, output)
            assert_array_equal(test_mask, mask)
        except ARTTestException as e:
            art_warning(e)

    @pytest.mark.skip_module("lingvo")
    @pytest.mark.skip_framework("pytorch", "tensorflow1", "tensorflow2", "mxnet", "kerastf", "non_dl_frameworks")
    def test_predict_batch(self, art_warning, audio_batch_padded):
        try:
            test_input, test_mask_frequency = audio_batch_padded
            test_target_dummy = np.array(["DUMMY"] * test_input.shape[0])

            lingvo = TensorFlowLingvoASR()
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
        except ARTTestException as e:
            art_warning(e)

    @pytest.mark.skip_module("lingvo")
    @pytest.mark.skip_framework("pytorch", "tensorflow1", "tensorflow2", "mxnet", "kerastf", "non_dl_frameworks")
    def test_predict(self, art_warning, audio_data):
        try:
            test_input = audio_data

            lingvo = TensorFlowLingvoASR()
            predictions = lingvo.predict(test_input, batch_size=2)
            assert predictions.shape == test_input.shape
            assert isinstance(predictions[0], np.str_)
        except ARTTestException as e:
            art_warning(e)

    @pytest.mark.skip_module("lingvo")
    @pytest.mark.skip_framework("pytorch", "tensorflow1", "tensorflow2", "mxnet", "kerastf", "non_dl_frameworks")
    def test_loss_gradient_tensor(self, art_warning, audio_batch_padded):
        try:
            test_input, test_mask_frequency = audio_batch_padded
            test_target_dummy = np.array(["DUMMY"] * test_input.shape[0])

            lingvo = TensorFlowLingvoASR()
            feed_dict = {
                lingvo._x_padded: test_input,
                lingvo._y_target: test_target_dummy,
                lingvo._mask_frequency: test_mask_frequency,
            }
            loss_gradient = lingvo._sess.run(lingvo._loss_gradient_op, feed_dict)
            assert test_input.shape == loss_gradient.shape
            assert loss_gradient.sum() == 0.0
        except ARTTestException as e:
            art_warning(e)

    @pytest.mark.skip_module("lingvo")
    @pytest.mark.skip_framework("pytorch", "tensorflow1", "tensorflow2", "mxnet", "kerastf", "non_dl_frameworks")
    @pytest.mark.parametrize("batch_mode", [True, False])
    def test_loss_gradient_batch_mode(self, art_warning, batch_mode, audio_data):
        try:
            test_input = audio_data
            test_target = np.array(["This", "is", "a dummy", "a dummy"])

            lingvo = TensorFlowLingvoASR()

            if batch_mode:
                gradients = lingvo._loss_gradient_per_batch(test_input, test_target)
            else:
                gradients = lingvo._loss_gradient_per_sequence(test_input, test_target)
            gradients_abs_sum = np.array([np.abs(g).sum() for g in gradients], dtype=object)

            # test shape, equal inputs have equal gradients, non-zero inputs have non-zero gradient sums
            assert [x.shape for x in test_input] == [g.shape for g in gradients]
            assert_allclose(np.abs(gradients[2]).sum(), np.abs(gradients[3]).sum(), rtol=1e-01)
            assert_array_equal(gradients_abs_sum > 0, [False, True, True, True])
        except ARTTestException as e:
            art_warning(e)


class TestTensorFlowLingvoASRLibriSpeechSamples:
    # specify LibriSpeech samples for download and with transcriptions
    samples = {
        "3575-170457-0013.wav": {
            "uri": (
                "https://github.com/tensorflow/cleverhans/blob/6ef939059172901db582c7702eb803b7171e3db5/"
                "examples/adversarial_asr/LibriSpeech/test-clean/3575/170457/3575-170457-0013.wav?raw=true"
            ),
            "transcript": (
                "THE MORE SHE IS ENGAGED IN HER PROPER DUTIES THE LAST LEISURE WILL SHE HAVE FOR IT EVEN AS"
                " AN ACCOMPLISHMENT AND A RECREATION"
            ),
        },
        "5105-28241-0006.wav": {
            "uri": (
                "https://github.com/tensorflow/cleverhans/blob/6ef939059172901db582c7702eb803b7171e3db5/"
                "examples/adversarial_asr/LibriSpeech/test-clean/5105/28241/5105-28241-0006.wav?raw=true"
            ),
            "transcript": (
                "THE LOG AND THE COMPASS THEREFORE WERE ABLE TO BE CALLED UPON TO DO THE WORK OF THE SEXTANT WHICH"
                " HAD BECOME UTTERLY USELESS"
            ),
        },
        "2300-131720-0015.wav": {
            "uri": (
                "https://github.com/tensorflow/cleverhans/blob/6ef939059172901db582c7702eb803b7171e3db5/"
                "examples/adversarial_asr/LibriSpeech/test-clean/2300/131720/2300-131720-0015.wav?raw=true"
            ),
            "transcript": (
                "HE OBTAINED THE DESIRED SPEED AND LOAD WITH A FRICTION BRAKE ALSO REGULATOR OF SPEED BUT WAITED FOR AN"
                " INDICATOR TO VERIFY IT"
            ),
        },
    }

    @pytest.mark.skip_module("lingvo")
    @pytest.mark.skip_framework("pytorch", "tensorflow1", "tensorflow2", "mxnet", "kerastf", "non_dl_frameworks")
    def test_predict(self, art_warning):
        try:
            transcripts = list()
            audios = list()
            for filename, sample in self.samples.items():
                file_path = get_file(filename, sample["uri"])
                _, audio = read(file_path)
                audios.append(audio)
                transcripts.append(sample["transcript"])

            audio_batch = np.array(audios, dtype=object)

            lingvo = TensorFlowLingvoASR()
            prediction = lingvo.predict(audio_batch, batch_size=1)
            assert prediction[0] == transcripts[0]
        except ARTTestException as e:
            art_warning(e)

    @pytest.mark.skip_module("lingvo")
    @pytest.mark.skip_framework("pytorch", "tensorflow1", "tensorflow2", "mxnet", "kerastf", "non_dl_frameworks")
    @pytest.mark.xfail(reason="Known issue that needs further investigation")
    def test_loss_gradient(self, art_warning):
        try:
            transcripts = list()
            audios = list()
            for filename, sample in self.samples.items():
                file_path = get_file(filename, sample["uri"])
                _, audio = read(file_path)
                audios.append(audio)
                transcripts.append(sample["transcript"])

            audio_batch = np.array(audios, dtype=object)
            target_batch = np.array(transcripts)

            lingvo = TensorFlowLingvoASR()
            gradient_batch = lingvo._loss_gradient_per_batch(audio_batch, target_batch)
            gradient_sequence = lingvo._loss_gradient_per_sequence(audio_batch, target_batch)

            gradient_batch_sum = np.array([np.abs(gb).sum() for gb in gradient_batch], dtype=object)
            gradient_sequence_sum = np.array([np.abs(gs).sum() for gs in gradient_sequence], dtype=object)

            # test loss gradients per batch and per sequence are the same
            assert_allclose(gradient_sequence_sum, gradient_batch_sum, rtol=1e-05)
            # test gradient_batch, gradient_sequence and audios items have same shapes
            assert (
                [gb.shape for gb in gradient_batch]
                == [gs.shape for gs in gradient_sequence]
                == [a.shape for a in audios]
            )
        except ARTTestException as e:
            art_warning(e)
