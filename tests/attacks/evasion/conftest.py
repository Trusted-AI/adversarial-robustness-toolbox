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

import numpy as np
import pytest

from art.estimators.pytorch import PyTorchEstimator
from art.estimators.speech_recognition.speech_recognizer import SpeechRecognizerMixin
from art.estimators.tensorflow import TensorFlowV2Estimator
from tests.utils import ARTTestFixtureNotImplemented


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
    test_target = ["DUMMY"] * test_input.shape[0]
    return test_input, test_target


@pytest.fixture
def audio_batch_padded():
    """
    Create audio fixtures of shape (batch_size=2,) with elements of variable length.
    """
    sample_rate = 16000
    test_input = np.ones((2, sample_rate)) * 2e3
    return test_input


@pytest.fixture
def asr_dummy_estimator(framework):
    def _asr_dummy_estimator(**kwargs):
        asr_dummy = None
        if framework in ("tensorflow2v1", "tensorflow2"):

            class TensorFlowV2ASRDummy(TensorFlowV2Estimator, SpeechRecognizerMixin):
                def get_activations(self):
                    pass

                def predict(self, x):
                    pass

                def loss_gradient(self, x, y, **kwargs):
                    return x

                @property
                def input_shape(self):
                    return None

                def compute_loss(self, x, y, **kwargs):
                    pass

            asr_dummy = TensorFlowV2ASRDummy(channels_first=None, model=None, clip_values=None)
        if framework == "pytorch":

            class PyTorchASRDummy(PyTorchEstimator, SpeechRecognizerMixin):
                def get_activations(self):
                    pass

                def predict(self, x):
                    pass

                def loss_gradient(self, x, y, **kwargs):
                    return x

                @property
                def input_shape(self):
                    return None

                def compute_loss(self, x, y, **kwargs):
                    pass

            asr_dummy = PyTorchASRDummy(channels_first=None, model=None, clip_values=None)
        if asr_dummy is None:
            raise ARTTestFixtureNotImplemented(
                "ASR dummy estimator not available for this framework", asr_dummy_estimator.__name__, framework
            )
        return asr_dummy

    return _asr_dummy_estimator
