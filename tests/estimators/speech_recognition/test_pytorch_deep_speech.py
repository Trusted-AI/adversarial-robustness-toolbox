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

from art.config import ART_NUMPY_DTYPE
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.mark.skip_module("deepspeech_pytorch")
@pytest.mark.skip_framework("tensorflow", "tensorflow2v1", "keras", "kerastf", "mxnet", "non_dl_frameworks")
@pytest.mark.parametrize("use_amp", [False, True])
@pytest.mark.parametrize("device_type", ["cpu", "gpu"])
def test_pytorch_deep_speech(art_warning, expected_values, use_amp, device_type):
    # Only import if deepspeech_pytorch module is available
    import torch

    from art.estimators.speech_recognition.pytorch_deep_speech import PyTorchDeepSpeech

    try:
        # Skip test if gpu is not available and use_amp is true
        if use_amp and not torch.cuda.is_available():
            return

        # Initialize a speech recognizer
        speech_recognizer = PyTorchDeepSpeech(pretrained_model="librispeech", device_type=device_type, use_amp=use_amp)

        # Get version of DeepSpeech
        version = "v{}".format(speech_recognizer._version)

        # Load data for testing
        expected_data = expected_values()

        x1 = expected_data["x1"]
        x2 = expected_data["x2"]
        x3 = expected_data["x3"]
        expected_sizes = expected_data["expected_sizes"]
        expected_transcriptions1 = expected_data["expected_transcriptions1"]
        expected_transcriptions2 = expected_data["expected_transcriptions2"]
        expected_probs = expected_data["expected_probs"][version]
        expected_gradients1 = expected_data["expected_gradients1"][version]
        expected_gradients2 = expected_data["expected_gradients2"][version]
        expected_gradients3 = expected_data["expected_gradients3"][version]

        # Create signal data
        x = np.array(
            [
                np.array(x1 * 100, dtype=ART_NUMPY_DTYPE),
                np.array(x2 * 100, dtype=ART_NUMPY_DTYPE),
                np.array(x3 * 100, dtype=ART_NUMPY_DTYPE),
            ]
        )

        # Create labels
        y = np.array(["SIX", "HI", "GOOD"])

        # Test probability outputs
        probs, sizes = speech_recognizer.predict(x, batch_size=2, transcription_output=False)

        np.testing.assert_array_almost_equal(probs[1][1], expected_probs, decimal=3)
        np.testing.assert_array_almost_equal(sizes, expected_sizes)

        # Test transcription outputs
        transcriptions = speech_recognizer.predict(x, batch_size=2, transcription_output=True)

        assert (expected_transcriptions1 == transcriptions).all()

        # Test transcription outputs, corner case
        transcriptions = speech_recognizer.predict(np.array([x[0]]), batch_size=2, transcription_output=True)

        assert (expected_transcriptions2 == transcriptions).all()

        # Now test loss gradients
        # Compute gradients
        grads = speech_recognizer.loss_gradient(x, y)

        assert grads[0].shape == (1300,)
        assert grads[1].shape == (1500,)
        assert grads[2].shape == (1400,)

        np.testing.assert_array_almost_equal(grads[0][:20], expected_gradients1, decimal=-2)
        np.testing.assert_array_almost_equal(grads[1][:20], expected_gradients2, decimal=-2)
        np.testing.assert_array_almost_equal(grads[2][:20], expected_gradients3, decimal=-2)

        # Now test fit function
        # Create the optimizer
        parameters = speech_recognizer.model.parameters()
        speech_recognizer._optimizer = torch.optim.SGD(parameters, lr=0.01)

        # Before train
        transcriptions1 = speech_recognizer.predict(x, batch_size=2, transcription_output=True)

        # Train the estimator
        speech_recognizer.fit(x=x, y=y, batch_size=2, nb_epochs=5)

        # After train
        transcriptions2 = speech_recognizer.predict(x, batch_size=2, transcription_output=True)

        assert not ((transcriptions1 == transcriptions2).all())

    except ARTTestException as e:
        art_warning(e)
