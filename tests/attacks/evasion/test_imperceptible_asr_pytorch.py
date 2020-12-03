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


@pytest.mark.skipModule("apex.amp", "deepspeech_pytorch", "torchaudio")
@pytest.mark.skipMlFramework("tensorflow", "keras", "kerastf", "mxnet", "non_dl_frameworks")
@pytest.mark.parametrize("use_amp", [False, True])
@pytest.mark.parametrize("device_type", ["cpu", "gpu"])
def test_imperceptible_asr_pytorch(art_warning, expected_values, use_amp, device_type):
    # Only import if deepspeech_pytorch module is available
    import torch

    from art.estimators.speech_recognition.pytorch_deep_speech import PyTorchDeepSpeech
    from art.attacks.evasion.imperceptible_asr.imperceptible_asr_pytorch import ImperceptibleASRPyTorch
    from art.defences.preprocessor import LFilterPyTorch

    try:
        # Load data for testing
        expected_data = expected_values()

        x1 = expected_data[0]
        x2 = expected_data[1]
        x3 = expected_data[2]

        # Create signal data
        x = np.array(
            [
                np.array(x1 * 100, dtype=ART_NUMPY_DTYPE),
                np.array(x2 * 100, dtype=ART_NUMPY_DTYPE),
                np.array(x3 * 100, dtype=ART_NUMPY_DTYPE),
            ]
        )

        # Create labels
        y = np.array(["S", "I", "GD"])

        # Create DeepSpeech estimator with preprocessing
        numerator_coef = np.array([0.0000001, 0.0000002, -0.0000001, -0.0000002])
        denominator_coef = np.array([1.0, 0.0, 0.0, 0.0])
        audio_filter = LFilterPyTorch(
            numerator_coef=numerator_coef, denominator_coef=denominator_coef, device_type=device_type
        )

        speech_recognizer = PyTorchDeepSpeech(
            pretrained_model="librispeech",
            device_type=device_type,
            use_amp=use_amp,
            preprocessing_defences=audio_filter,
        )

        # Create attack
        asr_attack = ImperceptibleASRPyTorch(
            estimator=speech_recognizer,
            initial_eps=0.001,
            max_iter_1st_stage=5,
            max_iter_2nd_stage=5,
            learning_rate_1st_stage=0.00001,
            learning_rate_2nd_stage=0.001,
            optimizer_1st_stage=torch.optim.SGD,
            optimizer_2nd_stage=torch.optim.SGD,
            global_max_length=2000,
            initial_rescale=1.0,
            rescale_factor=0.8,
            num_iter_adjust_rescale=5,
            initial_alpha=0.01,
            increase_factor_alpha=1.2,
            num_iter_increase_alpha=5,
            decrease_factor_alpha=0.8,
            num_iter_decrease_alpha=5,
            batch_size=2,
            use_amp=use_amp,
            opt_level="O1",
        )

        # Test transcription output
        transcriptions_preprocessing = speech_recognizer.predict(x, batch_size=2, transcription_output=True)

        expected_transcriptions = np.array(["", "", ""])

        assert (expected_transcriptions == transcriptions_preprocessing).all()

        # Generate attack
        x_adv_preprocessing = asr_attack.generate(x, y)

        # Test shape
        assert x_adv_preprocessing[0].shape == x[0].shape
        assert x_adv_preprocessing[1].shape == x[1].shape
        assert x_adv_preprocessing[2].shape == x[2].shape

    except ARTTestException as e:
        art_warning(e)
