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
import importlib

import torch
import numpy as np
import pytest

from tests.utils import master_seed

deepspeech_pytorch_spec = importlib.util.find_spec("deepspeech_pytorch")
deepspeech_pytorch_found = deepspeech_pytorch_spec is not None

apex_spec = importlib.util.find_spec("apex")
if apex_spec is not None:
    amp_spec = importlib.util.find_spec("apex.amp")
else:
    amp_spec = None
amp_found = amp_spec is not None

logger = logging.getLogger(__name__)


@pytest.mark.skipif(
    not deepspeech_pytorch_found,
    reason="Skip unittests if deep speech module is not found because of pre-trained model.",
)
@pytest.mark.skipif(not amp_found, reason="Skip unittests if apex module is not found.")
class TestImperceptibleASRPytorch:
    """
    This class tests the ImperceptibleASRPytorch attack.
    """

    @pytest.fixture
    def setup_class(self):
        master_seed(seed=1234)

        # Small data for testing
        x1 = np.array(
            [
                -1.0376293e-03,
                -1.0681478e-03,
                -1.0986663e-03,
                -1.1291848e-03,
                -1.1291848e-03,
                -1.1291848e-03,
                -1.1902219e-03,
                -1.1597034e-03,
                -1.1902219e-03,
                -1.1291848e-03,
                -1.1291848e-03,
                -1.0681478e-03,
                -9.1555528e-04,
            ]
            * 100
        )

        x2 = np.array(
            [
                -1.8311106e-04,
                -1.2207404e-04,
                -6.1037019e-05,
                0.0000000e00,
                3.0518509e-05,
                0.0000000e00,
                -3.0518509e-05,
                0.0000000e00,
                0.0000000e00,
                9.1555528e-05,
                2.1362957e-04,
                3.3570360e-04,
                4.2725913e-04,
                4.5777764e-04,
                -1.8311106e-04,
            ]
            * 100
        )

        x3 = np.array(
            [
                -8.2399976e-04,
                -7.0192572e-04,
                -5.4933317e-04,
                -4.2725913e-04,
                -3.6622211e-04,
                -2.7466659e-04,
                -2.1362957e-04,
                5.4933317e-04,
                5.7985168e-04,
                6.1037019e-04,
                6.7140721e-04,
                7.0192572e-04,
                6.7140721e-04,
                -1.5259255e-04,
            ]
            * 100
        )

        self.x = np.array([x1, x2, x3])

        # Create labels
        self.y = np.array(["S", "I", "GD"])

    @pytest.mark.only_with_platform("pytorch")
    def test_all(self, _test_all):
        pass

    @pytest.fixture(params=[False, True])
    def _test_all(self, request, setup_class):
        # Only import if deep speech module is available
        from art.estimators.speech_recognition.pytorch_deep_speech import PyTorchDeepSpeech
        from art.attacks.evasion.imperceptible_asr.imperceptible_asr_pytorch import ImperceptibleASRPytorch

        # Without amp
        if request.param is False:
            # Create DeepSpeech estimator
            speech_recognizer = PyTorchDeepSpeech(pretrained_model="librispeech")

            # Create attack
            asr_attack = ImperceptibleASRPytorch(
                estimator=speech_recognizer,
                initial_eps=0.001,
                max_iter_1st_stage=50,
                max_iter_2nd_stage=50,
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
                use_amp=False,
                opt_level="O1",
                loss_scale=1,
            )

        # With amp
        else:
            # Create DeepSpeech estimator
            speech_recognizer = PyTorchDeepSpeech(pretrained_model="librispeech", device_type="gpu", use_amp=True)

            # Create attack
            asr_attack = ImperceptibleASRPytorch(
                estimator=speech_recognizer,
                initial_eps=0.001,
                max_iter_1st_stage=50,
                max_iter_2nd_stage=50,
                learning_rate_1st_stage=0.00001,
                learning_rate_2nd_stage=0.001,
                optimizer_1st_stage=torch.optim.SGD,
                optimizer_2nd_stage=torch.optim.SGD,
                global_max_length=2000,
                initial_rescale=1.0,
                rescale_factor=0.8,
                num_iter_adjust_rescale=2,
                initial_alpha=0.01,
                increase_factor_alpha=1.2,
                num_iter_increase_alpha=2,
                decrease_factor_alpha=0.8,
                num_iter_decrease_alpha=2,
                batch_size=2,
                use_amp=True,
                opt_level="O1",
                loss_scale=1,
            )

        # Test transcription output
        transcriptions = speech_recognizer.predict(self.x, batch_size=2, transcription_output=True)

        expected_transcriptions = np.array(["", "", ""])
        assert (expected_transcriptions == transcriptions).all()

        # Generate attack
        x_adv = asr_attack.generate(self.x, self.y)

        # Test shape
        for i in range(3):
            assert x_adv[i].shape == self.x[i].shape

        # Test transcription adversarial output
        # This test is commented by now because of the difference in the prediction function of the estimator
        # in the eval() mode vs the train() mode. This test is already tested with the train() mode of the estimator
        # and it passed. For the eval() mode, we need to test on much larger data sets, i.e., with increasing
        # batch size to hundreds.

        # adv_transcriptions = speech_recognizer.predict(x_adv, batch_size=2, transcription_output=True)
        # assert (adv_transcriptions == self.y).all()


if __name__ == "__main__":
    pytest.cmdline.main("-q -s {} --mlFramework=pytorch --durations=0".format(__file__).split(" "))
