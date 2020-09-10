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
class TestPyTorchDeepSpeech:
    """
    This class tests the PyTorchDeepSpeech estimator.
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

    def test_all(self, _test_all):
        pass

    @pytest.fixture(params=[False, True])
    def _test_all(self, request, setup_class):
        # Only import if deep speech module is available
        from art.estimators.speech_recognition.pytorch_deep_speech import PyTorchDeepSpeech

        # Test probability outputs
        if request.param is True:
            self.speech_recognizer_amp = PyTorchDeepSpeech(
                pretrained_model="librispeech", device_type="gpu", use_amp=True
            )
            probs, sizes = self.speech_recognizer_amp.predict(self.x, batch_size=2)

        else:
            self.speech_recognizer = PyTorchDeepSpeech(pretrained_model="librispeech")
            probs, sizes = self.speech_recognizer.predict(self.x, batch_size=2)

        expected_sizes = np.asarray([5, 5, 5])
        np.testing.assert_array_almost_equal(sizes, expected_sizes)

        expected_probs = np.asarray(
            [
                1.0000000e00,
                7.0154901e-14,
                1.9170589e-13,
                8.2194836e-13,
                8.9967915e-13,
                1.8518193e-12,
                1.7883164e-10,
                1.8951663e-12,
                1.8818237e-13,
                3.2806991e-12,
                3.5664666e-16,
                3.3147299e-14,
                2.3439516e-13,
                8.4845603e-12,
                1.2017718e-13,
                1.1180213e-12,
                6.5572378e-15,
                3.0194697e-12,
                4.9065188e-15,
                1.9765363e-13,
                4.1670646e-11,
                2.6884213e-12,
                1.1436632e-13,
                7.1931783e-15,
                2.8135227e-11,
                4.5599673e-14,
                6.4587983e-13,
                2.4159567e-15,
                4.6668241e-13,
            ]
        )
        np.testing.assert_array_almost_equal(probs[1][1], expected_probs, decimal=3)

        # Test transcription outputs
        if request.param is True:
            transcriptions = self.speech_recognizer_amp.predict(self.x, batch_size=2, transcription_output=True)
        else:
            transcriptions = self.speech_recognizer.predict(self.x, batch_size=2, transcription_output=True)

        expected_transcriptions = np.array(["", "", ""])
        assert (expected_transcriptions == transcriptions).all()

        # Now test loss gradients
        # Create labels
        y = np.array(["SIX", "HI", "GOOD"])

        # Compute gradients
        if request.param is True:
            grads = self.speech_recognizer_amp.loss_gradient(self.x, y)
        else:
            grads = self.speech_recognizer.loss_gradient(self.x, y)

        assert grads[0].shape == (1300,)
        assert grads[1].shape == (1500,)
        assert grads[2].shape == (1400,)

        if request.param is True:
            expected_gradients1 = np.asarray(
                [
                    -3485.7,
                    659.0,
                    -111.7,
                    283.6,
                    1691.9,
                    715.0,
                    1480.4,
                    -3522.3,
                    -4087.9,
                    -8824.2,
                    -304.7,
                    2013.4,
                    -445.1,
                    4125.0,
                    1754.1,
                    -503.6,
                    1160.0,
                    7051.7,
                    -1992.2,
                    350.4,
                ]
            )

        else:
            expected_gradients1 = np.asarray(
                [
                    -3482.77892371,
                    665.64673575,
                    -116.24408896,
                    265.93803869,
                    1667.02236699,
                    688.33557577,
                    1455.14911883,
                    -3524.90476617,
                    -4082.06471587,
                    -8802.39419605,
                    -277.74274789,
                    2034.54679277,
                    -428.53153241,
                    4114.63683848,
                    1722.53840709,
                    -513.68916798,
                    1159.88786568,
                    7072.47761446,
                    -1963.71829047,
                    382.65287411,
                ]
            )
        np.testing.assert_array_almost_equal(grads[0][0:20], expected_gradients1, decimal=0)

        if request.param is True:
            expected_gradients2 = np.asarray(
                [
                    20924.5,
                    3046.3,
                    -7872.5,
                    15525.1,
                    -15766.9,
                    -18494.1,
                    19139.6,
                    6446.2,
                    26323.1,
                    4230.0,
                    -31122.4,
                    -2890.9,
                    12936.7,
                    13834.1,
                    17649.9,
                    8866.1,
                    -16454.6,
                    -6953.1,
                    -17899.6,
                    4100.7,
                ]
            )

        else:
            expected_gradients2 = np.asarray(
                [
                    20992.44844133,
                    3048.78701634,
                    -7849.13725934,
                    15557.59663939,
                    -15760.10725159,
                    -18422.9438386,
                    19132.22699435,
                    6508.51437337,
                    26292.5249963,
                    4232.62414548,
                    -31128.82664215,
                    -2894.85284984,
                    13008.74538039,
                    13845.08921681,
                    17657.67725957,
                    8807.42144017,
                    -16477.89414508,
                    -6977.8092622,
                    -17914.22352666,
                    4086.51150059,
                ]
            )
        np.testing.assert_array_almost_equal(grads[1][0:20], expected_gradients2, decimal=0)

        if request.param is True:
            expected_gradients3 = np.asarray(
                [
                    -1687.3,
                    6715.0,
                    16448.4,
                    -3848.9,
                    16521.1,
                    -15736.1,
                    -26204.0,
                    -8992.2,
                    9697.9,
                    13999.6,
                    -7595.3,
                    14181.0,
                    -24507.2,
                    5481.9,
                    7166.7,
                    -6182.3,
                    2510.3,
                    -7229.0,
                    -10821.9,
                    -11134.2,
                ]
            )

        else:
            expected_gradients3 = np.asarray(
                [
                    -1693.10472689,
                    6711.39788693,
                    16480.14166546,
                    -3786.95541286,
                    16448.3969823,
                    -15702.45621671,
                    -26162.89260564,
                    -8979.81601681,
                    9657.87483965,
                    13955.78845296,
                    -7552.01438108,
                    14170.60635269,
                    -24434.37243957,
                    5502.81163675,
                    7171.56926943,
                    -6154.06511686,
                    2483.93980406,
                    -7244.24618697,
                    -10798.70438903,
                    -11129.57632319,
                ]
            )
        np.testing.assert_array_almost_equal(grads[2][0:20], expected_gradients3, decimal=0)

        # Now test fit function
        if request.param is True:
            # Before train
            transcriptions1 = self.speech_recognizer_amp.predict(self.x, batch_size=2, transcription_output=True)

            # Train the estimator
            self.speech_recognizer_amp.fit(x=self.x, y=y, batch_size=2, nb_epochs=5)

            # After train
            transcriptions2 = self.speech_recognizer_amp.predict(self.x, batch_size=2, transcription_output=True)

            assert not ((transcriptions1 == transcriptions2).all())

        else:
            # Before train
            transcriptions1 = self.speech_recognizer.predict(self.x, batch_size=2, transcription_output=True)

            # Train the estimator
            self.speech_recognizer.fit(x=self.x, y=y, batch_size=2, nb_epochs=5)

            # After train
            transcriptions2 = self.speech_recognizer.predict(self.x, batch_size=2, transcription_output=True)

            assert not ((transcriptions1 == transcriptions2).all())


if __name__ == "__main__":
    pytest.cmdline.main("-q -s {} --mlFramework=pytorch --durations=0".format(__file__).split(" "))
