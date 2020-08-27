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
import unittest
import importlib

import numpy as np

from tests.utils import master_seed

deepspeech_pytorch_spec = importlib.util.find_spec("deepspeech_pytorch")
deepspeech_pytorch_found = deepspeech_pytorch_spec is not None

logger = logging.getLogger(__name__)


@unittest.skipIf(
    not deepspeech_pytorch_found,
    reason="Skip unittests if deep speech module is not found because of pre-trained model."
)
class TestPyTorchDeepSpeech(unittest.TestCase):
    """
    This class tests the PyTorchDeepSpeech estimator.
    """

    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234)

        # Only import if deep speech module is available
        import torch

        from art.estimators.speed_recognition.pytorch_deep_speech import PyTorchDeepSpeech

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
            ] * 100
        )

        x2 = np.array(
            [
                -1.8311106e-04,
                -1.2207404e-04,
                -6.1037019e-05,
                0.0000000e+00,
                3.0518509e-05,
                0.0000000e+00,
                -3.0518509e-05,
                0.0000000e+00,
                0.0000000e+00,
                9.1555528e-05,
                2.1362957e-04,
                3.3570360e-04,
                4.2725913e-04,
                4.5777764e-04,
                -1.8311106e-04,
            ] * 100
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
            ] * 100
        )
        cls.x = np.array([x1, x2, x3])

        # Define deep speech estimators
        cls.speed_recognizer = PyTorchDeepSpeech(pretrained_model='librispeech', device_type='cpu')
        cls.speed_recognizer_amp = PyTorchDeepSpeech(pretrained_model='librispeech', device_type='gpu', use_amp=True)

    def _test_all(self):
        self._test_predict()
        self._test_loss_gradient()
        self._test_fit()

    def test_all_amp(self):
        self._test_predict()
        self._test_loss_gradient()
        self._test_fit()

    def _test_predict(self):
        # Test probability outputs
        probs, sizes = self.speed_recognizer_amp.predict(self.x, batch_size=2)

        expected_sizes = np.asarray([5, 5, 5])
        np.testing.assert_array_almost_equal(sizes, expected_sizes)

        expected_probs = np.asarray(
            [
                1.0000000e+00,
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
        np.testing.assert_array_almost_equal(probs[1][1], expected_probs, decimal=6)

        # Test transcription outputs
        transcriptions = self.speed_recognizer_amp.predict(self.x, batch_size=2, transcription_output=True)

        expected_transcriptions = np.array(['', '', ''])
        self.assertTrue((expected_transcriptions == transcriptions).all())

    def _test_loss_gradient(self):
        # Create labels
        y = np.array(['SIX', 'HI', 'GOOD'])

        # Compute gradients
        grads = self.speed_recognizer_amp.loss_gradient(self.x, y)

        self.assertTrue(grads[0].shape == (1300,))
        self.assertTrue(grads[1].shape == (1500,))
        self.assertTrue(grads[2].shape == (1400,))

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
                382.65287411
            ]
        )
        np.testing.assert_array_almost_equal(grads[0][0 : 20], expected_gradients1, decimal=1)

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
                4086.51150059
            ]
        )
        np.testing.assert_array_almost_equal(grads[1][0: 20], expected_gradients2, decimal=1)

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
                -11129.57632319
            ]
        )
        np.testing.assert_array_almost_equal(grads[2][0: 20], expected_gradients3, decimal=1)

    def _test_fit(self):
        # Create labels
        y = np.array(['SIX', 'HI', 'GOOD'])

        # Before train
        transcriptions1 = self.speed_recognizer_amp.predict(self.x, batch_size=2, transcription_output=True)

        # Train the estimator
        self.speed_recognizer.fit(x=self.x, y=y, batch_size=2, nb_epochs=5)

        # After train
        transcriptions2 = self.speed_recognizer_amp.predict(self.x, batch_size=2, transcription_output=True)

        self.assertFalse((transcriptions1 == transcriptions2).all())


if __name__ == "__main__":
    unittest.main()
