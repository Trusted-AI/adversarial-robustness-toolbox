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

        # Define deep speech estimator
        cls.speed_recognizer = PyTorchDeepSpeech(pretrained_model='librispeech', device_type='cpu')

        # Create the optimizer
        parameters = cls.speed_recognizer.model.parameters()
        cls.speed_recognizer._optimizer = torch.optim.SGD(parameters, lr=0.01)

    def test_predict(self):
        # Test probability outputs
        probs, sizes = self.speed_recognizer.predict(self.x, batch_size=2)

        expected_sizes = np.asarray([5, 5, 5])
        np.testing.assert_array_almost_equal(sizes, expected_sizes)

        expected_probs = np.asarray(
            [
                9.9992490e-01,
                5.6676822e-23,
                3.9130475e-07,
                1.9610961e-18,
                4.4049504e-21,
                2.4876512e-17,
                2.7967793e-11,
                7.7903489e-20,
                5.1193400e-26,
                1.2637422e-25,
                1.1681875e-16,
                2.2417779e-19,
                2.0199957e-23,
                8.5335135e-17,
                1.0881066e-12,
                7.4728101e-05,
                9.5411921e-11,
                2.8458338e-18,
                1.2048823e-18,
                5.7043537e-18,
                3.1257310e-16,
                3.0971430e-13,
                2.8484861e-09,
                2.2766614e-21,
                7.3805555e-23,
                9.4372818e-23,
                1.0754367e-18,
                3.9642380e-21,
                2.7693408e-15,
            ]
        )
        np.testing.assert_array_almost_equal(probs[1][1], expected_probs, decimal=2)

        # Test transcription outputs
        transcriptions = self.speed_recognizer.predict(self.x, batch_size=2, transcription_output=True)

        expected_transcriptions = np.array(['', '', ''])
        self.assertTrue((expected_transcriptions == transcriptions).all())

    def test_loss_gradient(self):
        # Create labels
        y = np.array(['SIX', 'HI', 'GOOD'])

        # Compute gradients
        grads = self.speed_recognizer.loss_gradient(self.x, y)
        print(grads[0].shape, grads[1].shape, grads[2].shape)
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
        np.testing.assert_array_almost_equal(grads[0][0 : 20], expected_gradients1, decimal=2)

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
        np.testing.assert_array_almost_equal(grads[1][0: 20], expected_gradients2, decimal=2)

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
        np.testing.assert_array_almost_equal(grads[2][0: 20], expected_gradients3, decimal=2)

    def test_fit(self):
        # Create labels
        y = np.array(['SIX', 'HI', 'GOOD'])

        # Before train
        transcriptions1 = self.speed_recognizer.predict(self.x, batch_size=2, transcription_output=True)

        # Train the estimator
        self.speed_recognizer.fit(x=self.x, y=y, batch_size=2, nb_epochs=5)

        # After train
        transcriptions2 = self.speed_recognizer.predict(self.x, batch_size=2, transcription_output=True)

        self.assertFalse((transcriptions1 == transcriptions2).all())


if __name__ == "__main__":
    unittest.main()
