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
import logging
import unittest

import numpy as np

from art.defences.postprocessor import GaussianNoise
from art.utils import load_dataset

from tests.utils import master_seed, get_image_classifier_kr_tf, get_image_classifier_kr_tf_binary

logger = logging.getLogger(__name__)


class TestGaussianNoise(unittest.TestCase):
    """
    A unittest class for testing the GaussianNoise postprocessor.
    """

    @classmethod
    def setUpClass(cls):
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset("mnist")
        cls.mnist = (x_train, y_train), (x_test, y_test)

    def setUp(self):
        master_seed(seed=1234)

    def test_gaussian_noise(self):
        """
        Test Gaussian noise.
        """
        (_, _), (x_test, _) = self.mnist
        classifier = get_image_classifier_kr_tf()
        preds = classifier.predict(x_test[0:1])
        postprocessor = GaussianNoise(scale=0.1)
        post_preds = postprocessor(preds=preds)

        classifier_prediction_expected = np.asarray(
            [
                [
                    0.12109935,
                    0.0498215,
                    0.0993958,
                    0.06410096,
                    0.11366928,
                    0.04645343,
                    0.06419807,
                    0.30685693,
                    0.07616714,
                    0.05823757,
                ]
            ],
            dtype=np.float32,
        )
        post_classifier_prediction_expected = np.asarray(
            [[0.15412168, 0.0, 0.2222987, 0.03007976, 0.0381179, 0.12382449, 0.13755375, 0.22279163, 0.07121207, 0.0]],
            dtype=np.float32,
        )

        np.testing.assert_array_almost_equal(preds, classifier_prediction_expected, decimal=4)
        np.testing.assert_array_almost_equal(post_preds, post_classifier_prediction_expected, decimal=4)

    def test_gaussian_noise_binary(self):
        """
        Test Gaussian noise for binary classifier.
        """
        (_, _), (x_test, _) = self.mnist
        classifier = get_image_classifier_kr_tf_binary()
        preds = classifier.predict(x_test[0:1])
        postprocessor = GaussianNoise(scale=0.1)
        post_preds = postprocessor(preds=preds)

        classifier_prediction_expected = np.asarray([[0.5301345]], dtype=np.float32)
        post_classifier_prediction_expected = np.asarray([[0.577278]], dtype=np.float32)

        np.testing.assert_array_almost_equal(preds, classifier_prediction_expected, decimal=4)
        np.testing.assert_array_almost_equal(post_preds, post_classifier_prediction_expected, decimal=4)


if __name__ == "__main__":
    unittest.main()
