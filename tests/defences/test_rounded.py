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

from art.defences.postprocessor import Rounded
from art.utils import load_dataset

from tests.utils import master_seed, get_image_classifier_kr

logger = logging.getLogger(__name__)


class TestRounded(unittest.TestCase):
    """
    A unittest class for testing the Rounded postprocessor.
    """

    @classmethod
    def setUpClass(cls):
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset("mnist")
        cls.mnist = (x_train, y_train), (x_test, y_test)
        cls.classifier = get_image_classifier_kr()

    def setUp(self):
        master_seed(seed=1234)

    def test_decimals_2(self):
        """
        Test with 2 decimal places.
        """
        (_, _), (x_test, _) = self.mnist
        preds = self.classifier.predict(x_test[0:1])
        postprocessor = Rounded(decimals=2)
        post_preds = postprocessor(preds=preds)

        expected_predictions = np.asarray(
            [[0.12, 0.05, 0.1, 0.06, 0.11, 0.05, 0.06, 0.31, 0.08, 0.06]], dtype=np.float32
        )
        np.testing.assert_array_equal(post_preds, expected_predictions)

    def test_decimals_3(self):
        """
        Test with 3 decimal places.
        """
        (_, _), (x_test, _) = self.mnist
        preds = self.classifier.predict(x_test[0:1])
        postprocessor = Rounded(decimals=3)
        post_preds = postprocessor(preds=preds)

        expected_predictions = np.asarray(
            [[0.121, 0.05, 0.099, 0.064, 0.114, 0.046, 0.064, 0.307, 0.076, 0.058]], dtype=np.float32
        )
        np.testing.assert_array_equal(post_preds, expected_predictions)


if __name__ == "__main__":
    unittest.main()
