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

from art.utils import load_dataset
from art.defences.postprocessor import ReverseSigmoid

from tests.utils import master_seed, get_image_classifier_kr_tf, get_image_classifier_kr_tf_binary

logger = logging.getLogger(__name__)


class TestReverseSigmoid(unittest.TestCase):
    """
    A unittest class for testing the ReverseSigmoid postprocessor.
    """

    @classmethod
    def setUpClass(cls):
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset("mnist")
        cls.mnist = (x_train, y_train), (x_test, y_test)

    def setUp(self):
        master_seed(1234)

    def test_reverse_sigmoid(self):
        """
        Test reverse sigmoid.
        """
        (_, _), (x_test, _) = self.mnist
        classifier = get_image_classifier_kr_tf()
        preds = classifier.predict(x_test[0:1])
        postprocessor = ReverseSigmoid(beta=1.0, gamma=0.1)
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
            [
                [
                    0.10733664,
                    0.07743666,
                    0.09712707,
                    0.08230411,
                    0.10377649,
                    0.0764482,
                    0.08234023,
                    0.20600921,
                    0.08703023,
                    0.08019119,
                ]
            ],
            dtype=np.float32,
        )

        np.testing.assert_array_almost_equal(preds, classifier_prediction_expected, decimal=4)
        np.testing.assert_array_almost_equal(post_preds, post_classifier_prediction_expected, decimal=4)

    def test_reverse_sigmoid_beta(self):
        """
        Test reverse sigmoid parameter beta.
        """
        (_, _), (x_test, _) = self.mnist
        classifier = get_image_classifier_kr_tf()
        preds = classifier.predict(x_test[0:1])
        postprocessor = ReverseSigmoid(beta=0.75, gamma=0.1)
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
            [
                [
                    0.1097239,
                    0.07264659,
                    0.09752058,
                    0.07914664,
                    0.10549247,
                    0.07124537,
                    0.07919333,
                    0.22350204,
                    0.08514594,
                    0.07638316,
                ]
            ],
            dtype=np.float32,
        )

        np.testing.assert_array_almost_equal(preds, classifier_prediction_expected, decimal=4)
        np.testing.assert_array_almost_equal(post_preds, post_classifier_prediction_expected, decimal=4)

    def test_reverse_sigmoid_gamma(self):
        """
        Test reverse sigmoid parameter gamma.
        """
        (_, _), (x_test, _) = self.mnist
        classifier = get_image_classifier_kr_tf()
        preds = classifier.predict(x_test[0:1])
        postprocessor = ReverseSigmoid(beta=1.0, gamma=0.5)
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
            [
                [
                    0.09699764,
                    0.10062696,
                    0.09689676,
                    0.09873781,
                    0.0968849,
                    0.10121989,
                    0.0987279,
                    0.11275949,
                    0.09774373,
                    0.09940492,
                ]
            ],
            dtype=np.float32,
        )

        np.testing.assert_array_almost_equal(preds, classifier_prediction_expected, decimal=4)
        np.testing.assert_array_almost_equal(post_preds, post_classifier_prediction_expected, decimal=4)

    def test_reverse_sigmoid_binary(self):
        """
        Test reverse sigmoid for binary classifier.
        """
        (_, _), (x_test, _) = self.mnist
        classifier = get_image_classifier_kr_tf_binary()
        preds = classifier.predict(x_test[0:1])
        postprocessor = ReverseSigmoid(beta=1.0, gamma=0.1)
        post_preds = postprocessor(preds=preds)

        classifier_prediction_expected = np.asarray([[0.5301345]], dtype=np.float32)
        post_classifier_prediction_expected = np.asarray([[0.52711743]], dtype=np.float32)

        np.testing.assert_array_almost_equal(preds, classifier_prediction_expected, decimal=4)
        np.testing.assert_array_almost_equal(post_preds, post_classifier_prediction_expected, decimal=4)

    def test_reverse_sigmoid_beta_binary(self):
        """
        Test reverse sigmoid parameter beta for binary classifier
        """
        (_, _), (x_test, _) = self.mnist
        classifier = get_image_classifier_kr_tf_binary()
        preds = classifier.predict(x_test[0:1])
        postprocessor = ReverseSigmoid(beta=0.75, gamma=0.1)
        post_preds = postprocessor(preds=preds)

        classifier_prediction_expected = np.asarray([[0.5301345]], dtype=np.float32)
        post_classifier_prediction_expected = np.asarray([[0.5278717]], dtype=np.float32)

        np.testing.assert_array_almost_equal(preds, classifier_prediction_expected, decimal=4)
        np.testing.assert_array_almost_equal(post_preds, post_classifier_prediction_expected, decimal=4)

    def test_reverse_sigmoid_gamma_binary(self):
        """
        Test reverse sigmoid parameter gamma for binary classifier
        """
        (_, _), (x_test, _) = self.mnist
        classifier = get_image_classifier_kr_tf_binary()
        preds = classifier.predict(x_test[0:1])
        postprocessor = ReverseSigmoid(beta=1.0, gamma=0.5)
        post_preds = postprocessor(preds=preds)

        classifier_prediction_expected = np.asarray([[0.5301345]], dtype=np.float32)
        post_classifier_prediction_expected = np.asarray([[0.51505363]], dtype=np.float32)

        np.testing.assert_array_almost_equal(preds, classifier_prediction_expected, decimal=4)
        np.testing.assert_array_almost_equal(post_preds, post_classifier_prediction_expected, decimal=4)

    def test_check_params(self):
        with self.assertRaises(ValueError):
            _ = ReverseSigmoid(beta=-1.0, gamma=0.5)

        with self.assertRaises(ValueError):
            _ = ReverseSigmoid(beta=-1.0, gamma=-0.5)


if __name__ == "__main__":
    unittest.main()
