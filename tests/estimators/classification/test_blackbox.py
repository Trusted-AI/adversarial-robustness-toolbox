# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
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
import tempfile
import unittest

import numpy as np

from art.defences.preprocessor import FeatureSqueezing, JpegCompression, SpatialSmoothing

from tests.utils import TestBase, get_classifier_bb, get_classifier_bb_nn

logger = logging.getLogger(__name__)


class TestBlackBoxClassifier(TestBase):
    """
    This class tests the BlackBox classifier.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.test_dir = tempfile.mkdtemp()

    def test_fit(self):
        classifier = get_classifier_bb()
        self.assertRaises(
            NotImplementedError,
            lambda: classifier.fit(self.x_train_mnist, self.y_train_mnist, batch_size=self.batch_size, nb_epochs=2),
        )

    def test_shapes(self):
        classifier = get_classifier_bb()
        predictions = classifier.predict(self.x_test_mnist)
        self.assertEqual(predictions.shape, self.y_test_mnist.shape)
        self.assertEqual(classifier.nb_classes, 10)
        self.assertEqual(predictions.shape, self.y_test_mnist.shape)

    def test_defences_predict(self):
        clip_values = (0, 1)
        fs = FeatureSqueezing(clip_values=clip_values, bit_depth=2)
        jpeg = JpegCompression(clip_values=clip_values, apply_predict=True)
        smooth = SpatialSmoothing()
        classifier = get_classifier_bb(defences=[fs, jpeg, smooth])
        self.assertEqual(len(classifier.preprocessing_defences), 3)

        predictions_classifier = classifier.predict(self.x_test_mnist)

        # Apply the same defences by hand
        x_test_defense = self.x_test_mnist
        x_test_defense, _ = fs(x_test_defense, self.y_test_mnist)
        x_test_defense, _ = jpeg(x_test_defense, self.y_test_mnist)
        x_test_defense, _ = smooth(x_test_defense, self.y_test_mnist)
        classifier = get_classifier_bb()
        predictions_check = classifier.predict(x_test_defense)

        # Check that the prediction results match
        np.testing.assert_array_almost_equal(predictions_classifier, predictions_check, decimal=4)

    def test_save(self):
        path = "tmp"
        filename = "model.h5"

        classifier = get_classifier_bb()

        self.assertRaises(NotImplementedError, lambda: classifier.save(filename, path=path))

    def test_repr(self):
        classifier = get_classifier_bb()
        repr_ = repr(classifier)

        self.assertIn("BlackBoxClassifier", repr_)
        self.assertIn("clip_values=[  0. 255.]", repr_)
        self.assertIn("defences=None", repr_)
        self.assertIn(
            "preprocessing=StandardisationMeanStd(mean=0.0, std=1.0, apply_fit=True, apply_predict=True)", repr_
        )


class TestBlackBoxClassifierNeuralNetwork(TestBase):
    """
    This class tests the BlackBox Neural Network classifier.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.test_dir = tempfile.mkdtemp()

    def test_fit(self):
        classifier = get_classifier_bb_nn()
        self.assertRaises(
            NotImplementedError,
            lambda: classifier.fit(self.x_train_mnist, self.y_train_mnist, batch_size=self.batch_size, nb_epochs=2),
        )

    def test_shapes(self):
        classifier = get_classifier_bb_nn()
        predictions = classifier.predict(self.x_test_mnist)
        self.assertEqual(predictions.shape, self.y_test_mnist.shape)
        self.assertEqual(classifier.nb_classes, 10)
        self.assertEqual(predictions.shape, self.y_test_mnist.shape)

    def test_defences_predict(self):
        clip_values = (0, 1)
        fs = FeatureSqueezing(clip_values=clip_values, bit_depth=2)
        jpeg = JpegCompression(clip_values=clip_values, apply_predict=True)
        smooth = SpatialSmoothing()
        classifier = get_classifier_bb_nn(defences=[fs, jpeg, smooth])
        self.assertEqual(len(classifier.preprocessing_defences), 3)

        predictions_classifier = classifier.predict(self.x_test_mnist)

        # Apply the same defences by hand
        x_test_defense = self.x_test_mnist
        x_test_defense, _ = fs(x_test_defense, self.y_test_mnist)
        x_test_defense, _ = jpeg(x_test_defense, self.y_test_mnist)
        x_test_defense, _ = smooth(x_test_defense, self.y_test_mnist)
        classifier = get_classifier_bb_nn()
        predictions_check = classifier.predict(x_test_defense)

        # Check that the prediction results match
        np.testing.assert_array_almost_equal(predictions_classifier, predictions_check, decimal=4)

    def test_repr(self):
        classifier = get_classifier_bb_nn()
        repr_ = repr(classifier)

        self.assertIn("BlackBoxClassifier", repr_)
        self.assertIn("clip_values=[  0. 255.]", repr_)
        self.assertIn("defences=None", repr_)
        self.assertIn(
            "preprocessing=StandardisationMeanStd(mean=0.0, std=1.0, apply_fit=True, apply_predict=True)", repr_
        )


if __name__ == "__main__":
    unittest.main()
