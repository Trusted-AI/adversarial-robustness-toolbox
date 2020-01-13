# MIT License
#
# Copyright (C) IBM Corporation 2018
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

from art.defences import FeatureSqueezing, JpegCompression, SpatialSmoothing
from art.utils import load_dataset, master_seed
from tests.utils_test import get_classifier_bb

logger = logging.getLogger(__name__)

BATCH_SIZE = 10
NB_TRAIN = 500
NB_TEST = 100


class TestBlackBoxClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset('mnist')

        cls.x_train = x_train[:NB_TRAIN]
        cls.y_train = y_train[:NB_TRAIN]
        cls.x_test = x_test[:NB_TEST]
        cls.y_test = y_test[:NB_TEST]

        # Temporary folder for tests
        cls.test_dir = tempfile.mkdtemp()

    def setUp(self):
        master_seed(1234)

    def test_fit(self):
        classifier = get_classifier_bb()
        self.assertRaises(NotImplementedError, lambda: classifier.fit(self.x_train, self.y_train, batch_size=BATCH_SIZE,
                                                                      nb_epochs=2))

    def test_shapes(self):
        classifier = get_classifier_bb()
        predictions = classifier.predict(self.x_test)
        self.assertEqual(predictions.shape, self.y_test.shape)
        self.assertEqual(classifier.nb_classes(), 10)
        self.assertEqual(predictions.shape, self.y_test.shape)

    def test_defences_predict(self):
        clip_values = (0, 1)
        fs = FeatureSqueezing(clip_values=clip_values, bit_depth=2)
        jpeg = JpegCompression(clip_values=clip_values, apply_predict=True)
        smooth = SpatialSmoothing()
        classifier = get_classifier_bb(defences=[fs, jpeg, smooth])
        self.assertEqual(len(classifier.defences), 3)

        predictions_classifier = classifier.predict(self.x_test)

        # Apply the same defences by hand
        x_test_defense = self.x_test
        x_test_defense, _ = fs(x_test_defense, self.y_test)
        x_test_defense, _ = jpeg(x_test_defense, self.y_test)
        x_test_defense, _ = smooth(x_test_defense, self.y_test)
        classifier = get_classifier_bb()
        predictions_check = classifier.predict(x_test_defense)

        # Check that the prediction results match
        np.testing.assert_array_almost_equal(predictions_classifier, predictions_check, decimal=4)

    def test_save(self):
        path = 'tmp'
        filename = 'model.h5'

        classifier = get_classifier_bb()

        self.assertRaises(NotImplementedError, lambda: classifier.save(filename, path=path))

    def test_repr(self):
        classifier = get_classifier_bb()
        repr_ = repr(classifier)

        self.assertIn('BlackBoxClassifier', repr_)
        self.assertIn('clip_values=(0, 255)', repr_)
        self.assertIn('defences=None', repr_)
        self.assertIn('preprocessing=(0, 1)', repr_)
