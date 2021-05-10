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
import unittest

import keras.backend as k
import numpy as np

from art.estimators.classification.ensemble import EnsembleClassifier
from tests.utils import TestBase, get_image_classifier_kr

logger = logging.getLogger(__name__)


class TestEnsembleClassifier(TestBase):
    """
    This class tests the ensemble classifier.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # Use twice the same classifier for unit-testing, in application they would be different
        classifier_1 = get_image_classifier_kr()
        classifier_2 = get_image_classifier_kr()
        cls.ensemble = EnsembleClassifier(classifiers=[classifier_1, classifier_2], clip_values=(0, 1))

    @classmethod
    def tearDownClass(cls):
        k.clear_session()

    def test_fit(self):
        with self.assertRaises(NotImplementedError):
            self.ensemble.fit(self.x_train_mnist, self.y_train_mnist)

    def test_fit_generator(self):
        with self.assertRaises(NotImplementedError):
            self.ensemble.fit_generator(None)

    def test_layers(self):
        with self.assertRaises(NotImplementedError):
            self.ensemble.get_activations(self.x_test_mnist, layer=2)

    def test_predict(self):
        predictions = self.ensemble.predict(self.x_test_mnist, raw=False)
        self.assertTrue(predictions.shape, (self.n_test, 10))

        expected_predictions_1 = np.asarray(
            [
                0.12109935,
                0.0498215,
                0.0993958,
                0.06410097,
                0.11366927,
                0.04645343,
                0.06419807,
                0.30685693,
                0.07616713,
                0.05823759,
            ]
        )
        np.testing.assert_array_almost_equal(predictions[0, :], expected_predictions_1, decimal=4)

        predictions_raw = self.ensemble.predict(self.x_test_mnist, raw=True)
        self.assertEqual(predictions_raw.shape, (2, self.n_test, 10))

        expected_predictions_2 = np.asarray(
            [
                0.06054967,
                0.02491075,
                0.0496979,
                0.03205048,
                0.05683463,
                0.02322672,
                0.03209903,
                0.15342847,
                0.03808356,
                0.02911879,
            ]
        )
        np.testing.assert_array_almost_equal(predictions_raw[0, 0, :], expected_predictions_2, decimal=4)

    def test_loss_gradient(self):
        gradients = self.ensemble.loss_gradient(self.x_test_mnist, self.y_test_mnist, raw=False)
        self.assertEqual(gradients.shape, (self.n_test, 28, 28, 1))

        expected_predictions_1 = np.asarray(
            [
                0.0559206,
                0.05338925,
                0.0648919,
                0.07925165,
                -0.04029291,
                -0.11281465,
                0.01850601,
                0.00325054,
                0.08163195,
                0.03333949,
                0.031766,
                -0.02420463,
                -0.07815556,
                -0.04698735,
                0.10711591,
                0.04086434,
                -0.03441073,
                0.01071284,
                -0.04229195,
                -0.01386157,
                0.02827487,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        np.testing.assert_array_almost_equal(gradients[0, 14, :, 0], expected_predictions_1, decimal=4)

        gradients_2 = self.ensemble.loss_gradient(self.x_test_mnist, self.y_test_mnist, raw=True)
        self.assertEqual(gradients_2.shape, (2, self.n_test, 28, 28, 1))

        expected_predictions_2 = np.asarray(
            [
                -0.02444103,
                -0.06092717,
                -0.0449727,
                0.00737736,
                -0.0462507,
                -0.06225448,
                -0.08359106,
                -0.00270847,
                -0.009243,
                -0.00214317,
                -0.04728884,
                0.00369186,
                0.02211389,
                0.02094269,
                0.00219593,
                -0.02638348,
                0.00148741,
                -0.004582,
                -0.00621604,
                0.01604268,
                0.0174383,
                -0.01077293,
                -0.00548703,
                -0.01247547,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        np.testing.assert_array_almost_equal(gradients_2[0, 5, 14, :, 0], expected_predictions_2, decimal=4)

    def test_class_gradient(self):
        gradients = self.ensemble.class_gradient(self.x_test_mnist, None, raw=False)
        self.assertEqual(gradients.shape, (self.n_test, 10, 28, 28, 1))

        expected_predictions_1 = np.asarray(
            [
                -1.0557447e-03,
                -1.0079544e-03,
                -7.7426434e-04,
                1.7387432e-03,
                2.1773507e-03,
                5.0880699e-05,
                1.6497371e-03,
                2.6113100e-03,
                6.0904310e-03,
                4.1080985e-04,
                2.5268078e-03,
                -3.6661502e-04,
                -3.0568996e-03,
                -1.1665225e-03,
                3.8904310e-03,
                3.1726385e-04,
                1.3203260e-03,
                -1.1720930e-04,
                -1.4315104e-03,
                -4.7676818e-04,
                9.7251288e-04,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
            ]
        )
        np.testing.assert_array_almost_equal(gradients[0, 5, 14, :, 0], expected_predictions_1, decimal=4)

        gradients_2 = self.ensemble.class_gradient(self.x_test_mnist, raw=True)
        self.assertEqual(gradients_2.shape, (2, self.n_test, 10, 28, 28, 1))

        expected_predictions_2 = np.asarray(
            [
                -5.2787235e-04,
                -5.0397718e-04,
                -3.8713217e-04,
                8.6937158e-04,
                1.0886753e-03,
                2.5440349e-05,
                8.2486856e-04,
                1.3056550e-03,
                3.0452155e-03,
                2.0540493e-04,
                1.2634039e-03,
                -1.8330751e-04,
                -1.5284498e-03,
                -5.8326125e-04,
                1.9452155e-03,
                1.5863193e-04,
                6.6016300e-04,
                -5.8604652e-05,
                -7.1575522e-04,
                -2.3838409e-04,
                4.8625644e-04,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
                0.0000000e00,
            ]
        )
        np.testing.assert_array_almost_equal(gradients_2[0, 0, 5, 14, :, 0], expected_predictions_2, decimal=4)

    def test_repr(self):
        repr_ = repr(self.ensemble)
        self.assertIn("art.estimators.classification.ensemble.EnsembleClassifier", repr_)
        self.assertIn("classifier_weights=array([0.5, 0.5])", repr_)
        self.assertIn(
            "channels_first=False, clip_values=array([0., 1.], dtype=float32), "
            "preprocessing_defences=None, postprocessing_defences=None, preprocessing=StandardisationMeanStd(mean=0.0, "
            "std=1.0, apply_fit=True, apply_predict=True)",
            repr_,
        )


if __name__ == "__main__":
    unittest.main()
