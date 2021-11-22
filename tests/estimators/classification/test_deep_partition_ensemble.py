# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2021
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
import os
import unittest

import numpy as np
import tensorflow as tf

from art.utils import load_dataset
from art.estimators.classification.deep_partition_ensemble import DeepPartitionEnsemble

from tests.utils import master_seed, get_image_classifier_pt, get_image_classifier_kr, get_image_classifier_tf

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
logger = logging.getLogger(__name__)

BATCH_SIZE = 100
NB_TRAIN = 1000
NB_TEST = 10
ENSEMBLE_SIZE = 5


class TestDeepPartitionEnsemble(unittest.TestCase):
    """
    This class tests the deep partition ensemble classifier.
    """

    @classmethod
    def setUpClass(cls):
        # Get MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset("mnist")
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]
        cls.mnist = (x_train, y_train), (x_test, y_test)

    def setUp(self):
        master_seed(seed=1234)

    def test_1_tf(self):
        """
        Test with a TensorFlow Classifier.
        :return:
        """
        tf_version = list(map(int, tf.__version__.lower().split("+")[0].split(".")))
        if tf_version[0] == 2:

            # Build TensorFlowV2Classifier
            classifier, _ = get_image_classifier_tf()

            # Get MNIST
            (x_train, y_train), (x_test, y_test) = self.mnist

            # Initialize DPA Classifier
            dpa = DeepPartitionEnsemble(
                classifiers=classifier,
                ensemble_size=ENSEMBLE_SIZE,
                channels_first=classifier.channels_first,
                clip_values=classifier.clip_values,
                preprocessing_defences=classifier.preprocessing_defences,
                postprocessing_defences=classifier.postprocessing_defences,
                preprocessing=classifier.preprocessing,
            )

            # Check basic functionality of DPA Classifier
            # check predict
            y_test_dpa = dpa.predict(x=x_test)
            self.assertEqual(y_test_dpa.shape, y_test.shape)
            self.assertTrue((np.sum(y_test_dpa, axis=1) <= ENSEMBLE_SIZE * np.ones((NB_TEST,))).all())

            # loss gradient
            grad = dpa.loss_gradient(x=x_test, y=y_test, sampling=True)
            assert grad.shape == (10, 28, 28, 1)

            # fit
            dpa.fit(x=x_train, y=y_train)

    def test_2_pt(self):
        """
        Test with a PyTorch Classifier.
        :return:
        """
        # Build KerasClassifier
        classifier = get_image_classifier_pt()

        # Get MNIST
        (x_train, y_train), (x_test, y_test) = self.mnist

        x_test = x_test.transpose(0, 3, 1, 2).astype(np.float32)

        # Initialize DPA Classifier
        dpa = DeepPartitionEnsemble(
                classifiers=classifier,
                ensemble_size=ENSEMBLE_SIZE,
                channels_first=classifier.channels_first,
                clip_values=classifier.clip_values,
                preprocessing_defences=classifier.preprocessing_defences,
                postprocessing_defences=classifier.postprocessing_defences,
                preprocessing=classifier.preprocessing,
            )

        # Check basic functionality of DPA Classifier
        # check predict
        y_test_dpa = dpa.predict(x=x_test)
        self.assertEqual(y_test_dpa.shape, y_test.shape)
        self.assertTrue((np.sum(y_test_dpa, axis=1) <= ENSEMBLE_SIZE * np.ones((NB_TEST,))).all())

        # loss gradient
        grad = dpa.loss_gradient(x=x_test, y=y_test, sampling=True)
        assert grad.shape == (10, 1, 28, 28)

        # fit
        dpa.fit(x=x_train, y=y_train)

    def test_3_kr(self):
        """
        Test with a Keras Classifier.
        :return:
        """
        # Build KerasClassifier
        classifier = get_image_classifier_kr()

        # Get MNIST
        (x_train, y_train), (x_test, y_test) = self.mnist

        # Initialize DPA Classifier
        dpa = DeepPartitionEnsemble(
                classifiers=classifier,
                ensemble_size=ENSEMBLE_SIZE,
                channels_first=classifier.channels_first,
                clip_values=classifier.clip_values,
                preprocessing_defences=classifier.preprocessing_defences,
                postprocessing_defences=classifier.postprocessing_defences,
                preprocessing=classifier.preprocessing,
            )

        # Check basic functionality of DPA Classifier
        # check predict
        y_test_dpa = dpa.predict(x=x_test)
        self.assertEqual(y_test_dpa.shape, y_test.shape)
        self.assertTrue((np.sum(y_test_dpa, axis=1) <= ENSEMBLE_SIZE * np.ones((NB_TEST,))).all())

        # loss gradient
        grad = dpa.loss_gradient(x=x_test, y=y_test, sampling=True)
        assert grad.shape == (10, 28, 28, 1)

        # fit
        dpa.fit(x=x_train, y=y_train)


if __name__ == "__main__":
    unittest.main()
