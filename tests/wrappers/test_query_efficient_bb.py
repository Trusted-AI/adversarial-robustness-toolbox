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

from art.attacks.evasion.fast_gradient import FastGradientMethod
from art.estimators.classification.keras import KerasClassifier
from art.defences.preprocessor import FeatureSqueezing
from art.utils import load_dataset, get_labels_np_array
from art.wrappers.query_efficient_bb import QueryEfficientBBGradientEstimation

from tests.utils import master_seed, get_image_classifier_kr, get_tabular_classifier_kr

logger = logging.getLogger(__name__)

BATCH_SIZE = 10
NB_TRAIN = 100
NB_TEST = 11


class TestWrappingClassifierAttack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset("mnist")
        x_train, y_train, x_test, y_test = x_train[:NB_TRAIN], y_train[:NB_TRAIN], x_test[:NB_TEST], y_test[:NB_TEST]
        cls.mnist = (x_train, y_train), (x_test, y_test)

        # Keras classifier
        cls.classifier_k = get_image_classifier_kr()

    def setUp(self):
        master_seed(seed=1234)

    @classmethod
    def tearDownClass(cls):
        k.clear_session()

    def test_without_defences(self):
        (x_train, y_train), (x_test, y_test) = self.mnist

        # Get the ready-trained Keras model and wrap it in query efficient gradient estimator wrapper
        classifier = QueryEfficientBBGradientEstimation(self.classifier_k, 20, 1 / 64.0, round_samples=1 / 255.0)

        attack = FastGradientMethod(classifier, eps=1)
        x_train_adv = attack.generate(x_train)
        x_test_adv = attack.generate(x_test)

        self.assertFalse((x_train == x_train_adv).all())
        self.assertFalse((x_test == x_test_adv).all())

        train_y_pred = get_labels_np_array(classifier.predict(x_train_adv))
        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv))

        self.assertFalse((y_train == train_y_pred).all())
        self.assertFalse((y_test == test_y_pred).all())

        preds = classifier.predict(x_train_adv)
        acc = np.sum(np.argmax(preds, axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0]
        logger.info("Accuracy on adversarial train examples with limited query info: %.2f%%", (acc * 100))

        preds = classifier.predict(x_test_adv)
        acc = np.sum(np.argmax(preds, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info("Accuracy on adversarial test examples with limited query info: %.2f%%", (acc * 100))

    def test_with_defences(self):
        (x_train, y_train), (x_test, y_test) = self.mnist

        # Get the ready-trained Keras model
        model = self.classifier_k._model
        fs = FeatureSqueezing(bit_depth=1, clip_values=(0, 1))
        classifier = KerasClassifier(model=model, clip_values=(0, 1), preprocessing_defences=fs)
        # Wrap the classifier
        classifier = QueryEfficientBBGradientEstimation(classifier, 20, 1 / 64.0, round_samples=1 / 255.0)

        attack = FastGradientMethod(classifier, eps=1)
        x_train_adv = attack.generate(x_train)
        x_test_adv = attack.generate(x_test)

        self.assertFalse((x_train == x_train_adv).all())
        self.assertFalse((x_test == x_test_adv).all())

        train_y_pred = get_labels_np_array(classifier.predict(x_train_adv))
        test_y_pred = get_labels_np_array(classifier.predict(x_test_adv))

        self.assertFalse((y_train == train_y_pred).all())
        self.assertFalse((y_test == test_y_pred).all())

        preds = classifier.predict(x_train_adv)
        acc = np.sum(np.argmax(preds, axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0]
        logger.info(
            "Accuracy on adversarial train examples with feature squeezing and limited query info: %.2f%%", (acc * 100)
        )

        preds = classifier.predict(x_test_adv)
        acc = np.sum(np.argmax(preds, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info(
            "Accuracy on adversarial test examples with feature squeezing and limited query info: %.2f%%", (acc * 100)
        )


class TestQueryEfficientVectors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Get Iris
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset("iris")
        cls.iris = (x_train, y_train), (x_test, y_test)

    def setUp(self):
        master_seed(seed=1234)

    def test_iris_clipped(self):
        (_, _), (x_test, y_test) = self.iris

        classifier = get_tabular_classifier_kr()
        classifier = QueryEfficientBBGradientEstimation(classifier, 20, 1 / 64.0, round_samples=1 / 255.0)

        # Test untargeted attack
        attack = FastGradientMethod(classifier, eps=0.1)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1).all())
        self.assertTrue((x_test_adv >= 0).all())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info("Accuracy on Iris with limited query info: %.2f%%", (acc * 100))

    def test_iris_unbounded(self):
        (_, _), (x_test, y_test) = self.iris
        classifier = get_tabular_classifier_kr()

        # Recreate a classifier without clip values
        classifier = KerasClassifier(model=classifier._model, use_logits=False, channels_first=True)
        classifier = QueryEfficientBBGradientEstimation(classifier, 20, 1 / 64.0, round_samples=1 / 255.0)
        attack = FastGradientMethod(classifier, eps=1)
        x_test_adv = attack.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv > 1).any())
        self.assertTrue((x_test_adv < 0).any())

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == preds_adv).all())
        acc = np.sum(preds_adv == np.argmax(y_test, axis=1)) / y_test.shape[0]
        logger.info("Accuracy on Iris with limited query info: %.2f%%", (acc * 100))


if __name__ == "__main__":
    unittest.main()
