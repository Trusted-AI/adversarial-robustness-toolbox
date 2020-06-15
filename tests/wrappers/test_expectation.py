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
import numpy as np

from art.attacks.evasion.fast_gradient import FastGradientMethod
from art.estimators.classification.keras import KerasClassifier
from art.utils import load_dataset, random_targets
from art.wrappers.expectation import ExpectationOverTransformations
from tests.utils import master_seed, get_image_classifier_kr, get_tabular_classifier_kr

logger = logging.getLogger(__name__)

BATCH_SIZE = 100
NB_TRAIN = 5000
NB_TEST = 10


class TestExpectationOverTransformations(unittest.TestCase):
    """
    A unittest class for testing the Expectation over Transformations in attacks.
    """

    @classmethod
    def setUpClass(cls):
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset("mnist")
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]
        cls.mnist = (x_train, y_train), (x_test, y_test)

    def setUp(self):
        master_seed(seed=1234)

    def test_krclassifier(self):
        """
        Test with a KerasClassifier.
        :return:
        """
        # Build KerasClassifier
        krc = get_image_classifier_kr()

        # Get MNIST
        (_, _), (x_test, y_test) = self.mnist

        # First attack (without EoT):
        fgsm = FastGradientMethod(estimator=krc, targeted=True)
        params = {"y": random_targets(y_test, krc.nb_classes)}
        x_test_adv = fgsm.generate(x_test, **params)

        # Second attack (with EoT):
        def t(x):
            return x

        def transformation():
            while True:
                yield t

        eot = ExpectationOverTransformations(classifier=krc, sample_size=1, transformation=transformation)
        fgsm_with_eot = FastGradientMethod(estimator=eot, targeted=True)
        x_test_adv_with_eot = fgsm_with_eot.generate(x_test, **params)

        self.assertTrue((np.abs(x_test_adv - x_test_adv_with_eot) < 0.001).all())


class TestExpectationVectors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Get Iris
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset("iris")
        cls.iris = (x_train, y_train), (x_test, y_test)

    def setUp(self):
        master_seed(seed=1234)

    def test_iris_clipped(self):
        (_, _), (x_test, y_test) = self.iris

        def t(x):
            return x

        def transformation():
            while True:
                yield t

        classifier = get_tabular_classifier_kr()
        classifier = ExpectationOverTransformations(classifier, sample_size=1, transformation=transformation)

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

        def t(x):
            return x

        def transformation():
            while True:
                yield t

        # Recreate a classifier without clip values
        classifier = KerasClassifier(model=classifier._model, use_logits=False, channels_first=True)
        classifier = ExpectationOverTransformations(classifier, sample_size=1, transformation=transformation)
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
