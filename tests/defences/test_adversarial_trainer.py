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
import unittest

import numpy as np

from art.attacks import FastGradientMethod, DeepFool
from art.data_generators import DataGenerator
from art.defences import AdversarialTrainer
from art.utils import load_mnist, master_seed
from tests.utils_test import get_classifier_tf

logger = logging.getLogger(__name__)

BATCH_SIZE = 10
NB_TRAIN = 100
NB_TEST = 11


class TestAdversarialTrainer(unittest.TestCase):
    """
    Test cases for the AdversarialTrainer class.
    """

    @classmethod
    def setUpClass(cls):
        # MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train, x_test, y_test = x_train[:NB_TRAIN], y_train[:NB_TRAIN], x_test[:NB_TEST], y_test[:NB_TEST]
        cls.mnist = ((x_train, y_train), (x_test, y_test))

        cls.classifier, _ = get_classifier_tf()

    def setUp(self):
        master_seed(1234)

    def test_classifier_match(self):
        attack = FastGradientMethod(self.classifier)
        adv_trainer = AdversarialTrainer(self.classifier, attack)

        self.assertEqual(len(adv_trainer.attacks), 1)
        self.assertEqual(adv_trainer.attacks[0].classifier, adv_trainer.classifier)

    def test_fit_predict(self):
        (x_train, y_train), (x_test, y_test) = self.mnist
        x_test_original = x_test.copy()

        attack = FastGradientMethod(self.classifier)
        x_test_adv = attack.generate(x_test)
        predictions = np.argmax(self.classifier.predict(x_test_adv), axis=1)
        accuracy = np.sum(predictions == np.argmax(y_test, axis=1)) / NB_TEST

        adv_trainer = AdversarialTrainer(self.classifier, attack)
        adv_trainer.fit(x_train, y_train, nb_epochs=5, batch_size=128)

        predictions_new = np.argmax(adv_trainer.predict(x_test_adv), axis=1)
        accuracy_new = np.sum(predictions_new == np.argmax(y_test, axis=1)) / NB_TEST

        self.assertEqual(accuracy_new, 0.09090909090909091)
        self.assertEqual(accuracy, 0.18181818181818182)

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)

    def test_two_attacks(self):
        (x_train, y_train), (x_test, y_test) = self.mnist
        x_test_original = x_test.copy()

        attack1 = FastGradientMethod(self.classifier)
        attack2 = DeepFool(self.classifier)
        x_test_adv = attack1.generate(x_test)
        predictions = np.argmax(self.classifier.predict(x_test_adv), axis=1)
        accuracy = np.sum(predictions == np.argmax(y_test, axis=1)) / NB_TEST

        adv_trainer = AdversarialTrainer(self.classifier, attacks=[attack1, attack2])
        adv_trainer.fit(x_train, y_train, nb_epochs=5, batch_size=128)

        predictions_new = np.argmax(adv_trainer.predict(x_test_adv), axis=1)
        accuracy_new = np.sum(predictions_new == np.argmax(y_test, axis=1)) / NB_TEST
        self.assertGreaterEqual(accuracy_new, accuracy)

        self.assertEqual(accuracy_new, 0.18181818181818182)
        self.assertEqual(accuracy, 0.18181818181818182)

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)

    def test_two_attacks_with_generator(self):
        (x_train, y_train), (x_test, y_test) = self.mnist
        x_train_original = x_train.copy()
        x_test_original = x_test.copy()

        class MyDataGenerator(DataGenerator):
            def __init__(self, x, y, size, batch_size):
                super().__init__(size=size, batch_size=batch_size)
                self.x = x
                self.y = y
                self.size = size
                self.batch_size = batch_size

            def get_batch(self):
                ids = np.random.choice(self.size, size=min(self.size, self.batch_size), replace=False)
                return self.x[ids], self.y[ids]

        generator = MyDataGenerator(x_train, y_train, x_train.shape[0], 1)

        attack1 = FastGradientMethod(self.classifier)
        attack2 = DeepFool(self.classifier)
        x_test_adv = attack1.generate(x_test)
        predictions = np.argmax(self.classifier.predict(x_test_adv), axis=1)
        accuracy = np.sum(predictions == np.argmax(y_test, axis=1)) / NB_TEST

        adv_trainer = AdversarialTrainer(self.classifier, attacks=[attack1, attack2])
        adv_trainer.fit_generator(generator, nb_epochs=5)

        predictions_new = np.argmax(adv_trainer.predict(x_test_adv), axis=1)
        accuracy_new = np.sum(predictions_new == np.argmax(y_test, axis=1)) / NB_TEST

        self.assertAlmostEqual(accuracy_new, 0.8181818181818182, delta=0.2)
        self.assertAlmostEqual(accuracy, 0.18181818181818182, delta=0.2)

        # Assert that the original training data has not changed
        self.assertAlmostEqual(float(np.max(np.abs(x_train_original - x_train))), 0.0, delta=0.00001)

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)

    def test_targeted_attack_error(self):
        """
        Test the adversarial trainer using a targeted attack, which will currently result in a NotImplementError.

        :return: None
        """
        (x_train, y_train), (_, _) = self.mnist
        params = {'nb_epochs': 2, 'batch_size': BATCH_SIZE}

        adv = FastGradientMethod(self.classifier, targeted=True)
        adv_trainer = AdversarialTrainer(self.classifier, attacks=adv)
        self.assertRaises(NotImplementedError, adv_trainer.fit, x_train, y_train, **params)


if __name__ == '__main__':
    unittest.main()
