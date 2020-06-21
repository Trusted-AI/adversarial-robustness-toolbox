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
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import numpy as np

from art.defences.trainer.adversarial_trainer_madry_pgd import AdversarialTrainerMadryPGD
from art.utils import load_mnist

from tests.utils import master_seed, get_image_classifier_tf

logger = logging.getLogger(__name__)

BATCH_SIZE = 10
NB_TRAIN = 100
NB_TEST = 100


class TestAdversarialTrainerMadryPGD(unittest.TestCase):
    """
    Test cases for the AdversarialTrainerMadryPGD class.
    """

    @classmethod
    def setUpClass(cls):
        # MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train, x_test, y_test = (
            x_train[:NB_TRAIN],
            y_train[:NB_TRAIN],
            x_test[:NB_TEST],
            y_test[:NB_TEST],
        )
        cls.mnist = ((x_train, y_train), (x_test, y_test))

        cls.classifier, _ = get_image_classifier_tf()

    def setUp(self):
        master_seed(seed=1234)

    def test_fit_predict(self):
        (x_train, y_train), (x_test, y_test) = self.mnist
        x_test_original = x_test.copy()

        adv_trainer = AdversarialTrainerMadryPGD(self.classifier, nb_epochs=1, batch_size=128)
        adv_trainer.fit(x_train, y_train)

        predictions_new = np.argmax(adv_trainer.trainer.get_classifier().predict(x_test), axis=1)
        accuracy_new = np.sum(predictions_new == np.argmax(y_test, axis=1)) / NB_TEST

        self.assertEqual(accuracy_new, 0.38)

        # Check that x_test has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_test_original - x_test))), 0.0, delta=0.00001)


if __name__ == "__main__":
    unittest.main()
