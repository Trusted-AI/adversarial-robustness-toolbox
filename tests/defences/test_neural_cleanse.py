# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2019
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
import keras

from art.defences.transformer.poisoning import NeuralCleanse
from art.utils import load_dataset

from tests.utils import master_seed, get_image_classifier_kr

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
logger = logging.getLogger(__name__)

BATCH_SIZE = 100
NB_TRAIN = 5000
NB_TEST = 10


class TestNeuralCleanse(unittest.TestCase):
    """
    A unittest class for testing Randomized Smoothing as a post-processing step for classifiers.
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

    def test_keras(self):
        """
        Test with a KerasClassifier.
        :return:
        """
        if keras.__version__ != "2.2.4":
            self.assertRaises(NotImplementedError)
        else:
            # Build KerasClassifier
            krc = get_image_classifier_kr()

            # Get MNIST
            (x_train, y_train), (x_test, y_test) = self.mnist

            krc.fit(x_train, y_train, nb_epochs=1)

            cleanse = NeuralCleanse(krc)
            defense_cleanse = cleanse(krc, steps=2)
            defense_cleanse.mitigate(x_test, y_test, mitigation_types=["filtering", "pruning", "unlearning"])


if __name__ == "__main__":
    unittest.main()
