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

from art.attacks import FastGradientMethod
from art.utils import load_mnist, random_targets, master_seed, get_classifier_kr
from art.wrappers.expectation import ExpectationOverTransformations

logger = logging.getLogger('testLogger')

BATCH_SIZE = 100
NB_TRAIN = 5000
NB_TEST = 10


class TestExpectationOverTransformations(unittest.TestCase):
    """
    A unittest class for testing the Expectation over Transformations in attacks.
    """

    @classmethod
    def setUpClass(cls):
        # Get MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]
        cls.mnist = (x_train, y_train), (x_test, y_test)

    def setUp(self):
        # Set master seed
        master_seed(1234)

    def test_krclassifier(self):
        """
        Test with a KerasClassifier.
        :return:
        """
        # Build KerasClassifier
        krc, sess = get_classifier_kr()

        # Get MNIST
        (_, _), (x_test, y_test) = self.mnist

        # First attack (without EoT):
        fgsm = FastGradientMethod(classifier=krc, targeted=True)
        params = {'y': random_targets(y_test, krc.nb_classes)}
        x_test_adv = fgsm.generate(x_test, **params)

        # Second attack (with EoT):
        def t(x):
            return x

        def transformation():
            while True:
                yield t

        eot = ExpectationOverTransformations(classifier=krc, sample_size=1, transformation=transformation)
        fgsm_with_eot = FastGradientMethod(classifier=eot, targeted=True)
        x_test_adv_with_eot = fgsm_with_eot.generate(x_test, **params)

        self.assertTrue((np.abs(x_test_adv - x_test_adv_with_eot) < 0.001).all())


if __name__ == '__main__':
    unittest.main()
