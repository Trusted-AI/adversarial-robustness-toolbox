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
"""
A unittest class for testing the subset scanning detector.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import keras.backend as k
import numpy as np

from art.attacks.evasion.fast_gradient import FastGradientMethod
from art.defences.detector.evasion.subsetscanning import SubsetScanningDetector
from art.utils import load_dataset

from tests.utils import master_seed, get_image_classifier_kr

logger = logging.getLogger(__name__)

BATCH_SIZE = 100
NB_TRAIN = 100
NB_TEST = 100


class TestSubsetScanningDetector(unittest.TestCase):
    """
    A unittest class for testing the subset scanning detector.
    """

    def setUp(self):
        master_seed(seed=1234)

    def tearDown(self):
        k.clear_session()

    def test_subsetscan_detector(self):
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset("mnist")
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]

        # Keras classifier
        classifier = get_image_classifier_kr()

        # Generate adversarial samples:
        attacker = FastGradientMethod(classifier, eps=0.5)
        x_train_adv = attacker.generate(x_train)
        x_test_adv = attacker.generate(x_test)

        # Compile training data for detector:
        x_train_detector = np.concatenate((x_train, x_train_adv), axis=0)

        bgd = x_train
        clean = x_test
        anom = x_test_adv

        detector = SubsetScanningDetector(classifier, bgd, layer=1)

        _, _, dpwr = detector.scan(clean, clean)
        self.assertAlmostEqual(dpwr, 0.5)

        _, _, dpwr = detector.scan(clean, anom)
        self.assertGreater(dpwr, 0.5)

        _, _, dpwr = detector.scan(clean, x_train_detector, 85, 15)
        self.assertGreater(dpwr, 0.5)


if __name__ == "__main__":
    unittest.main()
