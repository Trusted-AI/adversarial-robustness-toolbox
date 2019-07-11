"""
A unittest class for testing the subset scanning detector.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import keras.backend as k
import numpy as np

from art.attacks.fast_gradient import FastGradientMethod
from art.detection.subsetscanning.detector import SubsetScanningDetector
from art.utils import master_seed, load_dataset
from art.utils_test import get_classifier_kr

logger = logging.getLogger('testLogger')

BATCH_SIZE = 100
NB_TRAIN = 100
NB_TEST = 100


class TestSubsetScanningDetector(unittest.TestCase):
    """
    A unittest class for testing the subset scanning detector.
    """

    def setUp(self):
        master_seed(1234)

    def tearDown(self):
        k.clear_session()

    def test_subsetscan_detector(self):
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset('mnist')
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]

        # Keras classifier
        classifier, _ = get_classifier_kr()

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
