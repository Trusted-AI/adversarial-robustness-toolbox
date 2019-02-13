from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

from art.attacks import AdversarialPatch
from art.utils import load_mnist, master_seed, get_classifier_tf, get_classifier_kr, get_classifier_pt

logger = logging.getLogger('testLogger')

BATCH_SIZE = 100
NB_TRAIN = 1000
NB_TEST = 10


class TestAdversarialPatch(unittest.TestCase):
    """
    A unittest class for testing Adversarial Patch attack.
    """

    def setUp(self):
        # Set master seed
        master_seed(1234)

    def test_tfclassifier(self):
        x = 1
