from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

# import numpy as np
# from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
# from keras.models import Sequential
# from keras.preprocessing.image import ImageDataGenerator

# from art.attacks import SamplingModelTheft
# from art.classifiers import KerasClassifier
from art.utils import load_dataset, master_seed

logger = logging.getLogger('testLogger')

BATCH_SIZE, NB_TRAIN, NB_TEST = 100, 1000, 10


class TestSamplingModelTheft(unittest.TestCase):
    """
    A unittest class for testing Spatial attack.
    """

    def setUp(self):
        # Set master seed
        master_seed(1234)

    def test_spatial(self):
        """
        First test with the spatial attack.
        :return:
        """


if __name__ == '__main__':
    unittest.main()
