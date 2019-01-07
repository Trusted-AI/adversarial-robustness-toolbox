from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import torch.nn as nn

# import numpy as np
# from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
# from keras.models import Sequential
# from keras.preprocessing.image import ImageDataGenerator

# from art.attacks import SamplingModelTheft
# from art.classifiers import KerasClassifier
from art.utils import load_dataset, master_seed

logger = logging.getLogger('testLogger')

BATCH_SIZE = 100
NB_TRAIN = 1000
NB_TEST = 10

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(2304, 10)

    def forward(self, x):
        import torch.nn.functional as f

        x = self.pool(f.relu(self.conv(x)))
        x = x.view(-1, 2304)
        logit_output = self.fc(x)

        return logit_output

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
