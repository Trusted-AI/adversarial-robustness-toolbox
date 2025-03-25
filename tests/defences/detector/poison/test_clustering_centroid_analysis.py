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

from keras.preprocessing.image import ImageDataGenerator
import numpy as np

from art.data_generators import KerasDataGenerator
from art.defences.detector.poison import ActivationDefence
from art.utils import load_mnist
from art.visualization import convert_to_rgb

from tests.utils import master_seed

logger = logging.getLogger(__name__)

NB_TRAIN, NB_TEST, BATCH_SIZE = 300, 10, 128

class TestClusteringCentroidAnalysis(unittest.TestCase):
    @classmethod
    def setUp(cls):
        pass

    def setUp(self):
        pass

    @unittest.expectedFailure
    def test_wrong_parameters_1(self):
        pass
