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

from art.utils import load_dataset, random_targets, master_seed, to_categorical
from art import NUMPY_DTYPE

logger = logging.getLogger(__name__)


BATCH_SIZE = 100
NB_TRAIN = 1000
NB_EPOCHS = 10
NB_STOLEN = 1000


class TestKnockoffNets(unittest.TestCase):
    """
    A unittest class for testing the KnockoffNets attack.
    """

    @classmethod
    def setUpClass(cls):
        (x_train, y_train), (_, _), _, _ = load_dataset('mnist')

        cls.x_train = x_train[:NB_TRAIN].astype(NUMPY_DTYPE)
        cls.y_train = y_train[:NB_TRAIN].astype(NUMPY_DTYPE)

    def setUp(self):
        master_seed(1234)


if __name__ == '__main__':
    unittest.main()
