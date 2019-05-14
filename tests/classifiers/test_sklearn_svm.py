from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import tensorflow as tf
import numpy as np

from sklearn.svm import SVC

from art.classifiers import SklearnSVC
from art.utils import load_mnist

logger = logging.getLogger('testLogger')
np.random.seed(seed=1234)
tf.set_random_seed(1234)

class TestSklearnLogisticRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print('setUpClass')

    def test_predict(self):
        print('test_predict')

    def test_class_gradient(self):
        print('test_class_gradient')

    def test_loss_gradient(self):
        print('test_loss_gradient')