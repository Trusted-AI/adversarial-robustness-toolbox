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

NB_TRAIN = 20


class TestSklearnSVC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        num_classes = 10
        cls.num_features = 784
        cls.num_samples = NB_TRAIN
        cls.num_samples_loss = 20

        (x_train, y_train), (_, _), min_, max_ = load_mnist()

    def test_predict(self):
        print('test_predict')

    def test_class_gradient(self):
        print('test_class_gradient')

    def test_loss_gradient(self):
        print('test_loss_gradient')
