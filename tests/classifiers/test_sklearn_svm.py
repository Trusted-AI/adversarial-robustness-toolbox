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

NB_TRAIN = 40


class TestSklearnSVC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        num_classes = 10
        cls.num_features = 784
        cls.num_samples = NB_TRAIN
        cls.num_samples_loss = 20

        (x_train, y_train), (_, _), min_, max_ = load_mnist()

        cls.x_train = x_train[0:cls.num_samples].reshape((cls.num_samples, 1, cls.num_features, 1))
        cls.y_train = y_train[0:cls.num_samples]

        clip_values = (0, 1)

        sklearn_model = SVC()
        cls.classifier = SklearnSVC(clip_values=clip_values, model=sklearn_model)
        # cls.classifier.fit(x=cls.x_train.reshape((cls.num_samples, cls.num_features)), y=cls.y_train)

    def test_predict(self):
        print('test_predict')

    def test_class_gradient(self):
        print('test_class_gradient')

    def test_loss_gradient(self):
        print('test_loss_gradient')

        grad = self.classifier.loss_gradient(
            self.x_train[0:self.num_samples_loss].reshape(self.num_samples_loss, self.num_features),
            self.y_train[0:self.num_samples_loss])
