from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import unittest

from art.classifiers import Classifier


class ClassifierInstance(Classifier):
    def __init__(self, clip_values, channel_index=1):
        super(ClassifierInstance, self).__init__(clip_values=clip_values, channel_index=channel_index)

    def class_gradient(self, x, label=None, logits=False):
        pass

    def fit(self, x, y, batch_size=128, nb_epochs=20):
        pass

    def get_activations(self, x, layer):
        pass

    def loss_gradient(self, x, y):
        pass

    def predict(self, x, logits=False):
        pass


class TestClassifier(unittest.TestCase):
    def test_processing(self):
        classifier = ClassifierInstance((0, 1))

        x = np.random.rand(100, 200)
        new_x = classifier._apply_processing(x)
        self.assertTrue(np.sum(x - new_x) == 0)
