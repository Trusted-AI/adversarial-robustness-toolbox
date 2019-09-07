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

import os
import tempfile
import logging
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from art.classifiers import PyTorchClassifier, DetectorClassifier
from art.utils import load_dataset, master_seed
from art.utils_test import get_classifier_pt

logger = logging.getLogger('testLogger')

NB_TRAIN = 1000
NB_TEST = 2


class Model(nn.Module):
    def __init__(self, model):
        super(Model, self).__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)
        x = x - 100000

        return x


class Flatten(nn.Module):
    def forward(self, x):
        n, _, _, _ = x.size()
        result = x.view(n, -1)

        return result


class TestDetectorClassifier(unittest.TestCase):
    """
    This class tests the functionalities of the DetectorClassifier.
    """

    @classmethod
    def setUpClass(cls):
        (x_train, y_train), (x_test, y_test), _, _ = load_dataset('mnist')

        x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
        x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)

        cls.x_train = x_train[:NB_TRAIN]
        cls.y_train = y_train[:NB_TRAIN]
        cls.x_test = x_test[:NB_TEST]
        cls.y_test = y_test[:NB_TEST]

        # Define the internal classifier
        classifier = get_classifier_pt()

        # Define the internal detector
        conv = nn.Conv2d(1, 16, 5)
        linear = nn.Linear(2304, 1)
        torch.nn.init.xavier_uniform_(conv.weight)
        torch.nn.init.xavier_uniform_(linear.weight)
        model = nn.Sequential(conv, nn.ReLU(), nn.MaxPool2d(2, 2), Flatten(), linear)
        model = Model(model)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        detector = PyTorchClassifier(model=model, loss=loss_fn, optimizer=optimizer, input_shape=(1, 28, 28),
                                     nb_classes=1, clip_values=(0, 1))

        # Define the detector-classifier
        cls.detector_classifier = DetectorClassifier(classifier=classifier, detector=detector)

    def setUp(self):
        master_seed(1234)

    def test_predict(self):
        predictions = self.detector_classifier.predict(x=self.x_test[0:1])
        predictions_expected = 7
        self.assertEqual(predictions.shape, (1, 11))
        self.assertEqual(np.argmax(predictions, axis=1)[0], predictions_expected)

    def test_nb_classes(self):
        self.assertEqual(self.detector_classifier.nb_classes(), 11)

    def test_input_shape(self):
        self.assertEqual(self.detector_classifier.input_shape, (1, 28, 28))

    def test_class_gradient_1(self):
        # Test label = None
        gradients = self.detector_classifier.class_gradient(x=self.x_test[0:1], label=None)
        self.assertEqual(gradients.shape, (1, 11, 1, 28, 28))

    def test_class_gradient_2(self):
        # Test label = 5
        gradients = self.detector_classifier.class_gradient(x=self.x_test, label=5)
        self.assertEqual(gradients.shape, (NB_TEST, 1, 1, 28, 28))

    def test_class_gradient_3(self):
        # Test label = 10
        gradients = self.detector_classifier.class_gradient(x=self.x_test, label=10)
        self.assertEqual(gradients.shape, (NB_TEST, 1, 1, 28, 28))

    def test_class_gradient_4(self):
        # Test label = array
        label = np.array([2, 10])
        gradients = self.detector_classifier.class_gradient(x=self.x_test, label=label)
        self.assertEqual(gradients.shape, (NB_TEST, 1, 1, 28, 28))

    def test_set_learning(self):
        self.assertTrue(self.detector_classifier.classifier._model.training)
        self.assertTrue(self.detector_classifier.detector._model.training)
        self.assertIs(self.detector_classifier.learning_phase, None)

        self.detector_classifier.set_learning_phase(False)
        self.assertFalse(self.detector_classifier.classifier._model.training)
        self.assertFalse(self.detector_classifier.detector._model.training)
        self.assertFalse(self.detector_classifier.learning_phase)

        self.detector_classifier.set_learning_phase(True)
        self.assertTrue(self.detector_classifier.classifier._model.training)
        self.assertTrue(self.detector_classifier.detector._model.training)
        self.assertTrue(self.detector_classifier.learning_phase)

    def test_save(self):
        model = self.detector_classifier
        t_file = tempfile.NamedTemporaryFile()
        full_path = t_file.name
        t_file.close()
        base_name = os.path.basename(full_path)
        dir_name = os.path.dirname(full_path)
        model.save(base_name, path=dir_name)

        self.assertTrue(os.path.exists(full_path + "_classifier.optimizer"))
        self.assertTrue(os.path.exists(full_path + "_classifier.model"))
        os.remove(full_path + '_classifier.optimizer')
        os.remove(full_path + '_classifier.model')

        self.assertTrue(os.path.exists(full_path + "_detector.optimizer"))
        self.assertTrue(os.path.exists(full_path + "_detector.model"))
        os.remove(full_path + '_detector.optimizer')
        os.remove(full_path + '_detector.model')

    def test_repr(self):
        repr_ = repr(self.detector_classifier)
        self.assertIn('art.classifiers.detector_classifier.DetectorClassifier', repr_)
        self.assertIn('preprocessing=(0, 1)', repr_)


if __name__ == '__main__':
    unittest.main()
