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

import os
import tempfile
import logging
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from art.estimators.classification.pytorch import PyTorchClassifier
from art.estimators.classification.detector_classifier import DetectorClassifier

from tests.utils import TestBase, get_image_classifier_pt


logger = logging.getLogger(__name__)


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


class TestDetectorClassifier(TestBase):
    """
    This class tests the Detector Classifier.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.x_train_mnist = np.reshape(cls.x_train_mnist, (cls.x_train_mnist.shape[0], 1, 28, 28)).astype(np.float32)
        cls.x_test_mnist = np.reshape(cls.x_test_mnist, (cls.x_test_mnist.shape[0], 1, 28, 28)).astype(np.float32)

        # Define the internal classifier
        classifier = get_image_classifier_pt()

        # Define the internal detector
        conv = nn.Conv2d(1, 16, 5)
        linear = nn.Linear(2304, 1)
        torch.nn.init.xavier_uniform_(conv.weight)
        torch.nn.init.xavier_uniform_(linear.weight)
        model = nn.Sequential(conv, nn.ReLU(), nn.MaxPool2d(2, 2), Flatten(), linear)
        model = Model(model)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        detector = PyTorchClassifier(
            model=model, loss=loss_fn, optimizer=optimizer, input_shape=(1, 28, 28), nb_classes=2, clip_values=(0, 1)
        )

        # Define the detector-classifier
        cls.detector_classifier = DetectorClassifier(classifier=classifier, detector=detector)

        cls.x_train_mnist = np.reshape(cls.x_train_mnist, (cls.x_train_mnist.shape[0], 28, 28, 1)).astype(np.float32)
        cls.x_test_mnist = np.reshape(cls.x_test_mnist, (cls.x_test_mnist.shape[0], 28, 28, 1)).astype(np.float32)

    def setUp(self):
        self.x_train_mnist = np.reshape(self.x_train_mnist, (self.x_train_mnist.shape[0], 1, 28, 28)).astype(np.float32)
        self.x_test_mnist = np.reshape(self.x_test_mnist, (self.x_test_mnist.shape[0], 1, 28, 28)).astype(np.float32)
        super().setUp()

    def tearDown(self):
        self.x_train_mnist = np.reshape(self.x_train_mnist, (self.x_train_mnist.shape[0], 28, 28, 1)).astype(np.float32)
        self.x_test_mnist = np.reshape(self.x_test_mnist, (self.x_test_mnist.shape[0], 28, 28, 1)).astype(np.float32)
        super().tearDown()

    def test_predict(self):
        predictions = self.detector_classifier.predict(x=self.x_test_mnist[0:1])
        predictions_expected = 7
        self.assertEqual(predictions.shape, (1, 11))
        self.assertEqual(np.argmax(predictions, axis=1)[0], predictions_expected)

    def test_nb_classes(self):
        self.assertEqual(self.detector_classifier.nb_classes, 11)

    def test_input_shape(self):
        self.assertEqual(self.detector_classifier.input_shape, (1, 28, 28))

    def test_class_gradient_1(self):
        # Test label = None
        gradients = self.detector_classifier.class_gradient(x=self.x_test_mnist[0:1], label=None)
        self.assertEqual(gradients.shape, (1, 11, 1, 28, 28))

    def test_class_gradient_2(self):
        # Test label = 5
        gradients = self.detector_classifier.class_gradient(x=self.x_test_mnist, label=5)
        self.assertEqual(gradients.shape, (self.n_test, 1, 1, 28, 28))

    def test_class_gradient_3(self):
        # Test label = 10 (detector classifier has 11 classes (10 + 1))
        gradients = self.detector_classifier.class_gradient(x=self.x_test_mnist, label=10)
        self.assertEqual(gradients.shape, (self.n_test, 1, 1, 28, 28))

    def test_class_gradient_4(self):
        # Test label = array
        n_test_local = 2
        label = np.array([2, 10])
        gradients = self.detector_classifier.class_gradient(x=self.x_test_mnist[0:n_test_local], label=label)
        self.assertEqual(gradients.shape, (n_test_local, 1, 1, 28, 28))

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
        os.remove(full_path + "_classifier.optimizer")
        os.remove(full_path + "_classifier.model")

        self.assertTrue(os.path.exists(full_path + "_detector.optimizer"))
        self.assertTrue(os.path.exists(full_path + "_detector.model"))
        os.remove(full_path + "_detector.optimizer")
        os.remove(full_path + "_detector.model")

    def test_repr(self):
        repr_ = repr(self.detector_classifier)
        self.assertIn("art.estimators.classification.detector_classifier.DetectorClassifier", repr_)
        self.assertIn(
            "preprocessing=StandardisationMeanStd(mean=0.0, std=1.0, apply_fit=True, apply_predict=True)", repr_
        )


if __name__ == "__main__":
    unittest.main()
