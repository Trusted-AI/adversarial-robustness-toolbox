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

import unittest

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from art.classifiers.pytorch import PyTorchClassifier
from art.utils import load_mnist


NB_TRAIN = 1000
NB_TEST = 20


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(2304, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = x.view(-1, 2304)
        logit_output = self.fc(x)
        output = F.softmax(logit_output, dim=1)

        return (logit_output, output)


class TestPyTorchClassifier(unittest.TestCase):
    """
    This class tests the functionalities of the PyTorch-based classifier.
    """
    def setUp(self):
        # Define the network
        self._model = Model()

        # Define a loss function and optimizer
        self._loss_fn = nn.CrossEntropyLoss()
        self._optimizer = optim.Adam(self._model.parameters(), lr=0.01)

    def test_fit_predict(self):
        # Get MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train = x_train[:NB_TRAIN], np.argmax(y_train[:NB_TRAIN], axis=1)
        x_test, y_test = x_test[:NB_TEST], np.argmax(y_test[:NB_TEST], axis=1)
        x_train = np.swapaxes(x_train, 1, 3)
        x_test = np.swapaxes(x_test, 1, 3)

        # Test fit and predict
        ptc = PyTorchClassifier(None, self._model, self._loss_fn, self._optimizer, (1, 28, 28), (10,))

        ptc.fit(x_train, y_train, batch_size=100, nb_epochs=1)
        preds = ptc.predict(x_test)
        preds_class = np.argmax(preds, axis=1)
        acc = np.sum(preds_class == y_test) / len(y_test)
        print("\nAccuracy: %.2f%%" % (acc * 100))
        self.assertGreater(acc, 0.1)

    def test_nb_classes(self):
        # Start to test
        ptc = PyTorchClassifier(None, self._model, self._loss_fn, self._optimizer, (1, 28, 28), (10,))
        self.assertTrue(ptc.nb_classes == 10)

    def test_input_shape(self):
        # Start to test
        ptc = PyTorchClassifier(None, self._model, self._loss_fn, self._optimizer, (1, 28, 28), (10,))
        self.assertTrue(np.array(ptc.input_shape == (1, 28, 28)).all())

    def test_class_gradient(self):
        # Get MNIST
        (_, _), (x_test, y_test), _, _ = load_mnist()
        x_test = x_test[:NB_TEST]
        x_test = np.swapaxes(x_test, 1, 3)

        # Test gradient
        ptc = PyTorchClassifier(None, self._model, self._loss_fn, self._optimizer, (1, 28, 28), (10,))
        grads = ptc.class_gradient(x_test)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 10, 1, 28, 28)).all())
        self.assertTrue(np.sum(grads) != 0)

    def test_loss_gradient(self):
        # Get MNIST
        (_, _), (x_test, y_test), _, _ = load_mnist()
        x_test, y_test = x_test[:NB_TEST], np.argmax(y_test[:NB_TEST], axis=1)
        x_test = np.swapaxes(x_test, 1, 3)

        # Test gradient
        ptc = PyTorchClassifier(None, self._model, self._loss_fn, self._optimizer, (1, 28, 28), (10,))
        grads = ptc.loss_gradient(x_test, y_test)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 28, 28)).all())
        self.assertTrue(np.sum(grads) != 0)


if __name__ == '__main__':
    unittest.main()



