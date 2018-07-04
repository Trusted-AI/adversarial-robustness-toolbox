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

        return logit_output


class Flatten(nn.Module):
    def forward(self, x):
        N, _, _, _  = x.size()
        result = x.view(N, -1)

        return result


class TestPyTorchClassifier(unittest.TestCase):
    """
    This class tests the functionalities of the PyTorch-based classifier.
    """
    def _model_setup_module(self):
        # Define the network
        model = Model()

        # Define a loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        return model, loss_fn, optimizer

    def _model_setup_sequential(self):
        # Define the network
        model = nn.Sequential(nn.Conv2d(1, 16, 5), nn.MaxPool2d(2, 2))

        # Define a loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        return model, loss_fn, optimizer


    def test_fit_predict(self):
        # Get MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]
        x_train = np.swapaxes(x_train, 1, 3)
        x_test = np.swapaxes(x_test, 1, 3)

        # Test fit and predict
        ptc = PyTorchClassifier((0, 1), self._model, self._loss_fn, self._optimizer, (1, 28, 28), 10)

        ptc.fit(x_train, y_train, batch_size=100, nb_epochs=1)
        preds = ptc.predict(x_test)
        acc = np.sum(np.argmax(preds, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        print("\nAccuracy: %.2f%%" % (acc * 100))
        self.assertGreater(acc, 0.1)

    def test_nb_classes(self):
        # Start to test
        ptc = PyTorchClassifier((0, 1), self._model, self._loss_fn, self._optimizer, (1, 28, 28), 10)
        self.assertTrue(ptc.nb_classes == 10)

    def test_input_shape(self):
        # Start to test
        ptc = PyTorchClassifier((0, 1), self._model, self._loss_fn, self._optimizer, (1, 28, 28), 10)
        self.assertTrue(np.array(ptc.input_shape == (1, 28, 28)).all())

    def test_class_gradient(self):
        # Get MNIST
        (_, _), (x_test, y_test), _, _ = load_mnist()
        x_test = x_test[:NB_TEST]
        x_test = np.swapaxes(x_test, 1, 3)

        # Test gradient
        ptc = PyTorchClassifier((0, 1), self._model, self._loss_fn, self._optimizer, (1, 28, 28), 10)
        grads = ptc.class_gradient(x_test)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 10, 1, 28, 28)).all())
        self.assertTrue(np.sum(grads) != 0)

    def test_loss_gradient(self):
        # Get MNIST
        (_, _), (x_test, y_test), _, _ = load_mnist()
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]
        x_test = np.swapaxes(x_test, 1, 3)

        # Test gradient
        ptc = PyTorchClassifier((0, 1), self._model, self._loss_fn, self._optimizer, (1, 28, 28), 10)
        grads = ptc.loss_gradient(x_test, y_test)

        self.assertTrue(np.array(grads.shape == (NB_TEST, 1, 28, 28)).all())
        self.assertTrue(np.sum(grads) != 0)


if __name__ == '__main__':
    unittest.main()





