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
import os
import pickle
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from art.config import ART_DATA_PATH
from art.data_generators import PyTorchDataGenerator
from art.estimators.classification.pytorch import PyTorchClassifier
from art.utils import Deprecated
from tests.utils import TestBase, get_image_classifier_pt, master_seed

logger = logging.getLogger(__name__)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(1, 2, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(288, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = x.view(-1, 288)
        logit_output = self.fc(x)
        return logit_output


class Flatten(nn.Module):
    def forward(self, x):
        n, _, _, _ = x.size()
        result = x.view(n, -1)

        return result


class TestPyTorchClassifier(TestBase):
    """
    This class tests the PyTorch classifier.
    """

    @classmethod
    def setUpClass(cls):
        master_seed(seed=1234)
        super().setUpClass()

        cls.x_train_mnist = np.reshape(cls.x_train_mnist, (cls.x_train_mnist.shape[0], 1, 28, 28)).astype(np.float32)
        cls.x_test_mnist = np.reshape(cls.x_test_mnist, (cls.x_test_mnist.shape[0], 1, 28, 28)).astype(np.float32)

        # Define the network
        model = nn.Sequential(nn.Conv2d(1, 2, 5), nn.ReLU(), nn.MaxPool2d(2, 2), Flatten(), nn.Linear(288, 10))

        # Define a loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        classifier = PyTorchClassifier(
            model=model, clip_values=(0, 1), loss=loss_fn, optimizer=optimizer, input_shape=(1, 28, 28), nb_classes=10
        )
        classifier.fit(cls.x_train_mnist, cls.y_train_mnist, batch_size=100, nb_epochs=1)
        cls.seq_classifier = classifier

        # Define the network
        model = Model()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        classifier_2 = PyTorchClassifier(
            model=model, clip_values=(0, 1), loss=loss_fn, optimizer=optimizer, input_shape=(1, 28, 28), nb_classes=10
        )
        classifier_2.fit(cls.x_train_mnist, cls.y_train_mnist, batch_size=100, nb_epochs=1)
        cls.module_classifier = classifier_2

        cls.x_train_mnist = np.reshape(cls.x_train_mnist, (cls.x_train_mnist.shape[0], 28, 28, 1)).astype(np.float32)
        cls.x_test_mnist = np.reshape(cls.x_test_mnist, (cls.x_test_mnist.shape[0], 28, 28, 1)).astype(np.float32)

    def setUp(self):
        master_seed(seed=1234)
        self.x_train_mnist = np.reshape(self.x_train_mnist, (self.x_train_mnist.shape[0], 1, 28, 28)).astype(np.float32)
        self.x_test_mnist = np.reshape(self.x_test_mnist, (self.x_test_mnist.shape[0], 1, 28, 28)).astype(np.float32)
        super().setUp()

    def tearDown(self):
        self.x_train_mnist = np.reshape(self.x_train_mnist, (self.x_train_mnist.shape[0], 28, 28, 1)).astype(np.float32)
        self.x_test_mnist = np.reshape(self.x_test_mnist, (self.x_test_mnist.shape[0], 28, 28, 1)).astype(np.float32)
        super().tearDown()

    def test_pickle(self):
        from art.config import ART_DATA_PATH
        full_path = os.path.join(ART_DATA_PATH, "my_classifier")
        folder = os.path.split(full_path)[0]
        if not os.path.exists(folder):
            os.makedirs(folder)







        import torch.nn.functional as F

        # class Model(nn.Module):
        #     def __init__(self):
        #         super(Model, self).__init__()
        #         self.conv = nn.Conv2d(1, 2, 5)
        #         self.pool = nn.MaxPool2d(2, 2)
        #         self.fc = nn.Linear(288, 10)
        #
        #     def forward(self, x):
        #         x = self.pool(F.relu(self.conv(x)))
        #         x = x.view(-1, 288)
        #         logit_output = self.fc(x)
        #         return logit_output

        # Define the network


        # model = Model()
        #         # loss_fn = nn.CrossEntropyLoss()
        #         # optimizer = optim.Adam(model.parameters(), lr=0.01)
        #         # classifier_2 = PyTorchClassifier(
        #         #     model=model, clip_values=(0, 1), loss=loss_fn, optimizer=optimizer, input_shape=(1, 28, 28), nb_classes=10
        #         # )
        #         # classifier_2.fit(self.x_train_mnist, self.y_train_mnist, batch_size=100, nb_epochs=1)
        #         # module_classifier = classifier_2

        # Define the network
        model = nn.Sequential(nn.Conv2d(1, 2, 5), nn.ReLU(), nn.MaxPool2d(2, 2), Flatten(), nn.Linear(288, 10))

        # Define a loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        classifier = PyTorchClassifier(
            model=model, clip_values=(0, 1), loss=loss_fn, optimizer=optimizer, input_shape=(1, 28, 28), nb_classes=10
        )
        classifier.fit(self.x_train_mnist, self.y_train_mnist, batch_size=100, nb_epochs=1)
        self.seq_classifier1 = classifier

        from art.config import ART_DATA_PATH
        full_path = os.path.join(ART_DATA_PATH, "my_classifier")
        folder = os.path.split(full_path)[0]
        if not os.path.exists(folder):
            os.makedirs(folder)

        # TODO the error is not coming from the classifier itself created but simply the fact that it's created
        #  within ghet get_image classifier_pt method
        pickle.dump(self.seq_classifier1, open(full_path, "wb"))
        # pickle.dump(model, open(full_path, "wb"))

        model = Model()

        # Define a loss function and optimizer
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Get classifier
        ptc = PyTorchClassifier(
            model=model, loss=loss_fn, optimizer=optimizer, input_shape=(1, 28, 28), nb_classes=10, clip_values=(0, 1)
        )

        # self.classifier = ptc
        self.classifier = get_image_classifier_pt(load_init=False)

        self.classifier.fit(self.x_train_mnist, self.y_train_mnist, batch_size=100, nb_epochs=1)

        # works with tensorflow
        from tests.utils import get_image_classifier_tf
        classifier_tf, sess = get_image_classifier_tf()

        # Define the network
        # model = Model()
        # loss_fn = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(model.parameters(), lr=0.01)
        # classifier_2 = PyTorchClassifier(
        #     model=model, clip_values=(0, 1), loss=loss_fn, optimizer=optimizer, input_shape=(1, 28, 28), nb_classes=10
        # )
        # classifier_2.fit(self.x_train_mnist, self.y_train_mnist, batch_size=100, nb_epochs=1)
        # self.module_classifier = classifier_2

        pickle.dump(self.module_classifier, open(full_path, "wb"))

        # Define the network
        model = Model()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        classifier_2 = PyTorchClassifier(
            model=model, clip_values=(0, 1), loss=loss_fn, optimizer=optimizer, input_shape=(1, 28, 28), nb_classes=10
        )
        classifier_2.fit(self.x_train_mnist, self.y_train_mnist, batch_size=100, nb_epochs=1)
        import copy
        deep_classifier_2 = copy.deepcopy(classifier_2)
        pickle.dump(deep_classifier_2, open(full_path, "wb"))

        # TODO the error is not coming from the classifier itself created but simply the fact that it's
        #  created within ghet get_image classifier_pt method
        pickle.dump(self.classifier, open(full_path, "wb"))

        # Unpickle:
        with open(full_path, "rb") as f:
            loaded = pickle.load(f)
            np.testing.assert_equal(self.module_classifier._clip_values, loaded._clip_values)
            self.assertEqual(self.module_classifier._channels_first, loaded._channels_first)
            self.assertEqual(set(self.module_classifier.__dict__.keys()), set(loaded.__dict__.keys()))

        # Test predict
        predictions_1 = self.module_classifier.predict(self.x_test_mnist)
        accuracy_1 = np.sum(np.argmax(predictions_1, axis=1) == np.argmax(self.y_test_mnist, axis=1)) / self.n_test
        predictions_2 = loaded.predict(self.x_test_mnist)
        accuracy_2 = np.sum(np.argmax(predictions_2, axis=1) == np.argmax(self.y_test_mnist, axis=1)) / self.n_test
        self.assertEqual(accuracy_1, accuracy_2)


if __name__ == "__main__":
    unittest.main()
