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
import unittest

import numpy as np
import torch.nn as nn
import torch.optim as optim

from art.estimators.classification.pytorch import PyTorchClassifier
from art.defences.preprocessor import PixelDefend
from art.utils import load_mnist

from tests.utils import master_seed

logger = logging.getLogger(__name__)


class ModelImage(nn.Module):
    def __init__(self):
        super(ModelImage, self).__init__()
        self.fc = nn.Linear(25, 6400)

    def forward(self, x):
        x = x.view(-1, 25)
        logit_output = self.fc(x)
        logit_output = logit_output.view(-1, 5, 5, 1, 256)

        return logit_output


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(4, 1024)

    def forward(self, x):
        x = x.view(-1, 4)
        logit_output = self.fc(x)
        logit_output = logit_output.view(-1, 4, 256)

        return logit_output


class TestPixelDefend(unittest.TestCase):
    def setUp(self):
        # Set master seed
        master_seed(seed=1234)

    def test_one_channel(self):
        (x_train, _), (_, _), _, _ = load_mnist()
        x_train = x_train[:2, 10:15, 15:20, :]
        x_train = x_train.astype(np.float32)
        x_train_original = x_train.copy()

        # Define the network
        model = ModelImage()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        self.pixelcnn = PyTorchClassifier(
            model=model, loss=loss_fn, optimizer=optimizer, input_shape=(1, 28, 28), nb_classes=10, clip_values=(0, 1)
        )
        preprocess = PixelDefend(eps=5, pixel_cnn=self.pixelcnn)
        x_defended, _ = preprocess(x_train)

        self.assertEqual(x_defended.shape, x_train.shape)
        self.assertTrue((x_defended <= 1.0).all())
        self.assertTrue((x_defended >= 0.0).all())

        # Check that x_train has not been modified by attack and classifier
        self.assertAlmostEqual(float(np.max(np.abs(x_train_original - x_train))), 0.0, delta=0.00001)

    def test_feature_vectors(self):
        # Define the network
        model = Model()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        pixel_cnn = PyTorchClassifier(
            model=model, loss=loss_fn, optimizer=optimizer, input_shape=(4,), nb_classes=2, clip_values=(0, 1)
        )

        x = np.random.rand(5, 4).astype(np.float32)
        preprocess = PixelDefend(eps=5, pixel_cnn=pixel_cnn)
        x_defended, _ = preprocess(x)

        self.assertEqual(x_defended.shape, x.shape)
        self.assertTrue((x_defended <= 1.0).all())
        self.assertTrue((x_defended >= 0.0).all())

    def test_check_params(self):
        model = Model()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        pixel_cnn = PyTorchClassifier(
            model=model, loss=loss_fn, optimizer=optimizer, input_shape=(4,), nb_classes=2, clip_values=(0, 1)
        )

        with self.assertRaises(TypeError):
            _ = PixelDefend(pixel_cnn="pixel_cnn")

        with self.assertRaises(ValueError):
            _ = PixelDefend(pixel_cnn=pixel_cnn, eps=-1)

        with self.assertRaises(ValueError):
            _ = PixelDefend(pixel_cnn=pixel_cnn, clip_values=(1, 0))

        with self.assertRaises(ValueError):
            _ = PixelDefend(pixel_cnn=pixel_cnn, clip_values=(-1, 1))

        with self.assertRaises(ValueError):
            _ = PixelDefend(pixel_cnn=pixel_cnn, clip_values=(0, 2))

        with self.assertRaises(ValueError):
            _ = PixelDefend(pixel_cnn=pixel_cnn, batch_size=-1)

        with self.assertRaises(ValueError):
            _ = PixelDefend(pixel_cnn=pixel_cnn, verbose="False")


if __name__ == "__main__":
    unittest.main()
