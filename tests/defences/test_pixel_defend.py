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

import logging
import unittest

import torch.nn as nn
import torch.optim as optim

from art.classifiers.pytorch import PyTorchClassifier
from art.defences.pixel_defend import PixelDefend
from art.utils import load_mnist, master_seed

logger = logging.getLogger('testLogger')


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(25, 6400)

    def forward(self, x):
        x = x.view(-1, 25)
        logit_output = self.fc(x)
        logit_output = logit_output.view(-1, 5, 5, 1, 256)

        return logit_output


class TestPixelDefend(unittest.TestCase):
    def setUp(self):
        # Set master seed
        master_seed(1234)

        # Define the network
        model = Model()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        self.pixelcnn = PyTorchClassifier((0, 1), model, loss_fn, optimizer, (1, 28, 28), 10)

    def test_one_channel(self):
        (x_train, _), (_, _), _, _ = load_mnist()
        x_train = x_train[:2]

        x_train = x_train[:, 10:15, 15:20, :]
        preprocess = PixelDefend()
        defended_x, _ = preprocess(x_train, eps=5, pixel_cnn=self.pixelcnn)

        self.assertTrue((defended_x.shape == x_train.shape))
        self.assertTrue((defended_x <= 1.0).all())
        self.assertTrue((defended_x >= 0.0).all())


if __name__ == '__main__':
    unittest.main()

