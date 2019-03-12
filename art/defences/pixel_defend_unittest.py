from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import torch.nn as nn
import torch.optim as optim

from art.defences.pixel_defend import PixelDefend
from art.classifiers.pytorch import PyTorchClassifier
from art.utils import master_seed

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
        mnist = input_data.read_data_sets("tmp/MNIST_data/")
        x = np.reshape(mnist.test.images[0:2], (-1, 28, 28, 1))
        x = x[:, 10:15, 15:20, :]
        preprocess = PixelDefend()
        defended_x = preprocess(x, eps=5, pixelcnn=self.pixelcnn)

        self.assertTrue((defended_x.shape == x.shape))
        self.assertTrue((defended_x <= 1.0).all())
        self.assertTrue((defended_x >= 0.0).all())


if __name__ == '__main__':
    unittest.main()

