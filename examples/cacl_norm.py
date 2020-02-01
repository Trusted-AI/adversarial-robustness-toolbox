# -*- coding: utf-8 -*-
"""
Trains a convolutional neural network on the CIFAR-10 dataset, then generated adversarial images using the
SimBA (pixel) attack and retrains the network on the training set augmented with the adversarial images.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np
from art.utils import load_dataset

#import matplotlib.pyplot as plt

# Configure a logger to capture ART outputs; these are printed in console and the level of detail is set to INFO
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Read CIFAR10 dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str('cifar10'))
x_train, y_train = x_train[:5000], y_train[:5000]
x_test, y_test = x_test[:500], y_test[:500]
im_shape = x_train[0].shape

norm = 0.0
for i in range(5000):
    norm = norm + np.linalg.norm(x_train[i].reshape(-1), ord=2)
print(norm / 5000)