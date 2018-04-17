# -*- coding: utf-8 -*-
"""Trains a convolutional neural network on the CIFAR10 dataset with feature squeezing as a defense.

Gets to 70.04% test accuracy after 10 epochs.
"""
from __future__ import absolute_import, division, print_function

from os.path import abspath
import sys
sys.path.append(abspath('.'))
from config import config_dict

from art.classifiers.cnn import CNN
from art.utils import load_dataset

# Read CIFAR10 dataset
(x_train, y_train), (x_test, y_test), _, _ = load_dataset('cifar10')
im_shape = x_train[0].shape

# Construct a convolutional neural network with feature squeezing activated
# For CIFAR10, squeezing the features to 3 bits works well
comp_params = {'loss': 'categorical_crossentropy',
               'optimizer': 'adam',
               'metrics': ['accuracy']}
classifier = CNN(im_shape, act='relu', dataset='cifar10', defences='featsqueeze3')
classifier.compile(comp_params)
classifier.fit(x_train, y_train, validation_split=.1, epochs=10, batch_size=128)

# Evaluate the classifier on the test set
scores = classifier.evaluate(x_test, y_test)
print("\nTest loss: %.2f%%\nTest accuracy: %.2f%%" % (scores[0], scores[1] * 100))
