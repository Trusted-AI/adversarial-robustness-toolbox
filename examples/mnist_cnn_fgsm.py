# -*- coding: utf-8 -*-
"""Trains a convolutional neural network on the MNIST dataset, then attacks it with the FGSM attack.

With 5 epochs of training, gets to 98.89% accuracy on the test data and 65.40% on the adversarial examples.
"""
from __future__ import absolute_import, division, print_function

from os.path import abspath
import sys
sys.path.append(abspath('.'))
from config import config_dict

import tensorflow as tf
import keras.backend as k

from art.attacks.fast_gradient import FastGradientMethod
from art.classifiers.cnn import CNN
from art.utils import load_dataset

# Get session
session = tf.Session()
k.set_session(session)

# Read MNIST dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset('mnist')
im_shape = x_train[0].shape

# Construct a convolutional neural network
comp_params = {'loss': 'categorical_crossentropy',
               'optimizer': 'adam',
               'metrics': ['accuracy']}
classifier = CNN(im_shape, act='relu', dataset='mnist')
classifier.compile(comp_params)
classifier.fit(x_train, y_train, validation_split=.1, epochs=5, batch_size=128)

# Evaluate the classifier on the test set
scores = classifier.evaluate(x_test, y_test)
print("\nTest loss: %.2f%%\nTest accuracy: %.2f%%" % (scores[0], scores[1] * 100))

# Craft adversarial samples with FGSM
epsilon = .1  # Maximum perturbation
adv_crafter = FastGradientMethod(classifier, sess=session)
x_test_adv = adv_crafter.generate(x_val=x_test, eps=epsilon, clip_min=min_, clip_max=max_)

# Evaluate the classifier on the adversarial examples
scores = classifier.evaluate(x_test_adv, y_test)
print("\nTest loss: %.2f%%\nTest accuracy: %.2f%%" % (scores[0], scores[1] * 100))
