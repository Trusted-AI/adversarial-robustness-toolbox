# -*- coding: utf-8 -*-
"""Trains a ResNet on the MNIST dataset, then generates adversarial images using DeepFool
and attacks a classic convolutional neural network (CNN) trained on MNIST with them. This is to show how to perform a
black-box attack.

The CNN obtains 98.57% accuracy on the adversarial samples when models are fitted for 5 epochs.
"""
from __future__ import absolute_import, division, print_function

from os.path import abspath
import sys
sys.path.append(abspath('.'))
from config import config_dict

import tensorflow as tf
import keras.backend as k

from art.attacks.deepfool import DeepFool
from art.classifiers.cnn import CNN
from art.classifiers.resnet import ResNet
from art.utils import load_dataset

# Get session
session = tf.Session()
k.set_session(session)

# Read MNIST dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset('mnist')
im_shape = x_train[0].shape

# Construct and train a Resnet convolutional neural network
comp_params = {'loss': 'categorical_crossentropy',
               'optimizer': 'adam',
               'metrics': ['accuracy']}
source = ResNet(im_shape, act='relu')
source.compile(comp_params)
source.fit(x_train, y_train, validation_split=.1, epochs=5, batch_size=128)

# Craft adversarial samples with DeepFool
epsilon = .1  # Maximum perturbation
adv_crafter = DeepFool(source, sess=session)
x_train_adv = adv_crafter.generate(x_val=x_train, eps=epsilon, clip_min=min_, clip_max=max_)
x_test_adv = adv_crafter.generate(x_val=x_test, eps=epsilon, clip_min=min_, clip_max=max_)

# Construct and train a convolutional neural network
target = CNN(im_shape, act='relu', dataset='mnist')
target.compile(comp_params)
target.fit(x_train, y_train, validation_split=.1, epochs=5, batch_size=128)

# Evaluate the CNN on the adversarial samples
scores = target.evaluate(x_test, y_test)
print("\nLoss on adversarial samples: %.2f%%\nAccuracy on adversarial samples: %.2f%%" % (scores[0], scores[1] * 100))
