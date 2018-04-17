# -*- coding: utf-8 -*-
"""Trains a convolutional neural network on the CIFAR-10 dataset, then generated adversarial images using the
DeepFool attack and retrains the network on the training set augmented with the adversarial images.

Gets to 56.80% accuracy on the adversarial samples after data augmentation over 10 epochs.
"""
from __future__ import absolute_import, division, print_function

from os.path import abspath
import sys
sys.path.append(abspath('.'))
from config import config_dict

from numpy import append
import tensorflow as tf
import keras.backend as k

from art.attacks.deepfool import DeepFool
from art.classifiers.cnn import CNN
from art.utils import load_dataset


# Get session
session = tf.Session()
k.set_session(session)

# Read CIFAR10 dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset('cifar10')
x_train, y_train = x_train[:5000], y_train[:5000]
x_test, y_test = x_test[:500], y_test[:500]
im_shape = x_train[0].shape

# Construct a convolutional neural network
comp_params = {'loss': 'categorical_crossentropy',
               'optimizer': 'adam',
               'metrics': ['accuracy']}
classifier = CNN(im_shape, act='relu', dataset='cifar10')
classifier.compile(comp_params)
classifier.fit(x_train, y_train, validation_split=.1, epochs=10, batch_size=128)

# Craft adversarial samples with DeepFool
print('Create DeepFool attack')
epsilon = .1  # Maximum perturbation
adv_crafter = DeepFool(classifier, sess=session)
print('Craft training examples')
x_train_adv = adv_crafter.generate(x_val=x_train, eps=epsilon, clip_min=min_, clip_max=max_)
print('Craft test examples')
x_test_adv = adv_crafter.generate(x_val=x_test, eps=epsilon, clip_min=min_, clip_max=max_)

# Evaluate the classifier on the adversarial samples
scores = classifier.evaluate(x_test, y_test)
print("\nClassifier before adversarial training")
print("\nLoss on adversarial samples: %.2f%%\nAccuracy on adversarial samples: %.2f%%" % (scores[0], scores[1] * 100))

# Data augmentation: expand the training set with the adversarial samples
x_train = append(x_train, x_train_adv, axis=0)
y_train = append(y_train, y_train, axis=0)

# Retrain the CNN on the extended dataset
classifier.compile(comp_params)
classifier.fit(x_train, y_train, validation_split=.1, epochs=10, batch_size=128)

# Evaluate the adversarially trained classifier on the test set
scores = classifier.evaluate(x_test, y_test)
print("\nClassifier with adversarial training")
print("\nLoss on adversarial samples: %.2f%%\nAccuracy on adversarial samples: %.2f%%" % (scores[0], scores[1] * 100))
