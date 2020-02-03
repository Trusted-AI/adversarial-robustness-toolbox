# -*- coding: utf-8 -*-
"""
Trains a convolutional neural network on the CIFAR-10 dataset, then generated adversarial images using the
SimBA (pixel) attack and retrains the network on the training set augmented with the adversarial images.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout
import numpy as np

from art.attacks import UniversalPerturbation, Universal_SimBA_pixel
from art.classifiers import KerasClassifier
from art.utils import load_dataset, random_sphere

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

# Create Keras convolutional neural network - basic architecture from Keras examples
# Source here: https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Create classifier wrapper
classifier = KerasClassifier(model=model, clip_values=(min_, max_))
classifier.fit(x_train, y_train, nb_epochs=5, batch_size=128)

# Evaluate the classifier on the train samples
preds = np.argmax(classifier.predict(x_train), axis=1)
acc = np.sum(preds == np.argmax(y_train, axis=1)) / y_train.shape[0]
logger.info('Accuracy on train samples: %.2f%%', (acc * 100))

x_train, y_train = x_train[:100], y_train[:100]
preds = np.argmax(classifier.predict(x_train), axis=1)

# Craft adversarial samples with SimBA for single image
logger.info('Create universal SimBA (pixel) attack')
adv_crafter = Universal_SimBA_pixel(classifier, epsilon=0.05)
logger.info('Craft attack on a training example')
x_train_adv_univ_simba = adv_crafter.generate(x_train)
logger.info('Craft attack the training example')
norm2 = np.linalg.norm((x_train_adv_univ_simba[0] - x_train[0]).reshape(-1), ord=2)
# compute fooling rate
preds_adv = np.argmax(classifier.predict(x_train_adv_univ_simba), axis=1)
acc = np.sum(preds != preds_adv) / y_train.shape[0]
logger.info('Fooling rate on universal SimBA adversarial examples: %.2f%%', (acc * 100))
logger.info('Perturbation norm: %.2f%%', norm2)

# Craft adversarial samples with random universal pertubation
x_train_adv_random = np.clip(x_train + random_sphere(nb_points=1, nb_dims=32*32*3, radius=norm2, norm=2).reshape(1,32,32,3), min_, max_)
preds_adv = np.argmax(classifier.predict(x_train_adv_random), axis=1)
acc = np.sum(preds != preds_adv) / y_train.shape[0]
logger.info('Fooling rate on random adversarial examples: %.2f%%', (acc * 100))
logger.info('Perturbation norm: %.2f%%', np.linalg.norm((x_train_adv_random[0]-x_train[0]).reshape(-1), ord=2))

# Craft adversarial samples with universal pertubation and FGSM
attack_params = {"attacker": "fgsm", "delta": 0.01, "max_iter": 1, "eps": norm2, "norm": 2}
adv_crafter_fgsm = UniversalPerturbation(classifier)
adv_crafter_fgsm.set_params(**attack_params)
x_train_adv_fgsm = adv_crafter_fgsm.generate(x_train, **attack_params)
np.linalg.norm(adv_crafter_fgsm.noise.reshape(-1), ord=2)
# compute fooling rate
preds_adv = np.argmax(classifier.predict(x_train_adv_fgsm), axis=1)
acc = np.sum(preds != preds_adv) / y_train.shape[0]
logger.info('Fooling rate on fgsm universal adversarial examples: %.2f%%', (acc * 100))
logger.info('Perturbation norm: %.2f%%', np.linalg.norm(adv_crafter_fgsm.noise.reshape(-1), ord=2))