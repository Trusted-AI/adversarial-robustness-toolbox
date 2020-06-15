# -*- coding: utf-8 -*-
"""Trains a convolutional neural network on the MNIST dataset, then attacks one of the hidden layers with the FGSM
attack."""
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
from os.path import abspath

sys.path.append(abspath("."))

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import numpy as np

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier
from art.utils import load_dataset

# Read MNIST dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str("mnist"))

# Create Keras convolutional neural network - basic architecture from Keras examples
# Source here: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=x_train.shape[1:]))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

classifier = KerasClassifier(model=model, clip_values=(min_, max_))
classifier.fit(x_train, y_train, nb_epochs=5, batch_size=128)

# Attack one of the inner layers, instead of the input one. In this example
# we are going to attack the second convolutional layer. To this aim, we need to
# split the network in 2 sub-nets, in order to have has input layer of the
# second network, the layer we want to attack

HL_model = Model(inputs=model.input, outputs=model.layers[2].output)

DL_input = Input(model.layers[3].input_shape[1:])
DL_model = DL_input
for layer in model.layers[3:]:
    DL_model = layer(DL_model)
DL_model = Model(inputs=DL_input, outputs=DL_model)

classifier = KerasClassifier(model=DL_model)

# Now we need to create the dataset for the DL_model, since the original one is
# suited only for the "model" network (and thus for the HL_model). Note that it
# is not needed to change the labels
x_test_inner = HL_model.predict(x_test)

# Evaluate the classifier on the test set
preds = np.argmax(classifier.predict(x_test_inner), axis=1)
acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
print("\nTest accuracy: %.2f%%" % (acc * 100))

# Craft adversarial samples with FGSM
epsilon = 0.1  # Maximum perturbation
adv_crafter = FastGradientMethod(classifier, eps=epsilon)
x_test_adv = adv_crafter.generate(x=x_test_inner)

# Evaluate the classifier on the adversarial examples
preds = np.argmax(classifier.predict(x_test_adv), axis=1)
acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
print("\nTest accuracy on adversarial sample: %.2f%%" % (acc * 100))
