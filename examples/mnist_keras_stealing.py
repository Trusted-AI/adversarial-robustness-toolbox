"""
An example of stealing and protecting a Keras classifier on MNIST.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential

from art.attacks import SamplingModelTheft
from art.classifiers import KerasClassifier
from art.defences import ReverseSigmoid
from art.utils import load_dataset


def build_model(input_shape):
    m = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')])
    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    k = KerasClassifier((min_, max_), model=m)
    return k

(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str('mnist'))

# This is the base model
k0 = build_model(x_train.shape[1:])
k0.fit(x_train, y_train, nb_epochs=5, batch_size=128)
k1 = build_model(x_train.shape[1:])

# Model theft attack on the base model
att = SamplingModelTheft(x_test)
k1 = att.steal(k0, k1, 10000, nb_epochs=5)
y0 = k0.predict(x_train)
y1 = k1.predict(x_train)
agree1 = np.sum(y0.argmax(axis=1) == y1.argmax(axis=1)) / len(x_train)

# Reverse sigmoid defense on the base model
rs = ReverseSigmoid()
k0p = rs(k0)
y0p = k0p.predict(x_train)
agree0 = np.sum(y0p.argmax(axis=1) == y0.argmax(axis=1)) / len(x_train)

# Model theft attack on the reverse-sigmoid-defended model
k2 = att.steal(k0p, k1, 10000, nb_epochs=5)
y2 = k2.predict(x_train)
agree2 = np.sum(y0p.argmax(axis=1) == y2.argmax(axis=1)) / len(x_train)

print("Agreement between the original and the protected models: %f" % agree0)
print("Agreement between the original and the stolen models: %f" % agree1)
print("Agreement between the protected model and the model stolen from the protected model: %f" % agree2)


