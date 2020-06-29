# -*- coding: utf-8 -*-
"""Trains a CNN on the MNIST dataset using the Keras backend, then generates adversarial images using DeepFool
and uses them to attack a CNN trained on MNIST using TensorFlow. This is to show how to perform a
black-box attack: the attack never has access to the parameters of the TensorFlow model.
"""
from __future__ import absolute_import, division, print_function

import keras
import keras.backend as k
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import numpy as np
import tensorflow as tf

from art.attacks.evasion import DeepFool
from art.estimators.classification import KerasClassifier, TensorFlowClassifier
from art.utils import load_mnist


def cnn_mnist_tf(input_shape):
    labels_tf = tf.placeholder(tf.float32, [None, 10])
    inputs_tf = tf.placeholder(tf.float32, [None] + list(input_shape))

    # Define the TensorFlow graph
    conv = tf.layers.conv2d(inputs_tf, 4, 5, activation=tf.nn.relu)
    conv = tf.layers.max_pooling2d(conv, 2, 2)
    fc = tf.contrib.layers.flatten(conv)

    # Logits layer
    logits = tf.layers.dense(fc, 10)

    # Train operator
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels_tf))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_tf = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    classifier = TensorFlowClassifier(
        clip_values=(0, 1), input_ph=inputs_tf, output=logits, loss=loss, train=train_tf, labels_ph=labels_tf, sess=sess
    )
    return classifier


def cnn_mnist_k(input_shape):
    # Create simple CNN
    model = Sequential()
    model.add(Conv2D(4, kernel_size=(5, 5), activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(10, activation="softmax"))

    model.compile(
        loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.01), metrics=["accuracy"]
    )

    classifier = KerasClassifier(model=model, clip_values=(0, 1))
    return classifier


# Get session
session = tf.Session()
k.set_session(session)

# Read MNIST dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_mnist()

# Construct and train a convolutional neural network on MNIST using Keras
source = cnn_mnist_k(x_train.shape[1:])
source.fit(x_train, y_train, nb_epochs=5, batch_size=128)

# Craft adversarial samples with DeepFool
adv_crafter = DeepFool(source)
x_train_adv = adv_crafter.generate(x_train)
x_test_adv = adv_crafter.generate(x_test)

# Construct and train a convolutional neural network
target = cnn_mnist_tf(x_train.shape[1:])
target.fit(x_train, y_train, nb_epochs=5, batch_size=128)

# Evaluate the CNN on the adversarial samples
preds = target.predict(x_test_adv)
acc = np.sum(np.equal(np.argmax(preds, axis=1), np.argmax(y_test, axis=1))) / y_test.shape[0]
print("\nAccuracy on adversarial samples: %.2f%%" % (acc * 100))
