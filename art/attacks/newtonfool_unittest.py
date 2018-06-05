# MIT License
#
# Copyright (C) IBM Corporation 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import tensorflow as tf
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from art.attacks.newtonfool import NewtonFool
from art.classifiers.tensorflow import TFClassifier
from art.classifiers.keras import KerasClassifier
from art.classifiers.pytorch import PyTorchClassifier
from art.utils import load_mnist


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(2304, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = x.view(-1, 2304)
        logit_output = self.fc(x)
        output = F.softmax(logit_output, dim=1)

        return logit_output, output


class TestNewtonFool(unittest.TestCase):
    """
    A unittest class for testing the NewtonFool attack.
    """
    def test_tfclassifier(self):
        """
        First test with the TFClassifier.
        :return:
        """
        # Build a TFClassifier
        # Define input and output placeholders
        self._input_ph = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        self._output_ph = tf.placeholder(tf.int32, shape=[None, 10])

        # Define the tensorflow graph
        conv = tf.layers.conv2d(self._input_ph, 4, 5, activation=tf.nn.relu)
        conv = tf.layers.max_pooling2d(conv, 2, 2)
        fc = tf.contrib.layers.flatten(conv)

        # Logits layer
        self._logits = tf.layers.dense(fc, 10)

        # Train operator
        self._loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=self._logits, onehot_labels=self._output_ph))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        self._train = optimizer.minimize(self._loss)

        # Tensorflow session and initialization
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

        # Get MNIST
        batch_size, nb_train, nb_test = 100, 1000, 10
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train = x_train[:nb_train], y_train[:nb_train]
        x_test, y_test = x_test[:nb_test], y_test[:nb_test]

        # Train the classifier
        tfc = TFClassifier((0, 1), self._input_ph, self._logits, self._output_ph, self._train, self._loss, None,
                           self._sess)
        tfc.fit(x_train, y_train, batch_size=batch_size, nb_epochs=2)

        # Attack
        nf = NewtonFool(tfc)
        nf.set_params(max_iter=5)
        x_test_adv = nf.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())

        y_pred = tfc.predict(x_test)
        y_pred_adv = tfc.predict(x_test_adv)
        y_pred_bool = y_pred.max(axis=1, keepdims=1) == y_pred
        y_pred_max = y_pred.max(axis=1)
        y_pred_adv_max = y_pred_adv[y_pred_bool]
        self.assertTrue((y_pred_max >= y_pred_adv_max).all())

    def test_krclassifier(self):
        """
        Second test with the KerasClassifier.
        :return:
        """
        # Get MNIST
        batch_size, nb_train, nb_test = 100, 1000, 10
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train = x_train[:nb_train], y_train[:nb_train]
        x_test, y_test = x_test[:nb_test], y_test[:nb_test]

        # Create simple CNN
        model = Sequential()
        model.add(Conv2D(4, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(10, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.01),
                      metrics=['accuracy'])

        # Get classifier
        krc = KerasClassifier((0, 1), model, use_logits=False)
        krc.fit(x_train, y_train, batch_size=batch_size, nb_epochs=2)

        # Attack
        nf = NewtonFool(krc)
        nf.set_params(max_iter=5)
        x_test_adv = nf.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())

        y_pred = krc.predict(x_test)
        y_pred_adv = krc.predict(x_test_adv)
        y_pred_bool = y_pred.max(axis=1, keepdims=1) == y_pred
        y_pred_max = y_pred.max(axis=1)
        y_pred_adv_max = y_pred_adv[y_pred_bool]
        self.assertTrue((y_pred_max >= y_pred_adv_max).all())

    def test_ptclassifier(self):
        """
        Third test with the PyTorchClassifier.
        :return:
        """
        # Get MNIST
        batch_size, nb_train, nb_test = 100, 1000, 10
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train = x_train[:nb_train], np.argmax(y_train[:nb_train], axis=1)
        x_test, y_test = x_test[:nb_test], np.argmax(y_test[:nb_test], axis=1)
        x_train = np.swapaxes(x_train, 1, 3)
        x_test = np.swapaxes(x_test, 1, 3)

        # Create simple CNN
        # Define the network
        model = Model()

        # Define a loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Get classifier
        ptc = PyTorchClassifier((0, 1), model, loss_fn, optimizer, (1, 28, 28), (10,))
        ptc.fit(x_train, y_train, batch_size=batch_size, nb_epochs=1)

        # Attack
        nf = NewtonFool(ptc)
        nf.set_params(max_iter=5)
        x_test_adv = nf.generate(x_test)
        self.assertFalse((x_test == x_test_adv).all())

        y_pred = ptc.predict(x_test)
        y_pred_adv = ptc.predict(x_test_adv)
        y_pred_bool = y_pred.max(axis=1, keepdims=1) == y_pred
        y_pred_max = y_pred.max(axis=1)
        y_pred_adv_max = y_pred_adv[y_pred_bool]
        self.assertTrue((y_pred_max >= y_pred_adv_max).all())


if __name__ == '__main__':
    unittest.main()
