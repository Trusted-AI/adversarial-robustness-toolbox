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

import tensorflow as tf
import unittest
import numpy as np
import keras
import keras.backend as k
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

from art.attacks.universal_perturbation import UniversalPerturbation
from art.classifiers.tensorflow import TFClassifier
from art.classifiers.keras import KerasClassifier
from art.utils import load_mnist


class TestUniversalPerturbation(unittest.TestCase):
    """
    A unittest class for testing the UniversalPerturbation attack.
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
        batch_size, nb_train, nb_test = 10, 10, 10
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train = x_train[:nb_train], y_train[:nb_train]
        x_test, y_test = x_test[:nb_test], y_test[:nb_test]

        # Train the classifier
        tfc = TFClassifier((0, 1), self._input_ph, self._logits, self._output_ph,
                           self._train, self._loss, None, self._sess)
        tfc.fit(x_train, y_train, batch_size=batch_size, nb_epochs=2)

        # Attack
        # TODO Launch with all possible attacks
        attack_params = {"attacker": "newtonfool", "attacker_params": {"max_iter": 20}}
        up = UniversalPerturbation(tfc)
        x_train_adv = up.generate(x_train, **attack_params)
        self.assertTrue((up.fooling_rate >= 0.2) or not up.converged)

        x_test_adv = x_test + up.v
        self.assertFalse((x_test == x_test_adv).all())

        train_y_pred = np.argmax(tfc.predict(x_train_adv), axis=1)
        test_y_pred = np.argmax(tfc.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == test_y_pred).all())
        self.assertFalse((np.argmax(y_train, axis=1) == train_y_pred).all())

    def test_krclassifier(self):
        """
        Second test with the KerasClassifier.
        :return:
        """
        # Initialize a tf session
        session = tf.Session()
        k.set_session(session)

        # Get MNIST
        batch_size, nb_train, nb_test = 10, 10, 10
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
        # TODO Launch with all possible attacks
        attack_params = {"attacker": "newtonfool", "attacker_params": {"max_iter": 20}}
        up = UniversalPerturbation(krc)
        x_train_adv = up.generate(x_train, **attack_params)
        self.assertTrue((up.fooling_rate >= 0.2) or not up.converged)

        x_test_adv = x_test + up.v
        self.assertFalse((x_test == x_test_adv).all())

        train_y_pred = np.argmax(krc.predict(x_train_adv), axis=1)
        test_y_pred = np.argmax(krc.predict(x_test_adv), axis=1)
        self.assertFalse((np.argmax(y_test, axis=1) == test_y_pred).all())
        self.assertFalse((np.argmax(y_train, axis=1) == train_y_pred).all())


if __name__ == '__main__':
    unittest.main()
