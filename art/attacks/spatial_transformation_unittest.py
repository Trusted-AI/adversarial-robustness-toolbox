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

import logging
import unittest

import keras
import keras.backend as k
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import numpy as np
import tensorflow as tf
import torch.nn as nn
import torch.optim as optim

from art.attacks.spatial_transformation import SpatialTransformation
from art.classifiers import KerasClassifier, PyTorchClassifier, TFClassifier
from art.utils import load_mnist, master_seed

logger = logging.getLogger('testLogger')
logger.setLevel(10)

BATCH_SIZE = 100
NB_TRAIN = 1000
NB_TEST = 10


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fullyconnected = nn.Linear(2304, 10)

    def forward(self, x):
        import torch.nn.functional as f

        x = self.pool(f.relu(self.conv(x)))
        x = x.view(-1, 2304)
        logit_output = self.fullyconnected(x)

        return logit_output


class TestSpatialTransformation(unittest.TestCase):
    """
    A unittest class for testing Spatial attack.
    """

    @classmethod
    def setUpClass(cls):
        # Get MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]
        cls.mnist = (x_train, y_train), (x_test, y_test)

    def setUp(self):
        # Set master seed
        master_seed(1234)

    def test_tfclassifier(self):
        """
        First test with the TFClassifier.
        :return:
        """
        # Build a TFClassifier
        # Define input and output placeholders
        input_ph = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        output_ph = tf.placeholder(tf.int32, shape=[None, 10])

        # Define the tensorflow graph
        conv = tf.layers.conv2d(input_ph, 4, 5, activation=tf.nn.relu)
        conv = tf.layers.max_pooling2d(conv, 2, 2)
        flattened = tf.contrib.layers.flatten(conv)

        # Logits layer
        logits = tf.layers.dense(flattened, 10)

        # Train operator
        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=output_ph))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train = optimizer.minimize(loss)

        # Tensorflow session and initialization
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # Get MNIST
        (x_train, y_train), (x_test, y_test) = self.mnist

        # Train the classifier
        tfc = TFClassifier((0, 1), input_ph, logits, output_ph, train, loss, None, sess)
        tfc.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epochs=2)

        # Attack
        attack_params = {"max_translation": 10.0, "num_translations": 3, "max_rotation": 30.0, "num_rotations": 3}
        attack_st = SpatialTransformation(tfc)
        x_train_adv = attack_st.generate(x_train, **attack_params)

        self.assertTrue(abs(x_train_adv[0, 8, 13, 0] - 0.8066048) <= 0.01)

        # self.assertTrue(abs(attack_st.fooling_rate - 0.948) <= 0.01)

        self.assertTrue(attack_st.attack_trans_x == -3)
        self.assertTrue(attack_st.attack_trans_y == -3)
        self.assertTrue(attack_st.attack_rot == -30.0)

        x_test_adv = attack_st.generate(x_test)

        self.assertTrue(abs(x_test_adv[0, 14, 14, 0] - 0.6941315) <= 0.01)

        sess.close()
        tf.reset_default_graph()

    def test_krclassifier(self):
        """
        Second test with the KerasClassifier.
        :return:
        """
        # Initialize a tf session
        session = tf.Session()
        k.set_session(session)

        # Get MNIST
        (x_train, y_train), (x_test, y_test) = self.mnist

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
        krc.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epochs=2)

        # Attack
        attack_params = {"max_translation": 10.0, "num_translations": 3, "max_rotation": 30.0, "num_rotations": 3}
        attack_st = SpatialTransformation(krc)
        x_train_adv = attack_st.generate(x_train, **attack_params)

        self.assertTrue(abs(x_train_adv[0, 8, 13, 0] - 0.8066048) <= 0.01)
        self.assertTrue(abs(attack_st.fooling_rate - 0.923) <= 0.01)

        self.assertTrue(attack_st.attack_trans_x == -3)
        self.assertTrue(attack_st.attack_trans_y == -3)
        self.assertTrue(attack_st.attack_rot == -30.0)

        x_test_adv = attack_st.generate(x_test)

        self.assertTrue(abs(x_test_adv[0, 14, 14, 0] - 0.6941315) <= 0.01)

        k.clear_session()

    def test_ptclassifier(self):
        """
        Third test with the PyTorchClassifier.
        :return:
        """
        # Get MNIST
        (x_train, y_train), (x_test, y_test) = self.mnist
        x_train = np.swapaxes(x_train, 1, 3)
        x_test = np.swapaxes(x_test, 1, 3)

        # Create simple CNN
        # Define the network
        model = Model()

        # Define a loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Get classifier
        ptc = PyTorchClassifier((0, 1), model, loss_fn, optimizer, (1, 28, 28), 10)
        ptc.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epochs=1)

        # Attack
        attack_params = {"max_translation": 10.0, "num_translations": 3, "max_rotation": 30.0, "num_rotations": 3}
        attack_st = SpatialTransformation(ptc)
        x_train_adv = attack_st.generate(x_train, **attack_params)

        self.assertTrue(abs(x_train_adv[0, 0, 13, 5] - 0.374206543) <= 0.01)
        # self.assertTrue(abs(attack_st.fooling_rate - 0.781) <= 0.01)

        self.assertTrue(attack_st.attack_trans_x == 0)
        self.assertTrue(attack_st.attack_trans_y == -3)
        self.assertTrue(attack_st.attack_rot == 30.0)

        x_test_adv = attack_st.generate(x_test)

        self.assertTrue(abs(x_test_adv[0, 0, 14, 14] - 0.008591662) <= 0.01)


if __name__ == '__main__':
    unittest.main()
