from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import tensorflow as tf
import numpy as np

from sklearn.linear_model import LogisticRegression

from art.classifiers import SklearnLogisticRegression
from art.utils import load_mnist

logger = logging.getLogger('testLogger')
np.random.seed(seed=1234)

BATCH_SIZE = 10
NB_TRAIN = 500
NB_TEST = 100


class TestSklearnLogisticRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        num_classes = 10

        (x_train, y_train), (_, _), min_, max_ = load_mnist()

        cls.x_train = x_train[:NB_TRAIN].reshape(NB_TRAIN, 784)
        cls.y_train = y_train[:NB_TRAIN]
        clip_values = (min_, max_)

        cls.model = tf.keras.models.Sequential()
        cls.model.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1,)))
        cls.model.add(tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax))

        cls.model.compile(optimizer='sgd',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

        cls.model.fit(x_train, y_train, epochs=1)

        cls.lr_weights = np.random.random((28 * 28, num_classes)) * 0.001
        cls.lr_biases = np.ones(num_classes)

        cls.model.layers[1].set_weights([cls.lr_weights, cls.lr_biases])

        loss_function = tf.keras.losses.categorical_crossentropy

        y_tensor = tf.keras.backend.placeholder(shape=(5, 10))
        loss = loss_function(y_tensor, cls.model.outputs[0])
        gradients = tf.keras.backend.gradients(loss, cls.model.input)

        with tf.keras.backend.get_session() as sess:
            evaluated_gradients = sess.run(gradients,
                                           feed_dict={cls.model.input: np.reshape(x_train[0:5], (5, 28, 28, 1)),
                                                      y_tensor: np.reshape(y_train[0:5], (5, 10))})

        cls.grad_tf_0 = evaluated_gradients[0].reshape(5, 784)[0, :]

        sklearn_model = LogisticRegression()
        cls.classifier = SklearnLogisticRegression(clip_values=clip_values, model=sklearn_model)
        cls.classifier.fit(x=cls.x_train, y=cls.y_train)

    def test_predict(self):
        y_pred = self.classifier.predict(self.x_train[0:1])

        y_target = [1.61375339e-03, 5.47272107e-03, 1.71608819e-03, 5.41703142e-02, 1.25176200e-06, 9.31320640e-01,
                    6.62234138e-04, 4.30541412e-03, 6.99393960e-04, 3.81892851e-05]

        for i in range(10):
            self.assertAlmostEqual(y_target[i], y_pred[0, i])

    def test_class_gradient(self):
        self.classifier.w = self.lr_weights.T

        label = np.zeros((NB_TRAIN, 10))
        label[:, 3] = 1

        grad = self.classifier.class_gradient(self.x_train, label=label)

        self.assertTrue(abs(grad[0, 0] - 2.79379165e-04) < 0.001)

    def test_loss_gradient(self):
        self.classifier.w = self.lr_weights.T
        grad = self.classifier.loss_gradient(self.x_train[0:5], self.y_train[0:5])

        # from matplotlib import pyplot as plt
        # plt.plot(self.grad_tf_0)
        # plt.plot(self.y2)
        # plt.show()

        for i in range(self.grad_tf_0.shape[0]):
            self.assertTrue(abs(self.grad_tf_0[i] - grad[0, i]) < 0.1)
