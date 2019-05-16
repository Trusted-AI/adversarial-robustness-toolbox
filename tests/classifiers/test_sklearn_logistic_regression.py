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
tf.set_random_seed(1234)

NB_TRAIN = 40

tf.keras.backend.set_floatx('float64')


class TestSklearnLogisticRegression(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        num_classes = 10
        cls.num_features = 784
        cls.num_samples = NB_TRAIN
        cls.num_samples_loss = 20

        (x_train, y_train), (_, _), min_, max_ = load_mnist()

        cls.x_train = x_train[0:cls.num_samples].reshape((cls.num_samples, 1, cls.num_features, 1))
        cls.y_train = y_train[0:cls.num_samples]

        clip_values = (0, 1)

        def unitnorm(x):
            return x / (tf.keras.backend.epsilon() + (tf.keras.backend.sum(x, keepdims=True, axis=1)))

        cls.model = tf.keras.models.Sequential()
        cls.model.add(tf.keras.layers.Flatten(input_shape=(1, cls.num_features, 1)))
        cls.model.add(tf.keras.layers.Dense(num_classes, use_bias=True, activation=tf.keras.activations.sigmoid,
                                            kernel_regularizer=tf.keras.regularizers.l2(0.5)))
        cls.model.add(tf.keras.layers.Lambda(unitnorm, name='unitnorm'))

        cls.model.compile(optimizer='adadelta',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

        cls.model.fit(cls.x_train, cls.y_train, epochs=400)

        (cls.lr_weights, cls.lr_biases) = cls.model.layers[1].get_weights()

        y_tensor = tf.keras.backend.placeholder(shape=(cls.num_samples_loss, num_classes))

        loss = tf.keras.losses.categorical_crossentropy(y_tensor, cls.model.outputs[0])
        loss_reg = cls.model.losses[0]
        loss += loss_reg

        # gradients = tf.keras.backend.gradients(loss, cls.model.input)

        # with tf.keras.backend.get_session() as sess:
        #     evaluated_gradients = sess.run(gradients,
        #                                    feed_dict={cls.model.input: cls.x_train[0:cls.num_samples_loss],
        #                                               y_tensor: cls.y_train[0:cls.num_samples_loss]})
        #
        #     evaluated_loss = sess.run(loss,
        #                               feed_dict={cls.model.input: cls.x_train[0:cls.num_samples_loss],
        #                                          y_tensor: cls.y_train[0:cls.num_samples_loss]})
        #
        #     evaluated_loss_reg = sess.run(loss_reg,
        #                                   feed_dict={cls.model.input: cls.x_train[0:cls.num_samples_loss],
        #                                              y_tensor: cls.y_train[0:cls.num_samples_loss]})

        # print('evaluated_loss:', evaluated_loss)
        # print('evaluated_loss mean:', np.mean(evaluated_loss))
        # print('evaluated_loss sum:', np.sum(evaluated_loss))
        # print('evaluated_loss_reg:', evaluated_loss_reg)

        # cls.grad_tf_0 = evaluated_gradients[0].reshape(cls.num_samples_loss, cls.num_features)

        sklearn_model = LogisticRegression(verbose=0, C=1, solver='newton-cg', dual=False, fit_intercept=True)
        cls.classifier = SklearnLogisticRegression(clip_values=clip_values, model=sklearn_model)
        cls.classifier.fit(x=cls.x_train.reshape((cls.num_samples, cls.num_features)), y=cls.y_train)

    def test_predict(self):
        y_pred = self.classifier.predict(self.x_train[0:self.num_samples_loss].reshape(self.num_samples_loss, self.num_features))

        y_target = [0.10596804, 0.09933693, 0.12362145, 0.16801073, 0.07376496, 0.1022053, 0.11010179, 0.06613702,
                    0.07016114, 0.08069265]

        for i in range(10):
            self.assertAlmostEqual(y_target[i], y_pred[0, i], places=1)

    def test_class_gradient(self):
        self.classifier.w = self.lr_weights.T

        label = np.zeros((NB_TRAIN, 10))
        label[:, 3] = 1

        grad = self.classifier.class_gradient(self.x_train[0:self.num_samples_loss].reshape(self.num_samples_loss, self.num_features), label=label)

        self.assertTrue(abs(grad[0, 0] - 2.79379165e-04) < 0.001)

    def test_loss_gradient(self):
        # from matplotlib import pyplot as plt

        self.classifier.w = self.lr_weights.T

        self.classifier.model.coef_ = self.lr_weights.T
        self.classifier.model.intercept_ = self.lr_biases

        grad = self.classifier.loss_gradient(
            self.x_train[0:self.num_samples_loss].reshape(self.num_samples_loss, self.num_features),
            self.y_train[0:self.num_samples_loss])

        # for i in range(self.num_samples_loss):
        #     plt.plot(self.grad_tf_0[i], label='tf')
        #     plt.plot(grad[i], label='sklearn')
        #     plt.legend()
        #     plt.show()
        #
        # plt.matshow(self.grad_tf_0[0].reshape((28, 28)))
        # plt.colorbar()
        # plt.clim(-0.004, 0.004)
        # plt.show()
        # plt.matshow(grad[0].reshape((28, 28)))
        # plt.colorbar()
        # plt.clim(-0.004, 0.004)
        # plt.show()
