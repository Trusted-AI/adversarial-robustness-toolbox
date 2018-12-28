from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import keras
import keras.backend as k
import numpy as np
import tensorflow as tf
import torch.nn as nn
import torch.optim as optim
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential

from art.attacks import ElasticNet
from art.classifiers import KerasClassifier, PyTorchClassifier, TFClassifier
from art.utils import load_mnist, random_targets, master_seed

logger = logging.getLogger('testLogger')

BATCH_SIZE, NB_TRAIN, NB_TEST = 100, 500, 10


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(2304, 10)

    def forward(self, x):
        import torch.nn.functional as f

        x = self.pool(f.relu(self.conv(x)))
        x = x.view(-1, 2304)
        logit_output = self.fc(x)

        return logit_output


class TestElasticNet(unittest.TestCase):
    """
    A unittest class for testing the ElasticNet attack.
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

    def test_failure_attack(self):
        """
        Test the corner case when attack is failed.
        :return:
        """
        # Build a TFClassifier
        # Define input and output placeholders
        input_ph = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        output_ph = tf.placeholder(tf.int32, shape=[None, 10])

        # Define the tensorflow graph
        conv = tf.layers.conv2d(input_ph, 4, 5, activation=tf.nn.relu)
        conv = tf.layers.max_pooling2d(conv, 2, 2)
        fc = tf.contrib.layers.flatten(conv)

        # Logits layer
        logits = tf.layers.dense(fc, 10)

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
        tfc.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epochs=10)

        # Failure attack
        eda = ElasticNet(classifier=tfc, targeted=True, max_iter=0, binary_search_steps=0, learning_rate=0,
                         initial_const=1)
        params = {'y': random_targets(y_test, tfc.nb_classes)}
        x_test_adv = eda.generate(x_test, **params)
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        np.testing.assert_almost_equal(x_test, x_test_adv, 3)

        # Kill TF
        sess.close()
        tf.reset_default_graph()

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
        fc = tf.contrib.layers.flatten(conv)

        # Logits layer
        logits = tf.layers.dense(fc, 10)

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
        tfc.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epochs=10)

        # First attack
        eda = ElasticNet(classifier=tfc, targeted=True, max_iter=10)
        params = {'y': random_targets(y_test, tfc.nb_classes)}
        x_test_adv = eda.generate(x_test, **params)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug('EDA Target: %s', target)
        logger.debug('EDA Actual: %s', y_pred_adv)
        logger.info('EDA Success Rate: %.2f', (sum(target == y_pred_adv) / float(len(target))))
        self.assertTrue((target == y_pred_adv).any())

        # Second attack
        eda = ElasticNet(classifier=tfc, targeted=False, max_iter=10)
        params = {'y': random_targets(y_test, tfc.nb_classes)}
        x_test_adv = eda.generate(x_test, **params)
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug('EDA Target: %s', target)
        logger.debug('EDA Actual: %s', y_pred_adv)
        logger.info('EDA Success Rate: %.2f', (sum(target != y_pred_adv) / float(len(target))))
        self.assertTrue((target != y_pred_adv).any())

        # Third attack
        eda = ElasticNet(classifier=tfc, targeted=False, max_iter=10)
        params = {}
        x_test_adv = eda.generate(x_test, **params)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        y_pred = np.argmax(tfc.predict(x_test), axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug('EDA Target: %s', y_pred)
        logger.debug('EDA Actual: %s', y_pred_adv)
        logger.info('EDA Success Rate: %.2f', (sum(y_pred != y_pred_adv) / float(len(y_pred))))
        self.assertTrue((y_pred != y_pred_adv).any())

        # First attack without batching
        edawob = ElasticNet(classifier=tfc, targeted=True, max_iter=10, batch_size=1)
        params = {'y': random_targets(y_test, tfc.nb_classes)}
        x_test_adv = edawob.generate(x_test, **params)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug('EDA Target: %s', target)
        logger.debug('EDA Actual: %s', y_pred_adv)
        logger.info('EDA Success Rate: %.2f', (sum(target == y_pred_adv) / float(len(target))))
        self.assertTrue((target == y_pred_adv).any())

        # Second attack without batching
        edawob = ElasticNet(classifier=tfc, targeted=False, max_iter=10, batch_size=1)
        params = {'y': random_targets(y_test, tfc.nb_classes)}
        x_test_adv = edawob.generate(x_test, **params)
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug('EDA Target: %s', target)
        logger.debug('EDA Actual: %s', y_pred_adv)
        logger.info('EDA Success Rate: %.2f', (sum(target != y_pred_adv) / float(len(target))))
        self.assertTrue((target != y_pred_adv).any())

        # Third attack without batching
        edawob = ElasticNet(classifier=tfc, targeted=False, max_iter=10, batch_size=1)
        params = {}
        x_test_adv = edawob.generate(x_test, **params)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        y_pred = np.argmax(tfc.predict(x_test), axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug('EDA Target: %s', y_pred)
        logger.debug('EDA Actual: %s', y_pred_adv)
        logger.info('EDA Success Rate: %.2f', (sum(y_pred != y_pred_adv) / float(len(y_pred))))
        self.assertTrue((y_pred != y_pred_adv).any())

        # Kill TF
        sess.close()
        tf.reset_default_graph()










































