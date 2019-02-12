from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest

import keras
import keras.backend as k
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential

from art.attacks import CarliniL2Method, CarliniLInfMethod
from art.classifiers import KerasClassifier, TFClassifier
from art.utils import load_mnist, random_targets, master_seed

logger = logging.getLogger('testLogger')

BATCH_SIZE, NB_TRAIN, NB_TEST = 100, 5000, 10


class TestCarliniL2(unittest.TestCase):
    """
    A unittest class for testing the Carlini L2 attack.
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
        cl2m = CarliniL2Method(classifier=tfc, targeted=True, max_iter=0, binary_search_steps=0,
                               learning_rate=0, initial_const=1)
        params = {'y': random_targets(y_test, tfc.nb_classes)}
        x_test_adv = cl2m.generate(x_test, **params)
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        self.assertTrue(np.allclose(x_test, x_test_adv, atol=1e-3))

        # Clean-up session
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
        cl2m = CarliniL2Method(classifier=tfc, targeted=True, max_iter=10)
        params = {'y': random_targets(y_test, tfc.nb_classes)}
        x_test_adv = cl2m.generate(x_test, **params)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug('CW2 Target: %s', target)
        logger.debug('CW2 Actual: %s', y_pred_adv)
        logger.info('CW2 Success Rate: %.2f', (np.sum(target == y_pred_adv) / float(len(target))))
        self.assertTrue((target == y_pred_adv).any())

        # Second attack, no batching
        cl2m = CarliniL2Method(classifier=tfc, targeted=False, max_iter=10, batch_size=1)
        x_test_adv = cl2m.generate(x_test)
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug('CW2 Target: %s', target)
        logger.debug('CW2 Actual: %s', y_pred_adv)
        logger.info('CW2 Success Rate: %.2f', (np.sum(target == y_pred_adv) / float(len(target))))
        self.assertTrue((target != y_pred_adv).any())

        # Clean-up session
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
        krc.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epochs=10)

        # First attack
        cl2m = CarliniL2Method(classifier=krc, targeted=True, max_iter=10)
        params = {'y': random_targets(y_test, krc.nb_classes)}
        x_test_adv = cl2m.generate(x_test, **params)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
        logger.debug('CW2 Target: %s', target)
        logger.debug('CW2 Actual: %s', y_pred_adv)
        logger.info('CW2 Success Rate: %.2f', (np.sum(target == y_pred_adv) / float(len(target))))
        self.assertTrue((target == y_pred_adv).any())

        # Second attack
        cl2m = CarliniL2Method(classifier=krc, targeted=False, max_iter=10)
        x_test_adv = cl2m.generate(x_test)
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
        logger.debug('CW2 Target: %s', target)
        logger.debug('CW2 Actual: %s', y_pred_adv)
        logger.info('CW2 Success Rate: %.2f', (np.sum(target != y_pred_adv) / float(len(target))))
        self.assertTrue((target != y_pred_adv).any())

        # Clean-up
        k.clear_session()

    # def test_ptclassifier(self):
    #     """
    #     Third test with the PyTorchClassifier.
    #     :return:
    #     """
    #     # Get MNIST
    #     (x_train, y_train), (x_test, y_test) = self.mnist
    #     x_train = np.swapaxes(x_train, 1, 3)
    #     x_test = np.swapaxes(x_test, 1, 3)
    #
    #     # Define the network
    #     model = Model()
    #
    #     # Define a loss function and optimizer
    #     loss_fn = nn.CrossEntropyLoss()
    #     optimizer = optim.Adam(model.parameters(), lr=0.01)
    #
    #     # Get classifier
    #     ptc = PyTorchClassifier((0, 1), model, loss_fn, optimizer, (1, 28, 28), 10)
    #     ptc.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epochs=10)
    #
    #     # First attack
    #     cl2m = CarliniL2Method(classifier=ptc, targeted=True, max_iter=10)
    #     params = {'y': random_targets(y_test, ptc.nb_classes)}
    #     x_test_adv = cl2m.generate(x_test, **params)
    #     self.assertFalse((x_test == x_test_adv).all())
    #     self.assertTrue((x_test_adv <= 1.0001).all())
    #     self.assertTrue((x_test_adv >= -0.0001).all())
    #     target = np.argmax(params['y'], axis=1)
    #     y_pred_adv = np.argmax(ptc.predict(x_test_adv), axis=1)
    #     self.assertTrue((target == y_pred_adv).any())
    #     logger.info('CW2 Success Rate: %.2f', (sum(target == y_pred_adv) / float(len(target))))
    #
    #     # Second attack
    #     cl2m = CarliniL2Method(classifier=ptc, targeted=False, max_iter=10)
    #     x_test_adv = cl2m.generate(x_test)
    #     self.assertTrue((x_test_adv <= 1.0001).all())
    #     self.assertTrue((x_test_adv >= -0.0001).all())
    #     target = np.argmax(params['y'], axis=1)
    #     y_pred_adv = np.argmax(ptc.predict(x_test_adv), axis=1)
    #     self.assertTrue((target != y_pred_adv).any())
    #     logger.info('CW2 Success Rate: %.2f', (sum(target != y_pred_adv) / float(len(target))))


class TestCarliniLInf(TestCarliniL2):
    """
    A unittest class for testing the Carlini LInf attack.
    """

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
        clinfm = CarliniLInfMethod(classifier=tfc, targeted=True, max_iter=0, learning_rate=0, eps=0.5)
        params = {'y': random_targets(y_test, tfc.nb_classes)}
        x_test_adv = clinfm.generate(x_test, **params)
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        self.assertTrue(np.allclose(x_test, x_test_adv, atol=1e-3))

        # Clean-up session
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
        clinfm = CarliniLInfMethod(classifier=tfc, targeted=True, max_iter=10, eps=0.5)
        params = {'y': random_targets(y_test, tfc.nb_classes)}
        x_test_adv = clinfm.generate(x_test, **params)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug('CW0 Target: %s', target)
        logger.debug('CW0 Actual: %s', y_pred_adv)
        logger.info('CW0 Success Rate: %.2f', (np.sum(target == y_pred_adv) / float(len(target))))
        self.assertTrue((target == y_pred_adv).any())

        # Second attack, no batching
        clinfm = CarliniLInfMethod(classifier=tfc, targeted=False, max_iter=10, eps=0.5, batch_size=1)
        x_test_adv = clinfm.generate(x_test)
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(tfc.predict(x_test_adv), axis=1)
        logger.debug('CW0 Target: %s', target)
        logger.debug('CW0 Actual: %s', y_pred_adv)
        logger.info('CW0 Success Rate: %.2f', (np.sum(target != y_pred_adv) / float(len(target))))
        self.assertTrue((target != y_pred_adv).any())

        # Clean-up session
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
        krc.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epochs=10)

        # First attack
        clinfm = CarliniLInfMethod(classifier=krc, targeted=True, max_iter=10, eps=0.5)
        params = {'y': random_targets(y_test, krc.nb_classes)}
        x_test_adv = clinfm.generate(x_test, **params)
        self.assertFalse((x_test == x_test_adv).all())
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
        logger.debug('CW0 Target: %s', target)
        logger.debug('CW0 Actual: %s', y_pred_adv)
        logger.info('CW0 Success Rate: %.2f', (np.sum(target == y_pred_adv) / float(len(target))))
        self.assertTrue((target == y_pred_adv).any())

        # Second attack
        clinfm = CarliniLInfMethod(classifier=krc, targeted=False, max_iter=10, eps=0.5)
        x_test_adv = clinfm.generate(x_test)
        self.assertTrue((x_test_adv <= 1.0001).all())
        self.assertTrue((x_test_adv >= -0.0001).all())
        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(krc.predict(x_test_adv), axis=1)
        logger.debug('CW0 Target: %s', target)
        logger.debug('CW0 Actual: %s', y_pred_adv)
        logger.info('CW0 Success Rate: %.2f', (np.sum(target != y_pred_adv) / float(len(target))))
        self.assertTrue((target != y_pred_adv).any())

        # Clean-up
        k.clear_session()

    # def test_ptclassifier(self):
    #     """
    #     Third test with the PyTorchClassifier.
    #     :return:
    #     """
    #     # Get MNIST
    #     (x_train, y_train), (x_test, y_test) = self.mnist
    #     x_train = np.swapaxes(x_train, 1, 3)
    #     x_test = np.swapaxes(x_test, 1, 3)
    #
    #     # Define the network
    #     model = Model()
    #
    #     # Define a loss function and optimizer
    #     loss_fn = nn.CrossEntropyLoss()
    #     optimizer = optim.Adam(model.parameters(), lr=0.01)
    #
    #     # Get classifier
    #     ptc = PyTorchClassifier((0, 1), model, loss_fn, optimizer, (1, 28, 28), 10)
    #     ptc.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epochs=10)
    #
    #     # First attack
    #     clinfm = CarliniLInfMethod(classifier=ptc, targeted=True, max_iter=10, eps=0.5)
    #     params = {'y': random_targets(y_test, ptc.nb_classes)}
    #     x_test_adv = clinfm.generate(x_test, **params)
    #     self.assertFalse((x_test == x_test_adv).all())
    #     self.assertTrue((x_test_adv <= 1.0001).all())
    #     self.assertTrue((x_test_adv >= -0.0001).all())
    #     target = np.argmax(params['y'], axis=1)
    #     y_pred_adv = np.argmax(ptc.predict(x_test_adv), axis=1)
    #     self.assertTrue((target == y_pred_adv).any())
    #
    #     # Second attack
    #     clinfm = CarliniLInfMethod(classifier=ptc, targeted=False, max_iter=10, eps=0.5)
    #     x_test_adv = clinfm.generate(x_test)
    #     self.assertTrue((x_test_adv <= 1.0001).all())
    #     self.assertTrue((x_test_adv >= -0.0001).all())
    #     target = np.argmax(params['y'], axis=1)
    #     y_pred_adv = np.argmax(ptc.predict(x_test_adv), axis=1)
    #     self.assertTrue((target != y_pred_adv).any())


if __name__ == '__main__':
    unittest.main()
