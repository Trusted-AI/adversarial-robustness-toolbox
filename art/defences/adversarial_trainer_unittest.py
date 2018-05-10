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

import keras.backend as k
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout
import numpy as np
import tensorflow as tf

from art.attacks import FastGradientMethod, DeepFool
from art.classifiers import TFClassifier, KerasClassifier
from art.defences.adversarial_trainer import AdversarialTrainer
from art.utils import load_mnist, get_labels_np_array

BATCH_SIZE = 10
NB_TRAIN = 100
NB_TEST = 11


class TestAdversarialTrainer(unittest.TestCase):
    """
    Test cases for the AdversarialTrainer class.
    """

    def setUp(self):
        k.set_learning_phase(1)

        # Get MNIST
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train, x_test, y_test = x_train[:NB_TRAIN], y_train[:NB_TRAIN], x_test[:NB_TEST], y_test[:NB_TEST]
        self.mnist = ((x_train, y_train), (x_test, y_test))

        self.classifier_k = self._cnn_mnist_k(x_train.shape[1:])
        self.classifier_k.fit(x_train, y_train, nb_epochs=2, batch_size=BATCH_SIZE)

        scores = self.classifier_k._model.evaluate(x_train, y_train)
        print("\n[Keras, MNIST] Accuracy on training set: %.2f%%" % (scores[1] * 100))
        scores = self.classifier_k._model.evaluate(x_test, y_test)
        print("\n[Keras, MNIST] Accuracy on test set: %.2f%%" % (scores[1] * 100))

        # Create basic CNN on MNIST using TensorFlow
        self.classifier_tf = self._cnn_mnist_tf(x_train.shape[1:])
        self.classifier_tf.fit(x_train, y_train, nb_epochs=2, batch_size=BATCH_SIZE)

        scores = get_labels_np_array(self.classifier_tf.predict(x_train))
        acc = np.sum(np.argmax(scores, axis=1) == np.argmax(y_train, axis=1)) / y_train.shape[0]
        print('\n[TF, MNIST] Accuracy on training set: %.2f%%' % (acc * 100))

        scores = get_labels_np_array(self.classifier_tf.predict(x_test))
        acc = np.sum(np.argmax(scores, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        print('\n[TF, MNIST] Accuracy on test set: %.2f%%' % (acc * 100))

    def test_one_attack_mnist(self):
        """
        Test the adversarial trainer using one FGSM attacker. The source and target models of the attack
        are two CNNs on MNIST (TensorFlow and Keras backends). The test cast check if accuracy on adversarial samples
        increases after adversarially training the model.

        :return: None
        """
        (x_train, y_train), (x_test, y_test) = self.mnist

        # Get source and target classifiers
        classifier_src = self.classifier_k
        classifier_tgt = self.classifier_tf

        # Create FGSM attacker
        adv = FastGradientMethod(classifier_src)
        x_adv = adv.generate(x_test)
        preds = classifier_tgt.predict(x_adv)
        acc = np.sum(np.argmax(preds, axis=1) == np.argmax(y_test, axis=1)) / x_adv.shape[0]

        # Perform adversarial training
        adv_trainer = AdversarialTrainer(classifier_tgt, adv)
        adv_trainer.fit(x_train, y_train, nb_epochs=1)

        # Evaluate that accuracy on adversarial sample has improved
        preds_adv_trained = adv_trainer.classifier.predict(x_adv)
        acc_adv_trained = np.sum(np.argmax(preds_adv_trained, axis=1) == np.argmax(y_test, axis=1)) / x_adv.shape[0]
        print('\nAccuracy before adversarial training: %.2f%%' % (acc * 100))
        print('\nAccuracy after adversarial training: %.2f%%' % (acc_adv_trained * 100))

    def test_multi_attack_mnist(self):
        """
        Test the adversarial trainer using two attackers: FGSM and DeepFool. The source and target models of the attack
        are two CNNs on MNIST trained for 5 epochs. FGSM and DeepFool both generate the attack images on the same
        source classifier. The test cast check if accuracy on adversarial samples increases
        after adversarially training the model.

        :return: None
        """
        (x_train, y_train), (x_test, y_test) = self.mnist

        # Get source and target classifiers
        classifier_tgt = self.classifier_k
        classifier_src = self.classifier_tf

        # Create FGSM and DeepFool attackers
        adv1 = FastGradientMethod(classifier_src)
        adv2 = DeepFool(classifier_src)
        x_adv = np.vstack((adv1.generate(x_test), adv2.generate(x_test)))
        y_adv = np.vstack((y_test, y_test))
        preds = classifier_tgt.predict(x_adv)
        acc = np.sum(np.argmax(preds, axis=1) == np.argmax(y_adv, axis=1)) / y_adv.shape[0]

        # Perform adversarial training
        adv_trainer = AdversarialTrainer(classifier_tgt, [adv1, adv2])
        params = {'nb_epochs': 2, 'batch_size': BATCH_SIZE}
        adv_trainer.fit(x_train, y_train, **params)

        # Evaluate that accuracy on adversarial sample has improved
        preds_adv_trained = adv_trainer.classifier.predict(x_adv)
        acc_adv_trained = np.sum(np.argmax(preds_adv_trained, axis=1) == np.argmax(y_adv, axis=1)) / y_adv.shape[0]
        print('\nAccuracy before adversarial training: %.2f%%' % (acc * 100))
        print('\nAccuracy after adversarial training: %.2f%%' % (acc_adv_trained * 100))

    def test_shared_model_mnist(self):
        """
        Test the adversarial trainer using one FGSM attacker. The source and target models of the attack are the same
        CNN on MNIST trained for 5 epochs. The test cast check if accuracy on adversarial samples increases
        after adversarially training the model.

        :return: None
        """
        (x_train, y_train), (x_test, y_test) = self.mnist

        # Create and fit classifier
        params = {'nb_epochs': 2, 'batch_size': BATCH_SIZE}
        classifier = self.classifier_k

        # Create FGSM attacker
        adv = FastGradientMethod(classifier)
        x_adv = adv.generate(x_test)
        preds = classifier.predict(x_adv)
        acc = np.sum(np.argmax(preds, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]

        # Perform adversarial training
        adv_trainer = AdversarialTrainer(classifier, adv)
        adv_trainer.fit(x_train, y_train, **params)

        # Evaluate that accuracy on adversarial sample has improved
        preds_adv_trained = adv_trainer.classifier.predict(x_adv)
        acc_adv_trained = np.sum(np.argmax(preds_adv_trained, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
        print('\nAccuracy before adversarial training: %.2f%%' % (acc * 100))
        print('\nAccuracy after adversarial training: %.2f%%' % (acc_adv_trained * 100))

    @staticmethod
    def _cnn_mnist_tf(input_shape):
        labels_tf = tf.placeholder(tf.float32, [None, 10])
        inputs_tf = tf.placeholder(tf.float32, [None] + list(input_shape))

        # Define the tensorflow graph
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

        classifier = TFClassifier((0, 1), inputs_tf, logits, loss=loss, train=train_tf, output_ph=labels_tf, sess=sess)
        return classifier

    @staticmethod
    def _cnn_mnist_k(input_shape):
        # Create simple CNN
        model = Sequential()
        model.add(Conv2D(4, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(10, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.01),
                      metrics=['accuracy'])

        classifier = KerasClassifier((0, 1), model, use_logits=False)
        return classifier
