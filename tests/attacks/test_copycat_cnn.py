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

import tensorflow as tf
import numpy as np

from art.attacks.extraction.copycat_cnn import CopycatCNN
from art.classifiers import TensorFlowClassifier
from art.utils import load_dataset, random_targets, master_seed
from art.utils_test import get_classifier_tf

logger = logging.getLogger(__name__)

BATCH_SIZE = 100
NB_TRAIN = 1000
NB_EPOCHS = 10
NB_STOLEN = 1000


class TestCopycatCNN(unittest.TestCase):
    """
    A unittest class for testing the CopycatCNN attack.
    """

    @classmethod
    def setUpClass(cls):
        (x_train, y_train), (_, _), _, _ = load_dataset('mnist')

        cls.x_train = x_train[:NB_TRAIN]
        cls.y_train = y_train[:NB_TRAIN]

    def setUp(self):
        master_seed(1234)

    def test_tfclassifier(self):
        """
        First test with the TensorFlowClassifier.
        :return:
        """
        # Build TensorFlowClassifiers
        victim_tfc, sess = get_classifier_tf()

        # Define input and output placeholders
        input_ph = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        output_ph = tf.placeholder(tf.int32, shape=[None, 10])

        # Define the tensorflow graph
        conv = tf.layers.conv2d(input_ph, 1, 7, activation=tf.nn.relu)
        conv = tf.layers.max_pooling2d(conv, 4, 4)
        flattened = tf.layers.flatten(conv)

        # Logits layer
        logits = tf.layers.dense(flattened, 10)

        # Train operator
        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=output_ph))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train = optimizer.minimize(loss)

        # TensorFlow session and initialization
        sess.run(tf.global_variables_initializer())

        # Create the classifier
        thieved_tfc = TensorFlowClassifier(clip_values=(0, 1), input_ph=input_ph, output=logits, labels_ph=output_ph,
                                           train=train, loss=loss, learning=None, sess=sess)

        # Create attack
        copycat_cnn = CopycatCNN(classifier=victim_tfc, batch_size=BATCH_SIZE, nb_epochs=NB_EPOCHS, nb_stolen=NB_STOLEN)
        thieved_tfc = copycat_cnn.generate(x=self.x_train, thieved_classifier=thieved_tfc)

        victim_preds = np.argmax(victim_tfc.predict(x=self.x_train[:100]), axis=1)
        thieved_preds = np.argmax(thieved_tfc.predict(x=self.x_train[:100]), axis=1)
        acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)

        self.assertGreater(acc, 0.5)

        # Clean-up session
        sess.close()



#     @unittest.skipIf(tf.__version__[0] == '2', reason='Skip unittests for Tensorflow v2 until Keras supports Tensorflow'
#                                                       ' v2 as backend.')
#     def test_krclassifier(self):
#         """
#         Second test with the KerasClassifier.
#         :return:
#         """
#
#     def test_ptclassifier(self):
#         """
#         Third test with the PyTorchClassifier.
#         :return:
#         """
#
#
#
# class TestCarliniL2Vectors(unittest.TestCase):
#     @classmethod
#     def setUpClass(cls):
#         (x_train, y_train), (x_test, y_test), _, _ = load_dataset('iris')
#
#         cls.x_train = x_train
#         cls.y_train = y_train
#         cls.x_test = x_test
#         cls.y_test = y_test
#
#     def setUp(self):
#         master_seed(1234)
#
#     def test_iris_tf(self):
#
#     def test_iris_pt(self):
#
#     def test_scikitlearn(self):


if __name__ == '__main__':
    unittest.main()
