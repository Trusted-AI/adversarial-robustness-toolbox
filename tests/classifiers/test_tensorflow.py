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
import pytest
# import os
# import shutil
import logging
import unittest
# import pickle

import tensorflow as tf
import numpy as np

# from art.config import ART_DATA_PATH
from art.data_generators import TFDataGenerator
from tests import utils_test

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def get_is_tf_version_2():
    if tf.__version__[0] == '2':
        yield True
    else:
        yield False

@pytest.fixture(scope="module")
def get_classifier():
    classifier, sess = utils_test.get_image_classifier_tf()
    yield classifier, sess


@pytest.fixture(scope="module")
def get_classifier_logits():
    classifier_logits, _ = utils_test.get_image_classifier_tf(from_logits=True)
    yield classifier_logits


@pytest.mark.only_with_platform("tensorflow")
def test_predict(get_classifier, get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    classifier, sess = get_classifier
    y_predicted = classifier.predict(x_test_mnist[0:1])
    y_expected = np.asarray([[0.12109935, 0.0498215, 0.0993958, 0.06410097, 0.11366927, 0.04645343, 0.06419806,
                              0.30685693, 0.07616713, 0.05823758]])
    np.testing.assert_array_almost_equal(y_predicted, y_expected, decimal=4)

@pytest.mark.only_with_platform("tensorflow")
def test_fit_generator(get_is_tf_version_2, get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset

    if not get_is_tf_version_2:
        classifier, sess = utils_test.get_image_classifier_tf()

        # Create TensorFlow data generator
        x_tensor = tf.convert_to_tensor(x_train_mnist.reshape(10, 100, 28, 28, 1))
        y_tensor = tf.convert_to_tensor(y_train_mnist.reshape(10, 100, 10))
        dataset = tf.data.Dataset.from_tensor_slices((x_tensor, y_tensor))
        iterator = dataset.make_initializable_iterator()
        data_gen = TFDataGenerator(sess=sess, iterator=iterator, iterator_type='initializable', iterator_arg={},
                                   size=1000, batch_size=100)

        # Test fit and predict
        classifier.fit_generator(data_gen, nb_epochs=2)
        predictions = classifier.predict(x_test_mnist)
        predictions_class = np.argmax(predictions, axis=1)
        true_class = np.argmax(y_test_mnist, axis=1)
        accuracy = np.sum(predictions_class == true_class) / len(true_class)

        logger.info('Accuracy after fitting TensorFlow classifier with generator: %.2f%%', (accuracy * 100))
        np.testing.assert_array_almost_equal(accuracy, 0.65, decimal=0.02)

@pytest.mark.only_with_platform("tensorflow")
def test_nb_classes(get_classifier):
   classifier, sess = get_classifier
   assert classifier.nb_classes() == 10

@pytest.mark.only_with_platform("tensorflow")
def test_input_shape(get_classifier):
   classifier, sess = get_classifier
   assert classifier.input_shape == (28, 28, 1)