# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
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

import numpy as np
import pytest
import tensorflow as tf

from art.data_generators import TensorFlowDataGenerator
from art.utils import Deprecated
from tests.classifiersFrameworks.utils import (
    backend_test_class_gradient,
    backend_test_fit_generator,
    backend_test_loss_gradient
)
from tests.utils import ExpectedValue

logger = logging.getLogger(__name__)


@pytest.mark.only_with_platform("tensorflow")
def test_fit_generator(is_tf_version_2, get_default_mnist_subset, get_image_classifier_list):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

    if not is_tf_version_2:
        classifier, sess = get_image_classifier_list(one_classifier=True)

        # Create TensorFlow data generator
        x_tensor = tf.convert_to_tensor(x_train_mnist.reshape(10, 100, 28, 28, 1))
        y_tensor = tf.convert_to_tensor(y_train_mnist.reshape(10, 100, 10))
        dataset = tf.data.Dataset.from_tensor_slices((x_tensor, y_tensor))

        iterator = dataset.make_initializable_iterator()
        data_gen = TensorFlowDataGenerator(
            sess=sess, iterator=iterator, iterator_type="initializable", iterator_arg={}, size=1000, batch_size=100
        )

        expected_values = {"post_fit_accuracy": ExpectedValue(0.65, 0.02)}

        backend_test_fit_generator(expected_values, classifier, data_gen, get_default_mnist_subset, nb_epochs=2)


@pytest.mark.only_with_platform("tensorflow")
def test_set_learning(is_tf_version_2, get_image_classifier_list):
    classifier, _ = get_image_classifier_list(one_classifier=True)
    if not is_tf_version_2:
        assert classifier._feed_dict == {}
        classifier.set_learning_phase(False)
        assert classifier._feed_dict[classifier._learning] is False
        classifier.set_learning_phase(True)
        assert classifier._feed_dict[classifier._learning]
        assert classifier.learning_phase


if __name__ == "__main__":
    pytest.cmdline.main("-q {} --mlFramework=tensorflow --durations=0".format(__file__).split(" "))
