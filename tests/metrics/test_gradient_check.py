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
import unittest

import numpy as np

from art.metrics.gradient_check import loss_gradient_check
from art.utils import load_mnist

from tests.utils import TestBase, master_seed, get_image_classifier_kr

logger = logging.getLogger(__name__)

BATCH_SIZE = 10
NB_TRAIN = 100
NB_TEST = 100


class TestGradientCheck(TestBase):
    def setUp(self):
        master_seed(seed=42)
        super().setUp()

    def test_loss_gradient_check(self):
        (x_train, y_train), (x_test, y_test), _, _ = load_mnist()
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        x_test, y_test = x_test[:NB_TEST], y_test[:NB_TEST]

        # Get classifier and train like normal
        classifier = get_image_classifier_kr()
        classifier.fit(x_train, y_train, batch_size=BATCH_SIZE, nb_epochs=2)

        # Now check that the function detects bad gradients
        # Set the weights of the convolution layer to zero
        weights = classifier._model.layers[0].get_weights()
        new_weights = [np.zeros(w.shape) for w in weights]
        classifier._model.layers[0].set_weights(new_weights)
        is_bad = loss_gradient_check(classifier, x_test, y_test, verbose=False)

        self.assertTrue(np.all(np.any(is_bad, 1)))

        # Set the weights of the convolution layer to nan
        weights = classifier._model.layers[0].get_weights()
        new_weights = [np.empty(w.shape) for w in weights]
        for i in range(len(new_weights)):
            new_weights[i][:] = np.nan
        classifier._model.layers[0].set_weights(new_weights)
        is_bad = loss_gradient_check(classifier, x_test, y_test, verbose=False)

        self.assertTrue(np.all(np.any(is_bad, 1)))


if __name__ == "__main__":
    unittest.main()
