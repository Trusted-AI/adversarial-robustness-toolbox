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
import logging
import numpy as np

logger = logging.getLogger(__name__)


def backend_test_fit_generator(expected_values, classifier, data_gen, get_default_mnist_subset, nb_epochs):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

    true_class = np.argmax(y_test_mnist, axis=1)

    predictions = classifier.predict(x_test_mnist)
    prediction_class = np.argmax(predictions, axis=1)
    pre_fit_accuracy = np.sum(prediction_class == true_class) / x_test_mnist.shape[0]
    logger.info("Accuracy: %.2f%%", (pre_fit_accuracy * 100))

    if "pre_fit_accuracy" in expected_values:
        np.testing.assert_array_almost_equal(
            pre_fit_accuracy,
            expected_values["pre_fit_accuracy"].value,
            decimal=expected_values["pre_fit_accuracy"].decimals,
        )

    classifier.fit_generator(generator=data_gen, nb_epochs=nb_epochs)
    predictions = classifier.predict(x_test_mnist)
    prediction_class = np.argmax(predictions, axis=1)
    post_fit_accuracy = np.sum(prediction_class == true_class) / x_test_mnist.shape[0]
    logger.info("Accuracy after fitting classifier with generator: %.2f%%", (post_fit_accuracy * 100))

    if "post_fit_accuracy" in expected_values:
        np.testing.assert_array_almost_equal(
            post_fit_accuracy,
            expected_values["post_fit_accuracy"].value,
            decimal=expected_values["post_fit_accuracy"].decimals,
        )
