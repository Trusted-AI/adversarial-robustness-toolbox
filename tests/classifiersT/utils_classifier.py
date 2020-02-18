import numpy as np
import pytest
import logging
from tests.utils_test import ExpectedValue

logger = logging.getLogger(__name__)

def backend_fit_generator(expected_values, data_gen, get_default_mnist_subset, classifier, nb_epochs):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

    true_class = np.argmax(y_test_mnist, axis=1)

    predictions = classifier.predict(x_test_mnist)
    prediction_class = np.argmax(predictions, axis=1)
    pre_fit_accuracy = np.sum(prediction_class == true_class) / x_test_mnist.shape[0]
    logger.info('Accuracy: %.2f%%', (pre_fit_accuracy * 100))

    if "pre_fit_accuracy" in expected_values:
        np.testing.assert_array_almost_equal(pre_fit_accuracy, expected_values["pre_fit_accuracy"].value, decimal=expected_values["pre_fit_accuracy"].decimals)

    classifier.fit_generator(generator=data_gen, nb_epochs=nb_epochs)
    predictions = classifier.predict(x_test_mnist)
    prediction_class = np.argmax(predictions, axis=1)
    post_fit_accuracy = np.sum(prediction_class == true_class) / x_test_mnist.shape[0]
    logger.info('Accuracy after fitting classifier with generator: %.2f%%', (post_fit_accuracy * 100))

    if "post_fit_accuracy" in expected_values:
        np.testing.assert_array_almost_equal(post_fit_accuracy, expected_values["post_fit_accuracy"].value, decimal=expected_values["post_fit_accuracy"].decimals)

