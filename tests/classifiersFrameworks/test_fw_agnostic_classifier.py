import logging
import numpy as np

from tests.classifiersFrameworks.utils import fw_agnostic_backend_test_nb_classes
from tests.classifiersFrameworks.utils import fw_agnostic_backend_test_input_shape

logger = logging.getLogger(__name__)


def test_predict(get_default_mnist_subset, get_image_classifier_list):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

    classifier, _ = get_image_classifier_list(one_classifier=True)

    if classifier is not None:
        y_predicted = classifier.predict(x_test_mnist[0:1])
        y_expected = np.asarray(
            [
                [
                    0.12109935,
                    0.0498215,
                    0.0993958,
                    0.06410097,
                    0.11366927,
                    0.04645343,
                    0.06419806,
                    0.30685693,
                    0.07616713,
                    0.05823758,
                ]
            ]
        )
        np.testing.assert_array_almost_equal(y_predicted, y_expected, decimal=4)


def test_nb_classes(get_image_classifier_list):
    fw_agnostic_backend_test_nb_classes(get_image_classifier_list)


def test_input_shape(framework, get_image_classifier_list):
    fw_agnostic_backend_test_input_shape(framework, get_image_classifier_list)
