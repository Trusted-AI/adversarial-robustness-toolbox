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
import pytest
import numpy as np
import logging

import keras.backend as k
from sklearn.tree import DecisionTreeClassifier

from art.utils import random_targets, get_labels_np_array
from art.exceptions import ClassifierError
from art.classifiers.classifier import ClassifierNeuralNetwork, ClassifierGradients, Classifier
from art.classifiers.scikitlearn import ScikitlearnDecisionTreeClassifier

from tests.utils import check_adverse_example_x, check_adverse_predicted_sample_y

logger = logging.getLogger(__name__)


def backend_targeted_images(attack, fix_get_mnist_subset):
    (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset
    targets = random_targets(y_test_mnist, attack.classifier.nb_classes())
    x_test_adv = attack.generate(x_test_mnist, y=targets)
    assert bool((x_test_mnist == x_test_adv).all()) is False

    y_test_pred_adv = get_labels_np_array(attack.classifier.predict(x_test_adv))

    assert targets.shape == y_test_pred_adv.shape
    assert (targets == y_test_pred_adv).sum() >= (x_test_mnist.shape[0] // 2)

    check_adverse_example_x(x_test_adv, x_test_mnist)

    y_pred_adv = np.argmax(attack.classifier.predict(x_test_adv), axis=1)

    target = np.argmax(targets, axis=1)
    assert (target == y_pred_adv).any()


def backend_test_defended_images(attack, mnist_dataset):
    (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = mnist_dataset
    x_train_adv = attack.generate(x_train_mnist)

    check_adverse_example_x(x_train_adv, x_train_mnist)

    y_train_pred_adv = get_labels_np_array(attack.classifier.predict(x_train_adv))
    y_train_labels = get_labels_np_array(y_train_mnist)

    check_adverse_predicted_sample_y(y_train_pred_adv, y_train_labels)

    x_test_adv = attack.generate(x_test_mnist)
    check_adverse_example_x(x_test_adv, x_test_mnist)

    y_test_pred_adv = get_labels_np_array(attack.classifier.predict(x_test_adv))
    check_adverse_predicted_sample_y(y_test_pred_adv, y_test_mnist)


def backend_test_random_initialisation_images(attack, mnist_dataset):
    (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = mnist_dataset
    x_test_adv = attack.generate(x_test_mnist)
    assert bool((x_test_mnist == x_test_adv).all()) is False


def backend_check_adverse_values(attack, mnist_dataset, expected_values):
    (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = mnist_dataset
    x_test_adv = attack.generate(x_test_mnist)
    y_test_pred_adv_matrix = attack.classifier.predict(x_test_adv)
    y_test_pred_adv = np.argmax(y_test_pred_adv_matrix, axis=1)

    if "x_test_mean" in expected_values:
        np.testing.assert_array_almost_equal(
            float(np.mean(x_test_adv - x_test_mnist)),
            expected_values["x_test_mean"].value,
            decimal=expected_values["x_test_mean"].decimals,
        )
    if "x_test_min" in expected_values:
        # utils_test.assert_almost_equal_min(x_test_mnist, x_test_adv,
        # expected_values["x_test_min"].value, decimal=expected_values["x_test_min"].decimals)
        np.testing.assert_array_almost_equal(
            float(np.min(x_test_adv - x_test_mnist)),
            expected_values["x_test_min"].value,
            decimal=expected_values["x_test_min"].decimals,
        )
    if "x_test_max" in expected_values:
        np.testing.assert_array_almost_equal(
            float(np.max(x_test_adv - x_test_mnist)),
            expected_values["x_test_max"].value,
            decimal=expected_values["x_test_max"].decimals,
        )
    if "y_test_pred_adv_expected_matrix" in expected_values:
        np.testing.assert_array_almost_equal(
            y_test_pred_adv_matrix,
            expected_values["y_test_pred_adv_expected_matrix"].value,
            decimal=expected_values["y_test_pred_adv_expected"].decimals,
        )
    if "y_test_pred_adv_expected" in expected_values:
        np.testing.assert_array_equal(y_test_pred_adv, expected_values["y_test_pred_adv_expected"].value)


def backend_test_classifier_type_check_fail(attack, classifier_expected_list=[], classifier=None):
    # Use a useless test classifier to test basic classifier properties
    class ClassifierNoAPI:
        pass

    noAPIClassifier = ClassifierNoAPI
    _backend_test_classifier_list_type_check_fail(attack, noAPIClassifier, [Classifier])

    if len(classifier_expected_list) > 0:
        # Testing additional types of classifiers expected
        if classifier is None:
            if ClassifierGradients in classifier_expected_list or ClassifierNeuralNetwork in classifier_expected_list:
                # Use a test classifier not providing gradients required by white-box attack
                classifier = ScikitlearnDecisionTreeClassifier(model=DecisionTreeClassifier())
            else:
                raise Exception(
                    "a test classifier must be provided if classifiers other than "
                    "ClassifierGradients and ClassifierNeuralNetwork are expected"
                )

        _backend_test_classifier_list_type_check_fail(attack, classifier, classifier_expected_list)


def _backend_test_classifier_list_type_check_fail(attack, classifier, classifier_expected_list):
    with pytest.raises(ClassifierError) as exception:
        _ = attack(classifier=classifier)

    for classifier_expected in classifier_expected_list:
        assert classifier_expected in exception.value.class_expected_list


def backend_targeted_tabular(attack, fix_get_iris):
    (x_train_iris, y_train_iris), (x_test_iris, y_test_iris) = fix_get_iris

    targets = random_targets(y_test_iris, nb_classes=3)
    x_test_adv = attack.generate(x_test_iris, **{"y": targets})

    check_adverse_example_x(x_test_adv, x_test_iris)

    y_pred_adv = np.argmax(attack.classifier.predict(x_test_adv), axis=1)
    target = np.argmax(targets, axis=1)
    assert (target == y_pred_adv).any()

    accuracy = np.sum(y_pred_adv == target) / y_test_iris.shape[0]
    logger.info("Success rate of targeted boundary on Iris: %.2f%%", (accuracy * 100))


def back_end_untargeted_images(attack, fix_get_mnist_subset, fix_mlFramework):
    (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset

    x_test_adv = attack.generate(x_test_mnist)

    check_adverse_example_x(x_test_adv, x_test_mnist)

    y_pred = np.argmax(attack.classifier.predict(x_test_mnist), axis=1)
    y_pred_adv = np.argmax(attack.classifier.predict(x_test_adv), axis=1)
    assert (y_pred != y_pred_adv).any()

    if fix_mlFramework in ["keras"]:
        k.clear_session()


def backend_untargeted_tabular(attack, iris_dataset, clipped):
    (x_train_iris, y_train_iris), (x_test_iris, y_test_iris) = iris_dataset

    x_test_adv = attack.generate(x_test_iris)

    # TODO remove that platform specific case
    # if mlFramework in ["scikitlearn"]:
    #     np.testing.assert_array_almost_equal(np.abs(x_test_adv - x_test_iris), .1, decimal=5)

    check_adverse_example_x(x_test_adv, x_test_iris)
    # utils_test.check_adverse_example_x(x_test_adv, x_test_iris, bounded=clipped)

    y_pred_test_adv = np.argmax(attack.classifier.predict(x_test_adv), axis=1)
    y_test_true = np.argmax(y_test_iris, axis=1)

    # assert (y_test_true == y_pred_test_adv).any(), "An untargeted attack should have changed SOME predictions"
    assert (
        bool((y_test_true == y_pred_test_adv).all()) is False
    ), "An untargeted attack should NOT have changed all predictions"

    accuracy = np.sum(y_pred_test_adv == y_test_true) / y_test_true.shape[0]
    logger.info(
        "Accuracy of " + attack.classifier.__class__.__name__ + " on Iris with FGM adversarial examples: " "%.2f%%",
        (accuracy * 100),
    )
