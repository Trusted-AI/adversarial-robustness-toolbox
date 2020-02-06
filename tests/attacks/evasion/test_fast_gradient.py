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
import numpy as np
from art.attacks import FastGradientMethod
from tests import utils_test
from tests.utils_test import ExpectedValue
from tests import utils_attack
import pytest
from art import utils
from art.classifiers.classifier import Classifier, ClassifierGradients
from art.classifiers.scikitlearn import ScikitlearnDecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)


@pytest.fixture()
def fix_get_mnist_subset(fix_get_mnist):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = fix_get_mnist
    n_train = 100
    n_test = 11
    yield (x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test])


def test_classifier_defended_images(fix_get_mnist_subset, image_classifier_list):
    (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset
    classifier_list = image_classifier_list(FastGradientMethod, defended=True)
    # TODO this if statement must be removed once we have a classifier for both image and tabular data
    if classifier_list is None:
        logging.warning("Couldn't perform  this test because no classifier is defined")
        return

    for classifier in classifier_list:
        attack = FastGradientMethod(classifier, eps=1, batch_size=128)
        x_train_adv = attack.generate(x_train_mnist)

        utils_test.check_adverse_example_x(x_train_adv, x_train_mnist)

        y_train_pred_adv = utils.get_labels_np_array(classifier.predict(x_train_adv))
        y_train_labels = utils.get_labels_np_array(y_train_mnist)

        utils_test.check_adverse_predicted_sample_y(y_train_pred_adv, y_train_labels)

        x_test_adv = attack.generate(x_test_mnist)
        utils_test.check_adverse_example_x(x_test_adv, x_test_mnist)

        y_test_pred_adv = utils.get_labels_np_array(classifier.predict(x_test_adv))
        utils_test.check_adverse_predicted_sample_y(y_test_pred_adv, y_test_mnist)


def test_random_initialisation_images(fix_get_mnist_subset, image_classifier_list):
    (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset
    classifier_list = image_classifier_list(FastGradientMethod)
    # TODO this if statement must be removed once we have a classifier for both image and tabular data
    if classifier_list is None:
        logging.warning("Couldn't perform  this test because no classifier is defined")
        return

    for classifier in classifier_list:
        attack = FastGradientMethod(classifier, num_random_init=3)
        x_test_adv = attack.generate(x_test_mnist)
        assert (x_test_mnist == x_test_adv).all() == False


def test_targeted_images(fix_get_mnist_subset, image_classifier_list):
    classifier_list = image_classifier_list(FastGradientMethod)
    # TODO this if statement must be removed once we have a classifier for both image and tabular data
    if classifier_list is None:
        logging.warning("Couldn't perform  this test because no classifier is defined")
        return

    for classifier in classifier_list:
        attack = FastGradientMethod(classifier, eps=1.0, targeted=True)
        attack_params = {"minimal": True, "eps_step": 0.01, "eps": 1.0}
        attack.set_params(**attack_params)

        utils_attack._backend_targeted_images(attack, classifier, fix_get_mnist_subset)


def test_minimal_perturbations_images(fix_get_mnist_subset, image_classifier_list):
    (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset
    classifier_list = image_classifier_list(FastGradientMethod)
    # TODO this if statement must be removed once we have a classifier for both image and tabular data
    if classifier_list is None:
        logging.warning("Couldn't perform  this test because no classifier is defined")
        return

    for classifier in classifier_list:
        attack = FastGradientMethod(classifier, eps=1.0, batch_size=11)
        attack_params = {"minimal": True, "eps_step": 0.1, "eps": 5.0}
        attack.set_params(**attack_params)

        x_test_adv = attack.generate(x_test_mnist)

        utils_test.assert_almost_equal_mean(x_test_mnist, x_test_adv, 0.03896513, decimal=0.01)
        utils_test.assert_almost_equal_min(x_test_mnist, x_test_adv, -0.30000000, decimal=0.00001)
        utils_test.assert_almost_equal_max(x_test_mnist, x_test_adv, 0.30000000, decimal=0.00001)

        y_test_pred = classifier.predict(x_test_adv)
        tmp = np.argmax(y_test_pred, axis=1)
        y_test_pred_expected = np.asarray([4, 2, 4, 7, 0, 4, 7, 2, 0, 7, 0])

        np.testing.assert_array_equal(tmp, y_test_pred_expected)


@pytest.mark.parametrize("norm", [np.inf, 1, 2])
@pytest.mark.mlFramework("pytorch")  # temporarily skipping for pytorch until find bug fix in bounded test
def test_norm_images(norm, fix_get_mnist_subset, image_classifier_list):
    classifier_list = image_classifier_list(FastGradientMethod)
    # TODO this if statement must be removed once we have a classifier for both image and tabular data
    if classifier_list is None:
        logging.warning("Couldn't perform  this test because no classifier is defined")
        return

    elif norm == np.inf:
        expected_values = {"x_test_mean": ExpectedValue(0.2346725, 0.002),
                           "x_test_min": ExpectedValue(-1.0, 0.00001),
                           "x_test_max": ExpectedValue(1.0, 0.00001),
                           "y_test_pred_adv_expected": ExpectedValue(np.asarray([[0.10271172, 0.08795895, 0.44324583,
                                                                                  0.11125648, 0.0380144, 0.02946785,
                                                                                  0.03945549, 0.11169701, 0.01919523,
                                                                                  0.01699707]]), 2)}

        # [[0.23493163 0.10521185 0.2238024  0.11960039 0.0432377  0.04333736
        #   0.05492055 0.10599796 0.03167986 0.03728036]]


    elif norm == 1:
        expected_values = {"x_test_mean": ExpectedValue(0.00051375, 0.002),
                           "x_test_min": ExpectedValue(-0.01486498, 0.001),
                           "x_test_max": ExpectedValue(0.014761963, 0.001),
                           "y_test_pred_adv_expected": ExpectedValue(
                               np.asarray([[0.17114946, 0.08205127, 0.07427921, 0.03722004, 0.28262928, 0.05035441,
                                            0.05271865, 0.12600125, 0.0811625, 0.0424339]]), 4)}
    elif norm == 2:
        expected_values = {"x_test_mean": ExpectedValue(0.007636424, 0.001),
                           "x_test_min": ExpectedValue(-0.211054801, 0.001),
                           "x_test_max": ExpectedValue(0.209592223, 0.001),
                           "y_test_pred_adv_expected": ExpectedValue(
                               np.asarray([[0.19395831, 0.11625732, 0.08293699, 0.04129186, 0.17826456, 0.06290703,
                                            0.06270657, 0.14066935, 0.07419015, 0.04681788]]), 2)}

    for classifier in classifier_list:
        attack = FastGradientMethod(classifier, eps=1, norm=norm, batch_size=128)

        utils_attack._backend_norm_images(attack, classifier, fix_get_mnist_subset, expected_values)


@pytest.mark.mlFramework("scikitlearn")  # temporarily skipping for scikitlearn until find bug fix in bounded test
@pytest.mark.parametrize("targeted, clipped", [(True, True), (True, False), (False, True), (False, False)])
def test_tabular(tabular_classifier_list, fix_mlFramework, fix_get_iris, targeted, clipped):
    (x_train_iris, y_train_iris), (x_test_iris, y_test_iris) = fix_get_iris
    classifier_list = tabular_classifier_list(FastGradientMethod, clipped=clipped)
    if classifier_list is None:
        logging.warning("Couldn't perform  this test because no classifier is defined")
        return
    for classifier in classifier_list:
        if fix_mlFramework in ["scikitlearn"]:
            classifier.fit(x=x_test_iris, y=y_test_iris)
        if targeted:
            attack = FastGradientMethod(classifier, targeted=True, eps=.1, batch_size=128)
            utils_attack._backend_targeted_tabular(attack, classifier, fix_get_iris, fix_mlFramework)
        else:
            attack = FastGradientMethod(classifier, eps=.1)
            utils_attack._backend_untargeted_tabular(attack, fix_get_iris, classifier, fix_mlFramework,
                                                     clipped=clipped)


def test_classifier_type_check_fail_gradients():
    # Use a test classifier not providing gradients required by white-box attack
    classifier = ScikitlearnDecisionTreeClassifier(model=DecisionTreeClassifier())
    with pytest.raises(utils.WrongClassifer) as exception:
        _ = FastGradientMethod(classifier=classifier)

    assert exception.value.class_expected == ClassifierGradients


def test_classifier_type_check_fail_classifier():
    # Use a useless test classifier to test basic classifier properties
    class ClassifierNoAPI:
        pass

    classifier = ClassifierNoAPI

    with pytest.raises(utils.WrongClassifer) as exception:
        _ = FastGradientMethod(classifier=classifier)

    assert exception.value.class_expected == Classifier


if __name__ == '__main__':
    unittest.main()
