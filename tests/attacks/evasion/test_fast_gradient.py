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


def test_no_norm_images(fix_get_mnist_subset, image_classifier_list):
    (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset
    classifier_list = image_classifier_list(FastGradientMethod)
    # TODO this if statement must be removed once we have a classifier for both image and tabular data
    if classifier_list is None:
        logging.warning("Couldn't perform  this test because no classifier is defined")
        return

    for classifier in classifier_list:
        attack = FastGradientMethod(classifier, eps=1.0, batch_size=11)
        x_test_adv = attack.generate(x_test_mnist)

        utils_test.assert_almost_equal_mean(x_test_mnist, x_test_adv, 0.2346725, decimal=0.002)
        utils_test.assert_almost_equal_min(x_test_mnist, x_test_adv, -1.0, decimal=0.00001)
        utils_test.assert_almost_equal_max(x_test_mnist, x_test_adv, 1.0, decimal=0.00001)

        y_test_pred = classifier.predict(x_test_adv)

        np.testing.assert_array_equal(np.argmax(y_test_mnist, axis=1), np.asarray([7, 2, 1, 0, 4, 1, 4, 9, 5, 9, 0]))

        y_test_pred_expected = np.asarray([[7.32060298e-02, 4.03153598e-02, 2.08138078e-01, 2.27986258e-02,
                                            4.08675969e-01, 1.64286494e-02, 8.81226882e-02, 2.71510370e-02,
                                            6.36906400e-02, 5.14728837e-02],
                                           [1.10022835e-01, 2.53075064e-04, 3.09050769e-01, 8.28748848e-03,
                                            4.23537999e-01, 1.58944018e-02, 3.54500744e-03, 7.03897625e-02,
                                            5.08272983e-02, 8.19133688e-03],
                                           [8.34077671e-02, 1.68634069e-04, 1.14863992e-01, 1.49999780e-03,
                                            7.81848907e-01, 2.06214096e-03, 1.57082418e-03, 7.90233351e-03,
                                            3.35145928e-03, 3.32383858e-03],
                                           [7.94695988e-02, 6.41014650e-02, 1.19662583e-01, 6.82745054e-02,
                                            5.87757975e-02, 5.54384440e-02, 4.47857119e-02, 4.73252147e-01,
                                            2.29432248e-02, 1.32965213e-02],
                                           [1.37778342e-01, 5.23229912e-02, 8.03085491e-02, 7.07063973e-02,
                                            1.13677077e-01, 7.50706568e-02, 4.73172851e-02, 3.50361735e-01,
                                            5.30573502e-02, 1.93995778e-02],
                                           [8.26486796e-02, 2.93200690e-04, 1.66191280e-01, 2.23751366e-03,
                                            7.05350637e-01, 8.26103613e-03, 3.88561003e-03, 1.66236982e-02,
                                            9.51580610e-03, 4.99255396e-03],
                                           [9.07047242e-02, 1.30164847e-01, 1.11855730e-01, 1.26194224e-01,
                                            9.42349583e-02, 7.18590096e-02, 7.08150640e-02, 2.04494953e-01,
                                            7.27845579e-02, 2.68919170e-02],
                                           [1.95148319e-01, 4.02570218e-02, 2.53095001e-01, 1.19175367e-01,
                                            7.29087070e-02, 6.70288056e-02, 3.26904431e-02, 1.72511339e-01,
                                            3.19005176e-02, 1.52844433e-02],
                                           [2.34931588e-01, 1.05211824e-01, 2.23802328e-01, 1.19600385e-01,
                                            4.32376936e-02, 4.33373451e-02, 5.49205467e-02, 1.05997942e-01,
                                            3.16798575e-02, 3.72803509e-02],
                                           [1.16207518e-01, 7.97201619e-02, 1.15341313e-01, 2.22322136e-01,
                                            6.16359413e-02, 1.39247745e-01, 5.34978770e-02, 1.17801160e-01,
                                            6.38158098e-02, 3.04102581e-02],
                                           [3.33993286e-01, 4.45333160e-02, 6.64125085e-02, 4.82672676e-02,
                                            4.61629629e-02, 7.41390288e-02, 2.49474458e-02, 3.12782317e-01,
                                            2.46306900e-02, 2.41311267e-02]])

        np.testing.assert_array_almost_equal(y_test_pred[0:3], y_test_pred_expected[0:3], decimal=2)


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
        # TODO Shouldn't the y_adv and y_expected labels be the same for the defence to be correct?
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

        x_test_adv_min = attack.generate(x_test_mnist)

        utils_test.assert_almost_equal_mean(x_test_mnist, x_test_adv_min, 0.03896513, decimal=0.01)
        utils_test.assert_almost_equal_min(x_test_mnist, x_test_adv_min, -0.30000000, decimal=0.00001)
        utils_test.assert_almost_equal_max(x_test_mnist, x_test_adv_min, 0.30000000, decimal=0.00001)

        y_test_pred = classifier.predict(x_test_adv_min)

        y_test_pred_expected = np.asarray([4, 2, 4, 7, 0, 4, 7, 2, 0, 7, 0])

        np.testing.assert_array_equal(np.argmax(y_test_pred, axis=1), y_test_pred_expected)


def test_l1_norm_images(fix_get_mnist_subset, image_classifier_list):
    (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset
    classifier_list = image_classifier_list(FastGradientMethod)
    # TODO this if statement must be removed once we have a classifier for both image and tabular data
    if classifier_list is None:
        logging.warning("Couldn't perform  this test because no classifier is defined")
        return

    for classifier in classifier_list:
        attack = FastGradientMethod(classifier, eps=1, norm=1, batch_size=128)
        x_test_adv = attack.generate(x_test_mnist)

        utils_test.assert_almost_equal_mean(x_test_mnist, x_test_adv, 0.00051375, decimal=0.002)
        utils_test.assert_almost_equal_min(x_test_mnist, x_test_adv, -0.01486498, decimal=0.001)
        utils_test.assert_almost_equal_max(x_test_mnist, x_test_adv, 0.014761963, decimal=0.001)

        y_test_pred = classifier.predict(x_test_adv[8:9])
        y_test_pred_expected = np.asarray([[0.17114946, 0.08205127, 0.07427921, 0.03722004, 0.28262928, 0.05035441,
                                            0.05271865, 0.12600125, 0.0811625, 0.0424339]])

        np.testing.assert_array_almost_equal(y_test_pred, y_test_pred_expected, decimal=4)




def test_l2_norm_images(fix_get_mnist_subset, image_classifier_list):

    classifier_list = image_classifier_list(FastGradientMethod)
    # TODO this if statement must be removed once we have a classifier for both image and tabular data
    if classifier_list is None:
        logging.warning("Couldn't perform  this test because no classifier is defined")
        return

    for classifier in classifier_list:
        attack = FastGradientMethod(classifier, eps=1, norm=2, batch_size=128)
        expected_values = {}

        expected_values["x_test_mean"] = ExpectedValue(0.007636424, 0.001)
        expected_values["x_test_min"] = ExpectedValue(-0.211054801, 0.001)
        expected_values["x_test_max"] = ExpectedValue(0.209592223, 0.001)
        expected_values["y_test_pred_adv_expected"] = ExpectedValue(
        np.asarray([[0.19395831, 0.11625732, 0.08293699, 0.04129186, 0.17826456, 0.06290703,
                     0.06270657, 0.14066935, 0.07419015, 0.04681788]]), 2)
        utils_attack._backend_norm_images(attack, classifier, fix_get_mnist_subset, expected_values)


@pytest.mark.mlFramework("scikitlearn") #temporarily skipping for scikitlearn until find bug fix in bounded test
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
