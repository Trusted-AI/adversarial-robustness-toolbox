import pytest
import numpy as np
import logging
import unittest
import keras.backend as k
from art.attacks import BoundaryAttack
from art.classifiers import KerasClassifier
from art.utils import random_targets
from art import utils
from tests import utils_attack
from art.classifiers.classifier import Classifier, ClassifierGradients
from tests import utils_test
from tests.utils_test import TestBase
from tests.utils_test import get_image_classifier_tf, get_image_classifier_kr, get_image_classifier_pt
from tests.utils_test import get_tabular_classifier_tf, get_tabular_classifier_kr, get_tabular_classifier_pt
logger = logging.getLogger(__name__)

@pytest.fixture()
def fix_get_mnist_subset(fix_get_mnist):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = fix_get_mnist
    n_train = 10
    n_test = 10
    yield (x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test])


@pytest.mark.parametrize("clipped_classifier, targeted", [(True, True), (True, False), (False, True), (False, False)])
def test_tabular(tabular_classifier_list, fix_mlFramework, fix_get_iris, clipped_classifier, targeted):

    classifier_list = tabular_classifier_list(clipped=clipped_classifier)
    if classifier_list is None:
        logging.warning("Couldn't perform  this test because no classifier is defined")
        return

    for classifier in classifier_list:
        if BoundaryAttack.is_valid_classifier_type(classifier) is False:
            continue

        attack = BoundaryAttack(classifier, targeted=targeted, max_iter=10)
        if targeted:
            utils_attack._backend_targeted_tabular(attack,  classifier, fix_get_iris, fix_mlFramework)
        else:
            utils_attack._backend_untargeted_tabular(attack, fix_get_iris, classifier, fix_mlFramework,
                                                     clipped=clipped_classifier)

@pytest.mark.parametrize("targeted", [(True), (False)])
def test_images(fix_get_mnist_subset, image_classifier_list, fix_mlFramework, targeted):

    if image_classifier_list is None:
        logging.warning("Couldn't perform  this test because no classifier is defined")
        return

    (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset

    for classifier in image_classifier_list:
        if BoundaryAttack.is_valid_classifier_type(classifier) is False:
            continue

        attack = BoundaryAttack(classifier=classifier, targeted=targeted, max_iter=20)
        if targeted:
            targets = random_targets(y_test_mnist, classifier.nb_classes())
            utils_attack._backend_targeted_images(attack, targets, classifier, fix_get_mnist_subset)
        else:
            utils_attack._back_end_untargeted_images(attack, classifier, fix_get_mnist_subset, fix_mlFramework)


def test_classifier_type_check_fail_classifier():
    # Use a useless test classifier to test basic classifier properties
    class ClassifierNoAPI:
        pass

    classifier = ClassifierNoAPI

    with pytest.raises(utils.WrongClassifer) as exception:
        _ = BoundaryAttack(classifier=classifier)

    assert exception.value.class_expected == Classifier
