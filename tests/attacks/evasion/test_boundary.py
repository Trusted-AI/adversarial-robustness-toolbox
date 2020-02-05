import pytest
import numpy as np
import logging
import unittest
import keras.backend as k
from art.attacks import BoundaryAttack
from art.classifiers import KerasClassifier
from art.utils import random_targets

from tests import utils_test
from tests.utils_test import TestBase
from tests.utils_test import get_image_classifier_tf, get_image_classifier_kr, get_image_classifier_pt
from tests.utils_test import get_tabular_classifier_tf, get_tabular_classifier_kr, get_tabular_classifier_pt

@pytest.fixture()
def fix_get_mnist_subset(fix_get_mnist):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = fix_get_mnist
    n_train = 10
    n_test = 10
    yield (x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test])

def test_targeted_images(fix_get_mnist_subset, image_classifier_list):
    """
    Second test with the KerasClassifier.
    :return:
    """
    (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset

    for classifier in image_classifier_list:

        # First targeted attack
        boundary = BoundaryAttack(classifier=classifier, targeted=True, max_iter=20)
        params = {'y': random_targets(y_test_mnist, classifier.nb_classes())}
        x_test_adv = boundary.generate(x_test_mnist, **params)

        utils_test.check_adverse_example(x_test_adv, x_test_mnist)

        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        assert (target == y_pred_adv).any()

        # Clean-up session
        k.clear_session()

def test_untargeted_images(fix_get_mnist_subset, image_classifier_list):
    """
    Second test with the KerasClassifier.
    :return:
    """
    (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset

    for classifier in image_classifier_list:
        boundary = BoundaryAttack(classifier=classifier, targeted=False, max_iter=20)
        x_test_adv = boundary.generate(x_test_mnist)

        utils_test.check_adverse_example(x_test_adv, x_test_mnist)

        y_pred = np.argmax(classifier.predict(x_test_mnist), axis=1)
        y_pred_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        assert (y_pred != y_pred_adv).any()

        # Clean-up session
        k.clear_session()