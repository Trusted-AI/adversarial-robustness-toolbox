import pytest
import numpy as np
import logging
import unittest
import keras.backend as k
from art.attacks import BoundaryAttack
from art.classifiers import KerasClassifier
from art.utils import random_targets
from art import utils
from tests import utils_test_attributes
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


def test_targeted_images(fix_get_mnist_subset, image_classifier_list, fix_mlFramework):

    if image_classifier_list is None:
        logging.warning("Couldn't perform  this test because no classifier is defined")
        return

    (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset

    for classifier in image_classifier_list:

        # First targeted attack
        attack = BoundaryAttack(classifier=classifier, targeted=True, max_iter=20)
        params = {'y': random_targets(y_test_mnist, classifier.nb_classes())}
        x_test_adv = attack.generate(x_test_mnist, **params)

        utils_test.check_adverse_example_x(x_test_adv, x_test_mnist)

        target = np.argmax(params['y'], axis=1)
        y_pred_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        assert (target == y_pred_adv).any()

        if fix_mlFramework in ["keras"]:
            k.clear_session()


def test_targeted_tabular(fix_get_iris, clipped_tabular_classifier_list, fix_mlFramework):
    for classifier in clipped_tabular_classifier_list:
        attack = BoundaryAttack(classifier, targeted=True, max_iter=10)
        utils_test_attributes._backend_targeted_tabular(attack, classifier, fix_get_iris, fix_mlFramework)



def test_untargeted_clipped_tabular(fix_get_iris, clipped_tabular_classifier_list, fix_mlFramework):

    if clipped_tabular_classifier_list is None:
        logging.warning("Couldn't perform  this test because no classifier is defined")
        return

    (x_train_iris, y_train_iris), (x_test_iris, y_test_iris) = fix_get_iris

    for classifier in clipped_tabular_classifier_list:

        if fix_mlFramework in ["scikitlearn"]:
            classifier.fit(x=x_test_iris, y=y_test_iris)

        attack = BoundaryAttack(classifier, targeted=False, max_iter=10)

        x_test_adv = attack.generate(x_test_iris.astype(np.float32))
        utils_test.check_adverse_example_x(x_test_adv, x_test_iris)

        y_pred = np.argmax(y_test_iris, axis=1)
        y_pred_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        assert(y_pred == y_pred_adv).all() == False
        accuracy = np.sum(y_pred_adv == y_pred) / y_test_iris.shape[0]
        logger.info('Accuracy on Iris with boundary adversarial examples: %.2f%%', (accuracy * 100))

def test_untargeted_unclipped_tabular(fix_get_iris, clipped_tabular_classifier_list, fix_mlFramework):
    if clipped_tabular_classifier_list is None:
        logging.warning("Couldn't perform  this test because no classifier is defined")
        return

    (x_train_iris, y_train_iris), (x_test_iris, y_test_iris) = fix_get_iris
    for classifier in clipped_tabular_classifier_list:

        if fix_mlFramework in ["scikitlearn"]:
            classifier.fit(x=x_test_iris, y=y_test_iris)

        # Recreate a classifier without clip values
        # classifier = KerasClassifier(model=classifier._model, use_logits=False, channel_index=1)
        attack = BoundaryAttack(classifier, targeted=False, max_iter=10)
        x_test_adv = attack.generate(x_test_iris)
        assert (x_test_iris == x_test_adv).all() == False

        preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        assert (np.argmax(y_test_iris, axis=1) == preds_adv).all() == False
        accuracy = np.sum(preds_adv == np.argmax(y_test_iris, axis=1)) / y_test_iris.shape[0]
        logger.info('Accuracy on Iris with boundary adversarial examples: %.2f%%', (accuracy * 100))

def test_untargeted_images(fix_get_mnist_subset, image_classifier_list, fix_mlFramework):
    if image_classifier_list is None:
        logging.warning("Couldn't perform  this test because no classifier is defined")
        return

    (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset

    for classifier in image_classifier_list:
        attack = BoundaryAttack(classifier=classifier, targeted=False, max_iter=20)

        x_test_adv = attack.generate(x_test_mnist)
        utils_test.check_adverse_example_x(x_test_adv, x_test_mnist)

        y_pred = np.argmax(classifier.predict(x_test_mnist), axis=1)
        y_pred_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
        assert (y_pred != y_pred_adv).any()

        if fix_mlFramework in ["keras"]:
            k.clear_session()


def test_classifier_type_check_fail_classifier():
    # Use a useless test classifier to test basic classifier properties
    class ClassifierNoAPI:
        pass

    classifier = ClassifierNoAPI

    with pytest.raises(utils.WrongClassifer) as exception:
        _ = BoundaryAttack(classifier=classifier)

    assert exception.value.class_expected == Classifier
