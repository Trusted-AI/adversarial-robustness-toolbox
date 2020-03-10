import pytest
import logging
import numpy as np
from art.attacks import BoundaryAttack
import unittest
from tests.attacks.utils import backend_targeted_tabular, backend_untargeted_tabular, backend_targeted_images
from tests.attacks.utils import back_end_untargeted_images, backend_test_classifier_type_check_fail

from art.utils import random_targets

from tests.utils import TestBase
from tests.utils import get_image_classifier_tf, get_image_classifier_kr, get_image_classifier_pt
from tests.utils import get_tabular_classifier_tf, get_tabular_classifier_kr, get_tabular_classifier_pt

logger = logging.getLogger(__name__)


@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 10
    n_test = 10
    yield (x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test])


@pytest.mark.parametrize("clipped_classifier, targeted", [(True, True), (True, False), (False, True), (False, False)])
def test_tabular(get_tabular_classifier_list, framework, get_iris_dataset, clipped_classifier, targeted):
    classifier_list = get_tabular_classifier_list(BoundaryAttack, clipped=clipped_classifier)
    if classifier_list is None:
        logging.warning("Couldn't perform  this test because no classifier is defined")
        return

    for classifier in classifier_list:

        attack = BoundaryAttack(classifier, targeted=targeted, max_iter=10)
        if targeted:
            backend_targeted_tabular(attack, get_iris_dataset)
        else:
            backend_untargeted_tabular(attack, get_iris_dataset, clipped=clipped_classifier)


def test_tensorflow_iris(get_iris_dataset):
    classifier, _ = get_tabular_classifier_tf()
    (x_train_iris, y_train_iris), (x_test_iris, y_test_iris) = get_iris_dataset
    # Test untargeted attack
    attack = BoundaryAttack(classifier, targeted=False, max_iter=3)
    x_test_adv = attack.generate(x_test_iris)
    unittest.TestCase.assertFalse((x_test_iris == x_test_adv).all())
    unittest.TestCase.assertTrue((x_test_adv <= 1).all())
    unittest.TestCase.assertTrue((x_test_adv >= 0).all())

    preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
    unittest.TestCase.assertFalse((np.argmax(y_test_iris, axis=1) == preds_adv).all())
    accuracy = np.sum(preds_adv == np.argmax(y_test_iris, axis=1)) / y_test_iris.shape[0]
    logger.info('Accuracy on Iris with boundary adversarial examples: %.2f%%', (accuracy * 100))

    # Test targeted attack
    targets = random_targets(y_test_iris, nb_classes=3)
    attack = BoundaryAttack(classifier, targeted=True, max_iter=3)
    x_test_adv = attack.generate(x_test_iris, **{'y': targets})
    unittest.TestCase.assertFalse((x_test_iris == x_test_adv).all())
    unittest.TestCase.assertTrue((x_test_adv <= 1).all())
    unittest.TestCase.assertTrue((x_test_adv >= 0).all())

    preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
    unittest.TestCase.assertTrue((np.argmax(targets, axis=1) == preds_adv).any())
    accuracy = np.sum(preds_adv == np.argmax(targets, axis=1)) / y_test_iris.shape[0]
    logger.info('Success rate of targeted boundary on Iris: %.2f%%', (accuracy * 100))

@pytest.mark.parametrize("targeted", [True, False])
def test_images(fix_get_mnist_subset, get_image_classifier_list_for_attack, framework, targeted):
    classifier_list = get_image_classifier_list_for_attack(BoundaryAttack)
    if classifier_list is None:
        logging.warning("Couldn't perform  this test because no classifier is defined")
        return

    for classifier in classifier_list:

        attack = BoundaryAttack(classifier=classifier, targeted=targeted, max_iter=20)
        if targeted:
            backend_targeted_images(attack, fix_get_mnist_subset)
        else:
            back_end_untargeted_images(attack, fix_get_mnist_subset, framework)


def test_classifier_type_check_fail():
    backend_test_classifier_type_check_fail(BoundaryAttack)
