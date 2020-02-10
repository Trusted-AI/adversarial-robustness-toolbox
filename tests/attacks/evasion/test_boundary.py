import pytest
import logging
from art.attacks import BoundaryAttack
from art import utils
from tests import utils_attack
from art.classifiers.classifier import Classifier, ClassifierGradients
logger = logging.getLogger(__name__)

@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 10
    n_test = 10
    yield (x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test])


@pytest.mark.parametrize("clipped_classifier, targeted", [(True, True), (True, False), (False, True), (False, False)])
def test_tabular(get_tabular_classifier_list, get_mlFramework, get_iris_dataset, clipped_classifier, targeted):

    classifier_list = get_tabular_classifier_list(BoundaryAttack, clipped=clipped_classifier)
    if classifier_list is None:
        logging.warning("Couldn't perform  this test because no classifier is defined")
        return

    for classifier in classifier_list:

        attack = BoundaryAttack(classifier, targeted=targeted, max_iter=10)
        if targeted:
            utils_attack.backend_targeted_tabular(attack, get_iris_dataset, get_mlFramework)
        else:
            utils_attack.backend_untargeted_tabular(attack, get_iris_dataset, get_mlFramework,
                                                    clipped=clipped_classifier)


@pytest.mark.parametrize("targeted", [True, False])
def test_images(fix_get_mnist_subset, get_image_classifier_list, get_mlFramework, targeted):
    classifier_list = get_image_classifier_list(BoundaryAttack)
    if classifier_list is None:
        logging.warning("Couldn't perform  this test because no classifier is defined")
        return

    for classifier in classifier_list:

        attack = BoundaryAttack(classifier=classifier, targeted=targeted, max_iter=20)
        if targeted:
            utils_attack.backend_targeted_images(attack, fix_get_mnist_subset)
        else:
            utils_attack.back_end_untargeted_images(attack, fix_get_mnist_subset, get_mlFramework)


def test_classifier_type_check_fail_classifier():
    # Use a useless test classifier to test basic classifier properties
    class ClassifierNoAPI:
        pass

    classifier = ClassifierNoAPI

    with pytest.raises(utils.WrongClassifer) as exception:
        _ = BoundaryAttack(classifier=classifier)

    assert exception.value.class_expected == Classifier
