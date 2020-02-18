import pytest
import logging
from tests import utils_test
from art.classifiers import KerasClassifier

logger = logging.getLogger(__name__)




@pytest.fixture
def get_image_classifier_list_for_attack(get_image_classifier_list):
    def get_image_classifier_list_for_attack(attack, defended=False):
        classifier_list = get_image_classifier_list(defended=defended)
        if classifier_list is None:
            return None

        return [potential_classier for potential_classier in classifier_list if
                attack.is_valid_classifier_type(potential_classier)]
    return get_image_classifier_list_for_attack



@pytest.fixture
def get_tabular_classifier_list(get_tabular_classifier_list):
    def _tabular_classifier_list(attack, clipped=True):
        classifier_list = get_tabular_classifier_list(clipped)
        if classifier_list is None:
            return None

        return [potential_classier for potential_classier in classifier_list if attack.is_valid_classifier_type(potential_classier)]


    return _tabular_classifier_list

