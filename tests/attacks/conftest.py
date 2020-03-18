import pytest
import logging
from tests import utils
from art.classifiers import KerasClassifier
from art.defences import FeatureSqueezing

logger = logging.getLogger(__name__)


@pytest.fixture
def get_image_classifier_list_defended(framework):
    def _get_image_classifier_list_defended(one_classifier=False, **kwargs):
        sess = None
        classifier_list = None
        if framework == "keras":
            classifier = utils.get_image_classifier_kr()
            # Get the ready-trained Keras model
            fs = FeatureSqueezing(bit_depth=1, clip_values=(0, 1))
            classifier_list = [KerasClassifier(model=classifier._model, clip_values=(0, 1), preprocessing_defences=fs)]

        if framework == "tensorflow":
            logging.warning("{0} doesn't have a defended image classifier defined yet".format(framework))

        if framework == "pytorch":
            logging.warning("{0} doesn't have a defended image classifier defined yet".format(framework))

        if framework == "scikitlearn":
            logging.warning("{0} doesn't have a defended image classifier defined yet".format(framework))

        if classifier_list is None:
            return None, None

        if one_classifier:
            return classifier_list[0], sess

        return classifier_list, sess

    return _get_image_classifier_list_defended


@pytest.fixture
def get_image_classifier_list_for_attack(get_image_classifier_list, get_image_classifier_list_defended):
    def get_image_classifier_list_for_attack(attack, defended=False, **kwargs):
        if defended:
            classifier_list, _ = get_image_classifier_list_defended(kwargs)
        else:
            classifier_list, _ = get_image_classifier_list()
        if classifier_list is None:
            return None

        return [
            potential_classier
            for potential_classier in classifier_list
            if attack.is_valid_classifier_type(potential_classier)
        ]

    return get_image_classifier_list_for_attack


@pytest.fixture
def get_tabular_classifier_list(get_tabular_classifier_list):
    def _tabular_classifier_list(attack, clipped=True):
        classifier_list = get_tabular_classifier_list(clipped)
        if classifier_list is None:
            return None

        return [
            potential_classier
            for potential_classier in classifier_list
            if attack.is_valid_classifier_type(potential_classier)
        ]

    return _tabular_classifier_list
