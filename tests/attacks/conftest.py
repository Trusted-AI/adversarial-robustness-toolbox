import pytest
import logging
from tests import utils_test
from art.defences import FeatureSqueezing
from art.classifiers import KerasClassifier

logger = logging.getLogger(__name__)

@pytest.fixture
def get_image_classifier_list_for_attack(get_mlFramework):
    def _image_classifier_list(attack, defended=False):
        if get_mlFramework == "keras":
            if defended:
                classifier = utils_test.get_image_classifier_kr()
                # Get the ready-trained Keras model
                fs = FeatureSqueezing(bit_depth=1, clip_values=(0, 1))
                classifier_list = [KerasClassifier(model=classifier._model, clip_values=(0, 1), defences=fs)]
            else:
                classifier_list = [utils_test.get_image_classifier_kr()]
        if get_mlFramework == "tensorflow":
            if defended:
                logging.warning("{0} doesn't have a defended image classifier defined yet".format(get_mlFramework))
                classifier_list = None
            else:
                classifier, sess = utils_test.get_image_classifier_tf()
                classifier_list = [classifier]
        if get_mlFramework == "pytorch":
            if defended:
                logging.warning("{0} doesn't have a defended image classifier defined yet".format(get_mlFramework))
                classifier_list =  None
            else:
                classifier_list =  [utils_test.get_image_classifier_pt()]
        if get_mlFramework == "scikitlearn":
            if defended:
                logging.warning("{0} doesn't have a defended image classifier defined yet".format(get_mlFramework))
                classifier_list = None
            else:
                logging.warning("{0} doesn't have an image classifier defined yet".format(get_mlFramework))
                classifier_list =  None

        if classifier_list is None:
            return None

        return [potential_classier for potential_classier in classifier_list if
                attack.is_valid_classifier_type(potential_classier)]
    return _image_classifier_list