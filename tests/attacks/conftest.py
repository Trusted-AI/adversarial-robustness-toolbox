# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
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
import pytest
import logging
from tests import utils
from art.estimators.classification import KerasClassifier
from art.defences.preprocessor import FeatureSqueezing

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
            potential_classifier
            for potential_classifier in classifier_list
            if all(t in type(potential_classifier).__mro__ for t in attack._estimator_requirements)
        ]

    return get_image_classifier_list_for_attack


@pytest.fixture
def get_tabular_classifier_list(get_tabular_classifier_list):
    def _tabular_classifier_list(attack, clipped=True):
        classifier_list = get_tabular_classifier_list(clipped)
        if classifier_list is None:
            return None

        return [
            potential_classifier
            for potential_classifier in classifier_list
            if all(t in type(potential_classifier).__mro__ for t in attack._estimator_requirements)
        ]

    return _tabular_classifier_list
