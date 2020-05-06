# MIT License
#
# Copyright (C) IBM Corporation 2020
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
import numpy as np
import pytest

from art.attacks.inference import AttributeInferenceWhiteBox

from tests.utils import ExpectedValue

logger = logging.getLogger(__name__)


def test_white_box(get_tabular_classifier_list, get_iris_dataset):
    classifier_list = get_tabular_classifier_list(AttributeInferenceWhiteBox)
    if classifier_list is None:
        logging.warning("Couldn't perform  this test because no classifier is defined")
        return

    attack_feature = 2 # petal length
    values = [1, 3, 4, 5, 6] # rounded down
    priors = [50/150, 11/150, 43/150, 35/150, 11/150]

    (x_train_iris, y_train_iris), (x_test_iris, y_test_iris) = get_iris_dataset
    x_train_for_attack = np.delete(x_train_iris, attack_feature, 1)
    x_train_feature = x_train_iris[:, attack_feature]
    x_test_for_attack = np.delete(x_test_iris, attack_feature, 1)
    x_test_feature = x_test_iris[:, attack_feature]

    for classifier in classifier_list:
        attack = AttributeInferenceWhiteBox(classifier, attack_feature=attack_feature)
        x_train_predictions = classifier.predict(x_train_iris)
        x_test_predictions = classifier.predict(x_test_iris)
        inferred_train = attack.infer(x_train_for_attack, x_train_predictions, values, priors)
        inferred_test = attack.infer(x_test_for_attack, x_test_predictions, values, priors)
        train_diff = np.abs(inferred_train - x_train_feature)
        test_diff = np.abs(inferred_test - x_test_feature)
        assert sum(train_diff) < sum(test_diff)
        #TODO: to be removed
        break

if __name__ == "__main__":
    pytest.cmdline.main("-q {} --mlFramework=scikitlearn --durations=0".format(__file__).split(" "))
