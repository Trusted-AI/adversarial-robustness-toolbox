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
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import pytest

from art.defences.transformer.poisoning import STRIP
from art.estimators.classification import TensorFlowClassifier, TensorFlowV2Classifier

from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.mark.framework_agnostic
def test_strip(art_warning, get_default_mnist_subset, image_dl_estimator):
    try:
        (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

        classifier, _ = image_dl_estimator()

        classifier.fit(x_train_mnist, y_train_mnist, nb_epochs=1)
        strip = STRIP(classifier)
        defense_cleanse = strip()
        defense_cleanse.mitigate(x_test_mnist)
        defense_cleanse.predict(x_test_mnist)
        stripped_classifier = strip.get_classifier()
        stripped_classifier._check_params()
        assert isinstance(stripped_classifier, TensorFlowV2Classifier) or isinstance(
            stripped_classifier, TensorFlowClassifier
        )
    except ARTTestException as e:
        art_warning(e)
