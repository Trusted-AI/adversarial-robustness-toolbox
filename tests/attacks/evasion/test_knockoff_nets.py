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
import numpy as np
import pytest

from art.attacks.extraction.knockoff_nets import KnockoffNets
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin

from tests.utils import ARTTestException
from tests.attacks.utils import backend_test_classifier_type_check_fail

logger = logging.getLogger(__name__)


@pytest.mark.skipMlFramework("mxnet", "non_dl_frameworks")
@pytest.mark.parametrize("sampling_strategy", ["random", "adaptive"])
@pytest.mark.framework_agnostic
def test_with_images(art_warning, mnist_subset_100_10, image_dl_estimator, sampling_strategy):
    try:
        (x_train, y_train, x_test, y_test) = mnist_subset_100_10

        # Build TensorFlowClassifier
        victim_tfc, sess = image_dl_estimator()

        # Create the thieved classifier
        thieved_tfc, _ = image_dl_estimator(load_init=False, sess=sess)

        back_end_knockoff_nets(0.4, victim_tfc, thieved_tfc, sampling_strategy, x_train, y_train)

    except ARTTestException as e:
        art_warning(e)




def back_end_knockoff_nets(min_accuracy, victim_tfc, thieved_tfc, sampling_strategy, x_train, y_train=None):
    # Create random attack

    attack = KnockoffNets(
        classifier=victim_tfc,
        batch_size_fit=10,
        batch_size_query=10,
        nb_epochs=10,
        nb_stolen=100,
        sampling_strategy=sampling_strategy,
    )

    thieved_tfc = attack.extract(x=x_train, y=y_train, thieved_classifier=thieved_tfc)

    victim_preds = np.argmax(victim_tfc.predict(x=x_train), axis=1)
    thieved_preds = np.argmax(thieved_tfc.predict(x=x_train), axis=1)
    acc = np.sum(victim_preds == thieved_preds) / len(victim_preds)
    assert acc > min_accuracy


def test_classifier_type_check_fail():
    backend_test_classifier_type_check_fail(KnockoffNets, [BaseEstimator, ClassifierMixin])
