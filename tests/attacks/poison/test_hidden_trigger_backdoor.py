# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2022
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

from art.attacks.poisoning.hidden_trigger_backdoor import HiddenTriggerBackdoor

from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.mark.skip_framework("non_dl_frameworks", "tensorflow", "mxnet", "keras", "kerastf")
def test_poison(art_warning, get_default_mnist_subset, image_dl_estimator):
    try:
        (x_train, y_train), (_, _) = get_default_mnist_subset
        classifier, _ = image_dl_estimator(functional=True)
        target = 0
        source = 1
        attack = HiddenTriggerBackdoor(classifier, eps=0.3, target=target, source=source, feature_layer=len(classifier.layers)-2, backdoor=backdoor, decay_coeff = .95, decay_iter = 1, max_iter=2, batch_size=1, poison_percent=.1)
        poison_data, poison_inds = attack.poison(x_train[:10], y_train[:10])

        np.testing.assert_equal(len(poison_data), 1)
        np.testing.assert_equal(len(poison_labels), 1)

        with pytest.raises(AssertionError):
            np.testing.assert_equal(poison_data, x_train[poison_inds])

    except ARTTestException as e:
        art_warning(e)
