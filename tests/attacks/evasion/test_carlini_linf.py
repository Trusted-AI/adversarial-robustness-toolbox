# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2026
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
import logging

import numpy as np
import pytest

from art.attacks.evasion import CarliniLInfMethod

from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    (_, _), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_test = 10
    yield x_test_mnist[:n_test], y_test_mnist[:n_test]


@pytest.mark.skip_framework("keras", "kerastf", "huggingface")
def test_targeted_returns_only_examples_classified_as_target(art_warning, fix_get_mnist_subset, image_dl_estimator):
    """
    In targeted mode an example may only be kept if it is classified as the target class: every input that the
    attack modifies must be predicted as its target, all others must be returned unchanged.
    """
    try:
        x_test, _ = fix_get_mnist_subset
        classifier, _ = image_dl_estimator()

        y_pred = np.argmax(classifier.predict(x_test), axis=1)
        y_target = (y_pred + 1) % 10

        attack = CarliniLInfMethod(
            classifier, targeted=True, max_iter=10, learning_rate=1.0, initial_const=1, largest_const=1.1, verbose=False
        )
        x_adv = attack.generate(x_test, y=y_target)

        y_pred_adv = np.argmax(classifier.predict(x_adv), axis=1)
        modified = np.any(np.abs(x_adv - x_test).reshape(len(x_test), -1) > 0, axis=1)

        assert modified.any()
        np.testing.assert_array_equal(y_pred_adv[modified], y_target[modified])
        np.testing.assert_array_almost_equal(x_adv[~modified], x_test[~modified])

    except ARTTestException as e:
        art_warning(e)
