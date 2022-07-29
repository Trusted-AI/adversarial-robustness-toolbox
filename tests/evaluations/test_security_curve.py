# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2021
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
import pytest

from art.evaluations.security_curve import SecurityCurve

from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 100
    n_test = 10
    yield x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test]


@pytest.mark.framework_agnostic
def test_generate_default(art_warning, fix_get_mnist_subset, image_dl_estimator):
    try:
        classifier, _ = image_dl_estimator(from_logits=True)

        sec = SecurityCurve(eps=3)

        (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset

        eps_list, accuracy_adv_list, accuracy = sec.evaluate(classifier=classifier, x=x_train_mnist, y=y_train_mnist)

        assert eps_list == [0.3333333333333333, 0.6666666666666666, 1.0]
        assert accuracy_adv_list == [0.0, 0.0, 0.0]
        assert accuracy == 0.27
        assert sec.detected_obfuscating_gradients is False

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_generate_list(art_warning, fix_get_mnist_subset, image_dl_estimator):
    try:
        classifier, _ = image_dl_estimator(from_logits=True)

        sec = SecurityCurve(eps=[0.3333333333333333, 0.6666666666666666, 1.0])

        (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset

        eps_list, accuracy_adv_list, accuracy = sec.evaluate(classifier=classifier, x=x_train_mnist, y=y_train_mnist)

        assert eps_list == [0.3333333333333333, 0.6666666666666666, 1.0]
        assert accuracy_adv_list == [0.0, 0.0, 0.0]
        assert accuracy == 0.27
        assert sec.detected_obfuscating_gradients is False

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_repr(art_warning):
    try:
        sec = SecurityCurve(eps=3)

        assert repr(sec) == "art.evaluations.security_curve.security_curve.SecurityCurve(eps=3)"

    except ARTTestException as e:
        art_warning(e)
