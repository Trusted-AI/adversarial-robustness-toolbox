# MIT License

# Copyright (c) 2022 Keiichiro Yamamura

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
import logging
import pytest

import numpy as np

from art.attacks.evasion import AutoConjugateGradient
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.classification.classifier import ClassifierMixin

from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 100
    n_test = 10
    yield x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test]


@pytest.mark.parametrize("loss_type", ["cross_entropy", "difference_logits_ratio"])
@pytest.mark.parametrize("norm", ["inf", np.inf, 1, 2])
@pytest.mark.skip_framework("keras", "non_dl_frameworks", "mxnet", "kerastf", "tensorflow1", "tensorflow2v1")
def test_generate(art_warning, fix_get_mnist_subset, image_dl_estimator_for_attack, framework, loss_type, norm):
    print("test_generate")
    try:
        classifier = image_dl_estimator_for_attack(AutoConjugateGradient, from_logits=True)

        print("framework", framework)

        if framework in ["tensorflow1", "tensorflow2v1"] and loss_type == "difference_logits_ratio":
            with pytest.raises(ValueError):
                _ = AutoConjugateGradient(
                    estimator=classifier,
                    norm=norm,
                    eps=0.3,
                    eps_step=0.1,
                    max_iter=5,
                    targeted=False,
                    nb_random_init=1,
                    batch_size=32,
                    loss_type=loss_type,
                    verbose=False,
                )
        else:

            attack = AutoConjugateGradient(
                estimator=classifier,
                norm=norm,
                eps=0.3,
                eps_step=0.1,
                max_iter=5,
                targeted=False,
                nb_random_init=1,
                batch_size=32,
                loss_type=loss_type,
                verbose=False,
            )

            (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset

            x_train_mnist_adv = attack.generate(x=x_train_mnist, y=y_train_mnist)

            assert np.max(np.abs(x_train_mnist_adv - x_train_mnist)) > 0.0

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_check_params(art_warning, image_dl_estimator_for_attack):
    try:

        classifier = image_dl_estimator_for_attack(AutoConjugateGradient, from_logits=True)

        with pytest.raises(ValueError):
            _ = AutoConjugateGradient(classifier, norm=0)

        with pytest.raises(ValueError):
            _ = AutoConjugateGradient(classifier, eps="1")
        with pytest.raises(ValueError):
            _ = AutoConjugateGradient(classifier, eps=-1.0)

        with pytest.raises(ValueError):
            _ = AutoConjugateGradient(classifier, eps_step="1")
        with pytest.raises(ValueError):
            _ = AutoConjugateGradient(classifier, eps_step=-1.0)

        with pytest.raises(ValueError):
            _ = AutoConjugateGradient(classifier, max_iter=1.0)
        with pytest.raises(ValueError):
            _ = AutoConjugateGradient(classifier, max_iter=-1)

        with pytest.raises(ValueError):
            _ = AutoConjugateGradient(classifier, targeted="true")

        with pytest.raises(ValueError):
            _ = AutoConjugateGradient(classifier, nb_random_init=1.0)
        with pytest.raises(ValueError):
            _ = AutoConjugateGradient(classifier, nb_random_init=-1)

        with pytest.raises(ValueError):
            _ = AutoConjugateGradient(classifier, batch_size=1.0)
        with pytest.raises(ValueError):
            _ = AutoConjugateGradient(classifier, batch_size=-1)

        with pytest.raises(ValueError):
            _ = AutoConjugateGradient(classifier, loss_type="test")

        with pytest.raises(ValueError):
            _ = AutoConjugateGradient(classifier, verbose="true")

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_classifier_type_check_fail(art_warning):
    try:
        backend_test_classifier_type_check_fail(
            AutoConjugateGradient, [BaseEstimator, LossGradientsMixin, ClassifierMixin]
        )
    except ARTTestException as e:
        art_warning(e)
