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

import numpy as np

from art.attacks.evasion.geometric_decision_based_attack import GeoDA
from art.estimators.estimator import BaseEstimator
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


@pytest.mark.framework_agnostic
def test_generate_2d_dct_basis(art_warning, image_dl_estimator):
    try:
        sub_dim = 5
        res = 224

        classifier, _ = image_dl_estimator(from_logits=True)
        attack = GeoDA(estimator=classifier, sub_dim=5, max_iter=4000, verbose=False)

        dct = attack._generate_2d_dct_basis(sub_dim=sub_dim, res=res)
        assert dct.shape == (res * res, sub_dim * sub_dim)
        dct_50000 = np.array(
            [
                [
                    0.00446429,
                    -0.0063133,
                    0.00631283,
                    -0.00631206,
                    0.00631097,
                    0.00490833,
                    -0.00694126,
                    0.00694075,
                    -0.00693989,
                    0.0069387,
                    0.00131841,
                    -0.00186447,
                    0.00186434,
                    -0.00186411,
                    0.00186379,
                    -0.00285836,
                    0.00404223,
                    -0.00404193,
                    0.00404143,
                    -0.00404074,
                    -0.00576281,
                    0.00814965,
                    -0.00814905,
                    0.00814805,
                    -0.00814664,
                ]
            ]
        )
        np.testing.assert_almost_equal(dct[50000], dct_50000)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_is_adversarial(art_warning, image_dl_estimator, fix_get_mnist_subset):
    try:
        (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset

        classifier, _ = image_dl_estimator(from_logits=True)
        attack = GeoDA(estimator=classifier, sub_dim=5, max_iter=4000, verbose=False)

        assert not attack._is_adversarial(x_adv=x_test_mnist[[0]], y_true=y_test_mnist[[0]])

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_find_random_adversarial(art_warning, image_dl_estimator, fix_get_mnist_subset):
    try:
        (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset

        classifier, _ = image_dl_estimator(from_logits=True)
        attack = GeoDA(estimator=classifier, sub_dim=5, max_iter=4000, verbose=False)

        x_adv = attack._find_random_adversarial(x=x_test_mnist[[0]], y=y_test_mnist[[0]])

        assert np.argmax(classifier.predict(x_adv)) != np.argmax(y_test_mnist[[0]])
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_opt_query_iteration(art_warning, image_dl_estimator):
    try:
        classifier, _ = image_dl_estimator(from_logits=True)
        attack = GeoDA(estimator=classifier, sub_dim=5, max_iter=4000, verbose=False)
        opt_q, var_t = attack._opt_query_iteration(var_nq=3800, var_t=8, lambda_param=0.6)

        opt_q_expected = [75, 106, 149, 210, 295, 414, 582, 818, 1150]
        var_t_expected = 9

        assert opt_q == opt_q_expected
        assert var_t == var_t_expected
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_sub_noise(art_warning, image_dl_estimator, fix_get_mnist_subset):
    try:
        (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset

        classifier, _ = image_dl_estimator(from_logits=True)
        attack = GeoDA(estimator=classifier, sub_dim=5, max_iter=4000, verbose=False)
        c = attack._sub_noise(num_noises=64, basis=np.ones((28 * 28, 25)))

        assert c[0].shape == x_train_mnist[0].shape
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_generate(art_warning, fix_get_mnist_subset, image_dl_estimator):
    try:
        (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset

        classifier, _ = image_dl_estimator(from_logits=True)
        attack = GeoDA(estimator=classifier, sub_dim=5, max_iter=400, verbose=False)
        x_train_mnist_adv = attack.generate(x=x_train_mnist, y=y_train_mnist)

        assert np.mean(np.abs(x_train_mnist_adv - x_train_mnist)) == pytest.approx(0.057965796, abs=0.01)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_check_params(art_warning, image_dl_estimator_for_attack):
    try:
        classifier = image_dl_estimator_for_attack(GeoDA, from_logits=True)

        with pytest.raises(ValueError):
            _ = GeoDA(classifier, batch_size=1.0)
        with pytest.raises(ValueError):
            _ = GeoDA(classifier, batch_size=-1)

        with pytest.raises(ValueError):
            _ = GeoDA(classifier, norm=0)

        with pytest.raises(ValueError):
            _ = GeoDA(classifier, sub_dim=1.0)
        with pytest.raises(ValueError):
            _ = GeoDA(classifier, sub_dim=-1)

        with pytest.raises(ValueError):
            _ = GeoDA(classifier, max_iter=1.0)
        with pytest.raises(ValueError):
            _ = GeoDA(classifier, max_iter=-1)

        with pytest.raises(ValueError):
            _ = GeoDA(classifier, max_iter=1.0)
        with pytest.raises(ValueError):
            _ = GeoDA(classifier, max_iter=-1)

        with pytest.raises(ValueError):
            _ = GeoDA(classifier, bin_search_tol="1")
        with pytest.raises(ValueError):
            _ = GeoDA(classifier, bin_search_tol=-1.0)

        with pytest.raises(ValueError):
            _ = GeoDA(classifier, lambda_param=1)
        with pytest.raises(ValueError):
            _ = GeoDA(classifier, lambda_param=-1.0)

        with pytest.raises(ValueError):
            _ = GeoDA(classifier, sigma=1)
        with pytest.raises(ValueError):
            _ = GeoDA(classifier, sigma=-1.0)

        # with pytest.raises(ValueError):
        #     _ = GeoDA(classifier, targeted="true")

        with pytest.raises(ValueError):
            _ = GeoDA(classifier, verbose="true")

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_classifier_type_check_fail(art_warning):
    try:
        backend_test_classifier_type_check_fail(GeoDA, [BaseEstimator, ClassifierMixin])
    except ARTTestException as e:
        art_warning(e)
