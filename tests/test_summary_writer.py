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

from art.summary_writer import SummaryWriterDefault

from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 10
    n_test = 10
    yield x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test]


@pytest.mark.skip_framework("scikitlearn", "mxnet")
def test_update_image_classification_sw(art_warning, fix_get_mnist_subset, image_dl_estimator):
    try:

        from art.attacks.evasion import ProjectedGradientDescent

        classifier, _ = image_dl_estimator(from_logits=False)

        swd = SummaryWriterDefault(summary_writer=True, ind_1=True, ind_2=True, ind_3=True, ind_4=True)

        attack = ProjectedGradientDescent(
            estimator=classifier, max_iter=10, eps=0.3, eps_step=0.03, batch_size=5, verbose=False, summary_writer=swd
        )

        (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset

        attack.generate(x=x_train_mnist, y=y_train_mnist)

        assert all(attack.summary_writer.i_1 == [False, False, False, False, False])
        if np.ndim(attack.summary_writer.i_2) != 0:
            assert len(attack.summary_writer.i_2) == 5
        np.testing.assert_almost_equal(attack.summary_writer.i_3["0"], np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
        np.testing.assert_almost_equal(attack.summary_writer.i_4["0"], np.array([0.0, 0.0, 0.0, 0.0, 0.0]))

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("scikitlearn", "mxnet")
@pytest.mark.parametrize("summary_writer", [True, "./"])
def test_update_image_classification_bool_str(art_warning, fix_get_mnist_subset, image_dl_estimator, summary_writer):
    try:

        from art.attacks.evasion import ProjectedGradientDescent

        classifier, _ = image_dl_estimator(from_logits=False)

        attack = ProjectedGradientDescent(
            estimator=classifier,
            max_iter=10,
            eps=0.3,
            eps_step=0.03,
            batch_size=5,
            verbose=False,
            summary_writer=summary_writer,
        )

        (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset

        attack.generate(x=x_train_mnist, y=y_train_mnist)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_update_image_object_detection_sw(art_warning, fix_get_mnist_subset, fix_get_rcnn):
    try:

        from art.attacks.evasion import ProjectedGradientDescent

        frcnn = fix_get_rcnn

        swd = SummaryWriterDefault(summary_writer=True, ind_1=False, ind_2=True, ind_3=True, ind_4=True)

        attack = ProjectedGradientDescent(
            estimator=frcnn, max_iter=10, eps=0.3, eps_step=0.03, batch_size=5, verbose=False, summary_writer=swd
        )

        (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset

        attack.generate(x=x_train_mnist, y=y_train_mnist)

        if np.ndim(attack.summary_writer.i_2) != 0:
            assert len(attack.summary_writer.i_2) == 5
        np.testing.assert_almost_equal(attack.summary_writer.i_3["0"], np.array([0.2265982]))
        np.testing.assert_almost_equal(attack.summary_writer.i_4["0"], np.array([0.0, 0.0, 0.0, 0.0, 0.0]))

    except ARTTestException as e:
        art_warning(e)
