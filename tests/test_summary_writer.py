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
def test_update_image_classification(art_warning, fix_get_mnist_subset, image_dl_estimator):
    try:

        from art.attacks.evasion import ProjectedGradientDescent

        classifier, _ = image_dl_estimator(from_logits=True)

        swd = SummaryWriterDefault(summary_writer=True, ind_1=True, ind_2=True, ind_3=True, ind_4=True)

        attack = ProjectedGradientDescent(
            estimator=classifier, max_iter=10, eps=0.3, eps_step=0.03, batch_size=5, summary_writer=swd
        )

        (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset

        attack.generate(x=x_train_mnist, y=y_train_mnist)

        assert all(attack.summary_writer.i_1 == [False, False, False, False, False])
        np.testing.assert_almost_equal(
            np.array(attack.summary_writer.i_2),
            np.array([4.8398972e-05, 5.5432320e-06, 9.8937750e-04, -4.7683716e-07, 1.4901161e-05]),
            decimal=3,
        )
        np.testing.assert_almost_equal(attack.summary_writer.i_3["0"], np.array([9.0, 9.0, 9.0, 9.0, 9.0]))
        np.testing.assert_almost_equal(attack.summary_writer.i_4["0"], np.array([0.0, 0.0, 0.0, 0.0, 0.0]))

    except ARTTestException as e:
        art_warning(e)
