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

import numpy as np
import pytest

from art.attacks.evasion.dpatch_robust_eot import EoTRobustDPatch, PatchOperator
from art.estimators.object_detection.pytorch_object_detector import PyTorchObjectDetector

from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 10
    n_test = 10
    yield x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test]


@pytest.mark.skip_framework("keras", "scikitlearn", "mxnet", "kerastf", "tensorflow")
def test_generate(art_warning, fix_get_mnist_subset, fix_get_rcnn, framework):
    try:
        (_, _, x_test_mnist, y_test_mnist) = fix_get_mnist_subset

        if framework == "pytorch":
            x_test_mnist = np.transpose(x_test_mnist, (0, 2, 3, 1))

        frcnn = fix_get_rcnn
        attack = EoTRobustDPatch(
            frcnn,
            distortion_scale_max = 0.3,
            patch_shape = (4,4,1),
            sample_size=1,
            learning_rate=1.0,
            max_iter=100,
            batch_size=5,
            verbose=False,
        )
        """
        patch = attack.generate(x=x_test_mnist)
        assert patch.shape == (4, 4, 1)

        with pytest.raises(ValueError):
            _ = attack.generate(x=np.repeat(x_test_mnist, axis=3, repeats=2))

        with pytest.raises(ValueError):
            _ = attack.generate(x=x_test_mnist, y=y_test_mnist)
        """
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.skip_framework("keras", "scikitlearn", "mxnet", "kerastf", "tensorflow")
def test_generate_targeted(art_warning, fix_get_mnist_subset, fix_get_rcnn, framework):
    try:
        (_, _, x_test_mnist, _) = fix_get_mnist_subset

        if framework == "pytorch":
            x_test_mnist = np.transpose(x_test_mnist, (0, 2, 3, 1))

        frcnn = fix_get_rcnn
        attack = EoTRobustDPatch(
            frcnn,
            patch_shape=(4, 4, 1),
            patch_location=(2, 2),
            crop_range=(0, 0),
            brightness_range=(1.0, 1.0),
            rotation_weights=(1, 0, 0, 0),
            sample_size=1,
            learning_rate=1.0,
            max_iter=1,
            batch_size=1,
            targeted=True,
            verbose=False,
        )
        """
        y = frcnn.predict(x_test_mnist)
        patch = attack.generate(x=x_test_mnist, y=y)
        assert patch.shape == (4, 4, 1)

        with pytest.raises(ValueError):
            _ = attack.generate(x=x_test_mnist, y=None)
    """

    except ARTTestException as e:
        art_warning(e)
