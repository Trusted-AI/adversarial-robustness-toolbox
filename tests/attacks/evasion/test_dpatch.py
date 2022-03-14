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

from art.attacks.evasion import DPatch
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.object_detection.object_detector import ObjectDetectorMixin
from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import master_seed
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 100
    n_test = 11
    yield x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test]


@pytest.mark.framework_agnostic
def test_generate(art_warning, fix_get_mnist_subset, fix_get_rcnn):
    try:
        (_, _, x_test_mnist, y_test_mnist) = fix_get_mnist_subset

        frcnn = fix_get_rcnn
        attack = DPatch(
            frcnn,
            patch_shape=(4, 4, 1),
            learning_rate=1.0,
            max_iter=1,
            batch_size=1,
            verbose=False,
        )
        patch = attack.generate(x=x_test_mnist[[0]])
        assert patch.shape == (4, 4, 1)

        attack.apply_patch(x=x_test_mnist)
        attack.apply_patch(x=x_test_mnist, patch_external=patch)
        attack.apply_patch(x=x_test_mnist, patch_external=patch, mask=np.ones((1, 28, 28)).astype(bool))
        attack.apply_patch(
            x=x_test_mnist, patch_external=patch, mask=np.ones((1, 28, 28)).astype(bool), random_location=True
        )

        patch = attack.generate(x=x_test_mnist[[0]], target_label=1)
        assert patch.shape == (4, 4, 1)
        patch = attack.generate(x=x_test_mnist[[0]], target_label=np.array([1]))
        assert patch.shape == (4, 4, 1)
        patch = attack.generate(x=x_test_mnist[[0]], target_label=[1])
        assert patch.shape == (4, 4, 1)

        with pytest.raises(ValueError):
            _ = attack.generate(x=np.repeat(x_test_mnist, axis=3, repeats=2))

        with pytest.raises(ValueError):
            _ = attack.generate(x=x_test_mnist, y=y_test_mnist)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.parametrize("random_location", [True, False])
@pytest.mark.parametrize("image_format", ["NHWC", "NCHW"])
@pytest.mark.framework_agnostic
def test_augment_images_with_patch(art_warning, random_location, image_format, fix_get_mnist_subset):
    try:
        (x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist) = fix_get_mnist_subset

        # TODO this master_seed should be removed as it is already set in conftest.py. expected values will
        # need to be updated accordingly
        master_seed()

        if image_format == "NHWC":
            patch = np.ones(shape=(4, 4, 1)) * 0.5
            x = x_train_mnist[0:3]
            channels_first = False
        elif image_format == "NCHW":
            patch = np.ones(shape=(1, 4, 4)) * 0.5
            x = np.transpose(x_train_mnist[0:3], (0, 3, 1, 2))
            channels_first = True

        patched_images, transformations = DPatch._augment_images_with_patch(
            x=x, patch=patch, random_location=random_location, channels_first=channels_first
        )

        if random_location:
            transformation_expected = {"i_x_1": 0, "i_y_1": 2, "i_x_2": 4, "i_y_2": 6}
            patched_images_column = [
                0.0,
                0.0,
                0.5,
                0.5,
                0.5,
                0.5,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        else:
            transformation_expected = {"i_x_1": 0, "i_y_1": 0, "i_x_2": 4, "i_y_2": 4}
            patched_images_column = [
                0.5,
                0.5,
                0.5,
                0.5,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]

        assert transformations[1] == transformation_expected

        if image_format == "NCHW":
            patched_images = np.transpose(patched_images, (0, 2, 3, 1))

        np.testing.assert_array_equal(patched_images[1, 2, :, 0], patched_images_column)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_check_params(art_warning, fix_get_rcnn):
    try:
        frcnn = fix_get_rcnn

        with pytest.raises(ValueError):
            _ = DPatch(frcnn, patch_shape=(1.0, 2.0, 3.0))
        with pytest.raises(ValueError):
            _ = DPatch(frcnn, patch_shape=(1, 2, 3, 4))

        with pytest.raises(ValueError):
            _ = DPatch(frcnn, learning_rate=1)
        with pytest.raises(ValueError):
            _ = DPatch(frcnn, learning_rate=-1.0)

        with pytest.raises(ValueError):
            _ = DPatch(frcnn, max_iter=1.0)
        with pytest.raises(ValueError):
            _ = DPatch(frcnn, max_iter=-1)

        with pytest.raises(ValueError):
            _ = DPatch(frcnn, batch_size=1.0)
        with pytest.raises(ValueError):
            _ = DPatch(frcnn, batch_size=-1)

        with pytest.raises(ValueError):
            _ = DPatch(frcnn, verbose="true")

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_classifier_type_check_fail(art_warning):
    try:
        backend_test_classifier_type_check_fail(DPatch, [BaseEstimator, LossGradientsMixin, ObjectDetectorMixin])
    except ARTTestException as e:
        art_warning(e)
