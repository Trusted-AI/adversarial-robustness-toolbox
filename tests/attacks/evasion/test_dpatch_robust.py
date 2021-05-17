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

from art.attacks.evasion import RobustDPatch
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.object_detection.object_detector import ObjectDetectorMixin

from tests.attacks.utils import backend_test_classifier_type_check_fail
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.fixture()
def fix_get_robust_dpatch():
    from abc import ABC

    class DummyObjectDetector(ObjectDetectorMixin, LossGradientsMixin, BaseEstimator, ABC):
        def __init__(self):
            self._clip_values = (0, 1)
            self.channels_first = False
            self._input_shape = None

        def loss_gradient(self, x: np.ndarray, y: None, **kwargs):
            raise NotImplementedError

        def fit(self, x: np.ndarray, y, batch_size: int = 128, nb_epochs: int = 20, **kwargs):
            raise NotImplementedError

        def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs):
            return [{"boxes": [], "labels": [], "scores": []}]

        @property
        def native_label_is_pytorch_format(self):
            return True

        @property
        def input_shape(self):
            return self._input_shape

    frcnn = DummyObjectDetector()
    attack = RobustDPatch(
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
        verbose=False,
    )
    yield attack


@pytest.mark.parametrize("image_format", ["NHWC", "NCHW"])
@pytest.mark.framework_agnostic
def test_augment_images_with_patch(art_warning, image_format, fix_get_robust_dpatch):
    try:
        attack = fix_get_robust_dpatch

        if image_format == "NHWC":
            patch = np.ones(shape=(4, 4, 1))
            x = np.zeros(shape=(1, 10, 10, 1))
            channels_first = False
        elif image_format == "NCHW":
            patch = np.ones(shape=(1, 4, 4))
            x = np.zeros(shape=(1, 1, 10, 10))
            channels_first = True

        patched_images, _, transformations = attack._augment_images_with_patch(
            x=x, y=None, patch=patch, channels_first=channels_first
        )

        transformation_expected = {"crop_x": 0, "crop_y": 0, "rot90": 0, "brightness": 1.0}
        patch_sum_expected = 16.0
        complement_sum_expected = 0.0

        if image_format == "NHWC":
            patch_sum = np.sum(patched_images[0, 2:7, 2:7, :])
        elif image_format == "NCHW":
            patch_sum = np.sum(patched_images[0, :, 2:7, 2:7])

        complement_sum = np.sum(patched_images[0]) - patch_sum

        assert transformations == transformation_expected

        assert patch_sum == patch_sum_expected

        assert complement_sum == complement_sum_expected

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_apply_patch(art_warning, fix_get_robust_dpatch):
    try:
        attack = fix_get_robust_dpatch

        patch = np.ones(shape=(4, 4, 1))
        x = np.zeros(shape=(1, 10, 10, 1))

        patched_images = attack.apply_patch(x=x, patch_external=patch)

        patch_sum_expected = 16.0
        complement_sum_expected = 0.0

        patch_sum = np.sum(patched_images[0, 2:7, 2:7, :])
        complement_sum = np.sum(patched_images[0]) - patch_sum

        assert patch_sum == patch_sum_expected

        assert complement_sum == complement_sum_expected

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_untransform_gradients(art_warning, fix_get_robust_dpatch):
    try:
        attack = fix_get_robust_dpatch

        gradients = np.zeros(shape=(1, 10, 10, 1))
        gradients[:, 2:7, 2:7, :] = 1

        crop_x = 1
        crop_y = 1
        rot90 = 3
        brightness = 1.0

        gradients = gradients[:, crop_x : 10 - crop_x, crop_y : 10 - crop_y, :]
        gradients = np.rot90(gradients, rot90, (1, 2))

        transforms = {"crop_x": crop_x, "crop_y": crop_y, "rot90": rot90, "brightness": brightness}

        gradients = attack._untransform_gradients(gradients=gradients, transforms=transforms, channels_first=False)
        gradients_sum = np.sum(gradients[0])
        gradients_sum_expected = 16.0

        assert gradients_sum == gradients_sum_expected

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_classifier_type_check_fail(art_warning):
    try:
        backend_test_classifier_type_check_fail(RobustDPatch, [BaseEstimator, LossGradientsMixin, ObjectDetectorMixin])
    except ARTTestException as e:
        art_warning(e)
