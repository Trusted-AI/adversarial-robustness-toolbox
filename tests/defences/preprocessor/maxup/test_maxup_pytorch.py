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
from numpy.testing import assert_array_equal, assert_array_almost_equal

from art.config import ART_NUMPY_DTYPE
from art.defences.preprocessor import CutoutPyTorch, MaxupPyTorch, MixupPyTorch
from tests.utils import ARTTestException, get_image_classifier_pt

logger = logging.getLogger(__name__)


@pytest.fixture(params=[1], ids=["grayscale"])
def image_batch(request):
    """
    Image fixtures of shape NHWC.
    """
    channels = request.param
    data_shape = (4, channels, 28, 28)
    return (0.5 * np.ones(data_shape)).astype(ART_NUMPY_DTYPE)


@pytest.fixture(params=[1], ids=["grayscale"])
def empty_image(request):
    """
    Empty image fixtures of shape NHWC.
    """
    channels = request.param
    data_shape = (4, channels, 28, 28)
    return np.zeros(data_shape).astype(ART_NUMPY_DTYPE)


@pytest.mark.only_with_platform("pytorch")
@pytest.mark.parametrize("num_trials", [1, 2, 4])
def test_maxup_single_augmentation(art_warning, image_batch, num_trials):
    classifier = get_image_classifier_pt(from_logits=True, use_maxpool=False)
    mixup = MixupPyTorch(num_classes=10)

    try:
        maxup = MaxupPyTorch(estimator=classifier, augmentations=mixup, num_trials=num_trials)
        x, y = maxup(image_batch, np.arange(len(image_batch)))
        assert_array_almost_equal(x, image_batch)
        assert_array_almost_equal(y.sum(axis=1), np.ones(len(image_batch)))
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
@pytest.mark.parametrize("num_trials", [1, 2, 4])
def test_mixup_multiple_augmentation(art_warning, empty_image, num_trials):
    classifier = get_image_classifier_pt(from_logits=True, use_maxpool=False)
    mixup = MixupPyTorch(num_classes=10)
    cutout = CutoutPyTorch(length=8, channels_first=True)

    try:
        maxup = MaxupPyTorch(estimator=classifier, augmentations=[mixup, cutout], num_trials=num_trials)
        x, y = maxup(empty_image, np.arange(len(empty_image)))
        assert_array_equal(x, empty_image)
        assert_array_almost_equal(y.sum(axis=1), np.ones(len(empty_image)))
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_missing_labels_error(art_warning, tabular_batch):
    classifier = get_image_classifier_pt(from_logits=True, use_maxpool=False)
    mixup = MixupPyTorch(num_classes=10)

    try:
        test_input = tabular_batch
        maxup = MaxupPyTorch(estimator=classifier, augmentations=mixup)
        exc_msg = "Labels `y` cannot be None."
        with pytest.raises(ValueError, match=exc_msg):
            maxup(test_input)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_check_params(art_warning):
    classifier = get_image_classifier_pt(from_logits=True, use_maxpool=False)
    mixup = MixupPyTorch(num_classes=10)

    try:
        with pytest.raises(ValueError):
            _ = MaxupPyTorch(estimator=classifier, augmentations=[])

        with pytest.raises(ValueError):
            _ = MaxupPyTorch(estimator=classifier, augmentations=mixup, num_trials=0)

        with pytest.raises(ValueError):
            _ = MaxupPyTorch(estimator=classifier, augmentations=mixup, num_trials=-1)

    except ARTTestException as e:
        art_warning(e)
