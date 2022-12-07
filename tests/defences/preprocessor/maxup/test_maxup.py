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
from art.defences.preprocessor import Maxup, Cutout, CutMix, Mixup, SpatialSmoothing
from art.utils import to_categorical
from tests.utils import ARTTestException, get_image_classifier_tf_v2, get_image_classifier_kr, get_image_classifier_pt

logger = logging.getLogger(__name__)


@pytest.fixture(params=[1], ids=["grayscale"])
def image_batch(request):
    """
    Image fixtures of shape NHWC.
    """
    channels = request.param
    data_shape = (4, 28, 28, channels)
    return (0.5 * np.ones(data_shape)).astype(ART_NUMPY_DTYPE)


@pytest.fixture(params=[1], ids=["grayscale"])
def empty_image(request):
    """
    Empty image fixtures of shape NHWC.
    """
    channels = request.param
    data_shape = (4, 28, 28, channels)
    return np.zeros(data_shape).astype(ART_NUMPY_DTYPE)


@pytest.mark.only_with_platform("tensorflow2")
@pytest.mark.parametrize("num_trials", [1, 2, 4])
def test_maxup_tensorflow(art_warning, empty_image, num_trials):
    import tensorflow as tf

    classifier = get_image_classifier_tf_v2(from_logits=True)
    classifier._loss_object = tf.keras.losses.CategoricalCrossentropy()
    classifier._reduce_labels = False
    cutout = Cutout(length=8, channels_first=False)
    smooth = SpatialSmoothing()
    mixup = Mixup(num_classes=10)
    cutmix = CutMix(num_classes=10, channels_first=False)
    labels = to_categorical(np.arange(len(empty_image)), 10)

    try:
        maxup = Maxup(estimator=classifier, augmentations=[cutout, smooth, mixup, cutmix], num_trials=num_trials)
        x, y = maxup(empty_image, labels)
        assert_array_equal(x, empty_image)
        assert_array_almost_equal(y.sum(axis=1), np.ones(len(empty_image)))
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("keras", "kerastf")
@pytest.mark.parametrize("num_trials", [1, 2, 4])
def test_maxup_keras(art_warning, empty_image, num_trials):
    classifier = get_image_classifier_kr(loss_name="categorical_crossentropy", from_logits=True)
    cutout = Cutout(length=8, channels_first=False)
    smooth = SpatialSmoothing()
    mixup = Mixup(num_classes=10)
    cutmix = CutMix(num_classes=10, channels_first=False)
    labels = to_categorical(np.arange(len(empty_image)), 10)

    try:
        maxup = Maxup(estimator=classifier, augmentations=[cutout, smooth, mixup, cutmix], num_trials=num_trials)
        x, y = maxup(empty_image, labels)
        assert_array_equal(x, empty_image)
        assert_array_almost_equal(y.sum(axis=1), np.ones(len(empty_image)))
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
@pytest.mark.parametrize("num_trials", [1, 2, 4])
def test_maxup_pytorch(art_warning, empty_image, num_trials):
    classifier = get_image_classifier_pt(from_logits=True, use_maxpool=False)
    cutout = Cutout(length=8, channels_first=True)
    smooth = SpatialSmoothing()
    mixup = Mixup(num_classes=10)
    cutmix = CutMix(num_classes=10, channels_first=True)
    empty_image = np.transpose(empty_image, (0, 3, 1, 2))
    labels = np.arange(len(empty_image))

    try:
        maxup = Maxup(estimator=classifier, augmentations=[cutout, smooth, mixup, cutmix], num_trials=num_trials)
        x, y = maxup(empty_image, labels)
        assert_array_equal(x, empty_image)
        assert_array_almost_equal(y.sum(axis=1), np.ones(len(empty_image)))
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("keras", "kerastf")
def test_missing_labels_error(art_warning, tabular_batch):
    classifier = get_image_classifier_kr(loss_name="categorical_crossentropy", from_logits=True)
    cutout = Cutout(length=8)

    try:
        test_input = tabular_batch
        maxup = Maxup(estimator=classifier, augmentations=cutout)
        exc_msg = "Labels `y` cannot be None."
        with pytest.raises(ValueError, match=exc_msg):
            maxup(test_input)
    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("keras", "kerastf")
def test_check_params(art_warning):
    classifier = get_image_classifier_kr(loss_name="categorical_crossentropy", from_logits=True)
    cutout = Cutout(length=8)

    try:
        with pytest.raises(ValueError):
            _ = Maxup(estimator=classifier, augmentations=[])

        with pytest.raises(ValueError):
            _ = Maxup(estimator=classifier, augmentations=cutout, num_trials=0)

        with pytest.raises(ValueError):
            _ = Maxup(estimator=classifier, augmentations=cutout, num_trials=-1)

    except ARTTestException as e:
        art_warning(e)
