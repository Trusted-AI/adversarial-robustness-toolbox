# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2023
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

from art.estimators.certification.object_seeker import PyTorchObjectSeeker
from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.mark.only_with_platform("pytorch")
def test_pytorch_train(art_warning, get_pytorch_faster_rcnn):
    object_detector, x_test, y_test = get_pytorch_faster_rcnn
    object_seeker = PyTorchObjectSeeker(
        model=object_detector.model,
        input_shape=object_detector.input_shape,
        channels_first=object_detector.channels_first,
        optimizer=object_detector.optimizer,
        clip_values=object_detector.clip_values,
        attack_losses=object_detector.attack_losses,
        detector_type="Faster-RCNN",
        num_lines=3,
        confidence_threshold=0.3,
        iou_threshold=0.4,
        prune_threshold=0.5,
        device_type="cpu",
    )

    try:
        # Compute loss before training
        loss1 = object_seeker.compute_loss(x=x_test, y=y_test)

        # Train for one epoch
        object_seeker.fit(x_test, y_test, nb_epochs=1)

        # Compute loss after training
        loss2 = object_seeker.compute_loss(x=x_test, y=y_test)

        assert loss1 != loss2

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_pytorch_predict(art_warning, get_pytorch_faster_rcnn):
    object_detector, x_test, _ = get_pytorch_faster_rcnn
    object_seeker = PyTorchObjectSeeker(
        model=object_detector.model,
        input_shape=object_detector.input_shape,
        channels_first=object_detector.channels_first,
        optimizer=object_detector.optimizer,
        clip_values=object_detector.clip_values,
        attack_losses=object_detector.attack_losses,
        detector_type="Faster-RCNN",
        num_lines=3,
        confidence_threshold=0.3,
        iou_threshold=0.4,
        prune_threshold=0.5,
        device_type="cpu",
    )

    try:
        result = object_seeker.predict(x=x_test)

        assert len(result) == len(x_test)
        assert list(result[0].keys()) == ["boxes", "labels", "scores"]
        assert np.all(result[0]["scores"] >= 0.3)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("pytorch")
def test_pytorch_certify(art_warning, get_pytorch_faster_rcnn):
    object_detector, x_test, _ = get_pytorch_faster_rcnn
    object_seeker = PyTorchObjectSeeker(
        model=object_detector.model,
        input_shape=object_detector.input_shape,
        channels_first=object_detector.channels_first,
        optimizer=object_detector.optimizer,
        clip_values=object_detector.clip_values,
        attack_losses=object_detector.attack_losses,
        detector_type="Faster-RCNN",
        num_lines=3,
        confidence_threshold=0.3,
        iou_threshold=0.4,
        prune_threshold=0.5,
        device_type="cpu",
    )

    try:
        result = object_seeker.certify(x=x_test, patch_size=0.01, offset=0.1)

        assert len(result) == len(x_test)
        assert np.any(result[0])

    except ARTTestException as e:
        art_warning(e)
