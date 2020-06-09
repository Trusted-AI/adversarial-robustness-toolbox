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
"""
This module implements ShapeShifter, a robust physical adversarial attack on Faster R-CNN object detector.

| Paper link: https://arxiv.org/abs/1804.05810
"""

import logging

import numpy as np

from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.object_detection.object_detector import ObjectDetectorMixin
from art.utils import Deprecated, deprecated_keyword_arg

logger = logging.getLogger(__name__)


class ShapeShifter(EvasionAttack):
    """
    Implementation of the ShapeShifter attack. This is a robust physical adversarial attack on Faster R-CNN object
    detector and is developed in TensorFlow.

    | Paper link: https://arxiv.org/abs/1804.05810
    """

    attack_params = EvasionAttack.attack_params + [
        "learning_rate",
        "batch_size",
    ]

    _estimator_requirements = (BaseEstimator, LossGradientsMixin, ObjectDetectorMixin)

    def __init__(self, estimator, learning_rate=5.0, batch_size=1):
        """
        Create an instance of the :class:`.ShapeShifter`.

        :param estimator: A trained object detector.
        :type estimator: :class:`.ObjectDetectorMixin`
        :param learning_rate: The learning rate of the optimization.
        :type learning_rate: `float`
        :param batch_size: The size of the training batch.
        :type batch_size: `int`
        """
        super(ShapeShifter, self).__init__(estimator=estimator)

        kwargs = {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
        }
        self.set_params(**kwargs)

    def generate(self, x, y=None, **kwargs):
        """
        Generate adversarial samples and return them in an array.

        :param x: Sample images.
        :type x: `np.ndarray`
        :param y: Target labels for object detector.
        :type y: `np.ndarray`
        :return: Adversarial patch.
        :rtype: `np.ndarray`
        """

        channel_index = 1 if self.estimator.channels_first else x.ndim - 1
        assert (
            x.shape[channel_index] == self.patch_shape[channel_index - 1]
        ), "The channels_first boolean of the estimator and the patch have to be identical."

        assert y is None, "The DPatch attack does not use target labels."

        assert x.ndim == 4, "The adversarial patch can only be applied to images."


        return self._patch

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param learning_rate: The learning rate of the optimization.
        :type learning_rate: `float`
        :param batch_size: The size of the training batch.
        :type batch_size: `int`
        """
        super(ShapeShifter, self).set_params(**kwargs)

        if not isinstance(self.learning_rate, float):
            raise ValueError("The learning rate must be of type float.")
        if not self.learning_rate > 0.0:
            raise ValueError("The learning rate must be greater than 0.0.")

        if not isinstance(self.batch_size, int):
            raise ValueError("The batch size must be of type int.")
        if not self.batch_size > 0:
            raise ValueError("The batch size must be greater than 0.")
