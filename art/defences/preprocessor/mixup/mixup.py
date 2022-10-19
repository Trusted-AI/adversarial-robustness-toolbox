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
"""
This module implements the Mixup data augmentation defence.

| Paper link: https://arxiv.org/abs/1710.09412

| Please keep in mind the limitations of defences. For more information on the limitations of this defence,
    see https://arxiv.org/abs/1803.09868 . For details on how to evaluate classifier security in general, see
    https://arxiv.org/abs/1902.06705
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Tuple

import numpy as np

from art.defences.preprocessor.preprocessor import Preprocessor
from art.utils import check_and_transform_label_format

logger = logging.getLogger(__name__)


class Mixup(Preprocessor):
    """
    Implement the Mixup data augmentation defence approach.

    | Paper link: https://arxiv.org/abs/1710.09412

    | Please keep in mind the limitations of defences. For more information on the limitations of this defence,
        see https://arxiv.org/abs/1803.09868 . For details on how to evaluate classifier security in general, see
        https://arxiv.org/abs/1902.06705
    """

    params = ["num_classes", "num_mix", "alpha"]

    def __init__(
        self,
        num_classes: int,
        num_mix: int = 2,
        alpha: float = 1.0,
        apply_fit: bool = False,
        apply_predict: bool = True,
    ) -> None:
        """
        Create an instance of a Mixup data augmentation object.

        :param num_classes: The number of classes used for one-hot encoding.
        :param num_samples: The number of samples to mix.
        :param alpha: The mixing factor parameter for drawing from the Dirichlet distribution.
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        """
        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
        self.num_classes = num_classes
        self.num_mix = num_mix
        self.alpha = alpha
        self._check_params()

    def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply Mixup data augmentation to samples `x` and labels `y`. The returned labels will be categorical
        probability vectors rather than integer labels.

        :param x: Sample to augment with shape `(length, channel)` or an array of sample arrays with shape
                  (length,) or (length, channel).
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :return: Data augmented sample.
        :raises `ValueError`: If no labels are provided.
        """
        if y is None:
            raise ValueError("Labels `y` cannot be None.")

        n = x.shape[0]

        # convert labels to one-hot encoding
        y_one_hot = check_and_transform_label_format(y, self.num_classes, return_one_hot=True)

        # generate the mixing factor from the Dirichlet distribution
        lmbs = np.random.dirichlet([self.alpha] * self.num_mix)

        # randomly draw indices for samples to mix
        indices = [np.random.permutation(n) for _ in range(self.num_mix)]

        x_aug = sum(lmb * x[i] for lmb, i in zip(lmbs, indices))
        y_aug = sum(lmb * y_one_hot[i] for lmb, i in zip(lmbs, indices))

        return x_aug, y_aug

    def _check_params(self) -> None:
        if self.num_classes <= 0:
            raise ValueError("Number of classes must be positive")

        if self.num_mix < 2:
            raise ValueError("Number of samples to mix must be at least 2.")

        if self.alpha <= 0:
            raise ValueError("Mixing factor parameter must be positive.")
