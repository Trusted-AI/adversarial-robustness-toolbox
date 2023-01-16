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
This module implements the Mixup data augmentation defence in TensorFlow.

| Paper link: https://arxiv.org/abs/1710.09412

| Please keep in mind the limitations of defences. For more information on the limitations of this defence,
    see https://arxiv.org/abs/1803.09868 . For details on how to evaluate classifier security in general, see
    https://arxiv.org/abs/1902.06705
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np

from art.defences.preprocessor.preprocessor import PreprocessorTensorFlowV2

if TYPE_CHECKING:
    # pylint: disable=C0412
    import tensorflow as tf

logger = logging.getLogger(__name__)


class MixupTensorFlowV2(PreprocessorTensorFlowV2):
    """
    Implement the Mixup data augmentation defence approach in TensorFlow v2.

    | Paper link: https://arxiv.org/abs/1710.09412

    | Please keep in mind the limitations of defences. For more information on the limitations of this defence,
        see https://arxiv.org/abs/1803.09868 . For details on how to evaluate classifier security in general, see
        https://arxiv.org/abs/1902.06705
    """

    params = ["num_classes", "alpha", "num_mix"]

    def __init__(
        self,
        num_classes: int,
        alpha: float = 1.0,
        num_mix: int = 2,
        apply_fit: bool = True,
        apply_predict: bool = False,
    ) -> None:
        """
        Create an instance of a Mixup data augmentation object.

        :param num_classes: The number of classes used for one-hot encoding.
        :param alpha: The hyperparameter for the mixing interpolation strength.
        :param num_mix: The number of samples to mix for k-way Mixup.
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        :param device_type: Type of device on which the classifier is run, either `gpu` or `cpu`.
        """
        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
        self.num_classes = num_classes
        self.alpha = alpha
        self.num_mix = num_mix
        self._check_params()

    def forward(self, x: "tf.Tensor", y: Optional["tf.Tensor"] = None) -> Tuple["tf.Tensor", Optional["tf.Tensor"]]:
        """
        Apply Mixup data augmentation to feature data `x` and labels `y`.

        :param x: Feature data to augment with shape `(batch_size, ...)`.
        :param y: Labels of `x` either one-hot or multi-hot encoded of shape `(nb_samples, nb_classes)`
                  or class indices of shape `(nb_samples,)`.
        :return: Data augmented sample. The returned labels will be probability vectors of shape
                 `(nb_samples, nb_classes)`.
        :raises `ValueError`: If no labels are provided.
        """
        import tensorflow as tf  # lgtm [py/repeated-import]

        if y is None:
            raise ValueError("Labels `y` cannot be None.")

        # convert labels to one-hot encoding
        if len(y.shape) == 2:
            y_one_hot = y
        elif len(y.shape) == 1:
            y_one_hot = tf.one_hot(y, self.num_classes, on_value=1.0, off_value=0.0)
        else:
            raise ValueError(
                "Shape of labels not recognised. "
                "Please provide labels in shape (nb_samples,) or (nb_samples, nb_classes)"
            )

        n = x.shape[0]

        # sample the mixing factor from the Dirichlet distribution
        lmbs = np.random.dirichlet([self.alpha] * self.num_mix)

        x_aug = lmbs[0] * x
        y_aug = lmbs[0] * y_one_hot
        for lmb in lmbs[1:]:
            # randomly choose indices for samples to mix
            indices = tf.random.shuffle(tf.range(n))
            x_aug = x_aug + lmb * tf.gather(x, indices, axis=None)
            y_aug = y_aug + lmb * tf.gather(y_one_hot, indices, axis=None)

        return x_aug, y_aug

    def _check_params(self) -> None:
        if self.num_classes <= 0:
            raise ValueError("The number of classes must be positive")

        if self.alpha <= 0:
            raise ValueError("The mixing interpolation strength must be positive.")

        if self.num_mix < 2:
            raise ValueError("The number of samples to mix must be at least 2.")
