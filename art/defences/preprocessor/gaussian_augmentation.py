# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
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
This module implements the Gaussian augmentation defence in `GaussianAugmentation`.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np

from art.config import ART_NUMPY_DTYPE
from art.defences.preprocessor.preprocessor import Preprocessor

if TYPE_CHECKING:
    from art.utils import CLIP_VALUES_TYPE

logger = logging.getLogger(__name__)


class GaussianAugmentation(Preprocessor):
    """
    Add Gaussian noise to a dataset in one of two ways: either add noise to each sample (keeping the size of the
    original dataset) or perform augmentation by keeping all original samples and adding noisy counterparts. When used
    as part of a :class:`.Classifier` instance, the defense will be applied automatically only when training if
    `augmentation` is true, and only when performing prediction otherwise.
    """

    params = [
        "sigma",
        "augmentation",
        "ratio",
        "clip_values",
        "_apply_fit",
        "_apply_predict",
    ]

    def __init__(
        self,
        sigma: float = 1.0,
        augmentation: bool = True,
        ratio: float = 1.0,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        apply_fit: bool = True,
        apply_predict: bool = False,
    ):
        """
        Initialize a Gaussian augmentation object.

        :param sigma: Standard deviation of Gaussian noise to be added.
        :param augmentation: If true, perform dataset augmentation using `ratio`, otherwise replace samples with noisy
                            counterparts.
        :param ratio: Percentage of data augmentation. E.g. for a rate of 1, the size of the dataset will double.
                      If `augmentation` is false, `ratio` value is ignored.
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        """
        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
        if augmentation and not apply_fit and apply_predict:
            raise ValueError(
                "If `augmentation` is `True`, then `apply_fit` must be `True` and `apply_predict` must be `False`."
            )
        if augmentation and not (apply_fit or apply_predict):
            raise ValueError("If `augmentation` is `True`, then `apply_fit` and `apply_predict` can't be both `False`.")

        self.sigma = sigma
        self.augmentation = augmentation
        self.ratio = ratio
        self.clip_values = clip_values
        self._check_params()

    def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Augment the sample `(x, y)` with Gaussian noise. The result is either an extended dataset containing the
        original sample, as well as the newly created noisy samples (augmentation=True) or just the noisy counterparts
        to the original samples.

        :param x: Sample to augment with shape `(batch_size, width, height, depth)`.
        :param y: Labels for the sample. If this argument is provided, it will be augmented with the corresponded
                  original labels of each sample point.
        :return: The augmented dataset and (if provided) corresponding labels.
        """
        logger.info("Original dataset size: %d", x.shape[0])

        # Select indices to augment
        if self.augmentation:
            size = int(x.shape[0] * self.ratio)
            indices = np.random.randint(0, x.shape[0], size=size)

            # Generate noisy samples
            x_aug = np.random.normal(x[indices], scale=self.sigma, size=(size,) + x.shape[1:]).astype(ART_NUMPY_DTYPE)
            x_aug = np.vstack((x, x_aug))
            if y is not None:
                y_aug = np.concatenate((y, y[indices]))
            else:
                y_aug = y
            logger.info("Augmented dataset size: %d", x_aug.shape[0])
        else:
            x_aug = np.random.normal(x, scale=self.sigma, size=x.shape).astype(ART_NUMPY_DTYPE)
            y_aug = y
            logger.info("Created %i samples with Gaussian noise.")

        if self.clip_values is not None:
            x_aug = np.clip(x_aug, self.clip_values[0], self.clip_values[1])

        return x_aug, y_aug

    def _check_params(self) -> None:
        if self.augmentation and self.ratio <= 0:
            raise ValueError("The augmentation ratio must be positive.")

        if self.clip_values is not None:

            if len(self.clip_values) != 2:
                raise ValueError(
                    "`clip_values` should be a tuple of 2 floats or arrays containing the allowed data range."
                )
            if np.array(self.clip_values[0] >= self.clip_values[1]).any():
                raise ValueError("Invalid `clip_values`: min >= max.")
