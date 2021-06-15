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
This module implements the black-box attack `SimBA`.

| Paper link: https://arxiv.org/abs/1905.07121
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, TYPE_CHECKING

import numpy as np
from scipy.fftpack import idct

from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.config import ART_NUMPY_DTYPE

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)


class SimBA(EvasionAttack):
    """
    This class implements the black-box attack `SimBA`.

    | Paper link: https://arxiv.org/abs/1905.07121
    """

    attack_params = EvasionAttack.attack_params + [
        "attack",
        "max_iter",
        "epsilon",
        "order",
        "freq_dim",
        "stride",
        "targeted",
        "batch_size",
    ]

    _estimator_requirements = (BaseEstimator, ClassifierMixin, NeuralNetworkMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_TYPE",
        attack: str = "dct",
        max_iter: int = 3000,
        order: str = "random",
        epsilon: float = 0.1,
        freq_dim: int = 4,
        stride: int = 1,
        targeted: bool = False,
        batch_size: int = 1,
    ):
        """
        Create a SimBA (dct) attack instance.

        :param classifier: A trained classifier.
        :param attack: attack type: pixel (px) or DCT (dct) attacks
        :param max_iter: The maximum number of iterations.
        :param epsilon: Overshoot parameter.
        :param order: order of pixel attacks: random or diagonal (diag)
        :param freq_dim: dimensionality of 2D frequency space (DCT).
        :param stride: stride for block order (DCT).
        :param targeted: perform targeted attack
        :param batch_size: Batch size (but, batch process unavailable in this implementation)
        """
        super().__init__(estimator=classifier)

        self.attack = attack
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.order = order
        self.freq_dim = freq_dim
        self.stride = stride
        self._targeted = targeted
        self.batch_size = batch_size
        self._check_params()

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :param y: An array with the original labels to be predicted.
        :return: An array holding the adversarial examples.
        """
        x = x.astype(ART_NUMPY_DTYPE)
        preds = self.estimator.predict(x, batch_size=self.batch_size)

        if self.estimator.nb_classes == 2 and preds.shape[1] == 1:
            raise ValueError(
                "This attack has not yet been tested for binary classification with a single output classifier."
            )

        if divmod(x.shape[2] - self.freq_dim, self.stride)[1] != 0:
            raise ValueError(
                "Incompatible value combination in image height/width, freq_dim and stride detected. "
                "Adapt these parameters to fulfill the following conditions: "
                "divmod(image_height - freq_dim, stride)[1] == 0 "
                "and "
                "divmod(image_width - freq_dim, stride)[1] == 0"
            )

        if y is None:
            if self.targeted:
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")

            # Use model predictions as correct outputs
            logger.info("Using the model prediction as the correct label for SimBA.")
            y_i = np.argmax(preds, axis=1)
        else:
            y_i = np.argmax(y, axis=1)

        desired_label = y_i[0]
        current_label = np.argmax(preds, axis=1)[0]
        last_prob = preds.reshape(-1)[desired_label]

        if self.estimator.channels_first:
            nb_channels = x.shape[1]
        else:
            nb_channels = x.shape[3]

        n_dims = np.prod(x.shape)

        if self.attack == "px":
            if self.order == "diag":
                indices = self.diagonal_order(x.shape[2], nb_channels)[: self.max_iter]
            elif self.order == "random":
                indices = np.random.permutation(n_dims)[: self.max_iter]
            indices_size = len(indices)
            while indices_size < self.max_iter:
                if self.order == "diag":
                    tmp_indices = self.diagonal_order(x.shape[2], nb_channels)
                elif self.order == "random":
                    tmp_indices = np.random.permutation(n_dims)
                indices = np.hstack((indices, tmp_indices))[: self.max_iter]
                indices_size = len(indices)
        elif self.attack == "dct":
            indices = self._block_order(x.shape[2], nb_channels, initial_size=self.freq_dim, stride=self.stride)[
                : self.max_iter
            ]
            indices_size = len(indices)
            while indices_size < self.max_iter:
                tmp_indices = self._block_order(x.shape[2], nb_channels, initial_size=self.freq_dim, stride=self.stride)
                indices = np.hstack((indices, tmp_indices))[: self.max_iter]
                indices_size = len(indices)

            def trans(var_z):
                return self._block_idct(var_z, block_size=x.shape[2])

        clip_min = -np.inf
        clip_max = np.inf
        if self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values

        term_flag = 1
        if self.targeted:
            if desired_label != current_label:
                term_flag = 0
        else:
            if desired_label == current_label:
                term_flag = 0

        nb_iter = 0
        while term_flag == 0 and nb_iter < self.max_iter:
            diff = np.zeros(n_dims).astype(ART_NUMPY_DTYPE)
            diff[indices[nb_iter]] = self.epsilon

            if self.attack == "dct":
                left_preds = self.estimator.predict(
                    np.clip(x - trans(diff.reshape(x.shape)), clip_min, clip_max), batch_size=self.batch_size
                )
            elif self.attack == "px":
                left_preds = self.estimator.predict(
                    np.clip(x - diff.reshape(x.shape), clip_min, clip_max), batch_size=self.batch_size
                )
            left_prob = left_preds.reshape(-1)[desired_label]

            if self.attack == "dct":
                right_preds = self.estimator.predict(
                    np.clip(x + trans(diff.reshape(x.shape)), clip_min, clip_max), batch_size=self.batch_size
                )
            elif self.attack == "px":
                right_preds = self.estimator.predict(
                    np.clip(x + diff.reshape(x.shape), clip_min, clip_max), batch_size=self.batch_size
                )
            right_prob = right_preds.reshape(-1)[desired_label]

            # Use (2 * int(self.targeted) - 1) to shorten code?
            if self.targeted:
                if left_prob > last_prob:
                    if left_prob > right_prob:
                        if self.attack == "dct":
                            x = np.clip(x - trans(diff.reshape(x.shape)), clip_min, clip_max)
                        elif self.attack == "px":
                            x = np.clip(x - diff.reshape(x.shape), clip_min, clip_max)
                        last_prob = left_prob
                        current_label = np.argmax(left_preds, axis=1)[0]
                    else:
                        if self.attack == "dct":
                            x = np.clip(x + trans(diff.reshape(x.shape)), clip_min, clip_max)
                        elif self.attack == "px":
                            x = np.clip(x + diff.reshape(x.shape), clip_min, clip_max)
                        last_prob = right_prob
                        current_label = np.argmax(right_preds, axis=1)[0]
                else:
                    if right_prob > last_prob:
                        if self.attack == "dct":
                            x = np.clip(x + trans(diff.reshape(x.shape)), clip_min, clip_max)
                        elif self.attack == "px":
                            x = np.clip(x + diff.reshape(x.shape), clip_min, clip_max)
                        last_prob = right_prob
                        current_label = np.argmax(right_preds, axis=1)[0]
            else:
                if left_prob < last_prob:
                    if left_prob < right_prob:
                        if self.attack == "dct":
                            x = np.clip(x - trans(diff.reshape(x.shape)), clip_min, clip_max)
                        elif self.attack == "px":
                            x = np.clip(x - diff.reshape(x.shape), clip_min, clip_max)
                        last_prob = left_prob
                        current_label = np.argmax(left_preds, axis=1)[0]
                    else:
                        if self.attack == "dct":
                            x = np.clip(x + trans(diff.reshape(x.shape)), clip_min, clip_max)
                        elif self.attack == "px":
                            x = np.clip(x + diff.reshape(x.shape), clip_min, clip_max)
                        last_prob = right_prob
                        current_label = np.argmax(right_preds, axis=1)[0]
                else:
                    if right_prob < last_prob:
                        if self.attack == "dct":
                            x = np.clip(x + trans(diff.reshape(x.shape)), clip_min, clip_max)
                        elif self.attack == "px":
                            x = np.clip(x + diff.reshape(x.shape), clip_min, clip_max)
                        last_prob = right_prob
                        current_label = np.argmax(right_preds, axis=1)[0]

            if self.targeted:
                if desired_label == current_label:
                    term_flag = 1
            else:
                if desired_label != current_label:
                    term_flag = 1

            nb_iter = nb_iter + 1

        if nb_iter < self.max_iter:
            logger.info("SimBA (%s) %s attack succeed", self.attack, ["non-targeted", "targeted"][int(self.targeted)])
        else:
            logger.info("SimBA (%s) %s attack failed", self.attack, ["non-targeted", "targeted"][int(self.targeted)])

        return x

    def _check_params(self) -> None:

        if not isinstance(self.max_iter, (int, np.int)) or self.max_iter <= 0:
            raise ValueError("The number of iterations must be a positive integer.")

        if self.epsilon < 0:
            raise ValueError("The overshoot parameter must not be negative.")

        if self.batch_size != 1:
            raise ValueError("The batch size `batch_size` has to be 1 in this implementation.")

        if not isinstance(self.stride, (int, np.int)) or self.stride <= 0:
            raise ValueError("The `stride` value must be a positive integer.")

        if not isinstance(self.freq_dim, (int, np.int)) or self.freq_dim <= 0:
            raise ValueError("The `freq_dim` value must be a positive integer.")

        if self.order != "random" and self.order != "diag":
            raise ValueError("The order of pixel attacks has to be `random` or `diag`.")

        if self.attack != "px" and self.attack != "dct":
            raise ValueError("The attack type has to be `px` or `dct`.")

        if not isinstance(self.targeted, int) or (self.targeted != 0 and self.targeted != 1):
            raise ValueError("`targeted` has to be a logical value.")

    def _block_order(self, img_size, channels, initial_size=2, stride=1):
        """
        Defines a block order, starting with top-left (initial_size x initial_size) submatrix
        expanding by stride rows and columns whenever exhausted
        randomized within the block and across channels.
        e.g. (initial_size=2, stride=1)
        [1, 3, 6]
        [2, 4, 9]
        [5, 7, 8]

        :param img_size: image size (i.e., width or height).
        :param channels: the number of channels.
        :param initial size: initial size for submatrix.
        :param stride: stride size for expansion.

        :return order: An array holding the block order of DCT attacks.
        """
        order = np.zeros((channels, img_size, img_size)).astype(ART_NUMPY_DTYPE)
        total_elems = channels * initial_size * initial_size
        perm = np.random.permutation(total_elems)
        order[:, :initial_size, :initial_size] = perm.reshape((channels, initial_size, initial_size))
        for i in range(initial_size, img_size, stride):
            num_elems = channels * (2 * stride * i + stride * stride)
            perm = np.random.permutation(num_elems) + total_elems
            num_first = channels * stride * (stride + i)
            order[:, : (i + stride), i : (i + stride)] = perm[:num_first].reshape((channels, -1, stride))
            order[:, i : (i + stride), :i] = perm[num_first:].reshape((channels, stride, -1))
            total_elems += num_elems

        if self.estimator.channels_first:
            return order.reshape(1, -1).squeeze().argsort()

        return order.transpose((1, 2, 0)).reshape(1, -1).squeeze().argsort()

    def _block_idct(self, x, block_size=8, masked=False, ratio=0.5):
        """
        Applies IDCT to each block of size block_size.

        :param x: An array with the inputs to be attacked.
        :param block_size: block size for DCT attacks.
        :param masked: use the mask.
        :param ratio: Ratio of the lowest frequency directions in order to make the adversarial perturbation in the low
                      frequency space.

        :return var_z: An array holding the order of DCT attacks.
        """
        if not self.estimator.channels_first:
            x = x.transpose(0, 3, 1, 2)
        var_z = np.zeros(x.shape).astype(ART_NUMPY_DTYPE)
        num_blocks = int(x.shape[2] / block_size)
        mask = np.zeros((x.shape[0], x.shape[1], block_size, block_size))
        if not isinstance(ratio, float):
            for i in range(x.shape[0]):
                mask[i, :, : int(block_size * ratio[i]), : int(block_size * ratio[i])] = 1
        else:
            mask[:, :, : int(block_size * ratio), : int(block_size * ratio)] = 1
        for i in range(num_blocks):
            for j in range(num_blocks):
                submat = x[:, :, (i * block_size) : ((i + 1) * block_size), (j * block_size) : ((j + 1) * block_size)]
                if masked:
                    submat = submat * mask
                var_z[
                    :, :, (i * block_size) : ((i + 1) * block_size), (j * block_size) : ((j + 1) * block_size)
                ] = idct(idct(submat, axis=3, norm="ortho"), axis=2, norm="ortho")

        if self.estimator.channels_first:
            return var_z

        return var_z.transpose((0, 2, 3, 1))

    def diagonal_order(self, image_size, channels):
        """
        Defines a diagonal order for pixel attacks.
        order is fixed across diagonals but are randomized across channels and within the diagonal
        e.g.
        [1, 2, 5]
        [3, 4, 8]
        [6, 7, 9]

        :param image_size: image size (i.e., width or height)
        :param channels: the number of channels

        :return order: An array holding the diagonal order of pixel attacks.
        """
        x = np.arange(0, image_size).cumsum()
        order = np.zeros((image_size, image_size)).astype(ART_NUMPY_DTYPE)
        for i in range(image_size):
            order[i, : (image_size - i)] = i + x[i:]
        for i in range(1, image_size):
            reverse = order[image_size - i - 1].take([i for i in range(i - 1, -1, -1)])  # pylint: disable=R1721
            order[i, (image_size - i) :] = image_size * image_size - 1 - reverse
        if channels > 1:
            order_2d = order
            order = np.zeros((channels, image_size, image_size))
            for i in range(channels):
                order[i, :, :] = 3 * order_2d + i

        if self.estimator.channels_first:
            return order.reshape(1, -1).squeeze().argsort()

        return order.transpose((1, 2, 0)).reshape(1, -1).squeeze().argsort()
