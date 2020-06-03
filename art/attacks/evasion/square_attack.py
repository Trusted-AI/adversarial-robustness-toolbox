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
This module implements the `SquareAttack` attack.

| Paper link: https://arxiv.org/abs/1912.00049
"""
import logging
import math
import bisect
import random

import numpy as np

from art.config import ART_NUMPY_DTYPE
from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator
from art.utils import check_and_transform_label_format

logger = logging.getLogger(__name__)


class SquareAttack(EvasionAttack):

    attack_params = EvasionAttack.attack_params + [
        "norm",
        "max_iter",
        "eps",
        "p_init",
        "nb_restarts",
    ]

    _estimator_requirements = (BaseEstimator,)

    def __init__(self, estimator, norm=np.inf, max_iter=100, eps=0.3, p_init=0.8, nb_restarts=1):
        super().__init__(estimator=estimator)

        kwargs = {
            "norm": norm,
            "max_iter": max_iter,
            "eps": eps,
            "p_init": p_init,
            "nb_restarts": nb_restarts,
        }
        self.set_params(**kwargs)

    def get_logits_diff(self, x, y):
        y_pred = self.estimator.predict(x)

        logit_correct = np.take_along_axis(y_pred, np.expand_dims(np.argmax(y, axis=1), axis=1), axis=1)
        logit_highest_incorrect = np.take_along_axis(
            y_pred, np.expand_dims(np.argsort(y_pred, axis=1)[:, -2], axis=1), axis=1
        )

        return (logit_correct - logit_highest_incorrect)[:, 0]

    def _get_p(self, i_iter):

        i_p = i_iter / self.max_iter

        intervals = [0.001, 0.005, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
        p_ratio = [1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 256, 1 / 512]

        i_ratio = bisect.bisect_left(intervals, i_p)

        return self.p_init * p_ratio[i_ratio]

    def generate(self, x, y=None, **kwargs):
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :type y: `np.ndarray`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        if x.ndim != 4:
            raise ValueError(
                "Unrecognized input dimension. Attack can only be applied to image data."
            )

        x_adv = x.astype(ART_NUMPY_DTYPE)

        y = check_and_transform_label_format(y, self.estimator.nb_classes)

        if self.estimator.channels_first:
            channels = x.shape[1]
            height = x.shape[2]
            width = x.shape[3]
        else:
            height = x.shape[1]
            width = x.shape[2]
            channels = x.shape[3]

        for i_restart in range(self.nb_restarts):

            # Determine correctly predicted samples
            y_pred = self.estimator.predict(x_adv)
            sample_is_robust = np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)

            if np.sum(sample_is_robust) == 0:
                break

            # x_robust = x_adv[sample_is_robust]
            x_robust = x[sample_is_robust]
            y_robust = y[sample_is_robust]
            sample_logits_diff_init = self.get_logits_diff(x_robust, y_robust)

            if self.norm == np.inf:

                if self.estimator.channels_first:
                    size = (x_robust.shape[0], channels, 1, width)
                else:
                    size = (x_robust.shape[0], 1, width, channels)

                # Add vertical stripe perturbations
                x_robust_new = np.clip(
                    x_robust + self.eps * np.random.choice([-1, 1], size=size),
                    a_min=self.estimator.clip_values[0],
                    a_max=self.estimator.clip_values[1],
                )

                sample_logits_diff_new = self.get_logits_diff(x_robust_new, y_robust)
                logits_diff_improved = (sample_logits_diff_new - sample_logits_diff_init) < 0.0

                x_robust[logits_diff_improved] = x_robust_new[logits_diff_improved]

                x_adv[sample_is_robust] = x_robust

                for i_iter in range(self.max_iter):

                    p = self._get_p(i_iter)

                    # Determine correctly predicted samples
                    y_pred = self.estimator.predict(x_adv)
                    sample_is_robust = np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)

                    if np.sum(sample_is_robust) == 0:
                        break

                    x_robust = x_adv[sample_is_robust]
                    x_init = x[sample_is_robust]
                    y_robust = y[sample_is_robust]

                    sample_logits_diff_init = self.get_logits_diff(x_robust, y_robust)

                    h_tile = max(int(round(math.sqrt(p * height * width))), 1)

                    h_mid = np.random.randint(0, height - h_tile)
                    w_start = np.random.randint(0, width - h_tile)

                    delta_new = np.zeros(self.estimator.input_shape)

                    if self.estimator.channels_first:
                        delta_new[:, h_mid : h_mid + h_tile, w_start : w_start + h_tile] = np.random.choice(
                            [-2 * self.eps, 2 * self.eps], size=[channels, 1, 1]
                        )
                    else:
                        delta_new[h_mid : h_mid + h_tile, w_start : w_start + h_tile, :] = np.random.choice(
                            [-2 * self.eps, 2 * self.eps], size=[1, 1, channels]
                        )

                    x_robust_new = x_robust + delta_new

                    x_robust_new = np.minimum(np.maximum(x_robust_new, x_init - self.eps), x_init + self.eps)

                    x_robust_new = np.clip(
                        x_robust_new, a_min=self.estimator.clip_values[0], a_max=self.estimator.clip_values[1]
                    )

                    sample_logits_diff_new = self.get_logits_diff(x_robust_new, y_robust)
                    logits_diff_improved = (sample_logits_diff_new - sample_logits_diff_init) < 0.0

                    x_robust[logits_diff_improved] = x_robust_new[logits_diff_improved]

                    x_adv[sample_is_robust] = x_robust

            elif self.norm == 2:

                n_tiles = 5

                h_tile = height // n_tiles

                def _get_perturbation(h):
                    delta = np.zeros([h, h])
                    gaussian_perturbation = np.zeros([h // 2, h])

                    x_c = h // 4
                    y_c = h // 2

                    for i_y in range(y_c):
                        gaussian_perturbation[
                            max(x_c, 0) : min(x_c + (2 * i_y + 1), h // 2), max(0, y_c) : min(y_c + (2 * i_y + 1), h)
                        ] += 1.0 / ((i_y + 1) ** 2)
                        x_c -= 1
                        y_c -= 1

                    gaussian_perturbation /= np.sqrt(np.sum(gaussian_perturbation ** 2))

                    delta[: h // 2] = gaussian_perturbation
                    delta[h // 2 : h // 2 + gaussian_perturbation.shape[0]] = -gaussian_perturbation

                    delta /= np.sqrt(np.sum(delta ** 2))

                    if random.random() > 0.5:
                        delta = np.transpose(delta)

                    if random.random() > 0.5:
                        delta = -delta

                    return delta

                delta_init = np.zeros(x_robust.shape)

                h_start = 0
                for _ in range(n_tiles):
                    w_start = 0
                    for _ in range(n_tiles):
                        if self.estimator.channels_first:
                            perturbation_size = (1, 1, h_tile, h_tile)
                            random_size = (x_robust.shape[0], channels, 1, 1)
                        else:
                            perturbation_size = (1, h_tile, h_tile, 1)
                            random_size = (x_robust.shape[0], 1, 1, channels)

                        perturbation = _get_perturbation(h_tile).reshape(perturbation_size) * np.random.choice(
                            [-1, 1], size=random_size
                        )

                        if self.estimator.channels_first:
                            delta_init[:, :, h_start : h_start + h_tile, w_start : w_start + h_tile] += perturbation
                        else:
                            delta_init[:, h_start : h_start + h_tile, w_start : w_start + h_tile, :] += perturbation
                        w_start += h_tile
                    h_start += h_tile

                x_robust_new = np.clip(
                    x_robust + delta_init / np.sqrt(np.sum(delta_init ** 2, axis=(1, 2, 3), keepdims=True)) * self.eps,
                    self.estimator.clip_values[0],
                    self.estimator.clip_values[1],
                )

                sample_logits_diff_new = self.get_logits_diff(x_robust_new, y_robust)
                logits_diff_improved = (sample_logits_diff_new - sample_logits_diff_init) < 0.0

                x_robust[logits_diff_improved] = x_robust_new[logits_diff_improved]

                x_adv[sample_is_robust] = x_robust

                for i_iter in range(self.max_iter):

                    p = self._get_p(i_iter)

                    # Determine correctly predicted samples
                    y_pred = self.estimator.predict(x_adv)
                    sample_is_robust = np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)

                    if np.sum(sample_is_robust) == 0:
                        break

                    x_robust = x_adv[sample_is_robust]
                    x_init = x[sample_is_robust]
                    y_robust = y[sample_is_robust]

                    sample_logits_diff_init = self.get_logits_diff(x_robust, y_robust)

                    delta_x_robust_init = x_robust - x_init

                    h_tile = max(int(round(math.sqrt(p * height * width))), 3)

                    if h_tile % 2 == 0:
                        h_tile += 1
                    h_tile_2 = h_tile

                    h_start = np.random.randint(0, height - h_tile)
                    w_start = np.random.randint(0, width - h_tile)

                    new_deltas_mask = np.zeros(x_init.shape)
                    if self.estimator.channels_first:
                        new_deltas_mask[:, :, h_start : h_start + h_tile, w_start : w_start + h_tile] = 1.0
                        W_1_norm = np.sqrt(
                            np.sum(
                                delta_x_robust_init[:, :, h_start : h_start + h_tile, w_start : w_start + h_tile] ** 2,
                                axis=(2, 3),
                                keepdims=True,
                            )
                        )
                    else:
                        new_deltas_mask[:, h_start : h_start + h_tile, w_start : w_start + h_tile, :] = 1.0
                        W_1_norm = np.sqrt(
                            np.sum(
                                delta_x_robust_init[:, h_start : h_start + h_tile, w_start : w_start + h_tile, :] ** 2,
                                axis=(1, 2),
                                keepdims=True,
                            )
                        )

                    h_2_start = np.random.randint(0, height - h_tile_2)
                    w_2_start = np.random.randint(0, width - h_tile_2)

                    new_deltas_mask_2 = np.zeros(x_init.shape)
                    if self.estimator.channels_first:
                        new_deltas_mask_2[
                            :, :, h_2_start : h_2_start + h_tile_2, w_2_start : w_2_start + h_tile_2
                        ] = 1.0
                    else:
                        new_deltas_mask_2[
                            :, h_2_start : h_2_start + h_tile_2, w_2_start : w_2_start + h_tile_2, :
                        ] = 1.0

                    norms_x_robust = np.sqrt(np.sum((x_robust - x_init) ** 2, axis=(1, 2, 3), keepdims=True))
                    W_norm = np.sqrt(
                        np.sum(
                            (delta_x_robust_init * np.maximum(new_deltas_mask, new_deltas_mask_2)) ** 2,
                            axis=(1, 2, 3),
                            keepdims=True,
                        )
                    )

                    if self.estimator.channels_first:
                        new_deltas_size = [x_init.shape[0], channels, h_tile, h_tile]
                        random_choice_size = [x_init.shape[0], channels, 1, 1]
                        perturbation_size = [1, 1, h_tile, h_tile]
                    else:
                        new_deltas_size = [x_init.shape[0], h_tile, h_tile, channels]
                        random_choice_size = [x_init.shape[0], 1, 1, channels]
                        perturbation_size = [1, h_tile, h_tile, 1]

                    delta_new = (
                        np.ones(new_deltas_size)
                        * _get_perturbation(h_tile).reshape(perturbation_size)
                        * np.random.choice([-1, 1], size=random_choice_size)
                    )

                    if self.estimator.channels_first:
                        delta_new += delta_x_robust_init[
                            :, :, h_start : h_start + h_tile, w_start : w_start + h_tile
                        ] / (np.maximum(1e-9, W_1_norm))
                    else:
                        delta_new += delta_x_robust_init[
                            :, h_start : h_start + h_tile, w_start : w_start + h_tile, :
                        ] / (np.maximum(1e-9, W_1_norm))

                    diff_norm = (self.eps * np.ones(delta_new.shape)) ** 2 - norms_x_robust ** 2
                    diff_norm[diff_norm < 0.0] = 0.0

                    if self.estimator.channels_first:
                        delta_new /= np.sqrt(np.sum(delta_new ** 2, axis=(2, 3), keepdims=True)) * np.sqrt(
                            diff_norm / channels + W_norm ** 2
                        )
                        delta_x_robust_init[
                            :, :, h_2_start : h_2_start + h_tile_2, w_2_start : w_2_start + h_tile_2
                        ] = 0.0
                        delta_x_robust_init[:, :, h_start : h_start + h_tile, w_start : w_start + h_tile] = delta_new
                    else:
                        delta_new /= np.sqrt(np.sum(delta_new ** 2, axis=(1, 2), keepdims=True)) * np.sqrt(
                            diff_norm / channels + W_norm ** 2
                        )
                        delta_x_robust_init[
                            :, h_2_start : h_2_start + h_tile_2, w_2_start : w_2_start + h_tile_2, :
                        ] = 0.0
                        delta_x_robust_init[:, h_start : h_start + h_tile, w_start : w_start + h_tile, :] = delta_new

                    x_robust_new = np.clip(
                        x_init
                        + self.eps
                        * delta_x_robust_init
                        / np.sqrt(np.sum(delta_x_robust_init ** 2, axis=(1, 2, 3), keepdims=True)),
                        self.estimator.clip_values[0],
                        self.estimator.clip_values[1],
                    )

                    sample_logits_diff_new = self.get_logits_diff(x_robust_new, y_robust)
                    logits_diff_improved = (sample_logits_diff_new - sample_logits_diff_init) < 0.0

                    x_robust[logits_diff_improved] = x_robust_new[logits_diff_improved]

                    x_adv[sample_is_robust] = x_robust

        return x_adv

    def set_params(self, **kwargs):
        super().set_params(**kwargs)
