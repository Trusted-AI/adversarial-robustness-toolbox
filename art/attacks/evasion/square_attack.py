# MIT License
#
# Copyright (C) IBM Corporation 2020
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
This module implements the `Square Attack` attack.

| Paper link: https://arxiv.org/abs/1912.00049
"""
import logging
import math
import bisect

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
        x_adv = x.astype(ART_NUMPY_DTYPE)

        y = check_and_transform_label_format(y, self.estimator.nb_classes)

        if self.estimator.channel_index == 1:
            channels = x.shape[1]
            height = x.shape[2]
            width = x.shape[3]
        elif self.estimator.channel_index == 3:
            height = x.shape[1]
            width = x.shape[2]
            channels = x.shape[3]

        for i_restart in range(self.nb_restarts):

            # Determine correctly predicted samples
            y_pred = self.estimator.predict(x_adv)
            sample_is_robust = np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)

            if np.sum(sample_is_robust) == 0:
                break

            x_robust = x_adv[sample_is_robust]
            y_robust = y[sample_is_robust]
            sample_logits_diff_init = self.get_logits_diff(x_robust, y_robust)

            if self.norm == np.inf:

                if self.estimator.channel_index == 1:
                    size = (x_robust.shape[0], channels, 1, width)
                elif self.estimator.channel_index == 3:
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

                    print("i_iter:", i_iter)

                    p = self._get_p(i_iter)

                    # Determine correctly predicted samples
                    y_pred = self.estimator.predict(x_adv)
                    sample_is_robust = np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)

                    print("np.sum(sample_is_robust):", np.sum(sample_is_robust))

                    if np.sum(sample_is_robust) == 0:
                        break

                    x_robust = x_adv[sample_is_robust]
                    x_init = x[sample_is_robust]
                    y_robust = y[sample_is_robust]

                    sample_logits_diff_init = self.get_logits_diff(x_robust, y_robust)

                    h = max(int(round(math.sqrt(p * height * width))), 1)

                    h_center = np.random.randint(0, height - h)
                    w_center = np.random.randint(0, width - h)

                    delta_new = np.zeros(self.estimator.input_shape)

                    if self.estimator.channel_index == 1:
                        delta_new[:, h_center : h_center + h, w_center : w_center + h] = np.random.choice(
                            [-2 * self.eps, 2 * self.eps], size=[channels, 1, 1]
                        )
                    elif self.estimator.channel_index == 3:
                        delta_new[h_center : h_center + h, w_center : w_center + h, :] = np.random.choice(
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
                pass

        return x_adv

    def set_params(self, **kwargs):
        super().set_params(**kwargs)
