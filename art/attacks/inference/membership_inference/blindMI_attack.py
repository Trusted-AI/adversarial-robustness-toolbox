# MIT License

# Copyright (c) 2023 Yisroel Mirsky

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
This module implements the Practical Blind Membership Inference Attack via Differential Comparison
| Paper link: https://arxiv.org/abs/2101.01341

Module author:
Shashank Priyadarshi

Contributed by:
The Offensive AI Research Lab
Ben-Gurion University, Israel
https://offensive-ai-lab.github.io/

Sponsored by INCD

"""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import TYPE_CHECKING
import tensorflow as tf
import numpy as np
from art.attacks.attack import MembershipInferenceAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.attacks.inference.membership_inference.utils import (
    sobel,
    mmd_loss,
)

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE, REGRESSOR_TYPE

logger = logging.getLogger(__name__)


class MembershipInferenceBlindMI(MembershipInferenceAttack):
    """
    Implementation of a Blind membership inference attack via Differential Comparison.

    This implementation can use as input to the learning process probabilities/logits or losses,
    depending on the type of model and provided configuration.
    """

    attack_params = MembershipInferenceAttack.attack_params
    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(
        self,
        estimator: "CLASSIFIER_TYPE",
    ):
        """
        Create a MembershipInferenceBlindMI attack instance.

        :param estimator: Target estimator.
        """

        super().__init__(estimator=estimator)

    def infer(self, x: np.ndarray, y: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, member) -> np.ndarray:
        """
        Infer membership in the training set of the target estimator.

        :param x: Input records to attack. Should be of shape B, C, H, W.
        :param y: True labels for `x`.
        :return: An array holding the inferred membership status, 1 indicates a member and 0 indicates non-member,
                 or class probabilities.
        """

        if x is None and x_test is None:
            raise ValueError("Must supply either x or pred")

        if self.estimator.input_shape is not None and x is not None:  # pragma: no cover
            if self.estimator.input_shape[0] != x.shape[1]:
                raise ValueError("Shape of x does not match input_shape of estimator")

        if y is None and y_test is None:
            raise ValueError("None value detected.")

        if x is not None and y.shape[0] != x.shape[0]:  # pragma: no cover
            raise ValueError("Number of rows in x and y do not match")

        x_ = np.r_[x, x_test]
        y_true = np.r_[y, y_test]
        # import ipdb;ipdb.set_trace()
        y_pred = self.estimator.predict(x_).astype(np.float32)

        mix = np.c_[y_pred[y_true.astype(bool)], np.sort(y_pred, axis=1)[:, ::-1][:, :2]]

        non_mem_idx = np.random.randint(0, x_.shape[0], size=20)
        non_mem_pred = self.estimator.predict(sobel(x_[non_mem_idx]))
        non_mem = tf.convert_to_tensor(
            np.c_[non_mem_pred[y_true[non_mem_idx].astype(bool)], np.sort(non_mem_pred, axis=1)[:, ::-1][:, :2]]
        )

        data = (
            tf.data.Dataset.from_tensor_slices((mix, member))
            .shuffle(buffer_size=x_.shape[0])
            .batch(20)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        m_pred, m_true = [], []
        mix_shuffled = []
        for mix_batch, m_true_batch in data:
            m_pred_batch = np.ones(mix_batch.shape[0])
            m_pred_epoch = np.ones(mix_batch.shape[0])
            non_mem_in_mix = True
            while non_mem_in_mix:
                mix_epoch_new = mix_batch[m_pred_epoch.astype(bool)]
                dis_ori = mmd_loss(non_mem, mix_epoch_new, 1)
                non_mem_in_mix = False
                # import ipdb;ipdb.set_trace()
                for index, item in enumerate(mix_batch):
                    if m_pred_batch[index] == 1:
                        non_mem_batch_new = tf.concat([non_mem, [mix_batch[index]]], axis=0)
                        mix_batch_new = tf.concat([mix_batch[:index], mix_batch[index + 1 :]], axis=0)
                        m_pred_without = np.r_[m_pred_batch[:index], m_pred_batch[index + 1 :]]
                        mix_batch_new = mix_batch_new[m_pred_without.astype(bool, copy=True)]
                        dis_new = mmd_loss(non_mem_batch_new, mix_batch_new, weight=1)
                        if dis_new > dis_ori:
                            non_mem_in_mix = True
                            m_pred_epoch[index] = 0
                m_pred_batch = m_pred_epoch.copy()

            mix_shuffled.append(mix_batch)
            m_pred.append(m_pred_batch)
            m_true.append(m_true_batch)
        return (
            np.concatenate(m_true, axis=0),
            np.concatenate(m_pred, axis=0),
            np.concatenate(mix_shuffled, axis=0),
            non_mem,
        )
