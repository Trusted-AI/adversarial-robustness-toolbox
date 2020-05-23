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
This module implements the evasion attack `ShadowAttackTensorFlowV2` for TensorFlow v2.

| Paper link: https://arxiv.org/abs/2003.08937
"""
import logging

import numpy as np

from art.config import ART_NUMPY_DTYPE
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.estimators.classification import TensorFlowV2Classifier, PyTorchClassifier
from art.estimators.certification.randomized_smoothing import (
    TensorFlowV2RandomizedSmoothing,
    PyTorchRandomizedSmoothing,
)
from art.attacks.attack import EvasionAttack
from art.utils import get_labels_np_array, check_and_transform_label_format

logger = logging.getLogger(__name__)


class ShadowAttack(EvasionAttack):
    """
    Implementation of the Shadow Attack.

    | Paper link: https://arxiv.org/abs/2003.08937
    """

    attack_params = EvasionAttack.attack_params + [
        "sigma",
        "nb_steps",
        "learning_rate",
        "lambda_tv",
        "lambda_c",
        "lambda_s",
        "batch_size",
        "targeted",
    ]

    _estimator_requirements = (BaseEstimator, LossGradientsMixin, ClassifierMixin)

    def __init__(
        self,
        estimator,
        sigma=0.5,
        nb_steps=300,
        learning_rate=0.1,
        lambda_tv=0.1,
        lambda_c=20.0,
        lambda_s=10.0,
        batch_size=32,
        targeted=False,
    ):
        """
        Create an instance of the :class:`.ShadowAttack`.

        :param estimator: A trained classifier.
        :type estimator: :class:`.Classifier`
        :param sigma: Standard deviation random Gaussian Noise.
        :type sigma: `float`
        :param nb_steps: Number of SGD steps.
        :type nb_steps: `int
        :param learning_rate: Learning rate for SGD.
        :type learning_rate: `float`
        :param lambda_tv: Scalar penalty weight for total variation of the perturbation.
        :type lambda_tv: `float`
        :param lambda_c: Scalar penalty weight for change in the mean of each color channel of the perturbation.
        :type lambda_c: `float`
        :param lambda_s: Scalar penalty weight for similarity of color channels in perturbation.
        :type lambda_s: `float`
        :param batch_size: The size of the training batch.
        :type batch_size: `int`
        :param targeted: True if the attack is targeted.
        :type targeted: `bool`
        """
        super().__init__(estimator=estimator)

        kwargs = {
            "sigma": sigma,
            "batch_size": batch_size,
            "nb_steps": nb_steps,
            "learning_rate": learning_rate,
            "lambda_tv": lambda_tv,
            "lambda_c": lambda_c,
            "lambda_s": lambda_s,
            "targeted": targeted,
        }

        self.set_params(**kwargs)

        if isinstance(self.estimator, TensorFlowV2Classifier) or isinstance(
            self.estimator, TensorFlowV2RandomizedSmoothing
        ):
            self.frame_work = "tensorflow"
        elif isinstance(self.estimator, PyTorchClassifier) or isinstance(self.estimator, PyTorchRandomizedSmoothing):
            self.frame_work = "pytorch"
        else:
            self.frame_work = None

    def generate(self, x, y=None, **kwargs):
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :type x: `np.ndarray`
        :param y: An array with the target labels.
        :type y: `np.ndarray`
        :return: An array with the adversarial examples.
        :rtype: `np.ndarray`
        """
        y = check_and_transform_label_format(y, self.estimator.nb_classes)

        if y is None:
            # Throw error if attack is targeted, but no targets are provided
            if self.targeted:
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")

            logger.info("Using model predictions as correct labels for FGM.")
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))
        else:
            self.targeted = True

        if x.ndim != 4:
            raise ValueError(
                "Feature vectors detected. The adversarial patch can only be applied to image data dimensions."
            )

        x = x.astype(ART_NUMPY_DTYPE)
        x_adv = np.zeros_like(x, dtype=ART_NUMPY_DTYPE)

        # Compute perturbation with implicit batching
        from tqdm import tqdm

        for i_batch in tqdm(range(int(np.ceil(x.shape[0] / self.batch_size)))):

            batch_index_1, batch_index_2 = i_batch * self.batch_size, (i_batch + 1) * self.batch_size
            x_batch = x[batch_index_1:batch_index_2]
            y_batch = y[batch_index_1:batch_index_2]

            perturbation = (
                np.random.uniform(
                    low=self.estimator.clip_values[0], high=self.estimator.clip_values[1], size=x_batch.shape
                ).astype(ART_NUMPY_DTYPE)
                - (self.estimator.clip_values[1] - self.estimator.clip_values[0]) / 2
            )

            for _ in range(self.nb_steps):

                gradients_ce = self.estimator.loss_gradient(x=x_batch + perturbation, y=y_batch) * (
                    1 - 2 * int(self.targeted)
                )

                gradients = gradients_ce - self._get_regularisation_loss_gradients(perturbation)

                perturbation += self.learning_rate * gradients

            x_p = x_batch + perturbation

            x_p = np.clip(x_p, a_min=self.estimator.clip_values[0], a_max=self.estimator.clip_values[1])

            perturbation = x_p - x_batch

            x_adv[batch_index_1:batch_index_2] = x_batch + perturbation

        return x_adv

    def _get_regularisation_loss_gradients(self, perturbation):

        if self.frame_work == "tensorflow":

            import tensorflow as tf

            if self.estimator.channel_index == 3:
                perturbation = perturbation.transpose(0, 3, 1, 2)

            if tf.executing_eagerly():
                with tf.GradientTape() as tape:

                    perturbation_t = tf.convert_to_tensor(perturbation)
                    tape.watch(perturbation_t)

                    x_ = perturbation_t[:, :, :, 1:] - perturbation_t[:, :, :, :-1]
                    y_ = perturbation_t[:, :, 1:, :] - perturbation_t[:, :, :-1, :]

                    loss_tv = tf.reduce_sum(x_ * x_, axis=(1, 2, 3)) + tf.reduce_sum(y_ * y_, axis=(1, 2, 3))

                    if perturbation_t.shape[1] == 1:
                        loss_s = 0.0
                    elif perturbation_t.shape[1] == 3:
                        loss_s = tf.norm(
                            (perturbation_t[:, 0, :, :] - perturbation_t[:, 1, :, :]) ** 2
                            + (perturbation_t[:, 1, :, :] - perturbation_t[:, 2, :, :]) ** 2
                            + (perturbation_t[:, 0, :, :] - perturbation_t[:, 2, :, :]) ** 2,
                            ord=2,
                            axis=(1, 2),
                        )

                    loss_c = tf.norm(tf.reduce_mean(tf.abs(perturbation_t), axis=[2, 3]), ord=2, axis=1) ** 2

                    loss = self.lambda_tv * loss_tv + self.lambda_s * loss_s + self.lambda_c * loss_c

                    gradients = tape.gradient(loss, perturbation_t).numpy()

                    if self.estimator.channel_index == 3:
                        gradients = gradients.transpose(0, 2, 3, 1)

            else:
                raise ValueError("Expecting eager execution.")

        elif self.frame_work == "pytorch":

            import torch

            perturbation_t = torch.from_numpy(perturbation)
            perturbation_t.requires_grad = True

            x_ = perturbation_t[:, :, :, 1:] - perturbation_t[:, :, :, :-1]
            y_ = perturbation_t[:, :, 1:, :] - perturbation_t[:, :, :-1, :]

            loss_tv = (x_ * x_).sum(dim=(1, 2, 3)) + (y_ * y_).sum(dim=(1, 2, 3))

            if perturbation_t.shape[1] == 1:
                loss_s = 0.0
            elif perturbation_t.shape[1] == 3:
                loss_s = (
                    (perturbation_t[:, 0, :, :] - perturbation_t[:, 1, :, :]) ** 2
                    + (perturbation_t[:, 1, :, :] - perturbation_t[:, 2, :, :]) ** 2
                    + (perturbation_t[:, 0, :, :] - perturbation_t[:, 2, :, :]) ** 2
                ).norm(p=2, dim=(1, 2))

            loss_c = perturbation_t.abs().mean([2, 3]).norm(dim=1) ** 2

            loss = torch.mean(self.lambda_tv * loss_tv + self.lambda_s * loss_s + self.lambda_c * loss_c)

            loss.backward()

            gradients = perturbation_t.grad.numpy()

        else:
            raise NotImplementedError

        return gradients

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and apply attack-specific checks before saving them as attributes.
        """
        super().set_params(**kwargs)

        if not isinstance(self.sigma, (int, float)):
            raise ValueError("The sigma must be of type int or float.")

        if self.sigma <= 0:
            raise ValueError("The sigma must larger than zero.")

        if not isinstance(self.nb_steps, int):
            raise ValueError("The number of steps must be of type int.")

        if self.nb_steps <= 0:
            raise ValueError("The number of steps must larger than zero.")

        if not isinstance(self.learning_rate, float):
            raise ValueError("The learning rate must be of type float.")

        if self.learning_rate <= 0:
            raise ValueError("The learning rate must larger than zero.")

        if not isinstance(self.lambda_tv, float):
            raise ValueError("The lambda_tv must be of type float.")

        if self.lambda_tv <= 0:
            raise ValueError("The lambda_tv must larger than zero.")

        if not isinstance(self.lambda_c, float):
            raise ValueError("The lambda_c must be of type float.")

        if self.lambda_c <= 0:
            raise ValueError("The lambda_c must larger than zero.")

        if not isinstance(self.lambda_s, float):
            raise ValueError("The lambda_s must be of type float.")

        if self.lambda_s <= 0:
            raise ValueError("The lambda_s must larger than zero.")

        if not isinstance(self.batch_size, int):
            raise ValueError("The batch size must be of type int.")

        if self.batch_size <= 0:
            raise ValueError("The sigma must larger than zero.")

        if not isinstance(self.targeted, bool):
            raise ValueError("The targeted argument must be of type bool.")

        return True
