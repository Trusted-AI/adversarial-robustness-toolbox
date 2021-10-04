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
This module implements the evasion attack `ShadowAttack`.

| Paper link: https://arxiv.org/abs/2003.08937
"""
import logging
from typing import Optional, Union

import numpy as np
from tqdm.auto import trange

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
        "verbose",
    ]

    _estimator_requirements = (BaseEstimator, LossGradientsMixin, ClassifierMixin)

    def __init__(
        self,
        estimator: Union[
            TensorFlowV2Classifier, TensorFlowV2RandomizedSmoothing, PyTorchClassifier, PyTorchRandomizedSmoothing
        ],
        sigma: float = 0.5,
        nb_steps: int = 300,
        learning_rate: float = 0.1,
        lambda_tv: float = 0.3,
        lambda_c: float = 1.0,
        lambda_s: float = 0.5,
        batch_size: int = 400,
        targeted: bool = False,
        verbose: bool = True,
    ):
        """
        Create an instance of the :class:`.ShadowAttack`.

        :param estimator: A trained classifier.
        :param sigma: Standard deviation random Gaussian Noise.
        :param nb_steps: Number of SGD steps.
        :param learning_rate: Learning rate for SGD.
        :param lambda_tv: Scalar penalty weight for total variation of the perturbation.
        :param lambda_c: Scalar penalty weight for change in the mean of each color channel of the perturbation.
        :param lambda_s: Scalar penalty weight for similarity of color channels in perturbation.
        :param batch_size: The size of the training batch.
        :param targeted: True if the attack is targeted.
        :param verbose: Show progress bars.
        """
        super().__init__(estimator=estimator)

        self.sigma = sigma
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.learning_rate = learning_rate
        self.lambda_tv = lambda_tv
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self._targeted = targeted
        self.verbose = verbose
        self._check_params()

        self.framework: Optional[str]
        if isinstance(self.estimator, (TensorFlowV2Classifier, TensorFlowV2RandomizedSmoothing)):
            self.framework = "tensorflow"
        elif isinstance(self.estimator, (PyTorchClassifier, PyTorchRandomizedSmoothing)):
            self.framework = "pytorch"
        else:
            self.framework = None

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array. This requires a lot of memory, therefore it accepts
        only a single samples as input, e.g. a batch of size 1.

        :param x: An array of a single original input sample.
        :param y: An array of a single target label.
        :return: An array with the adversarial examples.
        """
        y = check_and_transform_label_format(y, self.estimator.nb_classes)

        if y is None:
            # Throw error if attack is targeted, but no targets are provided
            if self.targeted:  # pragma: no cover
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")

            logger.info("Using model predictions as correct labels for FGM.")
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))
        else:
            self.targeted = True

        if self.estimator.nb_classes == 2 and y.shape[1] == 1:  # pragma: no cover
            raise ValueError(
                "This attack has not yet been tested for binary classification with a single output classifier."
            )

        if x.shape[0] > 1 or y.shape[0] > 1:  # pragma: no cover
            raise ValueError("This attack only accepts a single sample as input.")

        if x.ndim != 4:  # pragma: no cover
            raise ValueError("Unrecognized input dimension. Shadow Attack can only be applied to image data.")

        x = x.astype(ART_NUMPY_DTYPE)
        x_batch = np.repeat(x, repeats=self.batch_size, axis=0).astype(ART_NUMPY_DTYPE)
        x_batch = x_batch + np.random.normal(scale=self.sigma, size=x_batch.shape).astype(ART_NUMPY_DTYPE)
        y_batch = np.repeat(y, repeats=self.batch_size, axis=0)

        perturbation = (
            np.random.uniform(
                low=self.estimator.clip_values[0], high=self.estimator.clip_values[1], size=x.shape
            ).astype(ART_NUMPY_DTYPE)
            - (self.estimator.clip_values[1] - self.estimator.clip_values[0]) / 2
        )

        for _ in trange(self.nb_steps, desc="Shadow attack", disable=not self.verbose):
            gradients_ce = np.mean(
                self.estimator.loss_gradient(x=x_batch + perturbation, y=y_batch, sampling=False)
                * (1 - 2 * int(self.targeted)),
                axis=0,
                keepdims=True,
            )
            gradients = gradients_ce - self._get_regularisation_loss_gradients(perturbation)
            perturbation += self.learning_rate * gradients

        x_p = x + perturbation
        x_adv = np.clip(x_p, a_min=self.estimator.clip_values[0], a_max=self.estimator.clip_values[1]).astype(
            ART_NUMPY_DTYPE
        )

        return x_adv

    def _get_regularisation_loss_gradients(self, perturbation: np.ndarray) -> np.ndarray:
        """
        Get regularisation loss gradients.

        :param perturbation: The perturbation to be regularised.
        :return: The loss gradients of the perturbation.
        """
        if not self.estimator.channels_first:
            perturbation = perturbation.transpose((0, 3, 1, 2))

        if self.framework == "tensorflow":
            import tensorflow as tf

            if tf.executing_eagerly():
                with tf.GradientTape() as tape:

                    perturbation_t = tf.convert_to_tensor(perturbation)
                    tape.watch(perturbation_t)

                    x_t = perturbation_t[:, :, :, 1:] - perturbation_t[:, :, :, :-1]
                    y_t = perturbation_t[:, :, 1:, :] - perturbation_t[:, :, :-1, :]
                    loss_tv = tf.reduce_sum(x_t * x_t, axis=(1, 2, 3)) + tf.reduce_sum(y_t * y_t, axis=(1, 2, 3))

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

            else:  # pragma: no cover
                raise ValueError("Expecting eager execution.")

        elif self.framework == "pytorch":
            import torch

            perturbation_t = torch.from_numpy(perturbation).to("cpu")
            perturbation_t.requires_grad = True

            x_t = perturbation_t[:, :, :, 1:] - perturbation_t[:, :, :, :-1]
            y_t = perturbation_t[:, :, 1:, :] - perturbation_t[:, :, :-1, :]

            loss_tv = (x_t * x_t).sum(dim=(1, 2, 3)) + (y_t * y_t).sum(dim=(1, 2, 3))

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

        if not self.estimator.channels_first:
            gradients = gradients.transpose(0, 2, 3, 1)

        return gradients

    def _check_params(self) -> None:
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

        if self.lambda_tv < 0:
            raise ValueError("The lambda_tv must larger than zero.")

        if not isinstance(self.lambda_c, float):
            raise ValueError("The lambda_c must be of type float.")

        if self.lambda_c < 0:
            raise ValueError("The lambda_c must larger than zero.")

        if not isinstance(self.lambda_s, float):
            raise ValueError("The lambda_s must be of type float.")

        if self.lambda_s < 0:
            raise ValueError("The lambda_s must larger than zero.")

        if not isinstance(self.batch_size, int):
            raise ValueError("The batch size must be of type int.")

        if self.batch_size <= 0:
            raise ValueError("The sigma must larger than zero.")

        if not isinstance(self.targeted, bool):
            raise ValueError("The targeted argument must be of type bool.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")
