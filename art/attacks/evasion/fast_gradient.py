# MIT License
#
# Copyright (C) IBM Corporation 2018
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
This module implements the Fast Gradient Method attack. This implementation includes the original Fast Gradient Sign
Method attack and extends it to other norms, therefore it is called the Fast Gradient Method.

| Paper link: https://arxiv.org/abs/1412.6572
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional

import numpy as np

from art.config import ART_NUMPY_DTYPE
from art.classifiers.classifier import ClassifierGradients
from art.attacks.attack import EvasionAttack
from art.utils import (
    compute_success,
    get_labels_np_array,
    random_sphere,
    projection,
    check_and_transform_label_format,
)
from art.exceptions import ClassifierError

logger = logging.getLogger(__name__)


class FastGradientMethod(EvasionAttack):
    """
    This attack was originally implemented by Goodfellow et al. (2015) with the infinity norm (and is known as the "Fast
    Gradient Sign Method"). This implementation extends the attack to other norms, and is therefore called the Fast
    Gradient Method.

    | Paper link: https://arxiv.org/abs/1412.6572
    """

    attack_params = EvasionAttack.attack_params + [
        "norm",
        "eps",
        "eps_step",
        "targeted",
        "num_random_init",
        "batch_size",
        "minimal",
    ]

    def __init__(
        self,
        classifier: ClassifierGradients,
        norm: int = np.inf,
        eps: float = 0.3,
        eps_step: float = 0.1,
        targeted: bool = False,
        num_random_init: int = 0,
        batch_size: int = 1,
        minimal: bool = False,
    ) -> None:
        """
        Create a :class:`.FastGradientMethod` instance.

        :param classifier: A trained classifier.
        :param norm: The norm of the adversarial perturbation. Possible values: np.inf, 1 or 2.
        :param eps: Attack step size (input variation).
        :param eps_step: Step size of input variation for minimal perturbation computation.
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False)
        :param num_random_init: Number of random initialisations within the epsilon ball. For random_init=0 starting at
            the original input.
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :param minimal: Indicates if computing the minimal perturbation (True). If True, also define `eps_step` for
                        the step size and eps for the maximum perturbation.
        """
        super(FastGradientMethod, self).__init__(classifier)
        self.norm = norm
        self.eps = eps
        self.eps_step = eps_step
        self.targeted = targeted
        self.num_random_init = num_random_init
        self.batch_size = batch_size
        self.minimal = minimal
        self._project = True
        FastGradientMethod._check_params(self)

    @classmethod
    def is_valid_classifier_type(cls, classifier: ClassifierGradients) -> bool:
        """
        Checks whether the classifier provided is a classifier which this class can perform an attack on.

        :param classifier:
        :return: True if the candidate classifier can be used with this attack.
        """
        return isinstance(classifier, ClassifierGradients)

    def _minimal_perturbation(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Iteratively compute the minimal perturbation necessary to make the class prediction change. Stop when the
        first adversarial example was found.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes).
        :return: An array holding the adversarial examples.
        """
        adv_x = x.copy()

        # Compute perturbation with implicit batching
        for batch_id in range(int(np.ceil(adv_x.shape[0] / float(self.batch_size)))):
            batch_index_1, batch_index_2 = (
                batch_id * self.batch_size,
                (batch_id + 1) * self.batch_size,
            )
            batch = adv_x[batch_index_1:batch_index_2]
            batch_labels = y[batch_index_1:batch_index_2]

            # Get perturbation
            perturbation = self._compute_perturbation(batch, batch_labels)

            # Get current predictions
            active_indices = np.arange(len(batch))
            current_eps = self.eps_step
            while active_indices.size > 0 and current_eps <= self.eps:
                # Adversarial crafting
                current_x = self._apply_perturbation(
                    x[batch_index_1:batch_index_2], perturbation, current_eps
                )
                # Update
                batch[active_indices] = current_x[active_indices]
                adv_preds = self.classifier.predict(batch)
                # If targeted active check to see whether we have hit the target, otherwise head to anything but
                if self.targeted:
                    active_indices = np.where(
                        np.argmax(batch_labels, axis=1) != np.argmax(adv_preds, axis=1)
                    )[0]
                else:
                    active_indices = np.where(
                        np.argmax(batch_labels, axis=1) == np.argmax(adv_preds, axis=1)
                    )[0]

                current_eps += self.eps_step

            adv_x[batch_index_1:batch_index_2] = batch

        return adv_x

    def generate(
        self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs
    ) -> np.ndarray:
        """Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :return: An array holding the adversarial examples.
        """
        y = check_and_transform_label_format(y, self.classifier.nb_classes())

        if y is None:
            # Throw error if attack is targeted, but no targets are provided
            if self.targeted:
                raise ValueError(
                    "Target labels `y` need to be provided for a targeted attack."
                )

            # Use model predictions as correct outputs
            logger.info("Using model predictions as correct labels for FGM.")
            y = get_labels_np_array(
                self.classifier.predict(x, batch_size=self.batch_size)
            )
        y = y / np.sum(y, axis=1, keepdims=True)

        # Return adversarial examples computed with minimal perturbation if option is active
        adv_x_best: Optional[np.ndarray] = None
        rate_best: Optional[float] = None
        if self.minimal:
            logger.info("Performing minimal perturbation FGM.")
            adv_x_best = self._minimal_perturbation(x, y)
            rate_best = 100 * compute_success(
                self.classifier,
                x,
                y,
                adv_x_best,
                self.targeted,
                batch_size=self.batch_size,
            )
        else:
            for _ in range(max(1, self.num_random_init)):
                adv_x = self._compute(
                    x, x, y, self.eps, self.eps, self._project, self.num_random_init > 0
                )

                if self.num_random_init > 1:
                    rate = 100 * compute_success(
                        self.classifier,
                        x,
                        y,
                        adv_x,
                        self.targeted,
                        batch_size=self.batch_size,
                    )
                    if rate_best is None or rate > rate_best or adv_x_best is None:
                        rate_best = rate
                        adv_x_best = adv_x
                else:
                    adv_x_best = adv_x

        logger.info(
            "Success rate of FGM attack: %.2f%%",
            rate_best
            if rate_best is not None
            else 100
            * compute_success(
                self.classifier,
                x,
                y,
                adv_x_best,
                self.targeted,
                batch_size=self.batch_size,
            ),
        )

        return adv_x_best

    def _check_params(self) -> None:
        if not isinstance(self.classifier, ClassifierGradients):
            raise ClassifierError(
                self.__class__, [ClassifierGradients], self.classifier
            )

        # Check if order of the norm is acceptable given current implementation
        if self.norm not in [np.inf, int(1), int(2)]:
            raise ValueError("Norm order must be either `np.inf`, 1, or 2.")

        if self.eps <= 0:
            raise ValueError("The perturbation size `eps` has to be positive.")

        if self.eps_step <= 0:
            raise ValueError(
                "The perturbation step-size `eps_step` has to be positive."
            )

        if not isinstance(self.targeted, bool):
            raise ValueError("The flag `targeted` has to be of type bool.")

        if not isinstance(self.num_random_init, (int, np.int)):
            raise TypeError(
                "The number of random initialisations has to be of type integer"
            )

        if self.num_random_init < 0:
            raise ValueError(
                "The number of random initialisations `random_init` has to be greater than or equal to 0."
            )

        if self.batch_size <= 0:
            raise ValueError("The batch size `batch_size` has to be positive.")

        if not isinstance(self.minimal, bool):
            raise ValueError("The flag `minimal` has to be of type bool.")

    def _compute_perturbation(
        self, batch: np.ndarray, batch_labels: np.ndarray
    ) -> np.ndarray:
        # Pick a small scalar to avoid division by 0
        tol = 10e-8

        # Get gradient wrt loss; invert it if attack is targeted
        grad = self.classifier.loss_gradient(batch, batch_labels) * (
            1 - 2 * int(self.targeted)
        )

        # Apply norm bound
        if self.norm == np.inf:
            grad = np.sign(grad)
        elif self.norm == 1:
            ind = tuple(range(1, len(batch.shape)))
            grad = grad / (np.sum(np.abs(grad), axis=ind, keepdims=True) + tol)
        elif self.norm == 2:
            ind = tuple(range(1, len(batch.shape)))
            grad = grad / (
                np.sqrt(np.sum(np.square(grad), axis=ind, keepdims=True)) + tol
            )
        assert batch.shape == grad.shape

        return grad

    def _apply_perturbation(
        self, batch: np.ndarray, perturbation: np.ndarray, eps_step: float
    ) -> np.ndarray:
        batch = batch + eps_step * perturbation

        if (
            hasattr(self.classifier, "clip_values")
            and self.classifier.clip_values is not None
        ):
            clip_min, clip_max = self.classifier.clip_values
            batch = np.clip(batch, clip_min, clip_max)

        return batch

    def _compute(
        self,
        x: np.ndarray,
        x_init: np.ndarray,
        y: np.ndarray,
        eps: float,
        eps_step: float,
        project: bool,
        random_init: bool,
    ) -> np.ndarray:
        if random_init:
            n = x.shape[0]
            m = np.prod(x.shape[1:])
            x_adv = x.astype(ART_NUMPY_DTYPE) + (
                random_sphere(n, m, eps, self.norm)
                .reshape(x.shape)
                .astype(ART_NUMPY_DTYPE)
            )

            if (
                hasattr(self.classifier, "clip_values")
                and self.classifier.clip_values is not None
            ):
                clip_min, clip_max = self.classifier.clip_values
                x_adv = np.clip(x_adv, clip_min, clip_max)
        else:
            x_adv = x.astype(ART_NUMPY_DTYPE)

        # Compute perturbation with implicit batching
        for batch_id in range(int(np.ceil(x.shape[0] / float(self.batch_size)))):
            batch_index_1, batch_index_2 = (
                batch_id * self.batch_size,
                (batch_id + 1) * self.batch_size,
            )
            batch = x_adv[batch_index_1:batch_index_2]
            batch_labels = y[batch_index_1:batch_index_2]

            # Get perturbation
            perturbation = self._compute_perturbation(batch, batch_labels)

            # Apply perturbation and clip
            x_adv[batch_index_1:batch_index_2] = self._apply_perturbation(
                batch, perturbation, eps_step
            )

            if project:
                perturbation = projection(
                    x_adv[batch_index_1:batch_index_2]
                    - x_init[batch_index_1:batch_index_2],
                    eps,
                    self.norm,
                )
                x_adv[batch_index_1:batch_index_2] = (
                    x_init[batch_index_1:batch_index_2] + perturbation
                )

        return x_adv
