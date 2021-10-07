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
This module implements the white-box attack `NewtonFool`.

| Paper link: http://doi.acm.org/10.1145/3134600.3134635
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, TYPE_CHECKING

import numpy as np
from tqdm.auto import trange

from art.attacks.attack import EvasionAttack
from art.config import ART_NUMPY_DTYPE
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassGradientsMixin
from art.utils import to_categorical, compute_success

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE

logger = logging.getLogger(__name__)


class NewtonFool(EvasionAttack):
    """
    Implementation of the attack from Uyeong Jang et al. (2017).

    | Paper link: http://doi.acm.org/10.1145/3134600.3134635
    """

    attack_params = EvasionAttack.attack_params + ["max_iter", "eta", "batch_size", "verbose"]
    _estimator_requirements = (BaseEstimator, ClassGradientsMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE",
        max_iter: int = 100,
        eta: float = 0.01,
        batch_size: int = 1,
        verbose: bool = True,
    ) -> None:
        """
        Create a NewtonFool attack instance.

        :param classifier: A trained classifier.
        :param max_iter: The maximum number of iterations.
        :param eta: The eta coefficient.
        :param batch_size: Size of the batch on which adversarial samples are generated.
        :param verbose: Show progress bars.
        """
        super().__init__(estimator=classifier)
        self.max_iter = max_iter
        self.eta = eta
        self.batch_size = batch_size
        self.verbose = verbose
        self._check_params()

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in a Numpy array.

        :param x: An array with the original inputs to be attacked.
        :param y: An array with the original labels to be predicted.
        :return: An array holding the adversarial examples.
        """
        x_adv = x.astype(ART_NUMPY_DTYPE)

        # Initialize variables
        y_pred = self.estimator.predict(x, batch_size=self.batch_size)
        pred_class = np.argmax(y_pred, axis=1)

        if self.estimator.nb_classes == 2 and y_pred.shape[1] == 1:  # pragma: no cover
            raise ValueError(
                "This attack has not yet been tested for binary classification with a single output classifier."
            )

        # Compute perturbation with implicit batching
        for batch_id in trange(
            int(np.ceil(x_adv.shape[0] / float(self.batch_size))), desc="NewtonFool", disable=not self.verbose
        ):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch = x_adv[batch_index_1:batch_index_2]

            # Main algorithm for each batch
            norm_batch = np.linalg.norm(np.reshape(batch, (batch.shape[0], -1)), axis=1)
            l_batch = pred_class[batch_index_1:batch_index_2]
            l_b = to_categorical(l_batch, self.estimator.nb_classes).astype(bool)

            # Main loop of the algorithm
            for _ in range(self.max_iter):
                # Compute score
                score = self.estimator.predict(batch)[l_b]

                # Compute the gradients and norm
                grads = self.estimator.class_gradient(batch, label=l_batch)
                if grads.shape[1] == 1:
                    grads = np.squeeze(grads, axis=1)
                norm_grad = np.linalg.norm(np.reshape(grads, (batch.shape[0], -1)), axis=1)

                # Theta
                theta = self._compute_theta(norm_batch, score, norm_grad)

                # Perturbation
                di_batch = self._compute_pert(theta, grads, norm_grad)

                # Update xi and perturbation
                batch += di_batch

            # Apply clip
            if self.estimator.clip_values is not None:
                clip_min, clip_max = self.estimator.clip_values
                x_adv[batch_index_1:batch_index_2] = np.clip(batch, clip_min, clip_max)
            else:
                x_adv[batch_index_1:batch_index_2] = batch

        logger.info(
            "Success rate of NewtonFool attack: %.2f%%",
            100 * compute_success(self.estimator, x, y, x_adv, batch_size=self.batch_size),
        )
        return x_adv

    def _compute_theta(self, norm_batch: np.ndarray, score: np.ndarray, norm_grad: np.ndarray) -> np.ndarray:
        """
        Function to compute the theta at each step.

        :param norm_batch: Norm of a batch.
        :param score: Softmax value at the attacked class.
        :param norm_grad: Norm of gradient values at the attacked class.
        :return: Theta value.
        """
        equ1 = self.eta * norm_batch * norm_grad
        equ2 = score - 1.0 / self.estimator.nb_classes
        result = np.minimum.reduce([equ1, equ2])

        return result

    @staticmethod
    def _compute_pert(theta: np.ndarray, grads: np.ndarray, norm_grad: np.ndarray) -> np.ndarray:
        """
        Function to compute the perturbation at each step.

        :param theta: Theta value at the current step.
        :param grads: Gradient values at the attacked class.
        :param norm_grad: Norm of gradient values at the attacked class.
        :return: Computed perturbation.
        """
        # Pick a small scalar to avoid division by 0
        tol = 10e-8

        nom = -theta.reshape((-1,) + (1,) * (len(grads.shape) - 1)) * grads
        denom = norm_grad ** 2
        denom[denom < tol] = tol
        result = nom / denom.reshape((-1,) + (1,) * (len(grads.shape) - 1))

        return result

    def _check_params(self) -> None:
        if not isinstance(self.max_iter, (int, np.int)) or self.max_iter <= 0:
            raise ValueError("The number of iterations must be a positive integer.")

        if not isinstance(self.eta, (float, int, np.int)) or self.eta <= 0:
            raise ValueError("The eta coefficient must be a positive float.")

        if self.batch_size <= 0:
            raise ValueError("The batch size `batch_size` has to be positive.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")
