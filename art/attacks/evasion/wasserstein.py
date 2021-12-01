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
This module implements ``Wasserstein Adversarial Examples via Projected Sinkhorn Iterations`` as evasion attack.

| Paper link: https://arxiv.org/abs/1902.07906
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, TYPE_CHECKING

import numpy as np
from scipy.special import lambertw
from tqdm.auto import trange

from art.config import ART_NUMPY_DTYPE
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.attacks.attack import EvasionAttack
from art.utils import compute_success, get_labels_np_array, check_and_transform_label_format

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE

logger = logging.getLogger(__name__)

EPS_LOG = 10 ** -10


class Wasserstein(EvasionAttack):
    """
    Implements ``Wasserstein Adversarial Examples via Projected Sinkhorn Iterations`` as evasion attack.

    | Paper link: https://arxiv.org/abs/1902.07906
    """

    attack_params = EvasionAttack.attack_params + [
        "targeted",
        "regularization",
        "p",
        "kernel_size",
        "eps_step",
        "norm",
        "ball",
        "eps",
        "eps_iter",
        "eps_factor",
        "max_iter",
        "conjugate_sinkhorn_max_iter",
        "projected_sinkhorn_max_iter",
        "batch_size",
        "verbose",
    ]

    _estimator_requirements = (BaseEstimator, LossGradientsMixin, ClassifierMixin)

    def __init__(
        self,
        estimator: "CLASSIFIER_LOSS_GRADIENTS_TYPE",
        targeted: bool = False,
        regularization: float = 3000.0,
        p: int = 2,
        kernel_size: int = 5,
        eps_step: float = 0.1,
        norm: str = "wasserstein",
        ball: str = "wasserstein",
        eps: float = 0.3,
        eps_iter: int = 10,
        eps_factor: float = 1.1,
        max_iter: int = 400,
        conjugate_sinkhorn_max_iter: int = 400,
        projected_sinkhorn_max_iter: int = 400,
        batch_size: int = 1,
        verbose: bool = True,
    ):
        """
        Create a Wasserstein attack instance.

        :param estimator: A trained estimator.
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).
        :param regularization: Entropy regularization.
        :param p: The p-wasserstein distance.
        :param kernel_size: Kernel size for computing the cost matrix.
        :param eps_step: Attack step size (input variation) at each iteration.
        :param norm: The norm of the adversarial perturbation. Possible values: `inf`, `1`, `2` or `wasserstein`.
        :param ball: The ball of the adversarial perturbation. Possible values: `inf`, `1`, `2` or `wasserstein`.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param eps_iter: Number of iterations to increase the epsilon.
        :param eps_factor: Factor to increase the epsilon.
        :param max_iter: The maximum number of iterations.
        :param conjugate_sinkhorn_max_iter: The maximum number of iterations for the conjugate sinkhorn optimizer.
        :param projected_sinkhorn_max_iter: The maximum number of iterations for the projected sinkhorn optimizer.
        :param batch_size: Size of batches.
        :param verbose: Show progress bars.
        """
        super().__init__(estimator=estimator)

        self._targeted = targeted
        self.regularization = regularization
        self.p = p  # pylint: disable=C0103
        self.kernel_size = kernel_size
        self.eps_step = eps_step
        self.norm = norm
        self.ball = ball
        self.eps = eps
        self.eps_iter = eps_iter
        self.eps_factor = eps_factor
        self.max_iter = max_iter
        self.conjugate_sinkhorn_max_iter = conjugate_sinkhorn_max_iter
        self.projected_sinkhorn_max_iter = projected_sinkhorn_max_iter
        self.batch_size = batch_size
        self.verbose = verbose
        self._check_params()

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param cost_matrix: A non-negative cost matrix.
        :type cost_matrix: `np.ndarray`
        :return: An array holding the adversarial examples.
        """
        y = check_and_transform_label_format(y, self.estimator.nb_classes)
        x_adv = x.copy().astype(ART_NUMPY_DTYPE)

        if y is None:
            # Throw error if attack is targeted, but no targets are provided
            if self.targeted:
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")

            # Use model predictions as correct outputs
            targets = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))
        else:
            targets = y

        if self.estimator.nb_classes == 2 and targets.shape[1] == 1:
            raise ValueError(
                "This attack has not yet been tested for binary classification with a single output classifier."
            )

        # Compute the cost matrix if needed
        cost_matrix = kwargs.get("cost_matrix")
        if cost_matrix is None:
            cost_matrix = self._compute_cost_matrix(self.p, self.kernel_size)

        # Compute perturbation with implicit batching
        nb_batches = int(np.ceil(x.shape[0] / float(self.batch_size)))
        for batch_id in trange(nb_batches, desc="Wasserstein", disable=not self.verbose):
            logger.debug("Processing batch %i out of %i", batch_id, nb_batches)

            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch = x_adv[batch_index_1:batch_index_2]
            batch_labels = targets[batch_index_1:batch_index_2]

            x_adv[batch_index_1:batch_index_2] = self._generate_batch(batch, batch_labels, cost_matrix)

        logger.info(
            "Success rate of attack: %.2f%%",
            100 * compute_success(self.estimator, x, y, x_adv, self.targeted, batch_size=self.batch_size),
        )

        return x_adv

    def _generate_batch(self, x: np.ndarray, targets: np.ndarray, cost_matrix: np.ndarray) -> np.ndarray:
        """
        Generate a batch of adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param targets: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)`.
        :param cost_matrix: A non-negative cost matrix.
        :return: Adversarial examples.
        """
        adv_x = x.copy().astype(ART_NUMPY_DTYPE)
        adv_x_best = x.copy().astype(ART_NUMPY_DTYPE)

        if self.targeted:
            err = np.argmax(self.estimator.predict(adv_x, batch_size=x.shape[0]), axis=1) == np.argmax(targets, axis=1)
        else:
            err = np.argmax(self.estimator.predict(adv_x, batch_size=x.shape[0]), axis=1) != np.argmax(targets, axis=1)

        err_best = err
        eps_ = np.ones(x.shape[0]) * self.eps

        for i in range(self.max_iter):
            adv_x = self._compute(adv_x, x, targets, cost_matrix, eps_, err)

            if self.targeted:
                err = np.argmax(self.estimator.predict(adv_x, batch_size=x.shape[0]), axis=1) == np.argmax(
                    targets, axis=1
                )

            else:
                err = np.argmax(self.estimator.predict(adv_x, batch_size=x.shape[0]), axis=1) != np.argmax(
                    targets, axis=1
                )

            if np.mean(err) > np.mean(err_best):
                err_best = err
                adv_x_best = adv_x.copy()

            if np.mean(err) == 1:
                break

            if (i + 1) % self.eps_iter == 0:
                eps_[~err] *= self.eps_factor

        return adv_x_best

    def _compute(
        self,
        x_adv: np.ndarray,
        x_init: np.ndarray,
        y: np.ndarray,
        cost_matrix: np.ndarray,
        eps: np.ndarray,
        err: np.ndarray,
    ) -> np.ndarray:
        """
        Compute adversarial examples for one iteration.

        :param x_adv: Current adversarial examples.
        :param x_init: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param cost_matrix: A non-negative cost matrix.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param err: Current successful adversarial examples.
        :return: Adversarial examples.
        """
        # Compute and apply perturbation
        x_adv[~err] = self._compute_apply_perturbation(x_adv, y, cost_matrix)[~err]

        # Do projection
        x_adv[~err] = self._apply_projection(x_adv, x_init, cost_matrix, eps)[~err]

        # Clip x_adv
        if self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
            x_adv = np.clip(x_adv, clip_min, clip_max)

        return x_adv

    def _compute_apply_perturbation(self, x: np.ndarray, y: np.ndarray, cost_matrix: np.ndarray) -> np.ndarray:
        """
        Compute and apply perturbations.

        :param x: Current adversarial examples.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param cost_matrix: A non-negative cost matrix.
        :return: Adversarial examples.
        """
        # Pick a small scalar to avoid division by 0
        tol = 10e-8

        # Get gradient wrt loss; invert it if attack is targeted
        grad = self.estimator.loss_gradient(x, y) * (1 - 2 * int(self.targeted))

        # Apply norm bound
        if self.norm == "inf":
            grad = np.sign(grad)
            x_adv = x + self.eps_step * grad

        elif self.norm == "1":
            ind = tuple(range(1, len(x.shape)))
            grad = grad / (np.sum(np.abs(grad), axis=ind, keepdims=True) + tol)
            x_adv = x + self.eps_step * grad

        elif self.norm == "2":
            ind = tuple(range(1, len(x.shape)))
            grad = grad / (np.sqrt(np.sum(np.square(grad), axis=ind, keepdims=True)) + tol)
            x_adv = x + self.eps_step * grad

        elif self.norm == "wasserstein":
            x_adv = self._conjugate_sinkhorn(x, grad, cost_matrix)

        else:
            raise NotImplementedError(
                "Values of `norm` different from `1`, `2`, `inf` and `wasserstein` are currently not supported."
            )

        return x_adv

    def _apply_projection(
        self, x: np.ndarray, x_init: np.ndarray, cost_matrix: np.ndarray, eps: np.ndarray
    ) -> np.ndarray:
        """
        Apply projection on the ball of size `eps`.

        :param x: Current adversarial examples.
        :param x_init: An array with the original inputs.
        :param cost_matrix: A non-negative cost matrix.
        :param eps: Maximum perturbation that the attacker can introduce.
        :return: Adversarial examples.
        """
        # Pick a small scalar to avoid division by 0
        tol = 10e-8

        if self.ball == "2":
            values = x - x_init
            values_tmp = values.reshape((values.shape[0], -1))

            values_tmp = values_tmp * np.expand_dims(
                np.minimum(1.0, eps / (np.linalg.norm(values_tmp, axis=1) + tol)), axis=1
            )

            values = values_tmp.reshape(values.shape)

            x_adv = values + x_init

        elif self.ball == "1":
            values = x - x_init
            values_tmp = values.reshape((values.shape[0], -1))

            values_tmp = values_tmp * np.expand_dims(
                np.minimum(1.0, eps / (np.linalg.norm(values_tmp, axis=1, ord=1) + tol)), axis=1
            )

            values = values_tmp.reshape(values.shape)
            x_adv = values + x_init

        elif self.ball == "inf":
            values = x - x_init
            values_tmp = values.reshape((values.shape[0], -1))

            values_tmp = np.sign(values_tmp) * np.minimum(abs(values_tmp), np.expand_dims(eps, -1))

            values = values_tmp.reshape(values.shape)
            x_adv = values + x_init

        elif self.ball == "wasserstein":
            x_adv = self._projected_sinkhorn(x, x_init, cost_matrix, eps)

        else:
            raise NotImplementedError(
                "Values of `ball` different from `1`, `2`, `inf` and `wasserstein` are currently not supported."
            )

        return x_adv

    def _conjugate_sinkhorn(self, x: np.ndarray, grad: np.ndarray, cost_matrix: np.ndarray) -> np.ndarray:
        """
        The conjugate sinkhorn_optimizer.

        :param x: Current adversarial examples.
        :param grad: The loss gradients.
        :param cost_matrix: A non-negative cost matrix.
        :return: Adversarial examples.
        """
        # Normalize inputs
        normalization = x.reshape(x.shape[0], -1).sum(-1).reshape(x.shape[0], 1, 1, 1)
        x = x.copy() / normalization

        # Dimension size for each example
        m = np.prod(x.shape[1:])

        # Initialize
        alpha = np.log(np.ones(x.shape) / m) + 0.5
        exp_alpha = np.exp(-alpha)

        beta = -self.regularization * grad
        beta = beta.astype(np.float64)
        exp_beta = np.exp(-beta)

        # Check for overflow
        if (exp_beta == np.inf).any():
            raise ValueError("Overflow error in `_conjugate_sinkhorn` for exponential beta.")

        cost_matrix_new = cost_matrix.copy() + 1
        cost_matrix_new = np.expand_dims(np.expand_dims(cost_matrix_new, 0), 0)

        i_nonzero = self._batch_dot(x, self._local_transport(cost_matrix_new, grad, self.kernel_size)) != 0
        i_nonzero_ = np.zeros(alpha.shape).astype(bool)
        i_nonzero_[:, :, :, :] = np.expand_dims(np.expand_dims(np.expand_dims(i_nonzero, -1), -1), -1)

        psi = np.ones(x.shape[0])

        var_k = np.expand_dims(np.expand_dims(np.expand_dims(psi, -1), -1), -1)
        var_k = np.exp(-var_k * cost_matrix - 1)

        convergence = -np.inf

        for _ in range(self.conjugate_sinkhorn_max_iter):
            # Block coordinate descent iterates
            x[x == 0.0] = EPS_LOG  # Prevent divide by zero in np.log
            alpha[i_nonzero_] = (np.log(self._local_transport(var_k, exp_beta, self.kernel_size)) - np.log(x))[
                i_nonzero_
            ]
            exp_alpha = np.exp(-alpha)

            # Newton step
            var_g = -self.eps_step + self._batch_dot(
                exp_alpha, self._local_transport(cost_matrix * var_k, exp_beta, self.kernel_size)
            )

            var_h = -self._batch_dot(
                exp_alpha, self._local_transport(cost_matrix * cost_matrix * var_k, exp_beta, self.kernel_size)
            )

            delta = var_g / var_h

            # Ensure psi >= 0
            tmp = np.ones(delta.shape)
            neg = psi - tmp * delta < 0

            while neg.any() and np.min(tmp) > 1e-2:
                tmp[neg] /= 2
                neg = psi - tmp * delta < 0

            psi[i_nonzero] = np.maximum(psi - tmp * delta, 0)[i_nonzero]

            # Update K
            var_k = np.expand_dims(np.expand_dims(np.expand_dims(psi, -1), -1), -1)
            var_k = np.exp(-var_k * cost_matrix - 1)

            # Check for convergence
            next_convergence = self._conjugated_sinkhorn_evaluation(x, alpha, exp_alpha, exp_beta, psi, var_k)

            if (np.abs(convergence - next_convergence) <= 1e-4 + 1e-4 * np.abs(next_convergence)).all():
                break

            convergence = next_convergence

        result = exp_beta * self._local_transport(var_k, exp_alpha, self.kernel_size)
        result[~i_nonzero] = 0
        result *= normalization

        return result

    def _projected_sinkhorn(
        self, x: np.ndarray, x_init: np.ndarray, cost_matrix: np.ndarray, eps: np.ndarray
    ) -> np.ndarray:
        """
        The projected sinkhorn_optimizer.

        :param x: Current adversarial examples.
        :param x_init: An array with the original inputs.
        :param cost_matrix: A non-negative cost matrix.
        :param eps: Maximum perturbation that the attacker can introduce.
        :return: Adversarial examples.
        """
        # Normalize inputs
        normalization = x_init.reshape(x.shape[0], -1).sum(-1).reshape(x.shape[0], 1, 1, 1)
        x = x.copy() / normalization
        x_init = x_init.copy() / normalization

        # Dimension size for each example
        m = np.prod(x_init.shape[1:])

        # Initialize
        beta = np.log(np.ones(x.shape) / m)
        exp_beta = np.exp(-beta)

        psi = np.ones(x.shape[0])

        var_k = np.expand_dims(np.expand_dims(np.expand_dims(psi, -1), -1), -1)
        var_k = np.exp(-var_k * cost_matrix - 1)

        convergence = -np.inf

        for _ in range(self.projected_sinkhorn_max_iter):
            # Block coordinate descent iterates
            x_init[x_init == 0.0] = EPS_LOG  # Prevent divide by zero in np.log
            alpha = np.log(self._local_transport(var_k, exp_beta, self.kernel_size)) - np.log(x_init)
            exp_alpha = np.exp(-alpha)

            beta = (
                self.regularization
                * np.exp(self.regularization * x)
                * self._local_transport(var_k, exp_alpha, self.kernel_size)
            )
            beta[beta > 1e-10] = np.real(lambertw(beta[beta > 1e-10]))
            beta -= self.regularization * x
            exp_beta = np.exp(-beta)

            # Newton step
            var_g = -eps + self._batch_dot(
                exp_alpha, self._local_transport(cost_matrix * var_k, exp_beta, self.kernel_size)
            )

            var_h = -self._batch_dot(
                exp_alpha, self._local_transport(cost_matrix * cost_matrix * var_k, exp_beta, self.kernel_size)
            )

            delta = var_g / var_h

            # Ensure psi >= 0
            tmp = np.ones(delta.shape)
            neg = psi - tmp * delta < 0

            while neg.any() and np.min(tmp) > 1e-2:
                tmp[neg] /= 2
                neg = psi - tmp * delta < 0

            psi = np.maximum(psi - tmp * delta, 0)

            # Update K
            var_k = np.expand_dims(np.expand_dims(np.expand_dims(psi, -1), -1), -1)
            var_k = np.exp(-var_k * cost_matrix - 1)

            # Check for convergence
            next_convergence = self._projected_sinkhorn_evaluation(
                x,
                x_init,
                alpha,
                exp_alpha,
                beta,
                exp_beta,
                psi,
                var_k,
                eps,
            )

            if (np.abs(convergence - next_convergence) <= 1e-4 + 1e-4 * np.abs(next_convergence)).all():
                break

            convergence = next_convergence

        result = (beta / self.regularization + x) * normalization

        return result

    @staticmethod
    def _compute_cost_matrix(var_p: int, kernel_size: int) -> np.ndarray:
        """
        Compute the default cost matrix.

        :param var_p: The p-wasserstein distance.
        :param kernel_size: Kernel size for computing the cost matrix.
        :return: The cost matrix.
        """
        center = kernel_size // 2
        cost_matrix = np.zeros((kernel_size, kernel_size))

        for i in range(kernel_size):
            for j in range(kernel_size):
                # The code of the paper of this attack (https://arxiv.org/abs/1902.07906) implements the cost as:
                # cost_matrix[i, j] = (abs(i - center) ** 2 + abs(j - center) ** 2) ** (p / 2)
                # which only can reproduce L2-norm for p=1 correctly
                cost_matrix[i, j] = (abs(i - center) ** var_p + abs(j - center) ** var_p) ** (1 / var_p)

        return cost_matrix

    @staticmethod
    def _batch_dot(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute batch dot product.

        :param x: Sample batch.
        :param y: Sample batch.
        :return: Batch dot product.
        """
        batch_size = x.shape[0]
        assert batch_size == y.shape[0]

        var_x_ = x.reshape(batch_size, 1, -1)
        var_y_ = y.reshape(batch_size, -1, 1)

        result = np.matmul(var_x_, var_y_).reshape(batch_size)

        return result

    @staticmethod
    def _unfold(x: np.ndarray, kernel_size: int, padding: int) -> np.ndarray:
        """
        Extract sliding local blocks from a batched input.

        :param x: A batched input of shape `batch x channel x width x height`.
        :param kernel_size: Kernel size for computing the cost matrix.
        :param padding: Controls the amount of implicit zero-paddings on both sides for padding number of points
            for each dimension before reshaping.
        :return: Sliding local blocks.
        """
        # Do padding
        shape = tuple(np.array(x.shape[2:]) + padding * 2)
        x_pad = np.zeros(x.shape[:2] + shape)
        x_pad[:, :, padding : (shape[0] - padding), padding : (shape[1] - padding)] = x

        # Do unfolding
        res_dim_0 = x.shape[0]
        res_dim_1 = x.shape[1] * kernel_size ** 2
        res_dim_2 = (shape[0] - kernel_size + 1) * (shape[1] - kernel_size + 1)
        result = np.zeros((res_dim_0, res_dim_1, res_dim_2))

        for i in range(shape[0] - kernel_size + 1):
            for j in range(shape[1] - kernel_size + 1):
                patch = x_pad[:, :, i : (i + kernel_size), j : (j + kernel_size)]
                patch = patch.reshape(x.shape[0], -1)
                result[:, :, i * (shape[1] - kernel_size + 1) + j] = patch

        return result

    def _local_transport(self, var_k: np.ndarray, x: np.ndarray, kernel_size: int) -> np.ndarray:
        """
        Compute local transport.

        :param var_k: K parameter in Algorithm 2 of the paper ``Wasserstein Adversarial Examples via Projected
            Sinkhorn Iterations``.
        :param x: An array to apply local transport.
        :param kernel_size: Kernel size for computing the cost matrix.
        :return: Local transport result.
        """
        # Compute number of channels
        num_channels = x.shape[1 if self.estimator.channels_first else 3]

        # Expand channels
        var_k = np.repeat(var_k, num_channels, axis=1)

        # Swap channels to prepare for local transport computation
        if not self.estimator.channels_first:
            x = np.swapaxes(x, 1, 3)

        # Compute local transport
        unfold_x = self._unfold(x=x, kernel_size=kernel_size, padding=kernel_size // 2)
        unfold_x = unfold_x.swapaxes(-1, -2)
        unfold_x = unfold_x.reshape(*unfold_x.shape[:-1], num_channels, kernel_size ** 2)
        unfold_x = unfold_x.swapaxes(-2, -3)

        tmp_k = var_k.reshape(var_k.shape[0], num_channels, -1)
        tmp_k = np.expand_dims(tmp_k, -1)

        result = np.matmul(unfold_x, tmp_k)
        result = np.squeeze(result, -1)
        result = result.reshape(*result.shape[:-1], x.shape[-2], x.shape[-1])

        # Swap channels for final result
        if not self.estimator.channels_first:
            result = np.swapaxes(result, 1, 3)

        return result

    def _projected_sinkhorn_evaluation(
        self,
        x: np.ndarray,
        x_init: np.ndarray,
        alpha: np.ndarray,
        exp_alpha: np.ndarray,
        beta: np.ndarray,
        exp_beta: np.ndarray,
        psi: np.ndarray,
        var_k: np.ndarray,
        eps: np.ndarray,
    ) -> np.ndarray:
        """
        Function to evaluate the objective of the projected sinkhorn optimizer.

        :param x: Current adversarial examples.
        :param x_init: An array with the original inputs.
        :param alpha: Alpha parameter in Algorithm 2 of the paper ``Wasserstein Adversarial Examples via Projected
            Sinkhorn Iterations``.
        :param exp_alpha: Exponential of alpha.
        :param beta: Beta parameter in Algorithm 2 of the paper ``Wasserstein Adversarial Examples via Projected
            Sinkhorn Iterations``.
        :param exp_beta: Exponential of beta.
        :param psi: Psi parameter in Algorithm 2 of the paper ``Wasserstein Adversarial Examples via Projected
            Sinkhorn Iterations``.
        :param var_k: K parameter in Algorithm 2 of the paper ``Wasserstein Adversarial Examples via Projected
            Sinkhorn Iterations``.
        :param eps: Maximum perturbation that the attacker can introduce.
        :return: Evaluation result.
        """
        return (
            -0.5 / self.regularization * self._batch_dot(beta, beta)
            - psi * eps
            - self._batch_dot(np.minimum(alpha, 1e10), x_init)
            - self._batch_dot(np.minimum(beta, 1e10), x)
            - self._batch_dot(exp_alpha, self._local_transport(var_k, exp_beta, self.kernel_size))
        )

    def _conjugated_sinkhorn_evaluation(
        self,
        x: np.ndarray,
        alpha: np.ndarray,
        exp_alpha: np.ndarray,
        exp_beta: np.ndarray,
        psi: np.ndarray,
        var_k: np.ndarray,
    ) -> np.ndarray:
        """
        Function to evaluate the objective of the conjugated sinkhorn optimizer.

        :param x: Current adversarial examples.
        :param alpha: Alpha parameter in the conjugated sinkhorn optimizer of the paper ``Wasserstein Adversarial
            Examples via Projected Sinkhorn Iterations``.
        :param exp_alpha: Exponential of alpha.
        :param exp_beta: Exponential of beta parameter in the conjugated sinkhorn optimizer of the paper ``Wasserstein
            Adversarial Examples via Projected Sinkhorn Iterations``.
        :param psi: Psi parameter in the conjugated sinkhorn optimizer of the paper ``Wasserstein Adversarial
            Examples via Projected Sinkhorn Iterations``.
        :param var_k: K parameter in the conjugated sinkhorn optimizer of the paper ``Wasserstein Adversarial Examples
            via Projected Sinkhorn Iterations``.
        :return: Evaluation result.
        """
        return (
            -psi * self.eps_step
            - self._batch_dot(np.minimum(alpha, 1e38), x)
            - self._batch_dot(exp_alpha, self._local_transport(var_k, exp_beta, self.kernel_size))
        )

    def _check_params(self) -> None:
        if not isinstance(self.targeted, bool):
            raise ValueError("The flag `targeted` has to be of type bool.")

        if self.regularization <= 0:
            raise ValueError("The entropy regularization has to be greater than 0.")

        if not isinstance(self.p, (int, np.int)):
            raise TypeError("The p-wasserstein has to be of type integer.")

        if self.p < 1:
            raise ValueError("The p-wasserstein must be larger or equal to 1.")

        if not isinstance(self.kernel_size, (int, np.int)):
            raise TypeError("The kernel size has to be of type integer.")

        if self.kernel_size % 2 != 1:
            raise ValueError("Need odd kernel size.")

        # Check if order of the norm is acceptable given current implementation
        if self.norm not in ["inf", "1", "2", "wasserstein"]:
            raise ValueError("Norm order must be either `inf`, `1`, `2` or `wasserstein`.")

        # Check if order of the ball is acceptable given current implementation
        if self.ball not in ["inf", "1", "2", "wasserstein"]:
            raise ValueError("Ball order must be either `inf`, `1`, `2` or `wasserstein`.")

        if self.eps <= 0:
            raise ValueError("The perturbation size `eps` has to be positive.")

        if self.eps_step <= 0:
            raise ValueError("The perturbation step-size `eps_step` has to be positive.")

        if self.norm == "inf" and self.eps_step > self.eps:
            raise ValueError(
                "The iteration step `eps_step` has to be smaller than or equal to the total attack budget `eps`."
            )

        if self.eps_iter <= 0:
            raise ValueError("The number of epsilon iterations `eps_iter` has to be a positive integer.")

        if self.eps_factor <= 1:
            raise ValueError("The epsilon factor must be larger than 1.")

        if self.max_iter <= 0:
            raise ValueError("The number of iterations `max_iter` has to be a positive integer.")

        if self.conjugate_sinkhorn_max_iter <= 0:
            raise ValueError("The number of iterations `conjugate_sinkhorn_max_iter` has to be a positive integer.")

        if self.projected_sinkhorn_max_iter <= 0:
            raise ValueError("The number of iterations `projected_sinkhorn_max_iter` has to be a positive integer.")

        if self.batch_size <= 0:
            raise ValueError("The batch size `batch_size` has to be positive.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")
