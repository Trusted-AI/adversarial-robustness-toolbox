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

import numpy as np

from art.config import ART_NUMPY_DTYPE
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassGradientsMixin
from art.attacks.attack import EvasionAttack
from art.utils import compute_success, get_labels_np_array
from art.utils import check_and_transform_label_format

logger = logging.getLogger(__name__)


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
        "alpha",
        "norm",
        "ball",
        "epsilon",
        "epsilon_factor",
        "max_iter",
        "conjugate_sinkhorn_max_iter",
        "projected_sinkhorn_max_iter",
        "batch_size",
    ]

    _estimator_requirements = (BaseEstimator, ClassGradientsMixin)

    def __init__(
        self,
        estimator,
        targeted=False,
        regularization=3000,
        p=2,
        kernel_size=5,
        alpha=0.1,
        norm='wasserstein',
        ball='wasserstein',
        epsilon=0.001,
        epsilon_factor=1.17,
        max_iter=400,
        conjugate_sinkhorn_max_iter=400,
        projected_sinkhorn_max_iter=400,
        batch_size=1,
    ):
        """
        Create a Wasserstein attack instance.

        :param estimator: A trained estimator.
        :type estimator: :class:`.BaseEstimator`
        """
        super(Wasserstein, self).__init__(estimator=estimator)

        kwargs = {
            "targeted": targeted,
            "regularization": regularization,
            "p": p,
            "kernel_size": kernel_size,
            "alpha": alpha,
            "norm": norm,
            "ball": ball,
            "epsilon": epsilon,
            "epsilon_factor": epsilon_factor,
            "max_iter": max_iter,
            "conjugate_sinkhorn_max_iter": conjugate_sinkhorn_max_iter,
            "projected_sinkhorn_max_iter": projected_sinkhorn_max_iter,
            "batch_size": batch_size,
        }
        Wasserstein.set_params(self, **kwargs)

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
        :param cost_matrix: A non-negative cost matrix.
        :type cost_matrix: `np.ndarray`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        y = check_and_transform_label_format(y, self.estimator.nb_classes)
        x_adv = x.astype(ART_NUMPY_DTYPE)

        if y is None:
            # Throw error if attack is targeted, but no targets are provided
            if self.targeted:
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")

            # Use model predictions as correct outputs
            targets = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))
        else:
            targets = y

        # Compute the cost matrix if needed
        cost_matrix = kwargs.get("cost_matrix")
        if cost_matrix is None:
            cost_matrix = self._compute_cost_matrix(self.p, self.kernel_size)

        # Compute perturbation with implicit batching
        nb_batches = int(np.ceil(x.shape[0] / float(self.batch_size)))
        for batch_id in range(nb_batches):
            logger.debug("Processing batch %i out of %i", batch_id, nb_batches)
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch = x[batch_index_1: batch_index_2]
            batch_labels = targets[batch_index_1: batch_index_2]

            x_adv[batch_index_1:batch_index_2] = self._generate_batch(
                batch,
                batch_labels,
                cost_matrix,
                self.max_iter,
                self.norm,
                self.ball,
                self.targeted,
                self.eps,
                self.alpha,
                self.regularization,
                self.conjugate_sinkhorn_max_iter,
                self.projected_sinkhorn_max_iter,
            )

        return x_adv

    @staticmethod
    def _compute_cost_matrix(p, kernel_size):
        """
        Compute the default cost matrix.

        :param p: The p-wasserstein distance.
        :type p: `int`
        :param kernel_size: Kernel size for computing the cost matrix.
        :type kernel_size: `int`
        :return: The cost matrix.
        :rtype: `np.ndarray`
        """
        center = kernel_size // 2
        cost_matrix = np.zeros((kernel_size, kernel_size))

        for i in range(kernel_size):
            for j in range(kernel_size):
                cost_matrix[i, j] = (abs(i - center) ** 2 + abs(j - center) ** 2) ** (p / 2)

        return cost_matrix

    def _generate_batch(
            self,
            x,
            targets,
            cost_matrix,
            max_iter,
            norm,
            ball,
            targeted,
            eps,
            alpha,
            regularization,
            conjugate_sinkhorn_max_iter,
            projected_sinkhorn_max_iter,
    ):
        """
        Generate a batch of adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :type x: `np.ndarray`
        :param targets: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)`.
        :type targets: `np.ndarray`
        :param cost_matrix: A non-negative cost matrix.
        :type cost_matrix: `np.ndarray`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param norm: The norm of the adversarial perturbation. Possible values: `inf`, `1`, `2` or `wasserstein`.
        :type norm: `string`
        :param ball: The ball of the adversarial perturbation. Possible values: `inf`, `1`, `2` or `wasserstein`.
        :type ball: `string`
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False)
        :type targeted: `bool`
        :param eps: Maximum perturbation that the attacker can introduce.
        :type eps: `float`
        :param alpha: Attack step size (input variation) at each iteration.
        :type alpha: `float`
        :param regularization: Entropy regularization.
        :type regularization: `float`
        :param conjugate_sinkhorn_max_iter: The maximum number of iterations for the conjugate sinkhorn optimizer.
        :type conjugate_sinkhorn_max_iter: `int`
        :param projected_sinkhorn_max_iter: The maximum number of iterations for the projected sinkhorn optimizer.
        :type projected_sinkhorn_max_iter: `int`
        :return: Adversarial examples.
        :rtype: `np.ndarray`
        """
        adv_x = x.copy()
        for _ in range(max_iter):
            adv_x = self._compute(
                adv_x,
                x,
                targets,
                cost_matrix,
                norm,
                ball,
                targeted,
                eps,
                alpha,
                regularization,
                conjugate_sinkhorn_max_iter,
                projected_sinkhorn_max_iter,
            )

        return adv_x

    def _compute(
            self,
            x,
            x_init,
            y,
            cost_matrix,
            norm,
            ball,
            targeted,
            eps,
            alpha,
            regularization,
            conjugate_sinkhorn_max_iter,
            projected_sinkhorn_max_iter,
    ):
        """
        Compute adversarial examples for one iteration.

        :param x: Current adversarial examples.
        :type x: `np.ndarray`
        :param x_init: An array with the original inputs.
        :type x_init: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :type y: `np.ndarray`
        :param cost_matrix: A non-negative cost matrix.
        :type cost_matrix: `np.ndarray`
        :param norm: The norm of the adversarial perturbation. Possible values: `inf`, `1`, `2` or `wasserstein`.
        :type norm: `string`
        :param ball: The ball of the adversarial perturbation. Possible values: `inf`, `1`, `2` or `wasserstein`.
        :type ball: `string`
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False)
        :type targeted: `bool`
        :param eps: Maximum perturbation that the attacker can introduce.
        :type eps: `float`
        :param alpha: Attack step size (input variation) at each iteration.
        :type alpha: `float`
        :param regularization: Entropy regularization.
        :type regularization: `float`
        :param conjugate_sinkhorn_max_iter: The maximum number of iterations for the conjugate sinkhorn optimizer.
        :type conjugate_sinkhorn_max_iter: `int`
        :param projected_sinkhorn_max_iter: The maximum number of iterations for the projected sinkhorn optimizer.
        :type projected_sinkhorn_max_iter: `int`
        :return: Adversarial examples.
        :rtype: `np.ndarray`
        """
        # Compute and apply perturbation
        x_adv = self._compute_apply_perturbation(
            x,
            y,
            cost_matrix,
            norm,
            targeted,
            alpha,
            regularization,
            conjugate_sinkhorn_max_iter,
        )

        # Do projection
        x_adv = self._apply_projection(
            x_adv,
            x_init,
            cost_matrix,
            ball,
            eps,
            regularization,
            projected_sinkhorn_max_iter
        )

        # Clip x_adv
        if hasattr(self.estimator, "clip_values") and self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
            x_adv = np.clip(x_adv, clip_min, clip_max)

        return x_adv

    def _compute_apply_perturbation(
            self,
            x,
            y,
            cost_matrix,
            norm,
            targeted,
            alpha,
            regularization,
            conjugate_sinkhorn_max_iter,
    ):
        """
        Compute and apply perturbations.

        :param x: Current adversarial examples.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :type y: `np.ndarray`
        :param cost_matrix: A non-negative cost matrix.
        :type cost_matrix: `np.ndarray`
        :param norm: The norm of the adversarial perturbation. Possible values: `inf`, `1`, `2` or `wasserstein`.
        :type norm: `string`
        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False)
        :type targeted: `bool`
        :param alpha: Attack step size (input variation) at each iteration.
        :type alpha: `float`
        :param regularization: Entropy regularization.
        :type regularization: `float`
        :param conjugate_sinkhorn_max_iter: The maximum number of iterations for the conjugate sinkhorn optimizer.
        :type conjugate_sinkhorn_max_iter: `int`
        :return: Adversarial examples.
        :rtype: `np.ndarray`
        """
        # Pick a small scalar to avoid division by 0
        tol = 10e-8

        # Get gradient wrt loss; invert it if attack is targeted
        grad = self.estimator.loss_gradient(x, y) * (1 - 2 * int(targeted))

        # Apply norm bound
        if norm == 'inf':
            grad = np.sign(grad)
            x_adv = x + alpha * grad

        elif norm == '1':
            ind = tuple(range(1, len(x.shape)))
            grad = grad / (np.sum(np.abs(grad), axis=ind, keepdims=True) + tol)
            x_adv = x + alpha * grad

        elif norm == '2':
            ind = tuple(range(1, len(x.shape)))
            grad = grad / (np.sqrt(np.sum(np.square(grad), axis=ind, keepdims=True)) + tol)
            x_adv = x + alpha * grad

        elif norm == 'wasserstein':
            x_adv = self._conjugate_sinkhorn_optimizer(
                x,
                grad,
                cost_matrix,
                alpha,
                regularization,
                conjugate_sinkhorn_max_iter,
            )

        else:
            raise NotImplementedError(
                "Values of `norm` different from `1`, `2`, `inf` and `wasserstein` are currently not supported."
            )

        return x_adv

    def _conjugate_sinkhorn_optimizer(
            self,
            x,
            grad,
            cost_matrix,
            alpha,
            regularization,
            conjugate_sinkhorn_max_iter,
    ):
        """

        :return:
        """
        return 1

    def _apply_projection(
            self,
            x,
            x_init,
            cost_matrix,
            ball,
            eps,
            regularization,
            projected_sinkhorn_max_iter
    ):
        """
        Apply projection on the ball of size `eps`.

        :param x: Current adversarial examples.
        :type x: `np.ndarray`
        :param x_init: An array with the original inputs.
        :type x_init: `np.ndarray`
        :param cost_matrix: A non-negative cost matrix.
        :type cost_matrix: `np.ndarray`
        :param ball: The ball of the adversarial perturbation. Possible values: `inf`, `1`, `2` or `wasserstein`.
        :type ball: `string`
        :param eps: Maximum perturbation that the attacker can introduce.
        :type eps: `float`
        :param regularization: Entropy regularization.
        :type regularization: `float`
        :param projected_sinkhorn_max_iter: The maximum number of iterations for the projected sinkhorn optimizer.
        :type projected_sinkhorn_max_iter: `int`
        :return: Adversarial examples.
        :rtype: `np.ndarray`
        """
        # Pick a small scalar to avoid division by 0
        tol = 10e-8

        if ball == '2':
            values = x - x_init
            values_tmp = values.reshape((values.shape[0], -1))

            values_tmp = values_tmp * np.expand_dims(
                np.minimum(1.0, eps / (np.linalg.norm(values_tmp, axis=1) + tol)), axis=1
            )

            values = values_tmp.reshape(values.shape)
            x_adv = values + x_init

        elif ball == '1':
            values = x - x_init
            values_tmp = values.reshape((values.shape[0], -1))

            values_tmp = values_tmp * np.expand_dims(
                np.minimum(1.0, eps / (np.linalg.norm(values_tmp, axis=1, ord=1) + tol)), axis=1
            )

            values = values_tmp.reshape(values.shape)
            x_adv = values + x_init

        elif ball == 'inf':
            values = x - x_init
            values_tmp = values.reshape((values.shape[0], -1))

            values_tmp = np.sign(values_tmp) * np.minimum(abs(values_tmp), eps)

            values = values_tmp.reshape(values.shape)
            x_adv = values + x_init

        elif ball == 'wasserstein':
            x_adv = self._projected_sinkhorn_optimizer(
                x,
                x_init,
                cost_matrix,
                eps,
                regularization,
                projected_sinkhorn_max_iter
            )

        else:
            raise NotImplementedError(
                "Values of `ball` different from `1`, `2`, `inf` and `wasserstein` are currently not supported."
            )

        return x_adv

    def set_params(self, **kwargs):
        """Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param confidence: Confidence of adversarial examples: a higher value produces examples that are farther away,
               from the original input, but classified with higher confidence as the target class.
        :type confidence: `float`
        :param targeted: Should the attack target one specific class
        :type targeted: `bool`
        :param learning_rate: The learning rate for the attack algorithm. Smaller values produce better results but are
               slower to converge.
        :type learning_rate: `float`
        :param binary_search_steps: number of times to adjust constant with binary search (positive value)
        :type binary_search_steps: `int`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param initial_const: (optional float, positive) The initial trade-off constant c to use to tune the relative
               importance of distance and confidence. If binary_search_steps is large,
               the initial constant is not important. The default value 1e-4 is suggested in Carlini and Wagner (2016).
        :type initial_const: `float`
        :param max_halving: Maximum number of halving steps in the line search optimization.
        :type max_halving: `int`
        :param max_doubling: Maximum number of doubling steps in the line search optimization.
        :type max_doubling: `int`
        :param batch_size: Internal size of batches on which adversarial samples are generated.
        :type batch_size: `int`
        """
        # Save attack-specific parameters
        super(Wasserstein, self).set_params(**kwargs)


        if kernel_size % 2 != 1:
            raise ValueError("Need odd kernel size")


        return True


