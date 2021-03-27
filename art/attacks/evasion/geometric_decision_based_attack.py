# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2021
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
This module implements the Geometric Decision-based Attack (GeoDA), a black-box attack requiring class predictions.

| Paper link: https://arxiv.org/abs/2003.06468
"""
import os
import math
import logging
from typing import Optional, Union, TYPE_CHECKING

import numpy as np
from tqdm.auto import trange

from art.attacks.attack import EvasionAttack
from art.config import ART_NUMPY_DTYPE
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import check_and_transform_label_format

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)


class GeoDA(EvasionAttack):
    """
    Implementation of the Geometric Decision-based Attack (GeoDA), a black-box attack requiring class predictions.
    Based on reference implementation: https://github.com/thisisalirah/GeoDA

    | Paper link: https://arxiv.org/abs/2003.06468
    """

    attack_params = EvasionAttack.attack_params + [
        "batch_size",
        "norm",
        "sub_dim",
        "max_iter",
        "targeted",
        "verbose",
    ]

    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(
        self,
        estimator: "CLASSIFIER_TYPE",
        batch_size: int = 64,
        norm: Union[int, float, str] = 2,
        sub_dim: int = 10,
        max_iter: int = 4000,
        targeted: bool = True,
        verbose: bool = True,
    ) -> None:
        """
        Create a Geometric Decision-based Attack instance.

        :param estimator: A trained classifier.
        :param batch_size: The size of the batch used by the estimator during inference.
        :param norm: The norm of the adversarial perturbation. Possible values: "inf", np.inf, 1 or 2.
        :param sub_dim: dimensionality of 2D frequency space (DCT).
        :param max_iter: Maximum number of iterations.
        :param targeted: Should the attack target one specific class.
        :param verbose: Show progress bars.
        """
        super().__init__(estimator=estimator)

        self.batch_size = batch_size
        self.norm = norm
        self.sub_dim = sub_dim
        self.max_iter = max_iter
        self._targeted = targeted
        self.verbose = verbose
        self._check_params()

        self.sub_basis = None
        self.nb_calls = 0
        self.clip_min = 0.0
        self.clip_max = 0.0

        # Optimal number of iterations
        mu = 0.6
        iteration = round(self.max_iter / 500)
        q_opt_it = int(self.max_iter - iteration * 25)
        q_opt_iter, iterate = self._opt_query_iteration(q_opt_it, iteration, mu)
        q_opt_it = int(self.max_iter - iterate * 25)
        self.q_opt_iter, self.iterate = self._opt_query_iteration(q_opt_it, iteration, mu)

    @staticmethod
    def _generate_2d_dct_basis(sub_dim, res):
        def alpha(a, n):
            """
            Get alpha.
            """
            if a == 0:
                return math.sqrt(1.0 / n)
            else:
                return math.sqrt(2.0 / n)

        def dct(x, y, v, u, n):
            """
            Get 2D DCT.
            """
            return (
                alpha(u, n)
                * alpha(v, n)
                * math.cos(((2 * x + 1) * (u * math.pi)) / (2 * n))
                * math.cos(((2 * y + 1) * (v * math.pi)) / (2 * n))
            )

        # We can get different frequencies by setting u and v
        u_max = sub_dim
        v_max = sub_dim

        dct_basis = []
        for u in range(u_max):
            for v in range(v_max):
                basis = np.zeros((res, res))
                for y in range(res):
                    for x in range(res):
                        basis[y, x] = dct(x, y, v, u, max(res, v_max))
                dct_basis.append(basis)
        dct_basis = np.mat(np.reshape(dct_basis, (v_max * u_max, res * res))).transpose()
        return dct_basis

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples.

        :param x: An array with the original inputs to be attacked.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,). If `self.targeted` is true, then `y` represents the target labels.
        :return: The adversarial examples.
        """
        y = check_and_transform_label_format(y, self.estimator.nb_classes, return_one_hot=True)

        # Assert that, if attack is targeted, y is provided
        if self.targeted and y is None:
            raise ValueError("Target labels `y` need to be provided for a targeted attack.")

        # Get clip_min and clip_max from the classifier or infer them from data
        if self.estimator.clip_values is not None:
            self.clip_min, self.clip_max = self.estimator.clip_values
        else:
            self.clip_min, self.clip_max = np.min(x), np.max(x)

        # Check for square input images
        if (self.estimator.channels_first and x.shape[2] != x.shape[3]) or (
            not self.estimator.channels_first and not x.shape[1] == x.shape[2]
        ):
            raise ValueError("Input images `x` have to be square.")

        # Create or load DCT basis
        image_size = x.shape[2]
        logger.info("Create or load DCT basis.")
        path = "2d_dct_basis_{}_{}.npy".format(self.sub_dim, image_size)
        if os.path.exists(path):
            self.sub_basis = np.load(path).astype(ART_NUMPY_DTYPE)
        else:
            self.sub_basis = self._generate_2d_dct_basis(sub_dim=self.sub_dim, res=image_size).astype(ART_NUMPY_DTYPE)
            np.save(path, self.sub_basis)

        # Reset number of calls
        self.nb_calls = 0
        tol = 0.0001

        # Random search
        x_random = self._find_random_adversarial(x=x, y=y)
        logger.info("Random search adversarial example is adversarial: %r" % self._is_adversarial(x_random, y))

        # Binary search
        x_boundary = self._bin_search(x, y, x_random, tol=tol)
        logger.info("Binary search example at boundary is adversarial: %r" % self._is_adversarial(x_boundary, y))

        grad = 0
        sigma = 0.0002

        x_adv = x

        for i in trange(self.iterate, desc="GeoDA - steps", disable=not self.verbose):
            grad_oi, ratios = self._black_grad_batch(x_boundary, self.q_opt_iter[i], sigma, self.batch_size, y)
            grad = grad_oi + grad
            x_adv = self._go_to_boundary(x, y, grad)
            x_adv = self._bin_search(x, y, x_adv, tol=tol)
            x_boundary = x_adv

        x_adv = np.clip(x_adv, a_min=self.clip_min, a_max=self.clip_max)

        return x_adv

    def _is_adversarial(self, x_adv: np.ndarray, y_true: np.ndarray) -> bool:
        """
        Check if example is adversarial.
        """
        y_prediction = self.estimator.predict(x=x_adv)
        return np.argmax(y_prediction, axis=1)[0] != np.argmax(y_true, axis=1)[0]

    def _find_random_adversarial(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Find an adversarial example by random search.
        """
        nb_calls = 0
        step_size = 0.02
        x_perturbed = x

        while not self._is_adversarial(x_perturbed, y):
            nb_calls += 1
            perturbation = np.random.normal(size=x.shape).astype(ART_NUMPY_DTYPE)
            x_perturbed = x + nb_calls * step_size * perturbation
            x_perturbed = np.clip(x_perturbed, a_min=self.clip_min, a_max=self.clip_max)

        self.nb_calls += nb_calls

        return x_perturbed

    def _bin_search(self, x, y, x_random, tol):
        """
        Find example on decision boundary between input and random sample.
        """
        x_adv = x_random
        x_cln = x

        while np.linalg.norm(x_adv.flatten() - x_cln.flatten(), ord=2) >= tol:
            self.nb_calls += 1
            x_mid = (x_cln + x_adv) / 2.0
            if self._is_adversarial(x_mid, y):
                x_adv = x_mid
            else:
                x_cln = x_mid

        return x_adv

    def _opt_query_iteration(self, Nq, T, eta):
        """
        Determine optimal distribution of number of queries.
        """
        coefficients = [eta ** (-2 * i / 3) for i in range(0, T)]
        coefficients[0] = 1 * coefficients[0]
        sum_coefficients = sum(coefficients)
        opt_q = [round(Nq * coefficients[i] / sum_coefficients) for i in range(0, T)]

        if opt_q[0] > 80:
            T = T + 1
            opt_q, T = self._opt_query_iteration(Nq, T, eta)
        elif opt_q[0] < 50:
            T = T - 1
            opt_q, T = self._opt_query_iteration(Nq, T, eta)

        return opt_q, T

    def _black_grad_batch(self, x_boundary, q_max, sigma, batch_size, original_label):

        self.nb_calls += q_max

        grad_tmp = []  # estimated gradients in each estimate_batch
        z = []  # sign of grad_tmp
        outs = []
        num_batches = math.ceil(q_max / batch_size)
        last_batch = q_max - (num_batches - 1) * batch_size
        all_noises = []

        for j in range(num_batches):
            if j == num_batches - 1:
                current_batch = self._sub_noise(last_batch, self.sub_basis)
                noisy_boundary = [x_boundary[0, :, :, :]] * last_batch + sigma * current_batch
            else:
                current_batch = self._sub_noise(batch_size, self.sub_basis)
                noisy_boundary = [x_boundary[0, :, :, :]] * batch_size + sigma * current_batch

            all_noises.append(current_batch)
            predict_labels = np.argmax(self.estimator.predict(noisy_boundary), axis=1).astype(int)
            outs.append(predict_labels)

        all_noise = np.concatenate(all_noises, axis=0)
        outs = np.concatenate(outs, axis=0)

        for i, predict_label in enumerate(outs):
            if predict_label == np.argmax(original_label, axis=1)[0]:
                z.append(1)
                grad_tmp.append(all_noise[i])
            else:
                z.append(-1)
                grad_tmp.append(-all_noise[i])

        grad = -(1 / q_max) * sum(grad_tmp)
        grad_f = grad[None, :, :, :]

        return grad_f, sum(z)

    def _go_to_boundary(self, x, y, grad):
        """
        Move towards decision boundary.
        """
        epsilon = 5
        nb_calls = 0
        x_perturbed = x

        if self.norm in [np.inf, "inf"]:
            grads = np.sign(grad) / np.linalg.norm(grad.flatten(), ord=2)
        else:
            grads = grad  # self.norm in [1, 2]

        while not self._is_adversarial(x_perturbed, y):
            nb_calls += 1

            if nb_calls > 100:
                logger.info("Moving towards decision boundary failed because of too many iterations.")
                break

            x_perturbed = x + (nb_calls * epsilon * grads[0])
            x_perturbed = np.clip(x_perturbed, a_min=self.clip_min, a_max=self.clip_max)

        self.nb_calls += nb_calls

        return x_perturbed

    def _sub_noise(self, num_noises, x):
        """
        Create subspace random perturbation.
        """
        noise = np.random.normal(size=(x.shape[1], 3 * num_noises))
        sub_noise = np.matmul(x, noise).transpose((1, 0)).astype(ART_NUMPY_DTYPE)
        r_list = sub_noise.reshape((num_noises,) + self.estimator.input_shape)
        return r_list

    def _check_params(self) -> None:

        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("The batch size has to be a positive integer.")

        if self.norm not in [1, 2, np.inf, "inf"]:
            raise ValueError('The argument norm has to be either 1, 2, np.inf, or "inf".')

        if not isinstance(self.sub_dim, int) or self.sub_dim <= 0:
            raise ValueError("The subspace dimension has to be a positive integer.")

        if not isinstance(self.max_iter, int) or self.max_iter <= 0:
            raise ValueError("The maximum number of iterations has to be a positive integer.")

        if not isinstance(self.targeted, bool):
            raise ValueError("The argument `targeted` has to be of type bool.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")
