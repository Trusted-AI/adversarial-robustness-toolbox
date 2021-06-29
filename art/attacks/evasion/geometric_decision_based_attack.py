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
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

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
        "bin_search_tol",
        "lambda_param",
        "sigma",
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
        bin_search_tol: float = 0.1,
        lambda_param: float = 0.6,
        sigma: float = 0.0002,
        verbose: bool = True,
    ) -> None:
        """
        Create a Geometric Decision-based Attack instance.

        :param estimator: A trained classifier.
        :param batch_size: The size of the batch used by the estimator during inference.
        :param norm: The norm of the adversarial perturbation. Possible values: "inf", np.inf, 1 or 2.
        :param sub_dim: Dimensionality of 2D frequency space (DCT).
        :param max_iter: Maximum number of iterations.
        :param bin_search_tol: Maximum remaining L2 perturbation defining binary search convergence. Input images are
                               normalised by maximal estimator.clip_value[1] if available or maximal value in the input
                               image.
        :param lambda_param: The lambda of equation 19 with `lambda_param=0` corresponding to a single iteration and
                             `lambda_param=1` to a uniform distribution of iterations per step.
        :param sigma: Variance of the Gaussian perturbation.
        :param targeted: Should the attack target one specific class.
        :param verbose: Show progress bars.
        """
        super().__init__(estimator=estimator)

        self.batch_size = batch_size
        self.norm = norm
        self.sub_dim = sub_dim
        self.max_iter = max_iter
        self.bin_search_tol = bin_search_tol
        self.lambda_param = lambda_param
        self.sigma = sigma
        self._targeted = False

        self.verbose = verbose
        self._check_params()

        self.sub_basis = None
        self.nb_calls = 0
        self.clip_min = 0.0
        self.clip_max = 0.0
        if self.estimator.input_shape is None:
            raise ValueError("The `input_shape` of the is required but None.")
        self.nb_channels = (
            self.estimator.input_shape[0] if self.estimator.channels_first else self.estimator.input_shape[2]
        )

        # Optimal number of iterations
        iteration = round(self.max_iter / 500)
        q_opt_it = int(self.max_iter - iteration * 25)
        _, iterate = self._opt_query_iteration(q_opt_it, iteration, self.lambda_param)
        q_opt_it = int(self.max_iter - iterate * 25)
        self.q_opt_iter, self.iterate = self._opt_query_iteration(q_opt_it, iteration, self.lambda_param)

    @staticmethod
    def _generate_2d_dct_basis(sub_dim: int, res: int) -> np.ndarray:
        def alpha(var_a: int, num: int):
            """
            Get alpha.
            """
            if var_a == 0:
                return math.sqrt(1.0 / num)

            return math.sqrt(2.0 / num)

        def dct(i_x: int, i_y: int, i_v: int, i_u: int, num: int) -> float:
            """
            Get 2D DCT.
            """
            return (
                alpha(i_u, num)
                * alpha(i_v, num)
                * math.cos(((2 * i_x + 1) * (i_u * math.pi)) / (2 * num))
                * math.cos(((2 * i_y + 1) * (i_v * math.pi)) / (2 * num))
            )

        # We can get different frequencies by setting u and v
        u_max = sub_dim
        v_max = sub_dim

        dct_basis = []
        for i_u in range(u_max):
            for i_v in range(v_max):
                basis = np.zeros((res, res))
                for i_y in range(res):
                    for i_x in range(res):
                        basis[i_y, i_x] = dct(i_x, i_y, i_v, i_u, max(res, v_max))
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

        if y is not None and self.estimator.nb_classes == 2 and y.shape[1] == 1:
            raise ValueError(
                "This attack has not yet been tested for binary classification with a single output classifier."
            )

        x_adv = x.copy()

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
            not self.estimator.channels_first and x.shape[1] != x.shape[2]
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

        for i in trange(x.shape[0], desc="GeoDA - samples", disable=not self.verbose, position=0):
            x_i = x[[i]]
            y_i = y[[i]]

            # Reset number of calls
            self.nb_calls = 0

            # Random search
            x_random = self._find_random_adversarial(x=x_i, y=y_i)
            logger.info("Random search adversarial example is adversarial: %r", self._is_adversarial(x_random, y_i))

            # Binary search
            x_boundary = self._binary_search(x_i, y_i, x_random, tol=self.bin_search_tol)
            logger.info("Binary search example at boundary is adversarial: %r", self._is_adversarial(x_boundary, y_i))

            grad = 0

            x_adv_i = x_i

            for k in trange(self.iterate, desc="GeoDA - steps", disable=not self.verbose, position=1):
                grad_oi, _ = self._black_grad_batch(x_boundary, self.q_opt_iter[k], self.batch_size, y_i)
                grad = grad_oi + grad
                x_adv_i = self._go_to_boundary(x_i, y_i, grad)
                x_adv_i = self._binary_search(x_i, y_i, x_adv_i, tol=self.bin_search_tol)
                x_boundary = x_adv_i

            x_adv_i = np.clip(x_adv_i, a_min=self.clip_min, a_max=self.clip_max)

            x_adv[i] = x_adv_i

        return x_adv

    def _is_adversarial(self, x_adv: np.ndarray, y_true: np.ndarray) -> bool:
        """
        Check if example is adversarial.

        :param x_adv: Current example.
        :param y_true: True label of `x`.
        :return: Boolean if `x` is mis-classified.
        """
        y_prediction = self.estimator.predict(x=x_adv)

        if self.targeted:
            return np.argmax(y_prediction, axis=1)[0] == np.argmax(y_true, axis=1)[0]

        return np.argmax(y_prediction, axis=1)[0] != np.argmax(y_true, axis=1)[0]

    def _find_random_adversarial(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Find an adversarial example by random search.

        :param x: Current example.
        :param y: True label of `x`.
        :return: A random adversarial example for `x`.
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

    def _binary_search(self, x: np.ndarray, y: np.ndarray, x_random: np.ndarray, tol: float) -> np.ndarray:
        """
        Find example on decision boundary between input and random sample by binary search.

        :param x: Current example.
        :param y: True label of `x`.
        :param x_random: Random adversarial example of `x`.
        :return: The adversarial example at the decision boundary.
        """
        x_adv = x_random
        x_cln = x

        if self.estimator.clip_values is not None:
            max_value = self.estimator.clip_values[1]
        else:
            max_value = np.max(x)

        while np.linalg.norm((x_adv.flatten() - x_cln.flatten()) / max_value, ord=2) >= tol:
            self.nb_calls += 1
            x_mid = (x_cln + x_adv) / 2.0
            if self._is_adversarial(x_mid, y):
                x_adv = x_mid
            else:
                x_cln = x_mid

        return x_adv

    def _opt_query_iteration(self, var_nq: int, var_t: int, lambda_param: float) -> Tuple[List[int], int]:
        """
        Determine optimal distribution of number of queries.
        """
        coefficients = [lambda_param ** (-2 * i / 3) for i in range(0, var_t)]
        sum_coefficients = sum(coefficients)
        opt_q = [round(var_nq * coefficients[i] / sum_coefficients) for i in range(0, var_t)]

        if opt_q[0] > 80:
            var_t = var_t + 1
            opt_q, var_t = self._opt_query_iteration(var_nq, var_t, lambda_param)
        elif opt_q[0] < 50:
            var_t = var_t - 1
            opt_q, var_t = self._opt_query_iteration(var_nq, var_t, lambda_param)

        return opt_q, var_t

    def _black_grad_batch(
        self, x_boundary: np.ndarray, q_max: int, batch_size: int, original_label: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """
        Calculate gradient towards decision boundary.
        """
        self.nb_calls += q_max
        grad_tmp = []  # estimated gradients in each estimate_batch
        z_list = []  # sign of grad_tmp
        outs = []
        num_batches = math.ceil(q_max / batch_size)
        last_batch = q_max - (num_batches - 1) * batch_size
        all_noises = []

        for j in range(num_batches):
            if j == num_batches - 1:
                current_batch = self._sub_noise(last_batch, self.sub_basis)
                noisy_boundary = [x_boundary[0, :, :, :]] * last_batch + self.sigma * current_batch
            else:
                current_batch = self._sub_noise(batch_size, self.sub_basis)
                noisy_boundary = [x_boundary[0, :, :, :]] * batch_size + self.sigma * current_batch

            all_noises.append(current_batch)
            predict_labels = np.argmax(self.estimator.predict(noisy_boundary), axis=1).astype(int)
            outs.append(predict_labels)

        all_noise = np.concatenate(all_noises, axis=0)
        outs = np.concatenate(outs, axis=0)

        for i, predict_label in enumerate(outs):
            if predict_label == np.argmax(original_label, axis=1)[0]:
                z_list.append(1)
                grad_tmp.append(all_noise[i])
            else:
                z_list.append(-1)
                grad_tmp.append(-all_noise[i])

        grad: np.ndarray = -(1 / q_max) * sum(grad_tmp)
        grad_f = grad[None, :, :, :]

        return grad_f, sum(z_list)

    def _go_to_boundary(self, x: np.ndarray, y: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        Move towards decision boundary.

        :param x: Current example to be moved towards the decision boundary.
        :param y: The true label.
        :param grad: Gradient towards decision boundary.
        :return: Example moved towards decision boundary.
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

    def _sub_noise(self, num_noises: int, basis: np.ndarray):
        """
        Create subspace random perturbation.

        :param num_noises: Number of random subspace noises.
        :param basis: Subspace bases.
        :return: Random subspace perturbations.
        """
        noise = np.random.normal(size=(basis.shape[1], self.nb_channels * num_noises)) * (self.clip_max - self.clip_min)
        sub_noise = np.array(np.matmul(basis, noise).transpose((1, 0)).astype(ART_NUMPY_DTYPE))

        if self.estimator.channels_first:
            subspace_shape = (num_noises,) + self.estimator.input_shape
        else:
            subspace_shape = (
                num_noises,
                self.estimator.input_shape[2],
                self.estimator.input_shape[0],
                self.estimator.input_shape[1],
            )

        r_list = sub_noise.reshape(subspace_shape)

        if not self.estimator.channels_first:
            r_list = r_list.transpose((0, 2, 3, 1))

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

        if not isinstance(self.bin_search_tol, float) or self.bin_search_tol <= 0:
            raise ValueError("The binary search tolerance has to be a positive float.")

        if not isinstance(self.lambda_param, float) or self.lambda_param <= 0:
            raise ValueError("The lambda parameter has to be a positive float.")

        if not isinstance(self.sigma, float) or self.sigma <= 0:
            raise ValueError("The sigma has to be a positive float.")

        if not isinstance(self.targeted, bool):
            raise ValueError("The argument `targeted` has to be of type bool.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")
