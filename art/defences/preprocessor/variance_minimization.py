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
This module implements the total variance minimization defence `TotalVarMin`.

| Paper link: https://openreview.net/forum?id=SyJ7ClWCb

| Please keep in mind the limitations of defences. For more information on the limitations of this defence,
    see https://arxiv.org/abs/1802.00420 . For details on how to evaluate classifier security in general, see
    https://arxiv.org/abs/1902.06705
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize
from tqdm.auto import tqdm

from art.config import ART_NUMPY_DTYPE
from art.defences.preprocessor.preprocessor import Preprocessor

if TYPE_CHECKING:
    from art.utils import CLIP_VALUES_TYPE

logger = logging.getLogger(__name__)


class TotalVarMin(Preprocessor):
    """
    Implement the total variance minimization defence approach.

    | Paper link: https://openreview.net/forum?id=SyJ7ClWCb

    | Please keep in mind the limitations of defences. For more information on the limitations of this
        defence, see https://arxiv.org/abs/1802.00420 . For details on how to evaluate classifier security in general,
        see https://arxiv.org/abs/1902.06705
    """

    params = ["prob", "norm", "lamb", "solver", "max_iter", "clip_values", "verbose"]

    def __init__(
        self,
        prob: float = 0.3,
        norm: int = 2,
        lamb: float = 0.5,
        solver: str = "L-BFGS-B",
        max_iter: int = 10,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        apply_fit: bool = False,
        apply_predict: bool = True,
        verbose: bool = False,
    ):
        """
        Create an instance of total variance minimization.

        :param prob: Probability of the Bernoulli distribution.
        :param norm: The norm (positive integer).
        :param lamb: The lambda parameter in the objective function.
        :param solver: Current support: `L-BFGS-B`, `CG`, `Newton-CG`.
        :param max_iter: Maximum number of iterations when performing optimization.
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        :param verbose: Show progress bars.
        """
        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
        self.prob = prob
        self.norm = norm
        self.lamb = lamb
        self.solver = solver
        self.max_iter = max_iter
        self.clip_values = clip_values
        self.verbose = verbose
        self._check_params()

    def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply total variance minimization to sample `x`.

        :param x: Sample to compress with shape `(batch_size, width, height, depth)`.
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :return: Similar samples.
        """
        if len(x.shape) == 2:
            raise ValueError(
                "Feature vectors detected. Variance minimization can only be applied to data with spatial dimensions."
            )
        x_preproc = x.copy()

        # Minimize one input at a time
        for i, x_i in enumerate(tqdm(x_preproc, desc="Variance minimization", disable=not self.verbose)):
            mask = (np.random.rand(*x_i.shape) < self.prob).astype("int")
            x_preproc[i] = self._minimize(x_i, mask)

        if self.clip_values is not None:
            np.clip(x_preproc, self.clip_values[0], self.clip_values[1], out=x_preproc)

        return x_preproc.astype(ART_NUMPY_DTYPE), y

    def _minimize(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Minimize the total variance objective function.

        :param x: Original image.
        :param mask: A matrix that decides which points are kept.
        :return: A new image.
        """
        z_min = x.copy()

        for i in range(x.shape[2]):
            res = minimize(
                self._loss_func,
                z_min[:, :, i].flatten(),
                (x[:, :, i], mask[:, :, i], self.norm, self.lamb),
                method=self.solver,
                jac=self._deri_loss_func,
                options={"maxiter": self.max_iter},
            )
            z_min[:, :, i] = np.reshape(res.x, z_min[:, :, i].shape)

        return z_min

    @staticmethod
    def _loss_func(z_init: np.ndarray, x: np.ndarray, mask: np.ndarray, norm: int, lamb: float) -> float:
        """
        Loss function to be minimized.

        :param z_init: Initial guess.
        :param x: Original image.
        :param mask: A matrix that decides which points are kept.
        :param norm: The norm (positive integer).
        :param lamb: The lambda parameter in the objective function.
        :return: Loss value.
        """
        res = np.sqrt(np.power(z_init - x.flatten(), 2).dot(mask.flatten()))
        z_init = np.reshape(z_init, x.shape)
        res += lamb * np.linalg.norm(z_init[1:, :] - z_init[:-1, :], norm, axis=1).sum()
        res += lamb * np.linalg.norm(z_init[:, 1:] - z_init[:, :-1], norm, axis=0).sum()

        return res

    @staticmethod
    def _deri_loss_func(z_init: np.ndarray, x: np.ndarray, mask: np.ndarray, norm: int, lamb: float) -> float:
        """
        Derivative of loss function to be minimized.

        :param z_init: Initial guess.
        :param x: Original image.
        :param mask: A matrix that decides which points are kept.
        :param norm: The norm (positive integer).
        :param lamb: The lambda parameter in the objective function.
        :return: Derivative value.
        """
        # First compute the derivative of the first component of the loss function
        nor1 = np.sqrt(np.power(z_init - x.flatten(), 2).dot(mask.flatten()))
        if nor1 < 1e-6:
            nor1 = 1e-6
        der1 = ((z_init - x.flatten()) * mask.flatten()) / (nor1 * 1.0)

        # Then compute the derivative of the second component of the loss function
        z_init = np.reshape(z_init, x.shape)

        if norm == 1:
            z_d1 = np.sign(z_init[1:, :] - z_init[:-1, :])
            z_d2 = np.sign(z_init[:, 1:] - z_init[:, :-1])
        else:
            z_d1_norm = np.power(np.linalg.norm(z_init[1:, :] - z_init[:-1, :], norm, axis=1), norm - 1)
            z_d2_norm = np.power(np.linalg.norm(z_init[:, 1:] - z_init[:, :-1], norm, axis=0), norm - 1)
            z_d1_norm[z_d1_norm < 1e-6] = 1e-6
            z_d2_norm[z_d2_norm < 1e-6] = 1e-6
            z_d1_norm = np.repeat(z_d1_norm[:, np.newaxis], z_init.shape[1], axis=1)
            z_d2_norm = np.repeat(z_d2_norm[np.newaxis, :], z_init.shape[0], axis=0)
            z_d1 = norm * np.power(z_init[1:, :] - z_init[:-1, :], norm - 1) / z_d1_norm
            z_d2 = norm * np.power(z_init[:, 1:] - z_init[:, :-1], norm - 1) / z_d2_norm

        der2 = np.zeros(z_init.shape)
        der2[:-1, :] -= z_d1
        der2[1:, :] += z_d1
        der2[:, :-1] -= z_d2
        der2[:, 1:] += z_d2
        der2 = lamb * der2.flatten()

        # Total derivative
        return der1 + der2

    def _check_params(self) -> None:
        if not isinstance(self.prob, (float, int)) or self.prob < 0.0 or self.prob > 1.0:
            logger.error("Probability must be between 0 and 1.")
            raise ValueError("Probability must be between 0 and 1.")

        if not isinstance(self.norm, (int, np.int)) or self.norm <= 0:
            logger.error("Norm must be a positive integer.")
            raise ValueError("Norm must be a positive integer.")

        if not (self.solver == "L-BFGS-B" or self.solver == "CG" or self.solver == "Newton-CG"):
            logger.error("Current support only L-BFGS-B, CG, Newton-CG.")
            raise ValueError("Current support only L-BFGS-B, CG, Newton-CG.")

        if not isinstance(self.max_iter, (int, np.int)) or self.max_iter <= 0:
            logger.error("Number of iterations must be a positive integer.")
            raise ValueError("Number of iterations must be a positive integer.")

        if self.clip_values is not None:

            if len(self.clip_values) != 2:
                raise ValueError("`clip_values` should be a tuple of 2 floats containing the allowed data range.")

            if np.array(self.clip_values[0] >= self.clip_values[1]).any():
                raise ValueError("Invalid `clip_values`: min >= max.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")
