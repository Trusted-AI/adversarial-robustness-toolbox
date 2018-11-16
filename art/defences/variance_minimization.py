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
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np
from scipy.optimize import minimize

from art.defences.preprocessor import Preprocessor
from art import NUMPY_DTYPE

logger = logging.getLogger(__name__)


class TotalVarMin(Preprocessor):
    """
    Implement the total variance minimization defence approach. Defence method from [Guo et al., 2018].
    Paper link: https://openreview.net/forum?id=SyJ7ClWCb
    """
    params = ['prob', 'norm', 'lam', 'solver', 'maxiter']

    def __init__(self, prob=0.3, norm=2, lam=0.5, solver='L-BFGS-B', maxiter=10):
        """
        Create an instance of total variance minimization.

        :param prob: Probability of the Bernoulli distribution.
        :type prob: `float`
        :param norm: The norm (positive integer).
        :type norm: `int`
        :param lam: The lambda parameter in the objective function.
        :type lam: `float`
        :param solver: Current support: L-BFGS-B, CG, Newton-CG
        :type solver: `string`
        :param maxiter: Maximum number of iterations in an optimization.
        :type maxiter: `int`
        """
        super(TotalVarMin, self).__init__()
        self._is_fitted = True
        self.set_params(prob=prob, norm=norm, lam=lam, solver=solver, maxiter=maxiter)

    def __call__(self, x, y=None, prob=None, norm=None, lam=None, solver=None, maxiter=None, clip_values=(0, 1)):
        """
        Apply total variance minimization to sample `x`.

        :param x: Sample to compress with shape `(batch_size, width, height, depth)`.
        :type x: `np.ndarray`
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :type y: `np.ndarray`
        :param prob: Probability of the Bernoulli distribution.
        :type prob: `float`
        :param norm: The norm (positive integer).
        :type norm: `int`
        :param lam: The lambda parameter in the objective function.
        :type lam: `float`
        :param solver: Current support: L-BFGS-B, CG, Newton-CG
        :type solver: `string`
        :param maxiter: Maximum number of iterations in an optimization.
        :type maxiter: `int`
        :return: similar sample
        :rtype: `np.ndarray`
        """
        if prob is not None:
            self.set_params(prob=prob)

        if norm is not None:
            self.set_params(norm=norm)

        if lam is not None:
            self.set_params(lam=lam)

        if solver is not None:
            self.set_params(solver=solver)

        if maxiter is not None:
            self.set_params(maxiter=maxiter)

        x_ = x.copy()

        # Minimize one image per time
        for i, xi in enumerate(x_):
            mask = (np.random.rand(xi.shape[0], xi.shape[1], xi.shape[2]) < self.prob).astype('int')
            x_[i] = self._minimize(xi, mask)

        x_ = np.clip(x_, clip_values[0], clip_values[1])

        return x_.astype(NUMPY_DTYPE)

    def _minimize(self, x, mask):
        """
        Minimize the total variance objective function.

        :param x: Original image.
        :type x: `np.ndarray`
        :param mask: A matrix that decides which points are kept.
        :type mask: `np.ndarray`
        :return: A new image.
        :rtype: `np.ndarray`
        """
        z = x.copy()

        for i in range(x.shape[2]):
            res = minimize(self._loss_func, z[:, :, i].flatten(), (x[:, :, i], mask[:, :, i], self.norm, self.lam),
                           method=self.solver, jac=self._deri_loss_func, options={'maxiter': self.maxiter})
            z[:, :, i] = np.reshape(res.x, z[:, :, i].shape)

        return z

    def _loss_func(self, z, x, mask, norm, lam):
        """
        Loss function to be minimized.

        :param z: Initial guess.
        :type z: `np.ndarray`
        :param x: Original image.
        :type x: `np.ndarray`
        :param mask: A matrix that decides which points are kept.
        :type mask: `np.ndarray`
        :param norm: The norm (positive integer).
        :type norm: `int`
        :param lam: The lambda parameter in the objective function.
        :type lam: `float`
        :return: Loss value.
        :rtype: `float`
        """
        res = np.sqrt(np.power(z - x.flatten(), 2).dot(mask.flatten()))
        z = np.reshape(z, x.shape)
        res += lam * np.linalg.norm(z[1:, :] - z[:-1, :], norm, axis=1).sum()
        res += lam * np.linalg.norm(z[:, 1:] - z[:, :-1], norm, axis=0).sum()

        return res

    def _deri_loss_func(self, z, x, mask, norm, lam):
        """
        Derivative of loss function to be minimized.

        :param z: Initial guess.
        :type z: `np.ndarray`
        :param x: Original image.
        :type x: `np.ndarray`
        :param mask: A matrix that decides which points are kept.
        :type mask: `np.ndarray`
        :param norm: The norm (positive integer).
        :type norm: `int`
        :param lam: The lambda parameter in the objective function.
        :type lam: `float`
        :return: Derivative value.
        :rtype: `float`
        """
        # First compute the derivative of the first component of the loss function
        nor1 = np.sqrt(np.power(z - x.flatten(), 2).dot(mask.flatten()))
        if nor1 < 1e-6: nor1 = 1e-6
        der1 = ((z - x.flatten()) * mask.flatten()) / (nor1 * 1.0)

        # Then compute the derivative of the second component of the loss function
        z = np.reshape(z, x.shape)

        if norm == 1:
            z_d1 = np.sign(z[1:, :] - z[:-1, :])
            z_d2 = np.sign(z[:, 1:] - z[:, :-1])
        else:
            z_d1_norm = np.power(np.linalg.norm(z[1:, :] - z[:-1, :], norm, axis=1), norm - 1)
            z_d2_norm = np.power(np.linalg.norm(z[:, 1:] - z[:, :-1], norm, axis=0), norm - 1)
            z_d1_norm[z_d1_norm < 1e-6] = 1e-6
            z_d2_norm[z_d2_norm < 1e-6] = 1e-6
            z_d1_norm = np.repeat(z_d1_norm[:, np.newaxis], z.shape[1], axis=1)
            z_d2_norm = np.repeat(z_d2_norm[np.newaxis, :], z.shape[0], axis=0)
            z_d1 = norm * np.power(z[1:, :] - z[:-1, :], norm - 1) / z_d1_norm
            z_d2 = norm * np.power(z[:, 1:] - z[:, :-1], norm - 1) / z_d2_norm

        der2 = np.zeros(z.shape)
        der2[:-1, :] -= z_d1
        der2[1:, :] += z_d1
        der2[:, :-1] -= z_d2
        der2[:, 1:] += z_d2
        der2 = lam * der2.flatten()

        # Total derivative
        der = der1 + der2

        return der

    def fit(self, x, y=None, **kwargs):
        """
        No parameters to learn for this method; do nothing.
        """
        pass

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies defence-specific checks before saving them as attributes.

        :param prob: Probability of the Bernoulli distribution.
        :type prob: `float`
        :param norm: The norm (positive integer).
        :type norm: `int`
        :param lam: The lambda parameter in the objective function.
        :type lam: `float`
        :param solver: Current support: L-BFGS-B, CG, Newton-CG
        :type solver: `string`
        :param maxiter: Maximum number of iterations in an optimization.
        :type maxiter: `int`
        """
        # Save defense-specific parameters
        super(TotalVarMin, self).set_params(**kwargs)

        if type(self.prob) is not float or self.prob < 0.0 or self.prob > 1.0:
            logger.error('Probability must be between 0 and 1.')
            raise ValueError('Probability must be between 0 and 1.')

        if type(self.norm) is not int or self.norm <= 0:
            logger.error('Norm must be a positive integer.')
            raise ValueError('Norm must be a positive integer.')

        if not(self.solver == 'L-BFGS-B' or self.solver == 'CG' or self.solver == 'Newton-CG'):
            logger.error('Current support only L-BFGS-B, CG, Newton-CG.')
            raise ValueError('Current support only L-BFGS-B, CG, Newton-CG.')

        if type(self.maxiter) is not int or self.maxiter <= 0:
            logger.error('Number of iterations must be a positive integer.')
            raise ValueError('Number of iterations must be a positive integer.')

        return True





