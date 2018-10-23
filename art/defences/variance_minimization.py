from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np
from scipy.optimize import minimize

from art.defences.preprocessor import Preprocessor
from art import NUMPY_DTYPE

logger = logging.getLogger(__name__)


class TotalVarMin(Preprocessor):
    """
    Implement the total variance minimization defence approach. Defence method from
    https://openreview.net/forum?id=SyJ7ClWCb.
    """
    params = ['prob', 'norm', 'lam', 'solver', 'maxiter']

    def __init__(self, prob=0.5, norm=2, lam=0.01, solver='L-BFGS-B', maxiter=10):
        """
        Create an instance of total variance minimization.

        :param prob: Probability of the Bernoulli distribution.
        :type prob: `float`
        :param norm: Current support: 1, 2, np.inf
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

    def __call__(self, x, y=None, prob=None, norm=None, lam=None, solver=None, maxiter=None):
        """
        Apply total variance minimization to sample `x`.

        :param x: Sample to compress with shape `(batch_size, width, height, depth)`.
        :type x: `np.ndarray`
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :type y: `np.ndarray`
        :param prob: Probability of the Bernoulli distribution.
        :type prob: `float`
        :param norm: Current support: 1, 2, np.inf
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

        return x_

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
                           method=self.solver, jac=tv_l2_dx, options={'maxiter': self.maxiter}).x
            x_opt[:, :, i] = np.reshape(res, x_opt[:, :, i].shape)

        return x_

    def _loss_func(self, z, x, mask, norm, lam):
        """
        Loss function to be minimized.

        :param z: Initial guess.
        :type z: `np.ndarray`
        :param x: Original image.
        :type x: `np.ndarray`
        :param mask: A matrix that decides which points are kept.
        :type mask: `np.ndarray`
        :param norm: Current support: 1, 2, np.inf
        :type norm: `int`
        :param lam: The lambda parameter in the objective function.
        :type lam: `float`
        :return: Loss value.
        :rtype: `float`
        """
        res = np.linalg.norm((z - x.flatten()).dot(mask.flatten()), 2)
        res += lam * np.linalg.norm(z[1:, :] - z[:-1, :], norm, axis=1).sum()
        res += lam * np.linalg.norm(z[:, 1:] - z[:, :-1], norm, axis=0).sum()

        return res


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
        :param norm: Current support: 1, 2, np.inf
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

        if not(self.norm == 1 or self.norm == 2 or self.norm == np.inf):
            logger.error('Current support only 1, 2, np.inf.')
            raise ValueError('Current support only 1, 2, np.inf.')

        if not(self.solver == 'L-BFGS-B' or self.solver == 'CG' or self.solver == 'Newton-CG'):
            logger.error('Current support only L-BFGS-B, CG, Newton-CG.')
            raise ValueError('Current support only L-BFGS-B, CG, Newton-CG.')

        if type(self.maxiter) is not int or self.maxiter <= 0:
            logger.error('Number of iterations must be a positive integer.')
            raise ValueError('Number of iterations must be a positive integer.')

        return True



