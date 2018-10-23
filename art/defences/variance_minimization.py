from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

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
        x_ = x.copy()

        if self.norm == 1 or self.norm == 2:
                for i in range(img.shape[2]):
                    options = {'disp': verbose, 'maxiter': maxiter}
                    res = minimize(
                        tv_l2, x_opt[:, :, i], (img[:, :, i], w[:, :, i], lam, p),
                        method=solver, jac=tv_l2_dx, options=options).x
                    x_opt[:, :, i] = np.reshape(res, x_opt[:, :, i].shape)

        else:
            x_opt = np.copy(img)
            if solver == 'L-BFGS-B' or solver == 'CG' or solver == 'Newton-CG':
                for i in range(img.shape[2]):
                    options = {'disp': verbose, 'maxiter': maxiter}
                    lower = img[:, :, i] - tau
                    upper = img[:, :, i] + tau
                    lower[w[:, :, i] < 1e-6] = 0
                    upper[w[:, :, i] < 1e-6] = 1
                    bounds = np.array([lower.flatten(), upper.flatten()]).transpose()
                    res = minimize(
                        tv_inf, x_opt[:, :, i], (img[:, :, i], lam, p, tau),
                        method=solver, bounds=bounds, jac=tv_inf_dx, options=options).x
                    x_opt[:, :, i] = np.reshape(res, x_opt[:, :, i].shape)
            else:
                print('unsupported solver ' + solver)
                exit()
            return x_opt




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



