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
This module implements the Feature Adversaries attack.

| Paper link: https://arxiv.org/abs/1511.05122
"""
import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE

logger = logging.getLogger(__name__)


class FeatureAdversariesNumpy(EvasionAttack):
    """
    This class represent a Feature Adversaries evasion attack.

    | Paper link: https://arxiv.org/abs/1511.05122
    """

    attack_params = EvasionAttack.attack_params + [
        "delta",
        "layer",
        "batch_size",
    ]

    _estimator_requirements = (BaseEstimator, NeuralNetworkMixin)

    def __init__(
        self,
        classifier: "CLASSIFIER_NEURALNETWORK_TYPE",
        delta: Optional[float] = None,
        layer: Optional[int] = None,
        batch_size: int = 32,
    ):
        """
        Create a :class:`.FeatureAdversaries` instance.

        :param classifier: A trained classifier.
        :param delta: The maximum deviation between source and guide images.
        :param layer: Index of the representation layer.
        :param batch_size: Batch size.
        """
        super().__init__(estimator=classifier)

        self.delta = delta
        self.layer = layer
        self.batch_size = batch_size
        self._check_params()

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: Source samples.
        :param y: Guide samples.
        :param kwargs: The kwargs are used as `options` for the minimisation with `scipy.optimize.minimize` using
                       `method="L-BFGS-B"`. Valid options are based on the output of
                       `scipy.optimize.show_options(solver='minimize', method='L-BFGS-B')`:
                       Minimize a scalar function of one or more variables using the L-BFGS-B algorithm.

                       disp : None or int
                           If `disp is None` (the default), then the supplied version of `iprint`
                           is used. If `disp is not None`, then it overrides the supplied version
                           of `iprint` with the behaviour you outlined.
                       maxcor : int
                           The maximum number of variable metric corrections used to
                           define the limited memory matrix. (The limited memory BFGS
                           method does not store the full hessian but uses this many terms
                           in an approximation to it.)
                       ftol : float
                           The iteration stops when ``(f^k -
                           f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol``.
                       gtol : float
                           The iteration will stop when ``max{|proj g_i | i = 1, ..., n}
                           <= gtol`` where ``pg_i`` is the i-th component of the
                           projected gradient.
                       eps : float
                           Step size used for numerical approximation of the Jacobian.
                       maxfun : int
                           Maximum number of function evaluations.
                       maxiter : int
                           Maximum number of iterations.
                       iprint : int, optional
                           Controls the frequency of output. ``iprint < 0`` means no output;
                           ``iprint = 0``    print only one line at the last iteration;
                           ``0 < iprint < 99`` print also f and ``|proj g|`` every iprint iterations;
                           ``iprint = 99``   print details of every iteration except n-vectors;
                           ``iprint = 100``  print also the changes of active set and final x;
                           ``iprint > 100``  print details of every iteration including x and g.
                       callback : callable, optional
                           Called after each iteration, as ``callback(xk)``, where ``xk`` is the
                           current parameter vector.
                       maxls : int, optional
                           Maximum number of line search steps (per iteration). Default is 20.

                       The option `ftol` is exposed via the `scipy.optimize.minimize` interface,
                       but calling `scipy.optimize.fmin_l_bfgs_b` directly exposes `factr`. The
                       relationship between the two is ``ftol = factr * numpy.finfo(float).eps``.
                       I.e., `factr` multiplies the default machine floating-point precision to
                       arrive at `ftol`.
        :return: Adversarial examples.
        :raises KeyError: The argument {} in kwargs is not allowed as option for `scipy.optimize.minimize` using
                          `method="L-BFGS-B".`
        """
        from scipy.linalg import norm
        from scipy.optimize import Bounds, minimize

        if y is None:
            raise ValueError("The value of guide `y` cannot be None. Please provide a `np.ndarray` of guide inputs.")
        if x.shape != y.shape:
            raise ValueError("The shape of source `x` and guide `y` must be of same shape.")
        if x.shape[1:] != self.estimator.input_shape:  # pragma: no cover
            raise ValueError("Source and guide inputs must match `input_shape` of estimator.")

        l_b = x.flatten() - self.delta
        l_b[l_b < self.estimator.clip_values[0]] = self.estimator.clip_values[0]

        u_b = x.flatten() + self.delta
        u_b[u_b > self.estimator.clip_values[1]] = self.estimator.clip_values[1]

        bound = Bounds(lb=l_b, ub=u_b, keep_feasible=False)

        guide_representation = self.estimator.get_activations(
            x=y.reshape(-1, *self.estimator.input_shape),  # type: ignore
            layer=self.layer,
            batch_size=self.batch_size,
        )

        def func(x_i):
            x_i = x_i.astype(x.dtype)
            source_representation = self.estimator.get_activations(
                x=x_i.reshape(-1, *self.estimator.input_shape),
                layer=self.layer,
                batch_size=self.batch_size,
            )

            n = (
                norm(
                    source_representation.flatten() - guide_representation.flatten(),
                    ord=2,
                )
                ** 2
            )

            return n

        x_0 = x.copy()

        options = {"eps": 1e-3, "ftol": 1e-3}
        options_allowed_keys = [
            "disp",
            "maxcor",
            "ftol",
            "gtol",
            "eps",
            "maxfun",
            "maxiter",
            "iprint",
            "callback",
            "maxls",
        ]

        for key in kwargs:
            if key not in options_allowed_keys:
                raise KeyError(
                    "The argument `{}` in kwargs is not allowed as option for `scipy.optimize.minimize` using "
                    '`method="L-BFGS-B".`'.format(key)
                )
        options.update(kwargs)

        res = minimize(func, x_0, method="L-BFGS-B", bounds=bound, options=options)
        x_adv = res.x
        logger.info(res)

        return x_adv.reshape(-1, *self.estimator.input_shape).astype(x.dtype)

    def _check_params(self) -> None:
        """
        Apply attack-specific checks.
        """
        if self.delta is None:
            raise ValueError("The delta cannot be None.")
        if self.delta is not None and self.delta <= 0:
            raise ValueError("The maximum deviation `delta` has to be positive.")

        if not isinstance(self.layer, int):
            raise ValueError("The index of the representation layer `layer` has to be integer.")

        if self.batch_size <= 0:
            raise ValueError("The batch size `batch_size` has to be positive.")
