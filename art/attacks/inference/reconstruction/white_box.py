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
This module implements reconstruction attacks.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Optional, Tuple

import numpy as np
import sklearn
from scipy.optimize import fmin_l_bfgs_b

from art.attacks.attack import ReconstructionAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.estimators.classification.scikitlearn import ScikitlearnEstimator

logger = logging.getLogger(__name__)


class DatabaseReconstruction(ReconstructionAttack):
    """
    Implementation of a database reconstruction attack. In this case, the adversary is assumed to have in his/her
    possession a model trained on a dataset, and all but one row of that training dataset. This attack attempts to
    reconstruct the missing row.
    """

    _estimator_requirements = (BaseEstimator, ClassifierMixin, ScikitlearnEstimator)

    def __init__(self, estimator):
        """
        Create a DatabaseReconstruction instance.

        :param estimator: Trained target estimator.
        """
        super().__init__(estimator=estimator)

        self.params = self.estimator.get_trainable_attribute_names()

    @staticmethod
    def objective(x, y, x_train, y_train, private_estimator, parent_model, params):
        """Objective function which we seek to minimise"""

        model = sklearn.base.clone(parent_model.model, safe=True)
        model.fit(np.vstack((x_train, x)), np.hstack((y_train, y)))

        residual = 0.0

        for param in params:
            residual += np.sum((model.__getattribute__(param) - private_estimator.model.__getattribute__(param)) ** 2)

        return residual

    def reconstruct(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Infer the missing row from x, y with which `estimator` was trained with.

        :param x: Known records of the training set of `estimator`.
        :param y: Known labels of the training set of `estimator`.
        """
        if y is None:
            y = self.estimator.predict(x=x)

        if y.ndim == 2:
            y = np.argmax(y, axis=1)

        tol = float("inf")
        x_0 = x[0, :]
        x_guess = None
        y_guess = None

        for _y in range(self.estimator.nb_classes):
            args = (_y, x, y, self._estimator, self.estimator, self.params)
            _x, _tol, _ = fmin_l_bfgs_b(
                self.objective, x_0, args=args, approx_grad=True, factr=100, pgtol=1e-10, bounds=None
            )

            if _tol < tol:
                tol = _tol
                x_guess = _x
                y_guess = _y

        x_reconstructed = np.expand_dims(x_guess, axis=0)
        y_reconstructed = np.zeros(shape=(1, self.estimator.nb_classes))
        y_reconstructed[0, y_guess] = 1

        return x_reconstructed, y_reconstructed
