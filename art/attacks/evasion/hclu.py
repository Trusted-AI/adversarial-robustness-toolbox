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
"""
Implementation of the High-Confidence-Low-Uncertainty (HCLU) adversarial example formulation by Grosse et al. (2018)

| Paper link: https://arxiv.org/abs/1812.02606
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import logging
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from art.attacks.attack import EvasionAttack
from art.classifiers.GPy import GPyGaussianProcessClassifier
from art.utils import compute_success

logger = logging.getLogger(__name__)


class HighConfidenceLowUncertainty(EvasionAttack):
    """
    Implementation of the High-Confidence-Low-Uncertainty (HCLU) adversarial example formulation by Grosse et al. (2018)

    | Paper link: https://arxiv.org/abs/1812.02606
    """

    attack_params = ["conf", "unc_increase", "min_val", "max_val"]

    def __init__(
        self,
        classifier: GPyGaussianProcessClassifier,
        conf: float = 0.95,
        unc_increase: float = 100.0,
        min_val: float = 0.0,
        max_val: float = 1.0,
    ) -> None:
        """
        :param classifier: A trained model of type GPYGaussianProcessClassifier.
        :param conf: Confidence that examples should have, if there were to be classified as 1.0 maximally.
        :param unc_increase: Value uncertainty is allowed to deviate, where 1.0 is original value.
        :param min_val: minimal value any feature can take.
        :param max_val: maximal value any feature can take.
        """
        super(HighConfidenceLowUncertainty, self).__init__(classifier=classifier)
        if not isinstance(classifier, GPyGaussianProcessClassifier):
            raise TypeError("Model must be a GPy Gaussian Process classifier.")
        params = {
            "conf": conf,
            "unc_increase": unc_increase,
            "min_val": min_val,
            "max_val": max_val,
        }
        self.set_params(**params)

    def generate(
        self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs
    ) -> np.ndarray:
        """
        Generate adversarial examples and return them as an array. This method should be overridden by all concrete
        attack implementations.

        :param x: An array with the original inputs to be attacked.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :return: An array holding the adversarial examples.
        """
        x_adv = copy.copy(x)

        def minfun(x, args):  # minimize L2 norm
            return np.sum(np.sqrt((x - args["orig"]) ** 2))

        def constraint_conf(x, args):  # constraint for confidence
            pred = args["classifier"].predict(x.reshape(1, -1))[0, 0]
            if args["class_zero"]:
                pred = 1.0 - pred
            return (pred - args["conf"]).reshape(-1)

        def constraint_unc(x, args):  # constraint for uncertainty
            cur_unc = (
                args["classifier"].predict_uncertainty(x.reshape(1, -1))
            ).reshape(-1)
            return (args["max_uncertainty"] - cur_unc)[0]

        bounds = []
        # adding bounds, to not go away from original data
        for i in range(np.shape(x)[1]):
            bounds.append((self.min_val, self.max_val))
        for i in range(np.shape(x)[0]):  # go though data amd craft
            # get properties for attack
            max_uncertainty = self.unc_increase * self.classifier.predict_uncertainty(
                x_adv[i].reshape(1, -1)
            )
            class_zero = (
                not self.classifier.predict(x_adv[i].reshape(1, -1))[0, 0] < 0.5
            )
            init_args = {
                "classifier": self.classifier,
                "class_zero": class_zero,
                "max_uncertainty": max_uncertainty,
                "conf": self.conf,
            }
            constr_conf = {"type": "ineq", "fun": constraint_conf, "args": (init_args,)}
            constr_unc = {"type": "ineq", "fun": constraint_unc, "args": (init_args,)}
            args = {"args": init_args, "orig": x[i].reshape(-1)}
            # finally, run optimization
            x_adv[i] = minimize(
                minfun,
                x_adv[i],
                args=args,
                bounds=bounds,
                constraints=[constr_conf, constr_unc],
            )["x"]
        logger.info(
            "Success rate of HCLU attack: %.2f%%",
            100 * compute_success(self.classifier, x, y, x_adv),
        )
        return x_adv

    def set_params(self, **kwargs) -> bool:
        """
        Take in a dictionary of parameters and apply attack-specific checks before saving them as attributes.

        :param conf: Confidence that examples should have, if there were to be classified as 1.0 maximally.
        :param unc_increase: Value uncertainty is allowed to deviate, where 1.0 is original value.
        :param min_val: minimal value any feature can take.
        :param max_val: maximal value any feature can take.
        """
        super(HighConfidenceLowUncertainty, self).set_params(**kwargs)
        if self.conf <= 0.5 or self.conf > 1.0:
            raise ValueError("Confidence value has to be a value between 0.5 and 1.0.")
        if self.unc_increase <= 0.0:
            raise ValueError("Value to increase uncertainty must be positive.")
        if self.min_val > self.max_val:
            raise ValueError("Maximum has to be larger than minimum.")

        return True
