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

import logging
import copy

import numpy as np
from scipy.optimize import minimize

from art.attacks.attack import Attack
from art.classifiers.GPy import GPyGaussianProcessClassifier

logger = logging.getLogger(__name__)


class HighConfidenceLowUncertainty(Attack):
    """
    Implementation of the High-Confidence-Low-Uncertainty (HCLU) adversarial example formulation by Grosse et al. (2018)

    | Paper link: https://arxiv.org/abs/1812.02606
    """
    attack_params = ['conf', 'unc_increase', 'min_val', 'max_val']

    def __init__(self, classifier, conf=0.95, unc_increase=100.0, min_val=0.0, max_val=1.0):
        """
        :param classifier: A trained model of type GPYGaussianProcessClassifier.
        :type classifier: :class:`.Classifier.GPyGaussianProcessClassifier
        :param conf: Confidence that examples should have, if there were to be classified as 1.0 maximally
        :type conf: :float:
        :param unc_increase: Value uncertainty is allowed to deviate, where 1.0 is original value
        :type unc_increase: :float:
        :param min_val: minimal value any feature can take, defaults to 0.0
        :type min_val: :float:
        :param max_val: maximal value any feature can take, defaults to 1.0
        :type max_val: :float:
        """
        super(HighConfidenceLowUncertainty, self).__init__(classifier=classifier)
        if not isinstance(classifier, GPyGaussianProcessClassifier):
            raise TypeError('Model must be a GPy Gaussian Process classifier!')
        params = {'conf': conf,
                  'unc_increase': unc_increase,
                  'min_val': min_val,
                  'max_val': max_val
                  }
        self.set_params(**params)

    def generate(self, x, y=None, **kwargs):
        """
        Generate adversarial examples and return them as an array. This method should be overridden by all concrete
        attack implementations.

        :param x: An array with the original inputs to be attacked.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :type y: `np.ndarray`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        x_adv = copy.copy(x)

        def minfun(x, args):  # minimize L2 norm
            return np.sum(np.sqrt((x - args['orig']) ** 2))

        def constraint_conf(x, args):  # constraint for confidence
            pred = args['classifier'].predict(x.reshape(1, -1))[0, 0]
            if args['class_zero']:
                pred = 1.0 - pred
            return (pred - 0.95).reshape(-1)

        def constraint_unc(x, args):  # constraint for uncertainty
            return (args['max_uncertainty'] - (args['classifier'].predict_uncertainty(x.reshape(1, -1))).reshape(-1))[0]

        bounds = []
        # adding bounds, to not go away from original data
        for i in range(np.shape(x)[1]):
            bounds.append((self.min_val, self.max_val))
        for i in range(np.shape(x)[0]):  # go though data amd craft
            # get properties for attack
            max_uncertainty = self.unc_increase * self.classifier.predict_uncertainty(x_adv[i].reshape(1, -1))
            class_zero = not self.classifier.predict(x_adv[i].reshape(1, -1))[0, 0] < 0.5
            init_args = {'classifier': self.classifier, 'class_zero': class_zero, 'max_uncertainty': max_uncertainty}
            constr_conf = {'type': 'ineq', 'fun': constraint_conf, 'args': (init_args,)}
            constr_unc = {'type': 'ineq', 'fun': constraint_unc, 'args': (init_args,)}
            args = {'args': init_args, 'orig': x[i].reshape(-1)}
            # #finally, run optimization
            x_adv[i] = minimize(minfun, x_adv[i], args=args, bounds=bounds, constraints=[constr_conf, constr_unc])['x']
        return x_adv

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and apply attack-specific checks before saving them as attributes.

        :param kwargs: a dictionary of attack-specific parameters
        :type kwargs: `dict`
        :return: `True` when parsing was successful
        """
        super(HighConfidenceLowUncertainty, self).set_params(**kwargs)
        if self.conf <= 0.5 or self.conf > 1.0:
            raise ValueError(
                "Confidence value has to bea value between 0.5 and 1.0.")
        if self.unc_increase < 0.0:
            raise ValueError(
                "Uncertainty increase value has to be a positive number.")
        if self.min_val > self.max_val:
            raise ValueError("Maximum has to be larger than minimum.")
