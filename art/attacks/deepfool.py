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

from art.attacks.attack import Attack

logger = logging.getLogger(__name__)


class DeepFool(Attack):
    """
    Implementation of the attack from Moosavi-Dezfooli et al. (2015).
    Paper link: https://arxiv.org/abs/1511.04599
    """
    attack_params = Attack.attack_params + ['max_iter', 'epsilon']

    def __init__(self, classifier, max_iter=100, epsilon=1e-6):
        """
        Create a DeepFool attack instance.

        :param classifier: A trained model.
        :type classifier: :class:`Classifier`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param epsilon: Overshoot parameter.
        :type epsilon: `float`
        """
        super(DeepFool, self).__init__(classifier)
        params = {'max_iter': max_iter, 'epsilon': epsilon}
        self.set_params(**params)

    def generate(self, x, **kwargs):
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :type x: `np.ndarray`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param epsilon: Overshoot parameter.
        :type epsilon: `float`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        assert self.set_params(**kwargs)
        clip_min, clip_max = self.classifier.clip_values
        x_adv = x.copy()
        preds = self.classifier.predict(x, logits=True)

        # Pick a small scalar to avoid division by 0
        tol = 10e-8

        for j, val in enumerate(x_adv):
            xj = val[None, ...]
            f = preds[j]
            grd = self.classifier.class_gradient(xj, logits=True)[0]
            fk_hat = np.argmax(f)

            for _ in range(self.max_iter):
                grad_diff = grd - grd[fk_hat]
                f_diff = f - f[fk_hat]

                # Choose coordinate and compute perturbation
                norm = np.linalg.norm(grad_diff.reshape(self.classifier.nb_classes, -1), axis=1) + tol
                value = np.abs(f_diff) / norm
                value[fk_hat] = np.inf
                l = np.argmin(value)
                r = (abs(f_diff[l]) / (pow(np.linalg.norm(grad_diff[l]), 2) + tol)) * grad_diff[l]

                # Add perturbation and clip result
                xj = np.clip(xj + r, clip_min, clip_max)

                # Recompute prediction for new xj
                f = self.classifier.predict(xj, logits=True)[0]
                grd = self.classifier.class_gradient(xj, logits=True)[0]
                fk_i_hat = np.argmax(f)

                # Stop if misclassification has been achieved
                if fk_i_hat != fk_hat:
                    break

            # Apply overshoot parameter
            x_adv[j] = np.clip(x[j] + (1 + self.epsilon) * (xj[0] - x[j]), clip_min, clip_max)

        preds = np.argmax(preds, axis=1)
        preds_adv = np.argmax(self.classifier.predict(x_adv), axis=1)
        logger.info('Success rate of DeepFool attack: %.2f%%', (np.sum(preds != preds_adv) / x.shape[0]))

        return x_adv

    def set_params(self, **kwargs):
        """Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        """
        # Save attack-specific parameters
        super(DeepFool, self).set_params(**kwargs)

        if type(self.max_iter) is not int or self.max_iter <= 0:
            raise ValueError("The number of iterations must be a positive integer.")

        if self.epsilon < 0:
            raise ValueError("The overshoot parameter must not be negative.")

        return True

