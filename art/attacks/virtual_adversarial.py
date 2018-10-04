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


class VirtualAdversarialMethod(Attack):
    """
    This attack was originally proposed by Miyato et al. (2016) and was used for virtual adversarial training.
    Paper link: https://arxiv.org/abs/1507.00677
    """
    attack_params = Attack.attack_params + ['eps', 'finite_diff', 'max_iter']

    def __init__(self, classifier, max_iter=1, finite_diff=1e-6, eps=.1):
        """
        Create a VirtualAdversarialMethod instance.

        :param classifier: A trained model.
        :type classifier: :class:`Classifier`
        :param eps: Attack step (max input variation).
        :type eps: `float`
        :param finite_diff: The finite difference parameter.
        :type finite_diff: `float`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        """
        super(VirtualAdversarialMethod, self).__init__(classifier)

        kwargs = {'finite_diff': finite_diff, 'eps': eps, 'max_iter': max_iter}
        self.set_params(**kwargs)

    def generate(self, x, **kwargs):
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :type x: `np.ndarray`
        :param eps: Attack step (max input variation).
        :type eps: `float`
        :param finite_diff: The finite difference parameter.
        :type finite_diff: `float`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        # TODO Consider computing attack for a batch of samples at a time (no for loop)
        # Parse and save attack-specific parameters
        assert self.set_params(**kwargs)
        clip_min, clip_max = self.classifier.clip_values

        x_adv = np.copy(x)
        dims = list(x.shape[1:])
        preds = self.classifier.predict(x_adv, logits=False)
        tol = 1e-10

        for ind, val in enumerate(x_adv):
            d = np.random.randn(*dims)

            for _ in range(self.max_iter):
                d = self._normalize(d)
                preds_new = self.classifier.predict((val + d)[None, ...], logits=False)

                from scipy.stats import entropy
                kl_div1 = entropy(preds[ind], preds_new[0])

                # TODO remove for loop
                d_new = np.zeros_like(d)
                array_iter = np.nditer(d, op_flags=['readwrite'], flags=['multi_index'])
                for x in array_iter:
                    x[...] += self.finite_diff
                    preds_new = self.classifier.predict((val + d)[None, ...], logits=False)
                    kl_div2 = entropy(preds[ind], preds_new[0])
                    d_new[array_iter.multi_index] = (kl_div2 - kl_div1) / (self.finite_diff + tol)
                    x[...] -= self.finite_diff
                d = d_new

            # Apply perturbation and clip
            val = np.clip(val + self.eps * self._normalize(d), clip_min, clip_max)
            x_adv[ind] = val

        return x_adv

    @staticmethod
    def _normalize(x):
        """
        Apply L_2 batch normalization on `x`.

        :param x: The input array to normalize.
        :type x: `np.ndarray`
        :return: The normalized version of `x`.
        :rtype: `np.ndarray`
        """
        tol = 1e-10
        dims = x.shape

        x = x.flatten()
        inverse = (np.sum(x**2) + tol) ** -.5
        x = x * inverse
        x = np.reshape(x, dims)

        return x

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param eps: Attack step (max input variation).
        :type eps: `float`
        :param finite_diff: The finite difference parameter.
        :type finite_diff: `float`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        """
        # Save attack-specific parameters
        super(VirtualAdversarialMethod, self).set_params(**kwargs)

        return True
