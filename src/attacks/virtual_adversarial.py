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

import numpy as np
import tensorflow as tf

from src.attacks.attack import Attack, class_derivative


class VirtualAdversarialMethod(Attack):
    """
    This attack was originally proposed by Miyato et al. (2016) and was used for virtual adversarial training.
    Paper link: https://arxiv.org/abs/1507.00677
    """
    attack_params = ['eps', 'finite_diff', 'max_iter', 'clip_min', 'clip_max']

    def __init__(self, classifier, sess=None, max_iter=1, finite_diff=1e-6, eps=.1, clip_min=0., clip_max=1.):
        """
        Create a VirtualAdversarialMethod instance.

        :param classifier: A function that takes a symbolic input and returns the symbolic output for the classifier's
        predictions.
        :param sess: The tf session to run graphs in
        :param eps: (optional float) the epsilon (max input variation parameter)
        :param finite_diff: (optional float) The finite difference parameter
        :param max_iter: (optional integer) The maximum number of iterations
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        super(VirtualAdversarialMethod, self).__init__(classifier, sess)

        kwargs = {'finite_diff': finite_diff, 'eps': eps, 'max_iter': max_iter, 'clip_min': clip_min, 'clip_max': clip_max}
        self.set_params(**kwargs)

    def generate(self, x_val, **kwargs):
        """
        Generate adversarial samples and return them in a Numpy array.

        :param x_val: (required) A Numpy array with the original inputs
        :param eps: (optional float) the epsilon (max input variation parameter)
        :param finite_diff: (optional float) The finite difference parameter
        :param max_iter: (optinal integer) The maximum number of iterations
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        :return: A Numpy array holding the adversarial examples
        :rtype: np.ndarray
        """
        # TODO Consider computing attack for a batch of samples at a time (no for loop)
        # Parse and save attack-specific parameters
        assert self.set_params(**kwargs)

        x_adv = np.copy(x_val)
        dims = [None] + list(x_val.shape[1:])
        self._x = tf.placeholder(tf.float32, shape=dims)
        dims[0] = 1
        self._preds = self.classifier._get_predictions(self._x, log=False)
        preds_val = self.sess.run(self._preds, {self._x: x_adv})

        for ind, val in enumerate(x_adv):
            d = np.random.randn(*dims[1:])
            e = np.random.randn(*dims[1:])
            for _ in range(self.max_iter):
                d = self.finite_diff * self._normalize(d)
                e = self.finite_diff * self._normalize(e)
                preds_val_d = self.sess.run(self._preds, {self._x: [val + d]})[0]
                preds_val_e = self.sess.run(self._preds, {self._x: [val + e]})[0]

                # Compute KL divergence between logits
                from scipy.stats import entropy
                kl_div1 = entropy(preds_val[ind], preds_val_d)
                kl_div2 = entropy(preds_val[ind], preds_val_e)
                d = (kl_div1 - kl_div2) / np.abs(d - e)

            # Apply perturbation and clip
            val += self.eps * self._normalize(d)
            if self.clip_min is not None or self.clip_max is not None:
                val = np.clip(val, self.clip_min, self.clip_max)

        return x_adv

    def _normalize(self, x):
        """
        Apply L_2 batch normalization on `x`.

        :param x: (np.ndarray) The input array to normalize
        :return: The nornmalized version of `x`
        :rtype: np.ndarray
        """
        tol = 1e-12
        dims = x.shape

        x = x.flatten()
        x /= np.max(np.abs(x)) + tol
        inverse = (np.sum(x**2) + np.sqrt(tol)) ** -.5
        x = x * inverse
        x = np.reshape(x, dims)

        return x

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        Attack-specific parameters:
        :param eps: (optional float) the epsilon (max input variation parameter)
        :param finite_diff: (optional float) The finite difference parameter
        :param max_iter: (optional integer) The maximum number of iterations
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        # Save attack-specific parameters
        super(VirtualAdversarialMethod, self).set_params(**kwargs)

        return True
