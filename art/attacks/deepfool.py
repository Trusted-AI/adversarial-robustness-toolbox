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

from art.attacks.attack import Attack


class DeepFool(Attack):
    """
    Implementation of the attack from Moosavi-Dezfooli et al. (2015).
    Paper link: https://arxiv.org/abs/1511.04599
    """
    attack_params = ['max_iter']

    def __init__(self, classifier, max_iter=100):
        """
        Create a DeepFool attack instance.

        :param classifier: A trained model.
        :type classifier: :class:`Classifier`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        """
        super(DeepFool, self).__init__(classifier)
        params = {'max_iter': max_iter}
        self.set_params(**params)

    def generate(self, x, **kwargs):
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :type x: `np.ndarray`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        assert self.set_params(**kwargs)
        clip_min, clip_max = self.classifier.clip_values
        x_adv = x.copy()

        # Pick a small scalar to avoid division by 0
        tol = 10e-8

        for j, val in enumerate(x_adv):
            xj = val[None, ...]

            # TODO move prediction outside of for loop; add batching if `x` is too large?
            f = self.classifier.predict(xj)[0]
            grd = self.classifier.class_gradient(xj, logits=False)[0]
            fk_hat = np.argmax(f)
            fk_i_hat = fk_hat
            nb_iter = 0

            while fk_i_hat == fk_hat and nb_iter < self.max_iter:
                grad_diff = grd - grd[fk_hat]
                f_diff = f - f[fk_hat]

                # Masking true label
                mask = [0] * self.classifier.nb_classes
                mask[fk_hat] = 1
                norm = np.linalg.norm(grad_diff.reshape(self.classifier.nb_classes, -1), axis=1) + tol
                value = np.ma.array(np.abs(f_diff) / norm, mask=mask)

                l = value.argmin(fill_value=np.inf)
                r = (abs(f_diff[l]) / pow(np.linalg.norm(grad_diff[l]), 2)) * grad_diff[l]

                # Add perturbation and clip result
                xj = np.clip(xj + r, clip_min, clip_max)

                # Recompute prediction for new xj
                f = self.classifier.predict(xj)[0]
                grd = self.classifier.class_gradient(xj, logits=False)[0]
                fk_i_hat = np.argmax(f)

                nb_iter += 1

            x_adv[j] = xj[0]

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

        return True
