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


class NewtonFool(Attack):
    """
    Implementation of the attack from Uyeong Jang et al. (2017). Paper link: http://doi.acm.org/10.1145/3134600.3134635
    """
    attack_params = ["max_iter", "eta"]

    def __init__(self, classifier, max_iter=100, eta=0.01):
        """
        Create a NewtonFool attack instance.

        :param classifier: A trained model.
        :type classifier: :class:`Classifier`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param eta: The eta coefficient.
        :type eta: `float`
        """
        super(NewtonFool, self).__init__(classifier)
        params = {"max_iter": max_iter, "eta": eta}
        self.set_params(**params)

    def generate(self, x, **kwargs):
        """
        Generate adversarial samples and return them in a Numpy array.

        :param x: An array with the original inputs to be attacked.
        :type x: `np.ndarray`
        :param kwargs: Attack-specific parameters used by child classes.
        :type kwargs: `dict`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        assert self.set_params(**kwargs)
        nb_classes = self.classifier.nb_classes
        x_adv = x.copy()

        # Initialize variables
        clip_min, clip_max = self.classifier.clip_values
        y_pred = self.classifier.predict(x, logits=False)
        pred_class = np.argmax(y_pred, axis=1)

        # Main algorithm for each example
        for j, ex in enumerate(x_adv):
            norm_x0 = np.linalg.norm(np.reshape(ex, [-1]))
            l = pred_class[j]

            # Main loop of the algorithm
            for i in range(self.max_iter):
                # Compute score
                score = self.classifier.predict(np.array([ex]), logits=False)[0][l]

                # Compute the gradients and norm
                grads = self.classifier.class_gradient(np.array([ex]), logits=False)[0][l]
                norm_grad = np.linalg.norm(np.reshape(grads, [-1]))

                # Theta
                theta = self._compute_theta(norm_x0, score, norm_grad, nb_classes)

                # Pertubation
                di = self._compute_pert(theta, grads, norm_grad)

                # Update xi and pertubation
                ex += di

            # Apply clip
            x_adv[j] = np.clip(ex, clip_min, clip_max)

        return x_adv

    def set_params(self, **kwargs):
        """Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param eta: The eta coefficient.
        :type eta: `float`
        """
        # Save attack-specific parameters
        super(NewtonFool, self).set_params(**kwargs)

        if type(self.max_iter) is not int or self.max_iter <= 0:
            raise ValueError("The number of iterations must be a positive integer.")

        if type(self.eta) is not float or self.eta <= 0:
            raise ValueError("The eta coefficient must be a positive float.")

        return True

    def _compute_theta(self, norm_x0, score, norm_grad, nb_classes):
        """
        Function to compute the theta at each step.

        :param norm_x0: norm of x0
        :param score: softmax value at the attacked class.
        :param norm_grad: norm of gradient values at the attacked class.
        :param nb_classes: number of classes.
        :return: theta value.
        """
        equ1 = self.eta * norm_x0 * norm_grad
        equ2 = score - 1.0/nb_classes
        result = min(equ1, equ2)

        return result

    def _compute_pert(self, theta, grads, norm_grad):
        """
        Function to compute the pertubation at each step.

        :param theta: theta value at the current step.
        :param grads: gradient values at the attacked class.
        :param norm_grad: norm of gradient values at the attacked class.
        :return: pertubation.
        """
        nom = -theta * grads
        denom = norm_grad**2
        result = nom / float(denom)

        return result
