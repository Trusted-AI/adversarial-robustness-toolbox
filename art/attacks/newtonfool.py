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
from art.utils import to_categorical

logger = logging.getLogger(__name__)


class NewtonFool(Attack):
    """
    Implementation of the attack from Uyeong Jang et al. (2017). Paper link: http://doi.acm.org/10.1145/3134600.3134635
    """
    attack_params = Attack.attack_params + ["max_iter", "eta", "batch_size"]

    def __init__(self, classifier, max_iter=1000, eta=0.01, batch_size=128):
        """
        Create a NewtonFool attack instance.

        :param classifier: A trained model.
        :type classifier: :class:`Classifier`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param eta: The eta coefficient.
        :type eta: `float`
        :param batch_size: Batch size
        :type batch_size: `int`
        """
        super(NewtonFool, self).__init__(classifier)
        params = {"max_iter": max_iter, "eta": eta, "batch_size": batch_size}
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
        self.set_params(**kwargs)
        x_adv = x.copy()

        # Initialize variables
        clip_min, clip_max = self.classifier.clip_values
        y_pred = self.classifier.predict(x, logits=False)
        pred_class = np.argmax(y_pred, axis=1)

        # Compute perturbation with implicit batching
        for batch_id in range(int(np.ceil(x_adv.shape[0] / float(self.batch_size)))):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch = x_adv[batch_index_1:batch_index_2]

            # Main algorithm for each batch
            norm_batch = np.linalg.norm(np.reshape(batch, (batch.shape[0], -1)), axis=1)
            l = pred_class[batch_index_1:batch_index_2]
            l_b = to_categorical(l, self.classifier.nb_classes).astype(bool)

            # Main loop of the algorithm
            for i in range(self.max_iter):
                # Compute score
                score = self.classifier.predict(batch, logits=False)[l_b]

                # Compute the gradients and norm
                grads = self.classifier.class_gradient(batch, label=l, logits=False)
                grads = np.squeeze(grads, axis=1)
                norm_grad = np.linalg.norm(np.reshape(grads, (batch.shape[0], -1)), axis=1)

                # Theta
                theta = self._compute_theta(norm_batch, score, norm_grad)

                # Pertubation
                di_batch = self._compute_pert(theta, grads, norm_grad)

                # Update xi and pertubation
                batch += di_batch

            # Apply clip
            x_adv[batch_index_1:batch_index_2] = np.clip(batch, clip_min, clip_max)

        preds = np.argmax(self.classifier.predict(x), axis=1)
        preds_adv = np.argmax(self.classifier.predict(x_adv), axis=1)
        logger.info('Success rate of NewtonFool attack: %.2f%%', (np.sum(preds != preds_adv) / x.shape[0]))

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

        if self.batch_size <= 0:
            raise ValueError('The batch size `batch_size` has to be positive.')

        return True

    def _compute_theta(self, norm_batch, score, norm_grad):
        """
        Function to compute the theta at each step.

        :param norm_batch: norm of a batch.
        :type norm_batch: `np.ndarray`
        :param score: softmax value at the attacked class.
        :type score: `np.ndarray`
        :param norm_grad: norm of gradient values at the attacked class.
        :type norm_grad: `np.ndarray`
        :return: theta value.
        :rtype: `np.ndarray`
        """
        equ1 = self.eta * norm_batch * norm_grad
        equ2 = score - 1.0 / self.classifier.nb_classes
        result = np.minimum.reduce([equ1, equ2])

        return result

    @staticmethod
    def _compute_pert(theta, grads, norm_grad):
        """
        Function to compute the pertubation at each step.

        :param theta: theta value at the current step.
        :type theta: `np.ndarray`
        :param grads: gradient values at the attacked class.
        :type grads: `np.ndarray`
        :param norm_grad: norm of gradient values at the attacked class.
        :type norm_grad: `np.ndarray`
        :return: pertubation.
        :rtype: `np.ndarray`
        """
        # Pick a small scalar to avoid division by 0
        tol = 10e-8
        nom = -theta[:, None, None, None] * grads
        denom = norm_grad**2
        denom[denom < tol] = tol
        result = nom / denom[:, None, None, None]

        return result




