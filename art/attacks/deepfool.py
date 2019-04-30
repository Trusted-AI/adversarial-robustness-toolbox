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
    attack_params = Attack.attack_params + ['max_iter', 'epsilon', 'nb_grads', 'batch_size']

    def __init__(self, classifier, max_iter=100, epsilon=1e-6, nb_grads=10, batch_size=128):
        """
        Create a DeepFool attack instance.

        :param classifier: A trained model.
        :type classifier: :class:`.Classifier`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param epsilon: Overshoot parameter.
        :type epsilon: `float`
        :param nb_grads: The number of class gradients (top nb_grads w.r.t. prediction) to compute. This way only the
                         most likely classes are considered, speeding up the computation.
        :type nb_grads: `int`
        :param batch_size: Batch size
        :type batch_size: `int`
        """
        super(DeepFool, self).__init__(classifier=classifier)
        params = {'max_iter': max_iter, 'epsilon': epsilon, 'nb_grads': nb_grads, 'batch_size': batch_size}
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
        :param nb_grads: The number of class gradients (top nb_grads w.r.t. prediction) to compute. This way only the
                         most likely classes are considered, speeding up the computation.
        :type nb_grads: `int`
        :param batch_size: Batch size
        :type batch_size: `int`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        self.set_params(**kwargs)
        clip_min, clip_max = self.classifier.clip_values
        x_adv = x.copy()
        preds = self.classifier.predict(x, logits=True)

        # Determine the class labels for which to compute the gradients
        use_grads_subset = self.nb_grads < self.classifier.nb_classes
        if use_grads_subset:
            # TODO compute set of unique labels per batch
            grad_labels = np.argsort(-preds, axis=1)[:, :self.nb_grads]
            labels_set = np.unique(grad_labels)
        else:
            labels_set = np.arange(self.classifier.nb_classes)
        sorter = np.arange(len(labels_set))

        # Pick a small scalar to avoid division by 0
        tol = 10e-8

        # Compute perturbation with implicit batching
        for batch_id in range(int(np.ceil(x_adv.shape[0] / float(self.batch_size)))):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch = x_adv[batch_index_1:batch_index_2]

            # Get predictions and gradients for batch
            f = preds[batch_index_1:batch_index_2]
            fk_hat = np.argmax(f, axis=1)
            if use_grads_subset:
                # Compute gradients only for top predicted classes
                grd = np.array([self.classifier.class_gradient(batch, logits=True, label=_) for _ in labels_set])
                grd = np.squeeze(np.swapaxes(grd, 0, 2), axis=0)
            else:
                # Compute gradients for all classes
                grd = self.classifier.class_gradient(batch, logits=True)

            # Get current predictions
            active_indices = np.arange(len(batch))
            current_step = 0
            while len(active_indices) != 0 and current_step < self.max_iter:
                # Compute difference in predictions and gradients only for selected top predictions
                labels_indices = sorter[np.searchsorted(labels_set, fk_hat, sorter=sorter)]
                grad_diff = grd - grd[np.arange(len(grd)), labels_indices][:, None]
                f_diff = f[:, labels_set] - f[np.arange(len(f)), labels_indices][:, None]

                # Choose coordinate and compute perturbation
                norm = np.linalg.norm(grad_diff.reshape(len(grad_diff), len(labels_set), -1), axis=2) + tol
                value = np.abs(f_diff) / norm
                value[np.arange(len(value)), labels_indices] = np.inf
                l = np.argmin(value, axis=1)
                r = (abs(f_diff[np.arange(len(f_diff)), l]) / (pow(np.linalg.norm(grad_diff[np.arange(len(
                    grad_diff)), l].reshape(len(grad_diff), -1), axis=1), 2) + tol))[:, None, None, None] * \
                    grad_diff[np.arange(len(grad_diff)), l]

                # Add perturbation and clip result
                batch[active_indices] = np.clip(batch[active_indices] + r[active_indices], clip_min, clip_max)

                # Recompute prediction for new x
                f = self.classifier.predict(batch, logits=True)
                fk_i_hat = np.argmax(f, axis=1)

                # Recompute gradients for new x
                if use_grads_subset:
                    # Compute gradients only for (originally) top predicted classes
                    grd = np.array([self.classifier.class_gradient(batch, logits=True, label=_) for _ in labels_set])
                    grd = np.squeeze(np.swapaxes(grd, 0, 2), axis=0)
                else:
                    # Compute gradients for all classes
                    grd = self.classifier.class_gradient(batch, logits=True)

                # Stop if misclassification has been achieved
                active_indices = np.where(fk_i_hat == fk_hat)[0]

                current_step += 1

            # Apply overshoot parameter
            x_adv[batch_index_1:batch_index_2] = np.clip(x_adv[batch_index_1:batch_index_2] + (
                1 + self.epsilon) * (batch - x_adv[batch_index_1:batch_index_2]), clip_min, clip_max)

        logger.info('Success rate of DeepFool attack: %.2f%%',
                    (np.sum(np.argmax(preds, axis=1) != np.argmax(self.classifier.predict(x_adv), axis=1)) /
                     x.shape[0]))

        return x_adv

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param epsilon: Overshoot parameter.
        :type epsilon: `float`
        :param nb_grads: The number of class gradients (top nb_grads w.r.t. prediction) to compute. This way only the
                         most likely classes are considered, speeding up the computation.
        :type nb_grads: `int`
        :param batch_size: Internal size of batches on which adversarial samples are generated.
        :type batch_size: `int`
        """
        # Save attack-specific parameters
        super(DeepFool, self).set_params(**kwargs)

        if not isinstance(self.max_iter, (int, np.int)) or self.max_iter <= 0:
            raise ValueError("The number of iterations must be a positive integer.")

        if not isinstance(self.nb_grads, (int, np.int)) or self.nb_grads <= 0:
            raise ValueError("The number of class gradients to compute must be a positive integer.")

        if self.epsilon < 0:
            raise ValueError("The overshoot parameter must not be negative.")

        if self.batch_size <= 0:
            raise ValueError('The batch size `batch_size` has to be positive.')

        return True
