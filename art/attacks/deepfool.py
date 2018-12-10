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
    attack_params = Attack.attack_params + ['max_iter', 'epsilon', 'batch_size']

    def __init__(self, classifier, max_iter=100, epsilon=1e-6, batch_size=128):
        """
        Create a DeepFool attack instance.

        :param classifier: A trained model.
        :type classifier: :class:`Classifier`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param epsilon: Overshoot parameter.
        :type epsilon: `float`
        :param batch_size: Batch size
        :type batch_size: `int`
        """
        super(DeepFool, self).__init__(classifier)
        params = {'max_iter': max_iter, 'epsilon': epsilon, 'batch_size': batch_size}
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

        # Compute perturbation with implicit batching
        for batch_id in range(int(np.ceil(x_adv.shape[0] / float(self.batch_size)))):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch = x_adv[batch_index_1:batch_index_2]

            # Main algorithm for each batch
            f = preds[batch_index_1:batch_index_2]
            grd = self.classifier.class_gradient(batch, logits=True)
            fk_hat = np.argmax(f, axis=1)

            # Get current predictions
            active_indices = np.arange(len(batch))
            current_step = 0
            while len(active_indices) != 0 and current_step < self.max_iter:
                grad_diff = grd - grd[np.arange(len(grd)), fk_hat][:, None]
                f_diff = f - f[np.arange(len(f)), fk_hat][:, None]

                # Choose coordinate and compute perturbation
                norm = np.linalg.norm(grad_diff.reshape(len(grad_diff), self.classifier.nb_classes, -1), axis=2) + tol
                value = np.abs(f_diff) / norm







                # Adversarial crafting
                current_x = self._apply_perturbation(x[batch_index_1:batch_index_2], perturbation, current_eps)
                # Update
                batch[active_indices] = current_x[active_indices]
                adv_preds = self.classifier.predict(batch)
                # If targeted active check to see whether we have hit the target, otherwise head to anything but
                if self.targeted:
                    active_indices = np.where(np.argmax(batch_labels, axis=1) != np.argmax(adv_preds, axis=1))[0]
                else:
                    active_indices = np.where(np.argmax(batch_labels, axis=1) == np.argmax(adv_preds, axis=1))[0]

                current_eps += eps_step











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
