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

from art import NUMPY_DTYPE
from art.attacks.attack import Attack

logger = logging.getLogger(__name__)


class BoundaryAttack(Attack):
    """
    Implementation of the boundary attack from Wieland Brendel et al. (2018). This is a powerful black-box attack that
    only requires final class prediction. Paper link: https://arxiv.org/abs/1712.04248
    """
    attack_params = Attack.attack_params + ['targeted', 'delta', 'epsilon', 'step_adapt', 'max_iter', 'sample_size',
                                            'init_size']

    def __init__(self, classifier, targeted=True, delta=0.01, epsilon=0.01, step_adapt=0.9, max_iter=100,
                 sample_size=20, init_size=100):
        """
        Create a boundary attack instance.

        :param classifier: A trained model.
        :type classifier: :class:`.Classifier`
        :param targeted: Should the attack target one specific class.
        :type targeted: `bool`
        :param delta: Initial step size for the orthogonal step.
        :type delta: `float`
        :param epsilon: Initial step size for the step towards the target.
        :type epsilon: `float`
        :param step_adapt: Factor by which the step sizes are multiplied or divided, must be in the range (0, 1).
        :type step_adapt: `float`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param sample_size: Maximum number of trials per iteration.
        :type sample_size: `int`
        :param init_size: Maximum number of trials for initial generation of adversarial examples.
        :type init_size: `int`
        """
        super(BoundaryAttack, self).__init__(classifier=classifier)
        params = {'targeted': targeted,
                  'delta': delta,
                  'epsilon': epsilon,
                  'step_adapt': step_adapt,
                  'max_iter': max_iter,
                  'sample_size': sample_size,
                  'init_size': init_size}
        self.set_params(**params)

    def generate(self, x, y=None):
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :type x: `np.ndarray`
        :param y: If `self.targeted` is true, then `y` represents the target labels.
        :type y: `np.ndarray`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        # Assert that, if attack is targeted, y is provided
        if self.targeted and y is None:
            raise ValueError('Target labels `y` need to be provided for a targeted attack.')

        # Some initial setups
        x_adv = x.astype(NUMPY_DTYPE)
        if y is not None:
            y = np.argmax(y, axis=1)
        preds = np.argmax(self.classifier.predict(x), axis=1)

        # Generate the adversarial samples
        for ind, val in enumerate(x_adv):
            if self.targeted:
                x_adv[ind] = self._perturb(x=val, y_p=preds[ind], y=y[ind])
            else:
                x_adv[ind] = self._perturb(x=val, y_p=preds[ind])

        logger.info('Success rate of Boundary attack: %.2f%%',
                    (np.sum(preds != np.argmax(self.classifier.predict(x_adv), axis=1)) / x.shape[0]))

        return x_adv

    def _perturb(self, x, y_p, y=None):
        """
        Internal attack function for one example.

        :param x: An array with one original input to be attacked.
        :type x: `np.ndarray`
        :param y_p: The predicted label of x.
        :type y_p: `int`
        :param y: If `self.targeted` is true, then `y` represents the target label.
        :type y: `int`
        :return: an adversarial example.
        """
        # First, create an initial adversarial sample
        initial_sample = self._init_sample(x, y, y_p)

        # If an initial adversarial example is not found, then return the original image
        if initial_sample is None:
            return x

        # If an initial adversarial example found, then go with boundary attack
        if self.targeted:
            x_adv = self._attack(initial_sample, x, y, self.delta, self.epsilon)
        else:
            x_adv = self._attack(initial_sample, x, y_p, self.delta, self.epsilon)

        return x_adv

    def _attack(self, initial_sample, original_sample, target, initial_delta, initial_epsilon):
        """
        Main function for the boundary attack.

        :param initial_sample: An initial adversarial example.
        :type initial_sample: `np.ndarray`
        :param original_sample: The original input.
        :type original_sample: `np.ndarray`
        :param target: If `self.targeted` is true, then `target` represents the target label, otherwise the
        predicted label of the original sample.
        :type target: `int`
        :param initial_delta: Initial step size for the orthogonal step.
        :type initial_delta: `float`
        :param initial_epsilon: Initial step size for the step towards the target.
        :type initial_epsilon: `float`
        :return: an adversarial example.
        :rtype: `np.ndarray`
        """
        def compare(object1, object2):
            return object1 == object2 if self.targeted else object1 != object2

        # Get initialization for some variables
        x_adv = initial_sample
        delta = initial_delta
        epsilon = initial_epsilon

        # Main loop to wander around the boundary
        for _ in range(self.max_iter):
            # Trust region method to adjust delta
            for _ in range(self.max_iter):
                potential_advs = []
                for _ in range(self.sample_size):
                    potential_adv = x_adv + self._orthogonal_perturb(delta, x_adv, original_sample)
                    if hasattr(self.classifier, 'clip_values') and self.classifier.clip_values is not None:
                        np.clip(potential_adv, self.classifier.clip_values[0], self.classifier.clip_values[1],
                                out=potential_adv)
                    potential_advs.append(potential_adv)

                preds = np.argmax(self.classifier.predict(np.array(potential_advs)), axis=1)
                satisfied = compare(preds, target)
                delta_ratio = np.mean(satisfied)
                delta = delta * self.step_adapt if delta_ratio < .5 else delta / self.step_adapt

                if delta_ratio > 0:
                    x_adv = potential_advs[np.where(satisfied)[0][0]]
                    break

            else:
                logging.warning('Adversarial example found but not optimal.')
                return x_adv

            # Trust region method to adjust epsilon
            for _ in range(self.max_iter):
                perturb = original_sample - x_adv
                perturb *= epsilon
                potential_adv = x_adv + perturb
                if hasattr(self.classifier, 'clip_values') and self.classifier.clip_values is not None:
                    np.clip(potential_adv, self.classifier.clip_values[0], self.classifier.clip_values[1],
                            out=potential_adv)

                pred = np.argmax(self.classifier.predict(np.array([potential_adv])), axis=1)[0]

                if compare(pred, target):
                    x_adv = potential_adv
                    epsilon /= self.step_adapt
                    break
                else:
                    epsilon *= self.step_adapt

            else:
                logging.warning('Adversarial example found but not optimal.')
                return x_adv

        return x_adv

    def _orthogonal_perturb(self, delta, current_sample, original_sample):
        """
        Create an orthogonal perturbation.

        :param delta: Initial step size for the orthogonal step.
        :type delta: `float`
        :param current_sample: Current adversarial example.
        :type current_sample: `np.ndarray`
        :param original_sample: The original input.
        :type original_sample: `np.ndarray`
        :return: a possible perturbation.
        """
        # Generate perturbation randomly
        perturb = np.random.randn(*self.classifier.input_shape)

        # Rescale the perturbation
        perturb /= np.linalg.norm(perturb)
        perturb *= delta * np.linalg.norm(original_sample - current_sample)

        # Project the perturbation onto sphere
        direction = original_sample - current_sample
        perturb = np.swapaxes(perturb, 0, self.classifier.channel_index - 1)
        direction = np.swapaxes(direction, 0, self.classifier.channel_index - 1)

        for i in range(len(direction)):
            direction[i] /= np.linalg.norm(direction[i])
            perturb[i] -= np.dot(perturb[i], direction[i]) * direction[i]

        perturb = np.swapaxes(perturb, 0, self.classifier.channel_index - 1)

        # direction = original_sample - current_sample
        # norm_direction = np.linalg.norm(direction)
        # direction /= norm_direction
        # vdot = np.vdot(perturb, direction)
        # perturb -= vdot * direction
        # perturb *= delta * norm_direction / np.linalg.norm(perturb)
        # d = 1.0 / np.sqrt(delta ** 2 + 1)
        # direction = perturb - (original_sample - current_sample)
        # perturb = d * direction

        return perturb

    def _init_sample(self, x, y, y_pred):
        """
        Find initial adversarial example for the attack.

        :param x: An array with 1 original input to be attacked.
        :type x: `np.ndarray`
        :param y: If `self.targeted` is true, then `y` represents the target label.
        :type y: `int`
        :param y_pred: The predicted label of x.
        :type y_pred: `int`
        :return: an adversarial example.
        """
        nprd = np.random.RandomState()
        initial_sample = None

        # Attack satisfied
        if self.targeted and y == y_pred:
            return None

        # Attack unsatisfied yet
        for _ in range(self.init_size):
            if hasattr(self.classifier, 'clip_values') and self.classifier.clip_values is not None:
                random_sample = nprd.uniform(self.classifier.clip_values[0], self.classifier.clip_values[1],
                                             size=x.shape).astype(x.dtype)
            else:
                # TODO Adjust following feature-wise and for entire sample provided by user?
                mean_, std_ = np.mean(x), np.std(x)
                random_sample = nprd.normal(loc=mean_, scale=2 * std_, size=x.shape).astype(x.dtype)
            random_class = np.argmax(self.classifier.predict(np.array([random_sample])), axis=1)[0]

            if (self.targeted and random_class == y) or (not self.targeted and random_class != y_pred):
                initial_sample = random_sample
                logging.info('Found initial adversarial image for attack.')
                break

        return initial_sample

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param targeted: Should the attack target one specific class.
        :type targeted: `bool`
        :param delta: Initial step size for the orthogonal step.
        :type delta: `float`
        :param epsilon: Initial step size for the step towards the target.
        :type epsilon: `float`
        :param step_adapt: Factor by which the step sizes are multiplied or divided, must be in the range (0, 1).
        :type step_adapt: `float`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param sample_size: Maximum number of trials per iteration.
        :type sample_size: `int`
        :param init_size: Maximum number of trials for initial generation of adversarial examples.
        :type init_size: `int`
        """
        # Save attack-specific parameters
        super(BoundaryAttack, self).set_params(**kwargs)

        if not isinstance(self.max_iter, (int, np.int)) or self.max_iter <= 0:
            raise ValueError("The number of iterations must be a positive integer.")

        if not isinstance(self.sample_size, (int, np.int)) or self.sample_size <= 0:
            raise ValueError("The number of trials must be a positive integer.")

        if not isinstance(self.init_size, (int, np.int)) or self.init_size <= 0:
            raise ValueError("The number of trials must be a positive integer.")

        if self.epsilon <= 0:
            raise ValueError("The initial step size for the step towards the target must be positive.")

        if self.delta <= 0:
            raise ValueError("The initial step size for the orthogonal step must be positive.")

        if self.step_adapt <= 0 or self.step_adapt >= 1:
            raise ValueError("The adaptation factor must be in the range (0, 1).")

        return True
