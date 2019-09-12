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
This module implements the boundary attack `BoundaryAttack`. This is a black-box attack which only requires class
predictions.

| Paper link: https://arxiv.org/abs/1712.04248
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from art import NUMPY_DTYPE
from art.attacks.attack import Attack
from art.utils import compute_success, to_categorical, check_and_transform_label_format

logger = logging.getLogger(__name__)


class BoundaryAttack(Attack):
    """
    Implementation of the boundary attack from Brendel et al. (2018). This is a powerful black-box attack that
    only requires final class prediction.

    | Paper link: https://arxiv.org/abs/1712.04248
    """
    attack_params = Attack.attack_params + ['targeted', 'delta', 'epsilon', 'step_adapt', 'max_iter', 'num_trial',
                                            'sample_size', 'init_size', 'batch_size']

    def __init__(self, classifier, targeted=True, delta=0.01, epsilon=0.01, step_adapt=0.667, max_iter=5000,
                 num_trial=25, sample_size=20, init_size=100):
        """
        Create a boundary attack instance.

        :param classifier: A trained classifier.
        :type classifier: :class:`.Classifier`
        :param targeted: Should the attack target one specific class.
        :type targeted: `bool`
        :param delta: Initial step size for the orthogonal step.
        :type delta: `float`
        :param epsilon: Initial step size for the step towards the target.
        :type epsilon: `float`
        :param step_adapt: Factor by which the step sizes are multiplied or divided, must be in the range (0, 1).
        :type step_adapt: `float`
        :param max_iter: Maximum number of iterations.
        :type max_iter: `int`
        :param num_trial: Maximum number of trials per iteration.
        :type num_trial: `int`
        :param sample_size: Number of samples per trial.
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
                  'num_trial': num_trial,
                  'sample_size': sample_size,
                  'init_size': init_size,
                  'batch_size': 1
                  }
        self.set_params(**params)

    def generate(self, x, y=None, **kwargs):
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,). If `self.targeted` is true, then `y` represents the target labels.
        :type y: `np.ndarray` or `None`
        :param x_adv_init: Initial array to act as initial adversarial examples. Same shape as `x`.
        :type x_adv_init: `np.ndarray`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        y = check_and_transform_label_format(y, self.classifier.nb_classes(), return_one_hot=False)

        # Get clip_min and clip_max from the classifier or infer them from data
        if hasattr(self.classifier, 'clip_values') and self.classifier.clip_values is not None:
            clip_min, clip_max = self.classifier.clip_values
        else:
            clip_min, clip_max = np.min(x), np.max(x)

        # Prediction from the original images
        preds = np.argmax(self.classifier.predict(x, batch_size=self.batch_size), axis=1)

        # Prediction from the initial adversarial examples if not None
        x_adv_init = kwargs.get('x_adv_init')

        if x_adv_init is not None:
            init_preds = np.argmax(self.classifier.predict(x_adv_init, batch_size=self.batch_size), axis=1)
        else:
            init_preds = [None] * len(x)
            x_adv_init = [None] * len(x)

        # Assert that, if attack is targeted, y is provided
        if self.targeted and y is None:
            raise ValueError('Target labels `y` need to be provided for a targeted attack.')

        # Some initial setups
        x_adv = x.astype(NUMPY_DTYPE)

        # Generate the adversarial samples
        for ind, val in enumerate(x_adv):
            if self.targeted:
                x_adv[ind] = self._perturb(x=val, y=y[ind], y_p=preds[ind], init_pred=init_preds[ind],
                                           adv_init=x_adv_init[ind], clip_min=clip_min, clip_max=clip_max)
            else:
                x_adv[ind] = self._perturb(x=val, y=-1, y_p=preds[ind], init_pred=init_preds[ind],
                                           adv_init=x_adv_init[ind], clip_min=clip_min, clip_max=clip_max)

        if y is not None:
            y = to_categorical(y, self.classifier.nb_classes())

        logger.info('Success rate of Boundary attack: %.2f%%',
                    100 * compute_success(self.classifier, x, y, x_adv, self.targeted, batch_size=self.batch_size))

        return x_adv

    def _perturb(self, x, y, y_p, init_pred, adv_init, clip_min, clip_max):
        """
        Internal attack function for one example.

        :param x: An array with one original input to be attacked.
        :type x: `np.ndarray`
        :param y: If `self.targeted` is true, then `y` represents the target label.
        :type y: `int`
        :param y_p: The predicted label of x.
        :type y_p: `int`
        :param init_pred: The predicted label of the initial image.
        :type init_pred: `int`
        :param adv_init: Initial array to act as an initial adversarial example.
        :type adv_init: `np.ndarray`
        :param clip_min: Minimum value of an example.
        :type clip_min: `float`
        :param clip_max: Maximum value of an example.
        :type clip_max: `float`
        :return: an adversarial example.
        :rtype: `np.ndarray`
        """
        # First, create an initial adversarial sample
        initial_sample = self._init_sample(x, y, y_p, init_pred, adv_init, clip_min, clip_max)

        # If an initial adversarial example is not found, then return the original image
        if initial_sample is None:
            return x

        # If an initial adversarial example found, then go with boundary attack
        x_adv = self._attack(initial_sample[0], x, initial_sample[1], self.delta, self.epsilon, clip_min, clip_max)

        return x_adv

    def _attack(self, initial_sample, original_sample, target, initial_delta, initial_epsilon, clip_min, clip_max):
        """
        Main function for the boundary attack.

        :param initial_sample: An initial adversarial example.
        :type initial_sample: `np.ndarray`
        :param original_sample: The original input.
        :type original_sample: `np.ndarray`
        :param target: The target label.
        :type target: `int`
        :param initial_delta: Initial step size for the orthogonal step.
        :type initial_delta: `float`
        :param initial_epsilon: Initial step size for the step towards the target.
        :type initial_epsilon: `float`
        :param clip_min: Minimum value of an example.
        :type clip_min: `float`
        :param clip_max: Maximum value of an example.
        :type clip_max: `float`
        :return: an adversarial example.
        :rtype: `np.ndarray`
        """
        # Get initialization for some variables
        x_adv = initial_sample
        self.curr_delta = initial_delta
        self.curr_epsilon = initial_epsilon

        # Main loop to wander around the boundary
        for _ in range(self.max_iter):
            # Trust region method to adjust delta
            for _ in range(self.num_trial):
                potential_advs = []
                for _ in range(self.sample_size):
                    potential_adv = x_adv + self._orthogonal_perturb(self.curr_delta, x_adv, original_sample)
                    potential_adv = np.clip(potential_adv, clip_min, clip_max)
                    potential_advs.append(potential_adv)

                preds = np.argmax(self.classifier.predict(np.array(potential_advs), batch_size=self.batch_size), axis=1)
                satisfied = (preds == target)
                delta_ratio = np.mean(satisfied)

                if delta_ratio < 0.2:
                    self.curr_delta *= self.step_adapt
                elif delta_ratio > 0.5:
                    self.curr_delta /= self.step_adapt

                if delta_ratio > 0:
                    x_advs = np.array(potential_advs)[np.where(satisfied)[0]]
                    break
            else:
                logging.warning('Adversarial example found but not optimal.')
                return x_adv

            # Trust region method to adjust epsilon
            for _ in range(self.num_trial):
                perturb = np.repeat(np.array([original_sample]), len(x_advs), axis=0) - x_advs
                perturb *= self.curr_epsilon
                potential_advs = x_advs + perturb
                potential_advs = np.clip(potential_advs, clip_min, clip_max)
                preds = np.argmax(self.classifier.predict(potential_advs, batch_size=self.batch_size), axis=1)
                satisfied = (preds == target)
                epsilon_ratio = np.mean(satisfied)

                if epsilon_ratio < 0.2:
                    self.curr_epsilon *= self.step_adapt
                elif epsilon_ratio > 0.5:
                    self.curr_epsilon /= self.step_adapt

                if epsilon_ratio > 0:
                    x_adv = potential_advs[np.where(satisfied)[0][0]]
                    break
            else:
                logging.warning('Adversarial example found but not optimal.')
                return x_advs[0]

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
        :rtype: `np.ndarray`
        """
        # Generate perturbation randomly
        # input_shape = current_sample.shape
        perturb = np.random.randn(*self.classifier.input_shape).astype(NUMPY_DTYPE)

        # Rescale the perturbation
        perturb /= np.linalg.norm(perturb)
        perturb *= delta * np.linalg.norm(original_sample - current_sample)

        # Project the perturbation onto sphere
        direction = original_sample - current_sample

        if len(self.classifier.input_shape) == 3:
            perturb = np.swapaxes(perturb, 0, self.classifier.channel_index - 1)
            direction = np.swapaxes(direction, 0, self.classifier.channel_index - 1)
            for i in range(direction.shape[0]):
                direction[i] /= np.linalg.norm(direction[i])
                perturb[i] -= np.dot(perturb[i], direction[i]) * direction[i]
            perturb = np.swapaxes(perturb, 0, self.classifier.channel_index - 1)
        elif len(self.classifier.input_shape) == 1:
            direction /= np.linalg.norm(direction)
            perturb -= np.dot(perturb, direction.T) * direction
        else:
            raise ValueError('Input shape not recognised.')

        return perturb

    def _init_sample(self, x, y, y_p, init_pred, adv_init, clip_min, clip_max):
        """
        Find initial adversarial example for the attack.

        :param x: An array with 1 original input to be attacked.
        :type x: `np.ndarray`
        :param y: If `self.targeted` is true, then `y` represents the target label.
        :type y: `int`
        :param y_p: The predicted label of x.
        :type y_p: `int`
        :param init_pred: The predicted label of the initial image.
        :type init_pred: `int`
        :param adv_init: Initial array to act as an initial adversarial example.
        :type adv_init: `np.ndarray`
        :param clip_min: Minimum value of an example.
        :type clip_min: `float`
        :param clip_max: Maximum value of an example.
        :type clip_max: `float`
        :return: an adversarial example.
        :rtype: `np.ndarray`
        """
        nprd = np.random.RandomState()
        initial_sample = None

        if self.targeted:
            # Attack satisfied
            if y == y_p:
                return None

            # Attack unsatisfied yet and the initial image satisfied
            if adv_init is not None and init_pred == y:
                return adv_init.astype(NUMPY_DTYPE), init_pred

            # Attack unsatisfied yet and the initial image unsatisfied
            for _ in range(self.init_size):
                random_img = nprd.uniform(clip_min, clip_max, size=x.shape).astype(x.dtype)
                random_class = np.argmax(self.classifier.predict(np.array([random_img]), batch_size=self.batch_size),
                                         axis=1)[0]

                if random_class == y:
                    initial_sample = random_img, random_class

                    logging.info('Found initial adversarial image for targeted attack.')
                    break
            else:
                logging.warning('Failed to draw a random image that is adversarial, attack failed.')

        else:
            # The initial image satisfied
            if adv_init is not None and init_pred != y_p:
                return adv_init.astype(NUMPY_DTYPE), init_pred

            # The initial image unsatisfied
            for _ in range(self.init_size):
                random_img = nprd.uniform(clip_min, clip_max, size=x.shape).astype(x.dtype)
                random_class = np.argmax(self.classifier.predict(np.array([random_img]), batch_size=self.batch_size),
                                         axis=1)[0]

                if random_class != y_p:
                    initial_sample = random_img, random_class

                    logging.info('Found initial adversarial image for untargeted attack.')
                    break
            else:
                logging.warning('Failed to draw a random image that is adversarial, attack failed.')

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
        :param max_iter: Maximum number of iterations.
        :type max_iter: `int`
        :param num_trial: Maximum number of trials per iteration.
        :type num_trial: `int`
        :param sample_size: Number of samples per trial.
        :type sample_size: `int`
        :param init_size: Maximum number of trials for initial generation of adversarial examples.
        :type init_size: `int`
        """
        # Save attack-specific parameters
        super(BoundaryAttack, self).set_params(**kwargs)

        if not isinstance(self.max_iter, (int, np.int)) or self.max_iter < 0:
            raise ValueError("The number of iterations must be a non-negative integer.")

        if not isinstance(self.num_trial, (int, np.int)) or self.num_trial < 0:
            raise ValueError("The number of trials must be a non-negative integer.")

        if not isinstance(self.sample_size, (int, np.int)) or self.sample_size <= 0:
            raise ValueError("The number of samples must be a positive integer.")

        if not isinstance(self.init_size, (int, np.int)) or self.init_size <= 0:
            raise ValueError("The number of initial trials must be a positive integer.")

        if self.epsilon <= 0:
            raise ValueError("The initial step size for the step towards the target must be positive.")

        if self.delta <= 0:
            raise ValueError("The initial step size for the orthogonal step must be positive.")

        if self.step_adapt <= 0 or self.step_adapt >= 1:
            raise ValueError("The adaptation factor must be in the range (0, 1).")

        return True
