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


class HopSkipJump(Attack):
    """
    Implementation of the HopSkipJump attack from Jianbo et al. (2019). This is a powerful black-box attack that
    only requires final class prediction, and is an advanced version of the boundary attack.
    Paper link: https://arxiv.org/abs/1904.02144
    """
    attack_params = Attack.attack_params + ['targeted', 'norm', 'max_iter', 'max_eval', 'init_eval']

    def __init__(self, classifier, targeted=True, norm=2, max_iter=50, max_eval=1e4, init_eval=1e2):
        """
        Create a HopSkipJump attack instance.

        :param classifier: A trained model.
        :type classifier: :class:`.Classifier`
        :param targeted: Should the attack target one specific class.
        :type targeted: `bool`
        :param norm: Order of the norm. Possible values: np.inf or 2.
        :type norm: `int`
        :param max_iter: Maximum number of iterations.
        :type max_iter: `int`
        :param max_eval: Maximum number of evaluations for estimating gradient.
        :type max_eval: `int`
        :param init_eval: Initial number of evaluations for estimating gradient.
        :type init_eval: `int`
        """
        super(HopSkipJump, self).__init__(classifier=classifier)
        params = {'targeted': targeted,
                  'norm': norm,
                  'max_iter': max_iter,
                  'max_eval': max_eval,
                  'init_eval': init_eval,
                  }
        self.set_params(**params)

    def generate(self, x, y=None, x_adv_init=None):
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :type x: `np.ndarray`
        :param y: If `self.targeted` is true, then `y` represents the target labels.
        :type y: `np.ndarray`
        :param x_adv_init: Initial array to act as initial adversarial examples. Same shape as `x`.
        :type x_adv_init: `np.ndarray`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        # Get clip_min and clip_max from the classifier or infer them from data
        if hasattr(self.classifier, 'clip_values') and self.classifier.clip_values is not None:
            clip_min, clip_max = self.classifier.clip_values
        else:
            clip_min, clip_max = np.min(x), np.max(x)

        # Prediction from the original images
        preds = np.argmax(self.classifier.predict(x), axis=1)

        # Prediction from the initial adversarial examples if not None
        if x_adv_init is not None:
            init_preds = np.argmax(self.classifier.predict(x_adv_init), axis=1)
        else:
            init_preds = [None] * len(x)
            x_adv_init = [None] * len(x)

        # Assert that, if attack is targeted, y is provided
        if self.targeted and y is None:
            raise ValueError('Target labels `y` need to be provided for a targeted attack.')

        # Some initial setups
        x_adv = x.astype(NUMPY_DTYPE)
        if y is not None:
            y = np.argmax(y, axis=1)

        # Generate the adversarial samples
        for ind, val in enumerate(x_adv):
            if self.targeted:
                x_adv[ind] = self._perturb(x=val, y=y[ind], y_p=preds[ind], init_pred=init_preds[ind],
                                           adv_init=x_adv_init[ind], clip_min=clip_min, clip_max=clip_max)
            else:
                x_adv[ind] = self._perturb(x=val, y=-1, y_p=preds[ind], init_pred=init_preds[ind],
                                           adv_init=x_adv_init[ind], clip_min=clip_min, clip_max=clip_max)

        logger.info('Success rate of Boundary attack: %.2f%%',
                    (np.sum(preds != np.argmax(self.classifier.predict(x_adv), axis=1)) / x.shape[0]))

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

                preds = np.argmax(self.classifier.predict(np.array(potential_advs)), axis=1)
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
                preds = np.argmax(self.classifier.predict(potential_advs), axis=1)
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

        return perturb

    def _adversarial_satisfactory(self, samples, target, clip_min, clip_max):
        """
        Check whether an image is adversarial.

        :param samples: A batch of examples.
        :type samples: `np.ndarray`
        :param target: The target label.
        :type target: `int`
        :param clip_min: Minimum value of an example.
        :type clip_min: `float`
        :param clip_max: Maximum value of an example.
        :type clip_max: `float`
        :return: An array of 0/1.
        :rtype: `np.ndarray`
        """
        samples = np.clip(samples, clip_min, clip_max)
        preds = np.argmax(self.classifier.predict(samples), axis=1)

        if self.targeted:
            result = (preds == target)
        else:
            result = (preds != target)

        return result

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
                random_class = np.argmax(self.classifier.predict(np.array([random_img])), axis=1)[0]

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
                random_class = np.argmax(self.classifier.predict(np.array([random_img])), axis=1)[0]

                if random_class != y_p:
                    initial_sample = random_img, random_class

                    logging.info('Found initial adversarial image for untargeted attack.')
                    break
            else:
                logging.warning('Failed to draw a random image that is adversarial, attack failed.')

        return initial_sample

    def approximate_gradient(model, sample, num_evals, delta, params):
        clip_max, clip_min = params['clip_max'], params['clip_min']

        # Generate random vectors.
        noise_shape = [num_evals] + list(params['shape'])
        if params['constraint'] == 'l2':
            rv = np.random.randn(*noise_shape)
        elif params['constraint'] == 'linf':
            rv = np.random.uniform(low=-1, high=1, size=noise_shape)

        rv = rv / np.sqrt(np.sum(rv ** 2, axis=(1, 2, 3), keepdims=True))
        perturbed = sample + delta * rv
        perturbed = clip_image(perturbed, clip_min, clip_max)
        rv = (perturbed - sample) / delta

        # query the model.
        decisions = decision_function(model, perturbed, params)
        decision_shape = [len(decisions)] + [1] * len(params['shape'])
        fval = 2 * decisions.astype(float).reshape(decision_shape) - 1.0

        # Baseline subtraction (when fval differs)
        if np.mean(fval) == 1.0:  # label changes.
            gradf = np.mean(rv, axis=0)
        elif np.mean(fval) == -1.0:  # label not change.
            gradf = - np.mean(rv, axis=0)
        else:
            fval -= np.mean(fval)
            gradf = np.mean(fval * rv, axis=0)

        # Get the gradient direction.
        gradf = gradf / np.linalg.norm(gradf)

        return gradf

    def _interpolate(self, current_sample, original_sample, alpha):
        """
        Interpolate a new sample based on the original and the current samples.

        :param current_sample: Current adversarial example.
        :type current_sample: `np.ndarray`
        :param original_sample: The original input.
        :type original_sample: `np.ndarray`
        :param alpha: The coefficient of interpolation.
        :type alpha: `float`
        :return: an adversarial example.
        :rtype: `np.ndarray`
        """
        if self.norm == 2:
            result = (1 - alpha) * original_sample + alpha * current_sample

        else:
            result = np.clip(current_sample, original_sample - alpha, original_sample + alpha)

        return result

    def _binary_search(self, current_sample, original_sample, target, threshold=None):
        """
        Binary search to approach the boundary.

        :param current_sample: Current adversarial example.
        :type current_sample: `np.ndarray`
        :param original_sample: The original input.
        :type original_sample: `np.ndarray`
        :param target: The target label.
        :type target: `int`
        :param threshold: The upper threshold in binary search.
        :type threshold: `float`
        :return: an adversarial example.
        :rtype: `np.ndarray`
        """
        # First set upper and lower bounds as well as the threshold for the binary search
        if self.norm == 2:
            (upper_bound, lower_bound) = (1, 0)

            if threshold is None:
                threshold = self.theta

        else:
            (upper_bound, lower_bound) = (np.max(abs(original_sample - current_sample)), 0)

            if threshold is None:
                threshold = np.min(upper_bound * self.theta, self.theta)

        # Then start the binary search
        while (upper_bound - lower_bound) > threshold:
            # Interpolation point
            alpha = (upper_bound + lower_bound) / 2.0
            interpolated_sample = self._interpolate(current_sample=current_sample,
                                                    original_sample=original_sample,
                                                    alpha=alpha)

            # Update upper_bound and lower_bound
            decisions = decision_function(model, mid_images, params)
            lows = np.where(decisions == 0, mids, lows)
            highs = np.where(decisions == 1, mids, highs)

        out_images = project(original_image, perturbed_images, highs, params)

        # Compute distance of the output image to select the best choice.
        # (only used when stepsize_search is grid_search.)
        dists = np.array([
            compute_distance(
                original_image,
                out_image,
                params['constraint']
            )
            for out_image in out_images])
        idx = np.argmin(dists)

        dist = dists_post_update[idx]
        out_image = out_images[idx]

    return out_image, dist




    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param targeted: Should the attack target one specific class.
        :type targeted: `bool`
        :param norm: Order of the norm. Possible values: np.inf or 2.
        :type norm: `int`
        :param max_iter: Maximum number of iterations.
        :type max_iter: `int`
        :param max_eval: Maximum number of evaluations for estimating gradient.
        :type max_eval: `int`
        :param init_eval: Initial number of evaluations for estimating gradient.
        :type init_eval: `int`
        """
        # Save attack-specific parameters
        super(HopSkipJump, self).set_params(**kwargs)

        # Check if order of the norm is acceptable given current implementation
        if self.norm not in [np.inf, int(2)]:
            raise ValueError('Norm order must be either `np.inf` or 2.')

        if not isinstance(self.max_iter, (int, np.int)) or self.max_iter < 0:
            raise ValueError("The number of iterations must be a non-negative integer.")

        if not isinstance(self.max_eval, (int, np.int)) or self.max_eval <= 0:
            raise ValueError("The maximum number of evaluations must be a positive integer.")

        if not isinstance(self.init_eval, (int, np.int)) or self.init_eval <= 0:
            raise ValueError("The initial number of evaluations must be a positive integer.")

        if self.init_eval > self.max_eval:
            raise ValueError("The maximum number of evaluations must be larger than the initial number of evaluations.")

        return True
