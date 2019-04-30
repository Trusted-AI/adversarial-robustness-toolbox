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
import six

from art import NUMPY_DTYPE
from art.attacks.attack import Attack
from art.utils import compute_success, get_labels_np_array

logger = logging.getLogger(__name__)


class ElasticNet(Attack):
    """
    The elastic net attack of Pin-Yu Chen et al. (2018). Paper link: https://arxiv.org/abs/1709.04114.
    """
    attack_params = Attack.attack_params + ['confidence', 'targeted', 'learning_rate', 'max_iter', 'beta',
                                            'binary_search_steps', 'initial_const', 'batch_size', 'decision_rule']

    def __init__(self, classifier, confidence=0.0, targeted=True, learning_rate=1e-2, binary_search_steps=9,
                 max_iter=10000, beta=1e-3, initial_const=1e-3, batch_size=128, decision_rule='EN'):
        """
        Create an ElasticNet attack instance.

        :param classifier: A trained model.
        :type classifier: :class:`.Classifier`
        :param confidence: Confidence of adversarial examples: a higher value produces examples that are farther
               away, from the original input, but classified with higher confidence as the target class.
        :type confidence: `float`
        :param targeted: Should the attack target one specific class.
        :type targeted: `bool`
        :param learning_rate: The initial learning rate for the attack algorithm. Smaller values produce better
               results but are slower to converge.
        :type learning_rate: `float`
        :param binary_search_steps: Number of times to adjust constant with binary search (positive value).
        :type binary_search_steps: `int`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param beta: Hyperparameter trading off L2 minimization for L1 minimization.
        :type beta: `float`
        :param initial_const: The initial trade-off constant `c` to use to tune the relative importance of distance
               and confidence. If `binary_search_steps` is large, the initial constant is not important, as discussed in
               Carlini and Wagner (2016).
        :type initial_const: `float`
        :param batch_size: Internal size of batches on which adversarial samples are generated.
        :type batch_size: `int`
        :param decision_rule: Decision rule. 'EN' means Elastic Net rule, 'L1' means L1 rule, 'L2' means L2 rule.
        :type decision_rule: `string`
        """
        super(ElasticNet, self).__init__(classifier)

        kwargs = {'confidence': confidence,
                  'targeted': targeted,
                  'learning_rate': learning_rate,
                  'binary_search_steps': binary_search_steps,
                  'max_iter': max_iter,
                  'beta': beta,
                  'initial_const': initial_const,
                  'batch_size': batch_size,
                  'decision_rule': decision_rule
                  }
        assert self.set_params(**kwargs)

    def _loss(self, x, x_adv):
        """
        Compute the loss function values.

        :param x: An array with the original input.
        :type x: `np.ndarray`
        :param x_adv: An array with the adversarial input.
        :type x_adv: `np.ndarray`
        :return: A tuple holding the current logits, l1 distance, l2 distance and elastic net loss.
        :rtype: `(np.ndarray, float, float, float)`
        """
        l1dist = np.sum(np.abs(x - x_adv).reshape(x.shape[0], -1), axis=1)
        l2dist = np.sum(np.square(x - x_adv).reshape(x.shape[0], -1), axis=1)
        endist = self.beta * l1dist + l2dist
        z = self.classifier.predict(np.array(x_adv, dtype=NUMPY_DTYPE), logits=True)

        return np.argmax(z, axis=1), l1dist, l2dist, endist

    def _gradient_of_loss(self, target, x, x_adv, c):
        """
        Compute the gradient of the loss function.

        :param target: An array with the target class (one-hot encoded).
        :type target: `np.ndarray`
        :param x: An array with the original input.
        :type x: `np.ndarray`
        :param x_adv: An array with the adversarial input.
        :type x_adv: `np.ndarray`
        :param c: Weight of the loss term aiming for classification as target.
        :type c: `float`
        :return: An array with the gradient of the loss function.
        :type target: `np.ndarray`
        """
        # Compute the current logits
        z = self.classifier.predict(np.array(x_adv, dtype=NUMPY_DTYPE), logits=True)

        if self.targeted:
            i_sub = np.argmax(target, axis=1)
            i_add = np.argmax(z * (1 - target) + (np.min(z, axis=1) - 1)[:, np.newaxis] * target, axis=1)
        else:
            i_add = np.argmax(target, axis=1)
            i_sub = np.argmax(z * (1 - target) + (np.min(z, axis=1) - 1)[:, np.newaxis] * target, axis=1)

        loss_gradient = self.classifier.class_gradient(x_adv, label=i_add, logits=True)
        loss_gradient -= self.classifier.class_gradient(x_adv, label=i_sub, logits=True)
        loss_gradient = loss_gradient.reshape(x.shape)

        c_mult = c
        for _ in range(len(x.shape)-1):
            c_mult = c_mult[:, np.newaxis]

        loss_gradient *= c_mult
        loss_gradient += 2 * (x_adv - x)

        return loss_gradient

    def _decay_learning_rate(self, global_step, end_learning_rate, decay_steps):
        """
        Applies a square-root decay to the learning rate.

        :param global_step: Global step to use for the decay computation.
        :type global_step: `int`
        :param end_learning_rate: The minimal end learning rate.
        :type end_learning_rate: `float`
        :param decay_steps: Number of decayed steps.
        :type decay_steps: `int`
        :return: The decayed learning rate
        :rtype: `float`
        """
        decayed_learning_rate = (self.learning_rate - end_learning_rate) * (1 - global_step / decay_steps)**2 + \
            end_learning_rate

        return decayed_learning_rate

    def generate(self, x, **kwargs):
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :type x: `np.ndarray`
        :param y: If `self.targeted` is true, then `y` represents the target labels. Otherwise, the targets are the
                  original class labels.
        :type y: `np.ndarray`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        x_adv = x.astype(NUMPY_DTYPE)
        (clip_min, clip_max) = self.classifier.clip_values

        # Parse and save attack-specific parameters
        params_cpy = dict(kwargs)
        y = params_cpy.pop(str('y'), None)
        self.set_params(**params_cpy)

        # Assert that, if attack is targeted, y is provided:
        if self.targeted and y is None:
            raise ValueError('Target labels `y` need to be provided for a targeted attack.')

        # No labels provided, use model prediction as correct class
        if y is None:
            y = get_labels_np_array(self.classifier.predict(x, logits=False))

        # Compute adversarial examples with implicit batching
        nb_batches = int(np.ceil(x_adv.shape[0] / float(self.batch_size)))
        for batch_id in range(nb_batches):
            logger.debug('Processing batch %i out of %i', batch_id, nb_batches)

            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            x_batch = x_adv[batch_index_1:batch_index_2]
            y_batch = y[batch_index_1:batch_index_2]
            x_adv[batch_index_1:batch_index_2] = self._generate_batch(x_batch, y_batch)

        # Apply clip
        x_adv = np.clip(x_adv, clip_min, clip_max)

        # Compute success rate of the EAD attack
        logger.info('Success rate of EAD attack: %.2f%%',
                    100 * compute_success(self.classifier, x, y, x_adv, self.targeted))

        return x_adv

    def _generate_batch(self, x_batch, y_batch):
        """
        Run the attack on a batch of images and labels.

        :param x_batch: A batch of original examples.
        :type x_batch: `np.ndarray`
        :param y_batch: A batch of targets (0-1 hot).
        :type y_batch: `np.ndarray`
        :return: A batch of adversarial examples.
        :rtype: `np.ndarray`
        """
        # Initialize binary search:
        c = self.initial_const * np.ones(x_batch.shape[0])
        c_lower_bound = np.zeros(x_batch.shape[0])
        c_upper_bound = 10e10 * np.ones(x_batch.shape[0])

        # Initialize best distortions and best attacks globally
        o_best_dist = np.inf * np.ones(x_batch.shape[0])
        o_best_attack = x_batch.copy()

        # Start with a binary search
        for bss in range(self.binary_search_steps):
            logger.debug('Binary search step %i out of %i (c_mean==%f)', bss, self.binary_search_steps, np.mean(c))

            # Run with 1 specific binary search step
            best_dist, best_label, best_attack = self._generate_bss(x_batch, y_batch, c)

            # Update best results so far
            o_best_attack[best_dist < o_best_dist] = best_attack[best_dist < o_best_dist]
            o_best_dist[best_dist < o_best_dist] = best_dist[best_dist < o_best_dist]

            # Adjust the constant as needed
            c, c_lower_bound, c_upper_bound = self._update_const(y_batch, best_label, c, c_lower_bound, c_upper_bound)

        return o_best_attack

    def _update_const(self, y_batch, best_label, c, c_lower_bound, c_upper_bound):
        """
        Update constants.

        :param y_batch: A batch of targets (0-1 hot).
        :type y_batch: `np.ndarray`
        :param best_label: A batch of best labels.
        :type best_label: `np.ndarray`
        :param c: A batch of constants.
        :type c: `np.ndarray`
        :param c_lower_bound: A batch of lower bound constants.
        :type c_lower_bound: `np.ndarray`
        :param c_upper_bound: A batch of upper bound constants.
        :type c_upper_bound: `np.ndarray`
        :return: A tuple of three batches of updated constants and lower/upper bounds.
        :rtype: `tuple`
        """
        def compare(o1, o2):
            if self.targeted:
                return o1 == o2
            else:
                return o1 != o2

        for i in range(len(c)):
            if compare(best_label[i], np.argmax(y_batch[i])) and best_label[i] != -np.inf:
                # Successful attack
                c_upper_bound[i] = min(c_upper_bound[i], c[i])
                if c_upper_bound[i] < 1e9:
                    c[i] = (c_lower_bound[i] + c_upper_bound[i]) / 2.0

            else:
                # Failure attack
                c_lower_bound[i] = max(c_lower_bound[i], c[i])
                if c_upper_bound[i] < 1e9:
                    c[i] = (c_lower_bound[i] + c_upper_bound[i]) / 2.0
                else:
                    c[i] *= 10

        return c, c_lower_bound, c_upper_bound

    def _generate_bss(self, x_batch, y_batch, c):
        """
        Generate adversarial examples for a batch of inputs with a specific batch of constants.

        :param x_batch: A batch of original examples.
        :type x_batch: `np.ndarray`
        :param y_batch: A batch of targets (0-1 hot).
        :type y_batch: `np.ndarray`
        :param c: A batch of constants.
        :type c: `np.ndarray`
        :return: A tuple of best elastic distances, best labels, best attacks
        :rtype: `tuple`
        """
        def compare(o1, o2):
            if self.targeted:
                return o1 == o2
            else:
                return o1 != o2

        # Initialize best distortions and best changed labels and best attacks
        best_dist = np.inf * np.ones(x_batch.shape[0])
        best_label = [-np.inf] * x_batch.shape[0]
        best_attack = x_batch.copy()

        # Implement the algorithm 1 in the EAD paper
        x_adv = x_batch.copy()
        y_adv = x_batch.copy()
        for it in range(self.max_iter):
            logger.debug('Iteration step %i out of %i', it, self.max_iter)

            # Update learning rate
            lr = self._decay_learning_rate(global_step=it, end_learning_rate=0, decay_steps=self.max_iter)

            # Compute adversarial examples
            grad = self._gradient_of_loss(target=y_batch, x=x_batch, x_adv=y_adv, c=c)
            x_adv_next = self._shrinkage_threshold(y_adv - lr * grad, x_batch, self.beta)
            y_adv = x_adv_next + (1.0 * it / (it + 3)) * (x_adv_next - x_adv)
            x_adv = x_adv_next

            # Adjust the best result
            (z, l1dist, l2dist, endist) = self._loss(x=x_batch, x_adv=x_adv)

            if self.decision_rule == 'EN':
                zip_set = zip(endist, z)
            elif self.decision_rule == 'L1':
                zip_set = zip(l1dist, z)
            elif self.decision_rule == 'L2':
                zip_set = zip(l2dist, z)
            else:
                raise ValueError("The decision rule only supports `EN`, `L1`, `L2`.")

            for j, (d, s) in enumerate(zip_set):
                if d < best_dist[j] and compare(s, np.argmax(y_batch[j])):
                    best_dist[j] = d
                    best_attack[j] = x_adv[j]
                    best_label[j] = s

        return best_dist, best_label, best_attack

    @staticmethod
    def _shrinkage_threshold(z, x, beta):
        """
        Implement the element-wise projected shrinkage-threshold function.

        :param z: a batch of examples.
        :type z: `np.ndarray`
        :param x: a batch of original examples.
        :type x: `np.ndarray`
        :param beta: the shrink parameter.
        :type beta: `float`
        :return: a shrinked version of z.
        :rtype: `np.ndarray`
        """
        cond1 = (z - x) > beta
        cond2 = np.abs(z - x) <= beta
        cond3 = (z - x) < -beta

        upper = np.minimum(z - beta, 1.0)
        lower = np.maximum(z + beta, 0.0)

        result = cond1 * upper + cond2 * x + cond3 * lower

        return result

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param confidence: Confidence of adversarial examples: a higher value produces examples that are farther
               away, from the original input, but classified with higher confidence as the target class.
        :type confidence: `float`
        :param targeted: Should the attack target one specific class.
        :type targeted: `bool`
        :param learning_rate: The initial learning rate for the attack algorithm. Smaller values produce better
               results but are slower to converge.
        :type learning_rate: `float`
        :param binary_search_steps: Number of times to adjust constant with binary search (positive value).
        :type binary_search_steps: `int`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param beta: Hyperparameter trading off L2 minimization for L1 minimization.
        :type beta: `float`
        :param initial_const: The initial trade-off constant `c` to use to tune the relative importance of distance
               and confidence. If `binary_search_steps` is large, the initial constant is not important, as discussed in
               Carlini and Wagner (2016).
        :type initial_const: `float`
        :param batch_size: Internal size of batches on which adversarial samples are generated.
        :type batch_size: `int`
        :param decision_rule: Decision rule. 'EN' means Elastic Net rule, 'L1' means L1 rule, 'L2' means L2 rule.
        :type decision_rule: `string`
        """
        # Save attack-specific parameters
        super(ElasticNet, self).set_params(**kwargs)

        if type(self.binary_search_steps) is not int or self.binary_search_steps < 0:
            raise ValueError("The number of binary search steps must be a non-negative integer.")

        if type(self.max_iter) is not int or self.max_iter < 0:
            raise ValueError("The number of iterations must be a non-negative integer.")

        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("The batch size must be an integer greater than zero.")

        if not isinstance(self.decision_rule, six.string_types) or self.decision_rule not in ['EN', 'L1', 'L2']:
            raise ValueError("The decision rule only supports `EN`, `L1`, `L2`.")

        return True
