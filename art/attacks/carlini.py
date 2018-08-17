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

import sys

import numpy as np

from art.attacks.attack import Attack
from art.utils import get_labels_np_array, to_categorical


class CarliniL2Method(Attack):
    """
    The L_2 optimized attack of Carlini and Wagner (2016). This attack is the most efficient and should be used as the
    primary attack to evaluate potential defences (wrt the L_0 and L_inf attacks). This implementation is inspired by
    the one in Cleverhans, which reproduces the authors' original code (https://github.com/carlini/nn_robust_attacks).
    Paper link: https://arxiv.org/pdf/1608.04644.pdf
    """
    attack_params = Attack.attack_params + ['confidence', 'targeted', 'learning_rate', 'max_iter',
                                            'binary_search_steps', 'initial_const', 'decay']

    def __init__(self, classifier, confidence=5.0, targeted=True, learning_rate=1e-4, binary_search_steps=25,
                 max_iter=1000, initial_const=1e-4, decay=0.):
        """
        Create a Carlini L_2 attack instance.

        :param classifier: A trained model.
        :type classifier: :class:`Classifier`
        :param confidence: Confidence of adversarial examples: a higher value produces examples that are farther away,
                from the original input, but classified with higher confidence as the target class.
        :type confidence: `float`
        :param targeted: Should the attack target one specific class.
        :type targeted: `bool`
        :param learning_rate: The learning rate for the attack algorithm. Smaller values produce better results but are
                slower to converge.
        :type learning_rate: `float`
        :param binary_search_steps: number of times to adjust constant with binary search (positive value).
        :type binary_search_steps: `int`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param initial_const: The initial trade-off constant `c` to use to tune the relative importance of distance and
                confidence. If `binary_search_steps` is large, the initial constant is not important. The default value
                1e-4 is suggested in Carlini and Wagner (2016).
        :type initial_const: `float`
        :param decay: Coefficient for learning rate decay.
        :type decay: `float`
        """
        super(CarliniL2Method, self).__init__(classifier)

        kwargs = {'confidence': confidence,
                  'targeted': targeted,
                  'learning_rate': learning_rate,
                  'binary_search_steps': binary_search_steps,
                  'max_iter': max_iter,
                  'initial_const': initial_const,
                  'decay': decay
                  }
        assert self.set_params(**kwargs)

        # There are internal hyperparameters:
        # Abort binary search for c if it exceeds this threshold (suggested in Carlini and Wagner (2016)):
        self._c_upper_bound = 10e10
        # Smooth arguments of arctanh by multiplying with this constant to avoid division by zero:
        self._tanh_smoother = 0.999999

    def loss(self, x, x_adv, target, c):
        l2dist = np.sum(np.square(x-x_adv))
        z = self.classifier.predict(np.array([x_adv]), logits=True)[0]
        z_target = np.sum(z * target)
        z_other = np.max(z * (1 - target))

        # The following differs from the exact definition given in Carlini and Wagner (2016). There (page 9, left
        # column, last equation), the maximum is taken over Z_other - Z_target (or Z_target - Z_other respectively)
        # and -confidence. However, it doesn't seem that that would have the desired effect (loss term is <= 0 if and
        # only if the difference between the logit of the target and any other class differs by at least confidence).
        # Hence the rearrangement here.

        if self.targeted:
            # if targeted, optimize for making the target class most likely
            loss = max(z_other - z_target + self.confidence, 0)
        else:
            # if untargeted, optimize for making any other class most likely
            loss = max(z_target - z_other + self.confidence, 0)

        return z, l2dist, c*loss + l2dist

    def generate(self, x, **kwargs):
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs to be attacked.
        :type x: `np.ndarray`
        :param y: If `self.targeted` is true, then `y_val` represents the target labels. Otherwise, the targets are
                the original class labels.
        :type y: `np.ndarray`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        x_adv = x.copy()
        (clip_min, clip_max) = self.classifier.clip_values

        # Parse and save attack-specific parameters
        params_cpy = dict(kwargs)
        y = params_cpy.pop(str('y'), None)
        self.set_params(**params_cpy)

        # Assert that, if attack is targeted, y_val is provided:
        assert not (self.targeted and y is None)

        # No labels provided, use model prediction as correct class
        if y is None:
            y = get_labels_np_array(self.classifier.predict(x, logits=False))

        for j, (ex, target) in enumerate(zip(x_adv, y)):
            image = ex.copy()

            # The optimization is performed in tanh space to keep the
            # adversarial images bounded from clip_min and clip_max. To avoid division by zero (which occurs if
            # arguments of arctanh are +1 or -1), we multiply arguments with _tanh_smoother.
            # It appears this is what Carlini and Wagner
            # (2016) are alluding to in their footnote 8. However, it is not clear how their proposed trick
            # ("instead of scaling by 1/2 we cale by 1/2 + eps") would actually work.
            image_tanh = np.clip(image, clip_min, clip_max)
            image_tanh = (image_tanh - clip_min) / (clip_max - clip_min)
            image_tanh = np.arctanh(((image_tanh * 2) - 1) * self._tanh_smoother)

            # Initialize binary search:
            c = self.initial_const
            c_lower_bound = 0
            c_double = True

            # Initialize placeholders for best l2 distance and attack found so far
            best_l2dist = sys.float_info.max
            best_adv_image = image

            for _ in range(self.binary_search_steps):
                attack_success = False
                loss_prev = sys.float_info.max
                lr = self.learning_rate

                # Initialize perturbation in tanh space:
                perturbation_tanh = np.zeros(image_tanh.shape)

                for it in range(self.max_iter):
                    # First transform current adversarial sample from tanh to original space:
                    adv_image = image_tanh + perturbation_tanh
                    adv_image = (np.tanh(adv_image) / self._tanh_smoother + 1) / 2
                    adv_image = adv_image * (clip_max - clip_min) + clip_min

                    # Collect current logits, loss and l2 distance.
                    z, l2dist, loss = self.loss(image, adv_image, target, c)
                    last_attack_success = loss-l2dist <= 0
                    attack_success = attack_success or last_attack_success

                    if last_attack_success:
                        if l2dist < best_l2dist:
                            best_l2dist = l2dist
                            best_adv_image = adv_image
                        break
                    # elif loss >= loss_prev:
                    #    break
                    else:
                        if self.targeted:
                            i_sub, i_add = np.argmax(target), np.argmax(z * (1 - target))
                        else:
                            i_add, i_sub = np.argmax(target), np.argmax(z * (1 - target))

                        grad_l2p = self.classifier.class_gradient(np.array([adv_image]), label=i_add, logits=True)[0]
                        grad_l2p -= self.classifier.class_gradient(np.array([adv_image]), label=i_sub, logits=True)[0]
                        grad_l2p *= c
                        grad_l2p += 2 * (adv_image - image)
                        grad_l2p *= (clip_max - clip_min)
                        grad_l2p *= (1 - np.square(np.tanh(image_tanh + perturbation_tanh))) / (2 * self._tanh_smoother)

                        # Update the pertubation with decayed learning rate
                        lr *= (1. / (1. + self.decay * it))
                        perturbation_tanh -= lr * grad_l2p[0]
                        loss_prev = loss

                # Update binary search:
                if attack_success:
                    c_double = False
                    c = (c_lower_bound + c) / 2
                else:
                    c_old = c
                    if c_double:
                        c = 2 * c
                    else:
                        c = c + (c - c_lower_bound) / 2
                    c_lower_bound = c_old

                # Abort binary search if c exceeds upper bound:
                if c > self._c_upper_bound:
                    break

            x_adv[j] = best_adv_image

        return x_adv

    def set_params(self, **kwargs):
        """Take in a dictionary of parameters and applies attack-specific checks before saving them as attributes.

        :param confidence: Confidence of adversarial examples: a higher value produces examples that are farther away,
               from the original input, but classified with higher confidence as the target class.
        :type confidence: `float`
        :param targeted: Should the attack target one specific class
        :type targetd: `bool`
        :param learning_rate: The learning rate for the attack algorithm. Smaller values produce better results but are
               slower to converge.
        :type learning_rate: `float`
        :param binary_search_steps: number of times to adjust constant with binary search (positive value)
        :type binary_search_steps: `int`
        :param max_iter: The maximum number of iterations.
        :type max_iter: `int`
        :param initial_const: (optional float, positive) The initial trade-off constant c to use to tune the relative
               importance of distance and confidence. If binary_search_steps is large,
               the initial constant is not important. The default value 1e-4 is suggested in Carlini and Wagner (2016).
        :type initial_const: `float`
        :param decay: Coefficient for learning rate decay.
        :type decay: `float`
        """
        # Save attack-specific parameters
        super(CarliniL2Method, self).set_params(**kwargs)

        if type(self.binary_search_steps) is not int or self.binary_search_steps < 0:
            raise ValueError("The number of binary search steps must be a non-negative integer.")

        if type(self.max_iter) is not int or self.max_iter < 0:
            raise ValueError("The number of iterations must be a non-negative integer.")

        return True
