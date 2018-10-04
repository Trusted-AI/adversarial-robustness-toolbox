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
import sys

import numpy as np

from art.attacks.attack import Attack
from art.utils import get_labels_np_array

logger = logging.getLogger(__name__)


class CarliniL2Method(Attack):
    """
    The L_2 optimized attack of Carlini and Wagner (2016). This attack is the most efficient and should be used as the
    primary attack to evaluate potential defences (wrt the L_0 and L_inf attacks). This implementation is inspired by
    the one in Cleverhans, which reproduces the authors' original code (https://github.com/carlini/nn_robust_attacks).
    Paper link: https://arxiv.org/pdf/1608.04644.pdf
    """
    attack_params = Attack.attack_params + ['confidence', 'targeted', 'learning_rate', 'max_iter',
                                            'binary_search_steps', 'initial_const', 'max_halving', 'max_doubling']

    def __init__(self, classifier, confidence=0.0, targeted=True, learning_rate=0.01, binary_search_steps=10,
                 max_iter=10, initial_const=0.01, max_halving=5, max_doubling=5):
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
                confidence. If `binary_search_steps` is large, the initial constant is not important, as discussed in
                Carlini and Wagner (2016).
        :type initial_const: `float`
        :param max_halving: Maximum number of halving steps in the line search optimization.
        :type max_halving: `int`
        :param max_doubling: Maximum number of doubling steps in the line search optimization.
        :type max_doubling: `int`
        """
        super(CarliniL2Method, self).__init__(classifier)

        kwargs = {'confidence': confidence,
                  'targeted': targeted,
                  'learning_rate': learning_rate,
                  'binary_search_steps': binary_search_steps,
                  'max_iter': max_iter,
                  'initial_const': initial_const,
                  'max_halving': max_halving,
                  'max_doubling': max_doubling
                  }
        assert self.set_params(**kwargs)

        # There are internal hyperparameters:
        # Abort binary search for c if it exceeds this threshold (suggested in Carlini and Wagner (2016)):
        self._c_upper_bound = 10e10
        # Smooth arguments of arctanh by multiplying with this constant to avoid division by zero:
        self._tanh_smoother = 0.999999
    
    def _loss(self, x, x_adv, target, c):    
        """
        Compute the objective function value.

        :param x: An array with the original input.
        :type x: `np.ndarray`
        :param x_adv: An array with the adversarial input.
        :type x_adv: `np.ndarray`
        :param target: An array with the target class (one-hot encoded).
        :type target: `np.ndarray`
        :param c: Weight of the loss term aiming for classification as target.
        :type c: `float`
        :return: A tuple holding the current logits, l2 distance and overall loss.
        :rtype: `(float, float, float)`
        """    
        l2dist = np.sum(np.square(x-x_adv))        
        z = self.classifier.predict(np.array([x_adv]), logits=True)[0]
        z_target = np.sum(z * target)
        z_other = np.max(z * (1 - target) + (np.min(z)-1)*target)
        
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
    
    def _loss_gradient(self, z, target, x, x_adv, x_adv_tanh, c, clip_min, clip_max):  
        """
        Compute the gradient of the loss function.

        :param z: An array with the current logits.
        :type z: `np.ndarray`
        :param target: An array with the target class (one-hot encoded).
        :type target: `np.ndarray`
        :param x: An array with the original input.
        :type x: `np.ndarray`
        :param x_adv: An array with the adversarial input.
        :type x_adv: `np.ndarray`
        :param x_adv_tanh: An array with the adversarial input in tanh space.
        :type x_adv_tanh: `np.ndarray`
        :param c: Weight of the loss term aiming for classification as target.
        :type c: `float`
        :param clip_min: Minimum clipping value.
        :type clip_min: `float`
        :param clip_max: Maximum clipping value.
        :type clip_max: `float`
        :return: An array with the gradient of the loss function.
        :type target: `np.ndarray`
        """  
        if self.targeted:
            i_sub, i_add = np.argmax(target), np.argmax(z * (1 - target) + (np.min(z)-1)*target)
        else:
            i_add, i_sub = np.argmax(target), np.argmax(z * (1 - target) + (np.min(z)-1)*target)
        
        loss_gradient = self.classifier.class_gradient(np.array([x_adv]), label=i_add, logits=True)[0]
        loss_gradient -= self.classifier.class_gradient(np.array([x_adv]), label=i_sub, logits=True)[0]
        loss_gradient *= c
        loss_gradient += 2*(x_adv - x)
        loss_gradient *= (clip_max - clip_min) 
        loss_gradient *= (1-np.square(np.tanh(x_adv_tanh)))/(2*self._tanh_smoother)
        
        return loss_gradient[0]
                        
    
    def _original_to_tanh(self, x_original, clip_min, clip_max):
        """
        Transform input from original to tanh space.

        :param x_original: An array with the input to be transformed.
        :type x_original: `np.ndarray`
        :param clip_min: Minimum clipping value.
        :type clip_min: `float`
        :param clip_max: Maximum clipping value.
        :type clip_max: `float`
        :return: An array holding the transformed input.
        :rtype: `np.ndarray`
        """    
        # To avoid division by zero (which occurs if arguments of arctanh are +1 or -1),
        # we multiply arguments with _tanh_smoother. It appears this is what Carlini and Wagner
        # (2016) are alluding to in their footnote 8. However, it is not clear how their proposed trick
        # ("instead of scaling by 1/2 we scale by 1/2 + eps") works in detail.    
        x_tanh = np.clip(x_original, clip_min, clip_max)
        x_tanh = (x_tanh - clip_min) / (clip_max - clip_min)
        x_tanh = np.arctanh(((x_tanh * 2) - 1) * self._tanh_smoother)
        return x_tanh
        
    def _tanh_to_original(self, x_tanh, clip_min, clip_max):
        """
        Transform input from tanh to original space.

        :param x_tanh: An array with the input to be transformed.
        :type x_tanh: `np.ndarray`
        :param clip_min: Minimum clipping value.
        :type clip_min: `float`
        :param clip_max: Maximum clipping value.
        :type clip_max: `float`
        :return: An array holding the transformed input.
        :rtype: `np.ndarray`
        """        
        x_original = (np.tanh(x_tanh) / self._tanh_smoother + 1) / 2
        return x_original * (clip_max - clip_min) + clip_min
    
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
        if self.targeted and y is None:
            raise ValueError('Target labels `y` need to be provided for a targeted attack.')

        # No labels provided, use model prediction as correct class
        if y is None:
            y = get_labels_np_array(self.classifier.predict(x, logits=False))

        for j, (ex, target) in enumerate(zip(x_adv, y)):
            image = ex.copy()

            # The optimization is performed in tanh space to keep the
            # adversarial images bounded from clip_min and clip_max. 
            image_tanh = self._original_to_tanh(image, clip_min, clip_max)

            # Initialize binary search:
            c = self.initial_const
            c_lower_bound = 0
            c_double = True

            # Initialize placeholders for best l2 distance and attack found so far
            best_l2dist = sys.float_info.max
            best_adv_image = image            
            lr = self.learning_rate
            
            for _ in range(self.binary_search_steps):
                
                # Initialize perturbation in tanh space:
                perturbation_tanh = np.zeros(image_tanh.shape)
                adv_image = image
                adv_image_tanh = image_tanh
                z, l2dist, loss = self._loss(image, adv_image, target, c)
                attack_success = (loss-l2dist <= 0)
               
                for it in range(self.max_iter):                   
                    if attack_success:
                        break
                    
                    # compute gradient:
                    perturbation_tanh = -self._loss_gradient(z, target, image, adv_image, adv_image_tanh, 
                                                             c, clip_min, clip_max)
                    
                    # perform line search to optimize perturbation                     
                    # first, halve the learning rate until perturbation actually decreases the loss:                      
                    prev_loss = loss
                    halving = 0
                    while loss >= prev_loss and loss-l2dist > 0 and halving < self.max_halving: 
                        new_adv_image_tanh = adv_image_tanh + lr*perturbation_tanh
                        new_adv_image = self._tanh_to_original(new_adv_image_tanh, clip_min, clip_max)
                        _, l2dist, loss = self._loss(image, new_adv_image, target, c)
                        lr /= 2
                        halving += 1                        
                    lr *= 2
                    
                    # if no halving was actually required, double the learning rate as long as this
                    # decreases the loss:
                    if halving == 1:
                        doubling = 0
                        while loss <= prev_loss and doubling < self.max_doubling:  
                            prev_loss = loss
                            lr *= 2     
                            doubling += 1
                            new_adv_image_tanh = adv_image_tanh + lr*perturbation_tanh
                            new_adv_image = self._tanh_to_original(new_adv_image_tanh, clip_min, clip_max)
                            _, l2dist, loss = self._loss(image, new_adv_image, target, c)
                        lr /= 2
                    
                    # apply the optimal learning rate that was found and update the loss:
                    adv_image_tanh = adv_image_tanh + lr*perturbation_tanh
                    adv_image = self._tanh_to_original(adv_image_tanh, clip_min, clip_max)
                    z, l2dist, loss = self._loss(image, adv_image, target, c)                    
                    attack_success = (loss-l2dist <= 0)
                
                # Update depending on attack success:
                if attack_success:
                    if l2dist < best_l2dist:
                        best_l2dist = l2dist
                        best_adv_image =  adv_image 
                        
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

        adv_preds = np.argmax(self.classifier.predict(x_adv), axis=1)
        if self.targeted:
            rate = np.sum(adv_preds == np.argmax(y, axis=1)) / x_adv.shape[0]
        else:
            preds = np.argmax(self.classifier.predict(x), axis=1)
            rate = np.sum(adv_preds != preds) / x_adv.shape[0]
        logger.info('Success rate of C&W attack: %.2f%%', rate)

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
        :param max_halving: Maximum number of halving steps in the line search optimization.
        :type max_halving: `int`
        :param max_doubling: Maximum number of doubling steps in the line search optimization.
        :type max_doubling: `int`
        """
        # Save attack-specific parameters
        super(CarliniL2Method, self).set_params(**kwargs)

        if type(self.binary_search_steps) is not int or self.binary_search_steps < 0:
            raise ValueError("The number of binary search steps must be a non-negative integer.")

        if type(self.max_iter) is not int or self.max_iter < 0:
            raise ValueError("The number of iterations must be a non-negative integer.")
            
        if type(self.max_halving) is not int or self.max_halving < 1:
            raise ValueError("The number of halving steps must be an integer greater than zero.")
            
        if type(self.max_doubling) is not int or self.max_doubling < 1:
            raise ValueError("The number of doubling steps must be an integer greater than zero.")

        return True
