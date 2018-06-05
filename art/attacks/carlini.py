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
import tensorflow as tf

from art.attacks.attack import Attack
from art.utils import to_categorical


class CarliniL2Method(Attack):
    """
    The L_2 optimized attack of Carlini and Wagner (2016). This attack is the most efficient and should be used as the
    primary attack to evaluate potential defences (wrt the L_0 and L_inf attacks). This implementation is inspired by
    the one in Cleverhans, which reproduces the authors' original code (https://github.com/carlini/nn_robust_attacks).
    Paper link: https://arxiv.org/pdf/1608.04644.pdf
    """
    attack_params = ['confidence', 'targeted', 'learning_rate', 'max_iter', 'binary_search_steps', 'initial_const',
                     'decay']

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
                           
        # Next we are going to create a number of Tf variables and operations. We are doing this in the constructor in
        # order to not repeat this everytime we want to create an attack for a specific input.
        shape = classifier.input_shape
        num_labels = classifier.nb_classes
                
        # Create variable for perturbation in the tanh space. This is the variable wrt to which the loss function is
        # minimized. Also create operator for initializing this variable.
        perturbation_tanh = tf.Variable(np.zeros(shape), dtype=np.float32, name='perturbation_tanh')
        self._init_perturbation_tanh = tf.variables_initializer([perturbation_tanh])
                    
        # Next we define the loss function that we are going to minimize.
        # 1) external inputs to the loss function:
        image_tanh = tf.Variable(np.zeros(shape), dtype=tf.float32, name='image_tanh')
        target = tf.Variable(np.zeros(num_labels), dtype=tf.float32, name='target')
        c = tf.Variable(0, dtype=tf.float32, name='c')

        # 2) the resulting adversarial image. Note that the optimization is performed in tanh space to keep the
        #    adversarial images bounded from clip_min and clip_max. To avoid division by zero (which occurs if
        #    arguments of arctanh are +1 or -1), we multiply arguments with _tanh_smoother. Here, in the inverse
        #    transformation, we need to divide by _tanh_smoother. It appears this is what Carlini and Wagner
        #    (2016) are alluding to in their footnote 8. However, it is not clear how their proposed trick
        #    ("instead of scaling by 1/2 we cale by 1/2 + eps") would actually work.
        self._adv_image = tf.tanh(image_tanh + perturbation_tanh)
        self._adv_image = (self._adv_image / self._tanh_smoother + 1) / 2
        (clip_min, clip_max) = classifier.clip_values
        self._adv_image = self._adv_image * (clip_max - clip_min) + clip_min
               
        # 3) consider model logits Z for the adversarial image, temporarily as a placeholder:
        self._z = tf.placeholder(dtype=tf.float32, shape=[num_labels], name='logits')

        # 4) compute squared l2 distance between original and adversarial image. Need to transform original image from
        #    arctanh space first.
        image = (tf.tanh(image_tanh) / self._tanh_smoother + 1) / 2
        image = image * (clip_max - clip_min) + clip_min
        self._l2dist = tf.reduce_sum(tf.square(self._adv_image - image))
                                    
        # 5) Compute logit of the target, and maximum logit over all other classes:
        z_target = tf.reduce_sum(self._z * target)
        z_other = tf.reduce_max(self._z * (1 - target))
        
        # The following differs from the exact definition given in Carlini and Wagner (2016). There (page 9, left
        # column, last equation), the maximum is taken over Z_other - Z_target (or Z_target - Z_other respectively)
        # and -confidence. However, it doesn't seem that that would have the desired effect (loss term is <= 0 if and
        # only if the difference between the logit of the target and any other class differs by at least confidence).
        # Hence the rearrangement here.
        if self.targeted:
            # if targeted, optimize for making the target class most likely
            loss = tf.maximum(z_other - z_target + self.confidence, 0)
        else:
            # if untargeted, optimize for making any other class most likely
            loss = tf.maximum(z_target - z_other + self.confidence, 0)

        # 6) combine loss terms:
        self._loss = c * loss + self._l2dist

        # 7) compute partial gradients:
        self._grad_l2z = tf.gradients(self._loss, self._z)[0]

        # 8) create operators to assign values to external inputs of the loss function:
        self._image_tanh = tf.placeholder(tf.float32, shape, name='image_tanh')
        self._target = tf.placeholder(tf.float32, num_labels, name='target')
        self._c = tf.placeholder(tf.float32, name='c')

        self._assign_image_tanh = image_tanh.assign(self._image_tanh)
        self._assign_target = target.assign(self._target)
        self._assign_c = c.assign(self._c)

        # Create a gradient placeholder, a decayed learning rate placeholder and an operator to update the pertubation
        self._grad_l2p = tf.placeholder(tf.float32, shape, name='grad_l2p')
        self._lr = tf.placeholder(tf.float32, name='decayed_learning_rate')
        update = perturbation_tanh - self._lr * self._grad_l2p
        self._update_pert = perturbation_tanh.assign(update)

        # Finally create a tf session
        self._sess = tf.Session()

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
        # Parse and save attack-specific parameters
        params_cpy = dict(kwargs)
        y = params_cpy.pop(str('y'), None)
        
        # Assert that, if attack is targeted, y_val is provided:
        assert not (self.targeted and y is None)
                
        # No labels provided, use model prediction as correct class
        if y is None:
            y = np.argmax(self.classifier.predict(x, logits=False), axis=1)
            y = to_categorical(y, self.classifier.nb_classes)

        # Images to be attacked:
        x_adv = x.copy()

        # Transform images to tanh space:
        (clip_min, clip_max) = self.classifier.clip_values
        x_adv = np.clip(x_adv, clip_min, clip_max)
        x_adv = (x_adv - clip_min) / (clip_max - clip_min)
        x_adv = np.arctanh(((x_adv * 2) - 1) * self._tanh_smoother)

        for j, (ex, target) in enumerate(zip(x_adv, y)):
            # Assign the external inputs to the loss function:
            self._sess.run(self._assign_image_tanh, {self._image_tanh: ex})
            self._sess.run(self._assign_target, {self._target: target})

            # Initialize binary search:
            c = self.initial_const
            c_lower_bound = 0
            c_double = True

            # Initialize placeholders for best l2 distance and attack found so far
            best_l2dist = sys.float_info.max
            best_adv_image = ex

            for _ in range(self.binary_search_steps):
                attack_success = False
                loss_prev = sys.float_info.max
                lr = self.learning_rate
                
                # Initialize perturbation in tanh space:
                self._sess.run(self._init_perturbation_tanh)

                # Assign constant c of the loss function:
                self._sess.run(self._assign_c, {self._c: c})
               
                for it in range(self.max_iter):
                    # Collect current loss and l2 distance and compute gradients:
                    adv_image = self._sess.run(self._adv_image)
                    z = self.classifier.predict(np.array([adv_image]), logits=True)[0]
                    loss, l2dist, grad_l2z = self._sess.run([self._loss, self._l2dist, self._grad_l2z], {self._z: z})
                    grad_z2p = self.classifier.class_gradient(np.array([adv_image]), logits=True)[0]
                    grad_l2p = np.zeros(shape=self.classifier.input_shape)
                    for i in range(self.classifier.nb_classes):
                        grad_l2p += grad_z2p[i] * grad_l2z[i]

                    # Update the pertubation with decayed learning rate
                    lr *= (1. / (1. + self.decay * it))
                    self._sess.run(self._update_pert, {self._grad_l2p: grad_l2p, self._lr: lr})

                    # Check whether last attack was successful:
                    # Attack success criterion: first term of the loss function is <= 0
                    last_attack_success = loss-l2dist <= 0
                    attack_success = attack_success or last_attack_success
                    
                    if last_attack_success and l2dist < best_l2dist:
                        best_l2dist = l2dist
                        best_adv_image =  self._sess.run(self._adv_image)
                                       
                    # Check simple stopping criterion:
                    if loss > loss_prev:
                        break
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

            # Transform best_adv_image back into tanh space if attack is failed
            if (best_adv_image == ex).all():
                best_adv_image = (np.tanh(best_adv_image) / self._tanh_smoother + 1) / 2

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
