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

from keras import backend as k
import numpy as np
import tensorflow as tf

from art.attacks.attack import Attack


class FastGradientMethod(Attack):
    """
    This attack was originally implemented by Goodfellow et al. (2015) with the infinity norm (and is known as the "Fast
    Gradient Sign Method"). This implementation extends the attack to other norms, and is therefore called the Fast
    Gradient Method. Paper link: https://arxiv.org/abs/1412.6572
    """
    attack_params = ['ord', 'eps', 'y', 'y_val', 'targeted', 'clip_min', 'clip_max']

    def __init__(self, classifier, sess=None, ord=np.inf, eps=.3, y=None, targeted=False, clip_min=0, clip_max=1):
        """
        Create a FastGradientMethod instance.

        :param classifier: A trained model.
        :type classifier: :class:`Classifier`
        :param sess: The session to run graphs in.
        :type sess: `tf.Session`
        :param ord: Order of the norm. Possible values: np.inf, 1 or 2.
        :type ord: `int`
        :param eps: Attack step size (input variation)
        :type eps: `float`
        :param y: A placeholder for the model labels. Only provide this parameter if you'd like to use true
                  labels when crafting adversarial samples. Otherwise, model predictions are used as labels to avoid the
                  "label leaking" effect (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
                  Labels should be one-hot-encoded.
        :type y: `np.ndarray`
        :param targeted: Should the attack target one specific class
        :type targeted: `bool`
        :param clip_min: Minimum input component value.
        :type clip_min: `float`
        :param clip_max: Maximum input component value.
        :type clip_max: `float`
        """
        super(FastGradientMethod, self).__init__(classifier, sess)

        kwargs = {'ord': ord, 'eps': eps, 'targeted': targeted, 'clip_min': clip_min, 'clip_max': clip_max, 'y': y}
        self.set_params(**kwargs)

    def generate_graph(self, x_op, eps_op, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.

        :param x_op: The model's symbolic inputs.
        :type x_op: `tf.Placeholder`
        :param eps_op: The placeholder for input variation (noise amplitude)
        :type eps_op: `tf.Placeholder`
        :param ord: Order of the norm (mimics Numpy). Possible values: np.inf, 1 or 2.
        :type ord: `int`
        :param y: (optional) A placeholder for the model labels. Only provide this parameter if you'd like to use true
                  labels when crafting adversarial samples. Otherwise, model predictions are used as labels to avoid the
                  "label leaking" effect (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
                  Labels should be one-hot-encoded.
        :type y: `np.ndarray`
        :param clip_min: Minimum input component value.
        :type clip_min: `float`
        :param clip_max: Maximum input component value.
        :type clip_max: `float`
        :return: The computation graph for producing adversarial examples.
        :rtype: `tf.Tensor`
        """
        self.set_params(**kwargs)

        preds = self.classifier._get_predictions(x_op, log=False)

        if not hasattr(self, 'y') or self.y is None:
            # Use model predictions as correct outputs
            preds_max = tf.reduce_max(preds, 1, keep_dims=True)
            y = tf.to_float(tf.equal(preds, preds_max))
            y = tf.stop_gradient(y)
        else:
            y = self.y
        y = y / tf.reduce_sum(y, 1, keep_dims=True)

        loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=y)
        if self.targeted:
            loss = -loss
        grad, = tf.gradients(loss, x_op)

        # Apply norm bound
        if self.ord == np.inf:
            grad = tf.sign(grad)
        elif self.ord == 1:
            ind = list(range(1, len(x_op.get_shape())))
            grad = grad / tf.reduce_sum(tf.abs(grad), reduction_indices=ind, keep_dims=True)
        elif self.ord == 2:
            ind = list(range(1, len(x_op.get_shape())))
            grad = grad / tf.sqrt(tf.reduce_sum(tf.square(grad), reduction_indices=ind, keep_dims=True))

        # Apply perturbation and clip
        x_adv_op = x_op + eps_op * grad
        if self.clip_min is not None and self.clip_max is not None:
            x_adv_op = tf.clip_by_value(x_adv_op, self.clip_min, self.clip_max)

        return x_adv_op

    def minimal_perturbations(self, x, x_val, eps_step=0.1, eps_max=1., **kwargs):
        """Iteratively compute the minimal perturbation necessary to make the class prediction change. Stop when the
        first adversarial example was found.

        :param x: A placeholder for the input.
        :type x: `tf.Placeholder`
        :param x_val: An array with the original inputs.
        :type x_val: `np.ndarray`
        :param eps_step: The increase in the perturbation for each iteration
        :type eps_step: `float`
        :param eps_max: The maximum accepted perturbation
        :type eps_max: `float`
        :param kwargs: Other parameters to send to `generate_graph`
        :type kwargs: `dict`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """
        k.set_learning_phase(0)
        y = np.argmax(self.model.predict(x_val), 1)
        adv_x = x_val.copy()

        curr_indexes = np.arange(len(adv_x))
        eps = tf.placeholder(tf.float32, None)
        adv_x_op = self.generate_graph(x, eps, **kwargs)
        adv_y = tf.argmax(self.model(adv_x_op), 1)
        eps_val = eps_step

        while len(curr_indexes) != 0 and eps_val <= eps_max:
            # Adversarial crafting
            new_adv_x, new_y = self.sess.run([adv_x_op, adv_y], {x: x_val[curr_indexes], eps: eps_val})

            # Update
            adv_x[curr_indexes] = new_adv_x
            curr_indexes = np.where(y[curr_indexes] == new_y)[0]
            eps_val += eps_step

        return adv_x

    def generate(self, x_val, **kwargs):
        """Generate adversarial samples and return them in an array.

        :param x_val: An array with the original inputs.
        :type x_val: `np.ndarray`
        :param eps: Attack step size (input variation)
        :type eps: `float`
        :param ord: Order of the norm (mimics Numpy). Possible values: np.inf, 1 or 2.
        :type ord: `int`
        :param y: A placeholder for the model labels. Only provide this parameter if you'd like to use true
                  labels when crafting adversarial samples. Otherwise, model predictions are used as labels to avoid the
                  "label leaking" effect (explained in this paper: https://arxiv.org/abs/1611.01236). Default is None.
                  Labels should be one-hot-encoded.
        :type y: `np.ndarray`
        :param clip_min: Minimum input component value.
        :type clip_min: `float`
        :param clip_max: Maximum input component value.
        :type clip_max: `float`
        :return: An array holding the adversarial examples.
        :rtype: `np.ndarray`
        """

        input_shape = [None] + list(x_val.shape[1:])
        self._x = tf.placeholder(tf.float32, shape=input_shape)
        k.set_learning_phase(0)

        # Return adversarial examples computed with minimal perturbation if option is active
        if "minimal" in kwargs and kwargs["minimal"]:
            return self.minimal_perturbations(self._x, x_val, **kwargs)

        # Generate computation graph
        eps = tf.placeholder(tf.float32, None)
        self._x_adv = self.generate_graph(self._x, eps, **kwargs)

        # Run symbolic graph without or with true labels
        if 'y_val' not in kwargs or kwargs['y_val'] is None:
            feed_dict = {self._x: x_val, eps: self.eps}
        else:
            # Verify label placeholder was given in params if using true labels
            if self.y is None:
                raise Exception("True labels given but label placeholder not given.")
            feed_dict = {self._x: x_val, self.y: kwargs['y_val'], eps: self.eps}

        return self.sess.run(self._x_adv, feed_dict=feed_dict)

    def set_params(self, **kwargs):
        """Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        :param ord: Order of the norm. Possible values: np.inf, 1 or 2.
        :type ord: `int`
        :param eps: Attack step size (input variation)
        :type eps: `float`
        :param y: (optional) A placeholder for the model labels. Only provide this parameter if you'd like to use true
                  labels when crafting adversarial samples. Otherwise, model predictions are used as labels to avoid the
                  "label leaking" effect (explained in this paper: https://arxiv.org/abs/1611.01236). Default is None.
                  Labels should be one-hot-encoded.
        :type y: `np.ndarray`
        :param targeted: Should the attack target one specific class
        :type targeted: `bool`
        :param clip_min: Minimum input component value.
        :type clip_min: `float`
        :param clip_max: Maximum input component value.
        :type clip_max: `float`
        """
        # Save attack-specific parameters
        super(FastGradientMethod, self).set_params(**kwargs)

        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, int(1), int(2)]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")

        if self.clip_min is not None and self.clip_max is not None:
            if self.eps <= self.clip_min or self.eps > self.clip_max:
                raise ValueError('The amount of perturbation has to be in the data range.')

        return True
