from __future__ import absolute_import, division, print_function, unicode_literals

import abc
import sys

import numpy as np
import tensorflow as tf


def clip_perturbation(v, eps, p):
    """
    Clip the values in v if their L_p norm is larger than eps.
    :param v: array of perturbations to clip
    :param eps: maximum norm allowed
    :param p: L_p norm to use for clipping. Only p = 2 and p = Inf supported for now
    :return: clipped values of v
    """
    if p == 2:
        v *= min(1., eps/np.linalg.norm(v, axis=(1, 2)))
    elif p == np.inf:
        v = np.sign(v) * np.minimum(abs(v), eps)
    else:
        raise NotImplementedError('Values of p different from 2 and Inf are currently not supported.')

    return v


def class_derivative(preds, x, num_labels=10):
    """
    Computes per class derivatives.
    :param preds: the model's logits
    :param x: the input placeholder
    :param num_labels: the number of classes the model has
    :return: (list) class derivatives
    """
    return [tf.gradients(preds[:, i], x) for i in range(num_labels)]

# Ensure compatibility with Python 2 and 3 when using ABCMeta
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})


class Attack(ABC):
    """
    Abstract base class for all attack classes. Adapted from cleverhans (https://github.com/openai/cleverhans).
    """
    attack_params = ['classifier', 'session']

    def __init__(self, classifier, sess=None):
        """
        :param classifier: A function that takes a symbolic input and returns the symbolic output for the model's
                      predictions.
        :param sess: The tf session to run graphs in.
        """

        self.classifier = classifier
        self.model = classifier.model
        self.sess = sess
        self.inf_loop = False

    def generate_graph(self, x, **kwargs):
        """
        Generate the attack's symbolic graph for adversarial examples. This method should be overridden in any child
        class that implements an attack that is expressable symbolically. Otherwise, it will wrap the numerical
        implementation as a symbolic operator.
        :param x: The model's symbolic inputs.
        :param **kwargs: optional parameters used by child classes.
        :return: A symbolic representation of the adversarial examples.
        """
        if not self.inf_loop:
            self.inf_loop = True
            self.set_params(**kwargs)
            graph = tf.py_func(self.generate, [x], tf.float32)
            self.inf_loop = False
            return graph
        else:
            raise NotImplementedError("No symbolic or numeric implementation of attack.")

    def generate(self, x_val, **kwargs):
        """
        Generate adversarial examples and return them as a Numpy array. This method should be overridden in any child
        class that implements an attack that is not fully expressed symbolically.
        :param x_val: A Numpy array with the original inputs.
        :param **kwargs: optional parameters used by child classes.
        :return: A Numpy array holding the adversarial examples.
        """
        if not self.inf_loop:
            self.inf_loop = True
            self.set_params(**kwargs)
            input_shape = list(x_val.shape)
            input_shape[0] = None
            self._x = tf.placeholder(tf.float32, shape=input_shape)
            self._x_adv = self.generate_graph(self._x)
            self.inf_loop = False
        else:
            raise NotImplementedError("No symbolic or numeric implementation of attack.")

        return self.sess.run(self._x_adv, feed_dict={self._x: x_val})

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and apply attack-specific checks before saving them as attributes.
        :param kwargs: a dictionary of attack-specific parameters
        :return: True when parsing was successful
        """
        for key, value in kwargs.items():
            if key in self.attack_params:
                setattr(self, key, value)
        return True
