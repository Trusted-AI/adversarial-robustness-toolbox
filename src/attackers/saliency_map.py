from __future__ import absolute_import

from config import config_dict
from cleverhans.attacks_tf import jsma
from keras import backend as K

import numpy as np
import tensorflow as tf

from src.attackers.attack import Attack, class_derivative


class SaliencyMapMethod(Attack):
    """
    The Jacobian-based Saliency Map Method (Papernot et al. 2016). Adapted from Cleverhans.
    Paper link: https://arxiv.org/pdf/1511.07528.pdf
    """
    attack_params = ['theta', 'gamma', 'nb_classes', 'max_iter', 'clip_min', 'clip_max', 'y']

    def __init__(self, model, sess=None, theta=0.1, gamma=1., nb_classes=10, clip_min=0., clip_max=1., y=None):
        """
        Create a SaliencyMapMethod instance.

        Attack-specific parameters:
        :param theta: (optional float) Perturbation introduced to modified
                      components (can be positive or negative)
        :param gamma: (optional float) Maximum percentage of perturbed features
        :param nb_classes: (optional int) Number of model output classes
        :param clip_min: (optional float) Minimum component value for clipping
        :param clip_max: (optional float) Maximum component value for clipping
        :param y: (optional) Target placeholder if the attack is targeted
        """
        super(SaliencyMapMethod, self).__init__(model, sess)
        kwargs = {'theta': theta,
                  'gamma': gamma,
                  'nb_classes': nb_classes,
                  'clip_min': clip_min,
                  'clip_max': clip_max,
                  'y': y}
        self.set_params(**kwargs)

    def generate_graph(self, x, **kwargs):
        # Parse and save attack-specific parameters
        assert self.set_params(**kwargs)

        # Define Jacobian graph wrt to this input placeholder
        preds = self.classifier._get_predictions(x, log=False)
        grads = class_derivative(preds, x, self.nb_classes)

        # Define appropriate graph (targeted / random target labels)
        if self.y is not None:
            def jsma_wrap(x_val, y):
                return self._jsma_batch(x, preds, grads, x_val, self.theta, self.gamma, self.clip_min,
                                        self.clip_max, self.nb_classes, y=y)

            # Attack is targeted, target placeholder will need to be fed
            wrap = tf.py_func(jsma_wrap, [x, self.y], tf.float32)
        else:
            def jsma_wrap(x_val):
                return self._jsma_batch(x, preds, grads, x_val, self.theta, self.gamma, self.clip_min,
                                        self.clip_max, self.nb_classes, y=None)

            # Attack is untargeted, target values will be chosen at random
            wrap = tf.py_func(jsma_wrap, [x], tf.float32)

        return wrap

    def generate(self, x_val, **kwargs):
        """
        Generate adversarial samples and return them in a Numpy array.
        :param x_val: (required) A Numpy array with the original inputs.
        :param y: (optional) Target values if the attack is targeted
        """
        # Parse and save attack-specific parameters
        assert self.set_params(**kwargs)
        K.set_learning_phase(0)

        input_shape = list(x_val.shape)
        input_shape[0] = None
        self._x = tf.placeholder(tf.float32, shape=input_shape)
        self._x_adv = self.generate_graph(self._x, **kwargs)

        # Run symbolic graph without or with true labels
        if 'y_val' not in kwargs or kwargs['y_val'] is None:
            feed_dict = {self._x: x_val}
        else:
            if self.y is None:
                raise Exception("This attack was instantiated untargeted.")
            else:
                if len(kwargs['y_val'].shape) > 1:
                    nb_targets = len(kwargs['y_val'])
                else:
                    nb_targets = 1
                if nb_targets != len(x_val):
                    raise Exception("Specify exactly one target per input.")
            feed_dict = {self._x: x_val, self.y: kwargs['y_val']}
        return self.sess.run(self._x_adv, feed_dict=feed_dict)

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        Attack-specific parameters:
        :param theta: (optional float) Perturbation introduced to modified
                      components (can be positive or negative)
        :param gamma: (optional float) Maximum percentage of perturbed features
        :param nb_classes: (optional int) Number of model output classes
        :param clip_min: (optional float) Minimum component value for clipping
        :param clip_max: (optional float) Maximum component value for clipping
        :param y: (optional) Target placeholder if the attack is targeted
        """
        # Save attack-specific parameters
        super(SaliencyMapMethod, self).set_params(**kwargs)

        return True

    def _jsma_batch(self, x, pred, grads, X, theta, gamma, clip_min, clip_max, nb_classes, y=None):
        """
        Applies the JSMA to a batch of inputs
        :param x: the input placeholder
        :param pred: the model's symbolic output
        :param grads: symbolic gradients
        :param X: numpy array with sample inputs
        :param theta: delta for each feature adjustment
        :param gamma: a float between 0 - 1 indicating the maximum distortion percentage
        :param clip_min: minimum value for components of the example returned
        :param clip_max: maximum value for components of the example returned
        :param nb_classes: number of model output clSasses
        :param y: target class for sample input
        :return: adversarial examples
        """
        X_adv = np.zeros(X.shape)

        for ind, val in enumerate(X):
            val = np.expand_dims(val, axis=0)
            if y is None:
                # No targets provided, randomly choose from other classes
                from cleverhans.utils_tf import model_argmax
                gt = model_argmax(self.sess, x, pred, val)

                # Randomly choose from the incorrect classes for each sample
                from src.utils import random_targets
                target = random_targets(gt, nb_classes)[0]
            else:
                target = y[ind]

            X_adv[ind], _, _ = jsma(self.sess, x, pred, grads, val, np.argmax(target),
                                    theta, gamma, clip_min, clip_max)

        return np.asarray(X_adv, dtype=np.float32)
