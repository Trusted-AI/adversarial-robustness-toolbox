from config import config_dict

import numpy as np
import tensorflow as tf

from keras import backend as K

from cleverhans.attacks_tf import fgm

from src.attackers.attack import Attack


class FastGradientMethod(Attack):
    """
    This attack was originally implemented by Goodfellow et al. (2015) with the infinity norm (and is known as the "Fast
    Gradient Sign Method"). This implementation is inspired by the one in Cleverhans
    (https://github.com/tensorflow/cleverhans) which extends the attack to other norms, and is therefore called the Fast
    Gradient Method. Paper link: https://arxiv.org/abs/1412.6572
    """
    attack_params = ['ord', 'y', 'y_val', 'clip_min', 'clip_max']

    def __init__(self, classifier, sess=None, ord=np.inf, y=None, clip_min=None, clip_max=None):
        """
        Create a FastGradientMethod instance.
        :param ord: (optional) Order of the norm (mimics Numpy). Possible values: np.inf, 1 or 2.
        :param y: (optional) A placeholder for the model labels. Only provide this parameter if you'd like to use true
                  labels when crafting adversarial samples. Otherwise, model predictions are used as labels to avoid the
                  "label leaking" effect (explained in this paper: https://arxiv.org/abs/1611.01236). Default is None.
                  Labels should be one-hot-encoded.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        super(FastGradientMethod, self).__init__(classifier, sess)

        kwargs = {'ord': ord, 'clip_min': clip_min, 'clip_max': clip_max, 'y': y}
        self.set_params(**kwargs)

    def generate_graph(self, x, eps=0.3, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param eps: (optional float) attack step size (input variation)
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param y: (optional) A placeholder for the model labels. Only provide
                  this parameter if you'd like to use true labels when crafting
                  adversarial samples. Otherwise, model predictions are used as
                  labels to avoid the "label leaking" effect (explained in this
                  paper: https://arxiv.org/abs/1611.01236). Default is None.
                  Labels should be one-hot-encoded.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        self.set_params(**kwargs)

        return fgm(x, self._get_predictions(x, log=False), y=self.y, eps=eps, ord=self.ord,
                   clip_min=self.clip_min, clip_max=self.clip_max)

    def minimal_perturbations(self, x, x_val, eps_step=0.1, eps_max=1., **kwargs):
        """
        Iteratively compute the minimal perturbation necessary to make the class prediction change.
        :param x: (required) A Numpy array with the original inputs.
        :param eps_step: (optional float) The increase in the perturbation for each iteration
        :param eps_max: (optional float) The maximum accepted perturbation
        :param kwargs: Other parameters to send to generate_graph
        :return: A Numpy array holding the adversarial examples.
        """
        y = np.argmax(self.model.predict(x_val), 1)
        adv_x = x_val.copy()

        curr_indexes = np.arange(len(x_val))
        eps = eps_step

        while len(curr_indexes) != 0 and eps <= eps_max:

            # adversarial crafting
            adv_x_op = self.generate_graph(x, eps=eps, **kwargs)
            adv_y = tf.argmax(self.model(adv_x_op), 1)

            feed_dict = {x: x_val[curr_indexes], K.learning_phase(): 0}
            new_adv_x, new_y = self.sess.run([adv_x_op, adv_y], feed_dict=feed_dict)

            # update
            adv_x[curr_indexes] = new_adv_x
            curr_indexes = np.where(y[curr_indexes] == new_y)[0]

            eps += eps_step

        return adv_x

    def generate(self, x_val, **kwargs):
        """
        Generate adversarial samples and return them in a Numpy array.
        :param x_val: (required) A Numpy array with the original inputs.
        :param eps: (required float) attack step size (input variation)
        :param ord: (optional) Order of the norm (mimics Numpy). Possible values: np.inf, 1 or 2.
        :param y: (optional) A placeholder for the model labels. Only provide this parameter if you'd like to use true
                  labels when crafting adversarial samples. Otherwise, model predictions are used as labels to avoid the
                  "label leaking" effect (explained in this paper: https://arxiv.org/abs/1611.01236). Default is None.
                  Labels should be one-hot-encoded.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """

        input_shape = list(x_val.shape)
        input_shape[0] = None
        self._x = tf.placeholder(tf.float32, shape=input_shape)

        if "minimal" in kwargs and kwargs["minimal"]:
            return self.minimal_perturbations(self._x, x_val, **kwargs)

        else:

            self._x_adv = self.generate_graph(self._x, **kwargs)

            # Run symbolic graph without or with true labels
            if 'y_val' not in kwargs or kwargs['y_val'] is None:
                feed_dict = {self._x: x_val, K.learning_phase(): 0}
            else:
                # Verify label placeholder was given in params if using true labels
                if self.y is None:
                    raise Exception("True labels given but label placeholder not given.")
                feed_dict = {self._x: x_val, self.y: kwargs['y_val'], K.learning_phase(): 0}

        return self.sess.run(self._x_adv, feed_dict=feed_dict)

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        Attack-specific parameters:
        :param ord: (optional) Order of the norm (mimics Numpy). Possible values: np.inf, 1 or 2.
        :param y: (optional) A placeholder for the model labels. Only provide this parameter if you'd like to use true
                  labels when crafting adversarial samples. Otherwise, model predictions are used as labels to avoid the
                  "label leaking" effect (explained in this paper: https://arxiv.org/abs/1611.01236). Default is None.
                  Labels should be one-hot-encoded.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        # Save attack-specific parameters
        super(FastGradientMethod, self).set_params(**kwargs)

        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, int(1), int(2)]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")
        return True
