from config import config_dict
from cleverhans.attacks_tf import fgm
import numpy as np
import tensorflow as tf

from src.attackers.attack import Attack


class FastGradientMethod(Attack):
    """
    This attack was originally implemented by Goodfellow et al. (2015) with the infinity norm (and is known as the "Fast
    Gradient Sign Method"). This implementation extends the attack to other norms, and is therefore called the Fast
    Gradient Method. Paper link: https://arxiv.org/abs/1412.6572
    """
    attack_params = ['eps', 'ord', 'y', 'clip_min', 'clip_max']

    def __init__(self, model, sess=None, eps=0.3, ord=np.inf, y=None, clip_min=None, clip_max=None):
        """
        Create a FastGradientMethod instance.
        :param eps: (optional float) attack step size (input variation)
        :param ord: (optional) Order of the norm (mimics Numpy). Possible values: np.inf, 1 or 2.
        :param y: (optional) A placeholder for the model labels. Only provide this parameter if you'd like to use true
                  labels when crafting adversarial samples. Otherwise, model predictions are used as labels to avoid the
                  "label leaking" effect (explained in this paper: https://arxiv.org/abs/1611.01236). Default is None.
                  Labels should be one-hot-encoded.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        super(FastGradientMethod, self).__init__(model, sess)

        kwargs = {'eps': eps, 'ord': ord, 'clip_min': clip_min, 'clip_max': clip_max, 'y': y}
        self.set_params(**kwargs)

    def generate_graph(self, x, **kwargs):
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

        return fgm(x, self.model(x), y=self.y, eps=self.eps, ord=self.ord,
                   clip_min=self.clip_min, clip_max=self.clip_max)

    def generate(self, x_val, **kwargs):
        """
        Generate adversarial samples and return them in a Numpy array.
        :param x_val: (required) A Numpy array with the original inputs.
        :param eps: (required float) attack step size (input variation)
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

        # Generate this attack's graph if it hasn't been done previously
        if not hasattr(self, "_x"):
            input_shape = list(x_val.shape)
            input_shape[0] = None
            self._x = tf.placeholder(tf.float32, shape=input_shape)
            self._x_adv = self.generate_graph(self._x)

        # Run symbolic graph without or with true labels
        if 'y_val' not in kwargs or kwargs['y_val'] is None:
            feed_dict = {self._x: x_val}
        else:
            # Verify label placeholder was given in params if using true labels
            if self.y is None:
                raise Exception("True labels given but label placeholder not given.")
            feed_dict = {self._x: x_val, self.y: kwargs['y_val']}
        return self.sess.run(self._x_adv, feed_dict=feed_dict)

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        Attack-specific parameters:
        :param eps: (optional float) attack step size (input variation)
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
