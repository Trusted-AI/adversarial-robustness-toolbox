from config import config_dict

from keras import backend as K

from cleverhans.attacks_tf import vatm
import tensorflow as tf

from src.attackers.attack import Attack


class VirtualAdversarialMethod(Attack):
    """
    This attack was originally proposed by Miyato et al. (2016) and was used for virtual adversarial training.
    Paper link: https://arxiv.org/abs/1507.00677
    """
    attack_params = ['max_iter', 'xi', 'clip_min', 'clip_max']

    def __init__(self, model, sess=None, max_iter=5, xi=1e-6, clip_min=None, clip_max=None):
        super(VirtualAdversarialMethod, self).__init__(model, sess)

        kwargs = {'max_iter': max_iter, 'xi': xi, 'clip_min': clip_min, 'clip_max': clip_max}
        self.set_params(**kwargs)

    def generate_graph(self, x, eps=0.1, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param eps: (optional float ) the epsilon (input variation parameter)
        :param max_iter: (optional) the number of iterations
        :param xi: (optional float) the finite difference parameter
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        # Parse and save attack-specific parameters
        assert self.set_params(**kwargs)

        return vatm(self.model, x, self.model(x), eps=eps, num_iterations=self.max_iter, xi=self.xi,
                    clip_min=self.clip_min, clip_max=self.clip_max)

    def generate(self, x_val, eps=0.1, **kwargs):
        """
        Generate adversarial samples and return them in a Numpy array.
        :param x_val: (required) A Numpy array with the original inputs.
        :param eps: (optional float )the epsilon (input variation parameter)
        :param max_iter: (optional) the number of iterations
        :param xi: (optional float) the finite difference parameter
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        # Generate this attack's graph if it hasn't been done previously
        input_shape = list(x_val.shape)
        input_shape[0] = None
        self._x = tf.placeholder(tf.float32, shape=input_shape)
        self._x_adv = self.generate_graph(self._x, eps, **kwargs)

        return self.sess.run(self._x_adv, feed_dict={self._x: x_val, K.learning_phase(): 0})

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        Attack-specific parameters:
        :param max_iter: (optional) the number of iterations
        :param xi: (optional float) the finite difference parameter
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        """
        # Save attack-specific parameters
        super(VirtualAdversarialMethod, self).set_params(**kwargs)

        return True