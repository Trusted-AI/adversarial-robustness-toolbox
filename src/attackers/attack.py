from abc import ABCMeta
import tensorflow as tf


class Attack:
    """
    Abstract base class for all attack classes. Adapted from cleverhans (https://github.com/openai/cleverhans).
    """
    __metaclass__ = ABCMeta

    def __init__(self, model, sess=None):
        """
        :param model: A function that takes a symbolic input and returns the symbolic output for the model's
                      predictions.
        :param sess: The tf session to run graphs in.
        """
        if not hasattr(model, '__call__'):
            raise ValueError("model argument must be a function that returns "
                             "the symbolic output when given an input tensor.")

        self.model = model
        self.sess = sess
        self.inf_loop = False

    def generate_graph(self, x):
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
            input_shape = list(x_val.shape)
            input_shape[0] = None
            self._x = tf.placeholder(tf.float32, shape=input_shape)
            self._x_adv = self.generate_graph(self._x, **kwargs)
            self.inf_loop = False
        else:
            raise NotImplementedError("No symbolic or numeric implementation of attack.")

        return self.sess.run(self._x_adv, feed_dict={self._x: x_val})

    def _params_have_changed(self, **kwargs):
        """
        
        :param kwargs: 
        :return: True if any of the parameters in kwargs is different from the values set in the attack
        """

        return False

    def set_params(self, params=None):
        """
        Take in a dictionary of parameters and apply attack-specific checks before saving them as attributes.
        :param params: a dictionary of attack-specific parameters
        :return: True when parsing was successful
        """
        return True
