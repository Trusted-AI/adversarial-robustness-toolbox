from __future__ import absolute_import, division, print_function

from keras import backend as k
from keras.engine import Layer


class BoundedReLU(Layer):
    """
    Bounded Rectified Linear Unit, defined as:
    `f(x) = alpha * x for x < 0`,
    `f(x) = x for x >= 0`,
    `f(x) = max_value for x > max_value.`
    """
    def __init__(self, alpha=0., max_value=1., **kwargs):
        """

        :param alpha: Negative slope coefficient for leaky ReLU (positive value)
        :type alpha: `float`
        :param max_value: Maximum value of the function (strictly positive)
        :type max_value: `float`
        :param kwargs: input_shape: when using this layer as the first layer in a model.
        :type kwargs: `dict`
        """
        assert max_value > 0., "max_value must be positive"
        super(BoundedReLU, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha = k.cast_to_floatx(alpha)
        self.max_value = k.cast_to_floatx(max_value)

    def call(self, inputs):
        return k.relu(inputs, alpha=self.alpha, max_value=self.max_value)

    def get_config(self):
        """Get the parameters of the object

        :return: Dictionary of parameters and values
        :rtype: `dict`
        """
        config = {'alpha': self.alpha, 'max_value': self.max_value}
        base_config = super(BoundedReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
