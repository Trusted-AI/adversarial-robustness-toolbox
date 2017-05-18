from keras import backend as K
from keras.engine import Layer

class BoundedReLU(Layer):


    """Bounded Rectified Linear Unit.
    `f(x) = alpha * x for x < 0`,
    `f(x) = x for x >= 0`,
    `f(x) = max_value for x > max_value.`
    :param tuple input_shape: when using this layer as the first layer in a model.
    :param float alpha: >= 0, Negative slope coefficient
    :param float max_value: > 0, Maximal value of the function
        
    """

    def __init__(self, alpha=0, max_value=1, **kwargs):
        assert max_value > 0., "max_value must be positive"
        super(BoundedReLU, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha = K.cast_to_floatx(alpha)
        self.max_value = K.cast_to_floatx(max_value)

    def call(self, inputs):
        return K.relu(inputs, alpha=self.alpha, max_value=self.max_value)

    def get_config(self):
        config = {'alpha':self.alpha, 'max_value':self.max_value}
        base_config = super(BoundedReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))