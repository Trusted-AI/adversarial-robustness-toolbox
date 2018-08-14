from __future__ import absolute_import, division, print_function, unicode_literals

from art.defences.network_manipulator import NetworkManipulator
from art.classifiers import KerasClassifier


class ReverseSigmoid(NetworkManipulator):
    def __call__(self, m, max_perturb=1.0, scale=0.1, input_layer=0, output_layer=0):
        """
        Perform data preprocessing and return preprocessed data as tuple.

        :param m: Classifier to be manipulated.
        :type m: `Classifier`
        :return: Manipulated model
        """
        if isinstance(m, KerasClassifier):
            from keras.layers import Lambda
            from keras.models import Model
            _model = m._model
            # x = [Input(i) for i in _model.inputs]
            x = _model.inputs
            y = _model(x)
            y = Lambda(lambda z: ReverseSigmoid.keras_reverse_sigmoid_addition(z, max_perturb, scale))(y)
            manipulated_model = Model(inputs = x, outputs = y)
            return KerasClassifier(m._clip_values, manipulated_model, use_logits=False,
                                   channel_index=m._channel_index, defences=m.defences,
                                   preprocessing=m._preprocessing, input_layer=input_layer,
                                   output_layer=output_layer, logits=m._model.layers[-1].input)
        else:
            raise NotImplementedError

    @staticmethod
    def keras_reverse_sigmoid_addition(x, max_perturb, scale):
        import keras.backend as K
        import numpy as np
        if x.shape[-1] == 1:  # Binary classifier, with one sigmoid logit.
            p1 = (K.sigmoid(-scale * (K.log(x / (1.0001 - x)))) - 0.5) * max_perturb
            x2 = 1 - x
            p2 = (K.sigmoid(-scale * (K.log(x2 / (1.0001 - x2)))) - 0.5) * max_perturb
            e1 = x + p1
            e2 = x2 + p2
            e1 = K.clip(e1, 0, np.finfo(np.float32).max)
            e2 = K.clip(e2, 0, np.finfo(np.float32).max)
            s = e1 + e2
            return e1 / s
        else:  # K-ary classifier with K softmax logits.
            p = (K.sigmoid(-scale * (K.log(x / (1.0001 - x)))) - 0.5) * max_perturb
            e = x + p
            e = K.clip(e, 0, np.finfo(np.float32).max)
            s = K.sum(e, axis=-1, keepdims=True)
            return e / s




