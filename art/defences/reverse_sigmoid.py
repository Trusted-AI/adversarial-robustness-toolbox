from __future__ import absolute_import, division, print_function, unicode_literals

from art.defences.network_manipulator import NetworkManipulator
from art.classifiers import KerasClassifier


class ReverseSigmoid(NetworkManipulator):
    def __call__(self, classifier, max_perturb=1.0, scale=0.1, input_layer=0, output_layer=0):
        """
        Build a protected model with the reverse sigmoid layer.
        This is an implementation of 'Defending Against Model Stealing Attacks Using Deceptive Perturbations' (https://arxiv.org/abs/1806.00054).

        :param classifier: Classifier to be manipulated.
        :type classifier: `Classifier`
        :return: A manipulated model derived from the given model. While the given model will be unchanged, its component can be used in the manipulated model, and hence the manipulated model will be updated if the given model is updated, which does not break the ReverseSigmoid defense.
        """
        if isinstance(classifier, KerasClassifier):
            from keras.layers import Lambda
            from keras.models import Model
            _model = classifier._model
            # x = [Input(i) for i in _model.inputs]
            x = _model.inputs
            y = _model(x)
            y = Lambda(lambda z: ReverseSigmoid.__keras_reverse_sigmoid_addition(z, max_perturb, scale))(y)
            manipulated_model = Model(inputs = x, outputs = y)
            return KerasClassifier(classifier._clip_values, manipulated_model, use_logits=False,
                                   channel_index=classifier._channel_index, defences=classifier.defences,
                                   preprocessing=classifier._preprocessing, input_layer=input_layer,
                                   output_layer=output_layer, custom_activation=True)
        else:
            raise NotImplementedError

    @staticmethod
    def __keras_reverse_sigmoid_addition(x, max_perturb, scale):
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




