from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from types import MethodType

import numpy as np

from art.classifiers import KerasClassifier
from art.defences.network_manipulator import NetworkManipulator

logger = logging.getLogger(__name__)


class ReverseSigmoid(NetworkManipulator):
    """
    Build a protected model with the reverse sigmoid layer.
    This is an implementation of 'Defending Against Model Stealing Attacks Using Deceptive Perturbations'
    (https://arxiv.org/abs/1806.00054).
    """
    def __call__(self, classifier, max_perturb=1.0, scale=0.1, input_layer=0, output_layer=0, streamlined=False):
        """
        Apply the reverse sigmoid defence.

        :param classifier: Classifier to be manipulated.
        :type classifier: `Classifier`
        :return: A manipulated model derived from the given model. While the given model will be unchanged, its
                 component can be used in the manipulated model, and hence the manipulated model will be updated if the
                 given model is updated, which does not break the :class:`.ReverseSigmoid` defense.
        """
        if streamlined:
            if isinstance(classifier, KerasClassifier):
                logger.info('Using Keras specific code for the reverse sigmoid defence.')
                from keras.layers import Lambda
                from keras.models import Model
                _model = classifier._model
                x = _model.inputs
                y = _model(x)
                y = Lambda(lambda z: ReverseSigmoid._keras_reverse_sigmoid_addition(z, max_perturb, scale))(y)
                manipulated_model = Model(inputs=x, outputs=y)
                return KerasClassifier(classifier.clip_values, manipulated_model, use_logits=False,
                                       channel_index=classifier.channel_index, defences=classifier.defences,
                                       preprocessing=classifier.preprocessing, input_layer=input_layer,
                                       output_layer=output_layer, custom_activation=True)
            else:
                raise NotImplementedError("End-to-end network integration of the defense is implemented only for"
                                          "Keras.")
        else:  # Note that the Numpy version makes in-place change to the given classifier.
            logger.info('Using Numpy computation for the reverse sigmoid defence.')
            classifier._predict = classifier.predict
            classifier.predict = MethodType(lambda self, x, logits=False:
                                            ReverseSigmoid._numpy_reverse_sigmoid_addition(
                                                self._predict(x, logits), max_perturb, scale), classifier)
            return classifier

    @staticmethod
    def _keras_reverse_sigmoid_addition(x, max_perturb, scale):
        import keras.backend as k

        if x.shape[-1] == 1:  # Binary classifier, with one sigmoid logit.
            p1 = (k.sigmoid(-scale * (k.log(x / (1.0001 - x)))) - 0.5) * max_perturb
            x2 = 1 - x
            p2 = (k.sigmoid(-scale * (k.log(x2 / (1.0001 - x2)))) - 0.5) * max_perturb
            e1 = x + p1
            e2 = x2 + p2
            e1 = k.clip(e1, 0, np.finfo(np.float32).max)
            e2 = k.clip(e2, 0, np.finfo(np.float32).max)
            s = e1 + e2
            return e1 / s
        else:  # K-ary classifier with K softmax logits.
            p = (k.sigmoid(-scale * (k.log(x / (1.0001 - x)))) - 0.5) * max_perturb
            e = x + p
            e = k.clip(e, 0, np.finfo(np.float32).max)
            s = k.sum(e, axis=-1, keepdims=True)
            return e / s

    @staticmethod
    def _numpy_reverse_sigmoid_addition(x, max_perturb, scale):
        if x.shape[-1] == 1:  # Binary classifier, with one sigmoid logit.
            p1 = (ReverseSigmoid._numpy_sigmoid(-scale * (np.log(x / (1.0001 - x)))) - 0.5) * max_perturb
            x2 = 1 - x
            p2 = (ReverseSigmoid._numpy_sigmoid(-scale * (np.log(x2 / (1.0001 - x2)))) - 0.5) * max_perturb
            e1 = x + p1
            e2 = x2 + p2
            e1 = np.clip(e1, 0, np.finfo(np.float32).max)
            e2 = np.clip(e2, 0, np.finfo(np.float32).max)
            s = e1 + e2
            return e1 / s
        else:  # K-ary classifier with K softmax logits.
            p = (ReverseSigmoid._numpy_sigmoid(-scale * (np.log(x / (1.0001 - x)))) - 0.5) * max_perturb
            e = x + p
            e = np.clip(e, 0, np.finfo(np.float32).max)
            s = np.sum(e, axis=-1, keepdims=True)
            return e / s

    @staticmethod
    def _numpy_sigmoid(x):
        return 1 / (1 + np.exp(-x))
