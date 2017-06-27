from abc import ABCMeta

from keras.layers import Activation
from sklearn.base import BaseEstimator

from src.defences.preprocessings import label_smoothing
from src.layers.activations import BoundedReLU

class Classifier(BaseEstimator):

    __metaclass__ = ABCMeta

    def __init__(self, defences=None):
        self.comp_param = None

        if defences:
            self.parse_defences(defences)

    def compile(self, comp_param):
        self.comp_param = comp_param
        self.model.compile(**comp_param)

    def fit(self, inputs_val, outputs_val, **kwargs):

        if self.labsmooth:
            y = label_smoothing(outputs_val)

        else:
            y = outputs_val

        self.model.fit(inputs_val, y, **kwargs)

    def predict(self, y_val, **kwargs):
        return self.model.predict(y_val, **kwargs)

    def evaluate(self, x_val, y_val, **kwargs):
        return self.model.evaluate(x_val, y_val, **kwargs)

    def get_activation(self, act, **kwargs):
        """ Creates and returns the Layer object corresponding to `act` activation function
    
        :param str act: name of the activation function
        :return: 
        :rtype: keras.Layer
        """
        if act in ['relu']:
            return Activation(act)
        elif act == 'brelu':
            return BoundedReLU(**kwargs)
        else:
            raise Exception("Activation function not supported.")

    def parse_defences(self, defences):

        if "labsmooth" in defences:
            self.labsmooth = True
        else:
            self.labsmooth = False

        if "featsqueeze" in defences:
            self.featsqueeze = True
        else:
            self.featsqueeze = False