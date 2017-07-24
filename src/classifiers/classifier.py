import re

from keras.layers import Activation
from sklearn.base import BaseEstimator

from src.defences.preprocessings import label_smoothing, feature_squeezing
from src.layers.activations import BoundedReLU

class Classifier(BaseEstimator):

    def __init__(self, model, defences=None, preproc=None):

        if not hasattr(model, '__call__'):
            raise ValueError(
                "Model argument must be a function that returns the symbolic output when given an input tensor.")

        self.model = model
        self.comp_param = None
        self._parse_defences(defences)
        if callable(preproc):
            self._preproc = preproc
        elif preproc is None:
            self._preproc = None
        else:
            raise Exception("preproc must be a callable.")

    def compile(self, comp_param):
        self.comp_param = comp_param
        self.model.compile(**comp_param)

    def fit(self, inputs_val, outputs_val, **kwargs):

        if self.label_smooth:
            y = label_smoothing(outputs_val)

        else:
            y = outputs_val

        if self.feature_squeeze:
            x = feature_squeezing(inputs_val, self.bit_depth)

        else:
            x = inputs_val

        x = self._preprocess(x)
        self.model.fit(x, y, **kwargs)

    def predict(self, x_val, **kwargs):

        if self.feature_squeeze:
            x = feature_squeezing(x_val, self.bit_depth)

        else:
            x = x_val

        x = self._preprocess(x)
        return self.model.predict(x, **kwargs)

    def evaluate(self, x_val, y_val, **kwargs):

        if self.feature_squeeze:
            x = feature_squeezing(x_val, self.bit_depth)

        else:
            x = x_val

        x = self._preprocess(x)
        return self.model.evaluate(x, y_val, **kwargs)

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

    def _parse_defences(self, defences):

        self.label_smooth = False
        self.feature_squeeze = False
        self.defences = defences

        if defences:
            pattern = re.compile("featsqueeze[1-8]?")

            for d in defences:

                if pattern.match(d):
                    self.feature_squeeze = True

                    try:
                        self.bit_depth = int(d[-1])
                        print(self.bit_depth)
                    except:
                        raise ValueError("You must specify the bit depth for feature squeezing: featsqueeze[1-8]")

                if d == "labsmooth":
                    self.label_smooth = True

    def _preprocess(self, x):
        if self._preproc != None:
            return self._preproc(x.copy())
        else:
            return x