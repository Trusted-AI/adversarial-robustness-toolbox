# MIT License
#
# Copyright (C) IBM Corporation 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import absolute_import, division, print_function, unicode_literals

import abc
import re
import sys

from keras.layers import Activation
import tensorflow as tf

from art.defences.feature_squeezing import FeatureSqueezing
from art.defences.label_smoothing import LabelSmoothing
from art.layers.activations import BoundedReLU


# Ensure compatibility with Python 2 and 3 when using ABCMeta
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class Classifier(ABC):
    """
    Abstract base class for all classifiers.
    """
    def __init__(self, model, defences=None, preproc=None):
        """
        Create a classifier object.

        :param model: Model object
        :type model: `keras.Sequential`
        :param defences: Defences to be applied to the model
        :type defences: `list(str)`
        :param preproc: Preprocessing to be applied to the data
        :type preproc: `callable`
        """
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
            raise Exception("Preprocessing must be a callable.")

    def compile(self, comp_param):
        """Compile model using given parameters.

        :param comp_param: Compilation parameters
        :type comp_param: `dict`
        """
        self.comp_param = comp_param
        self.model.compile(**comp_param)

    def fit(self, inputs_val, outputs_val, **kwargs):
        """Fit the classifier on the training set (inputs_val, outputs_val)

        :param inputs_val: Training set
        :type inputs_val: `np.ndarray`
        :param outputs_val: Labels
        :type outputs_val: `np.ndarray`
        :param kwargs: Other parameters
        :type kwargs: `dict`
        :return: `None`
        """
        # Apply label smoothing if option is set
        if hasattr(self, 'label_smooth'):
            _, y = self.label_smooth(None, outputs_val)
        else:
            y = outputs_val

        # Apply feature squeezing if option is set
        if hasattr(self, 'feature_squeeze'):
            x = self.feature_squeeze(inputs_val)
        else:
            x = inputs_val

        x = self._preprocess(x)
        self.model.fit(x, y, **kwargs)
        self.is_fitted = True

    def predict(self, x_val, **kwargs):
        """Perform prediction using a fitted classifier.

        :param x_val: Test set
        :type x_val: `np.ndarray`
        :param kwargs: Other parameters
        :type kwargs: `dict`
        :return: Predictions for test set
        :rtype: `np.ndarray`
        """
        if hasattr(self, 'feature_squeeze'):
            x = self.feature_squeeze(x_val)

        else:
            x = x_val

        x = self._preprocess(x)
        return self.model.predict(x, **kwargs)

    def evaluate(self, x_val, y_val, **kwargs):
        """Evaluate the classifier on the test set (x_val, y_val)

        :param x_val: Test set
        :type x_val: `np.ndarray`
        :param y_val: True labels for test set
        :type y_val: `np.ndarray`
        :param kwargs: Other parameters
        :type kwargs: `dict`
        :return: The accuracy of the model on `(x_val, y_val)`
        :rtype: float
        """
        if hasattr(self, 'feature_squeeze'):
            x = self.feature_squeeze(x_val)
        else:
            x = x_val

        x = self._preprocess(x)
        return self.model.evaluate(x, y_val, **kwargs)

    @staticmethod
    def get_activation(act, **kwargs):
        """Create and return the Layer object corresponding to `act` activation function
    
        :param act: (string) name of the activation function, only 'relu' and 'brelu' supported for now
        :return: the activation layer
        :rtype: keras.Layer
        """
        if act in ['relu']:
            return Activation(act)
        elif act == 'brelu':
            return BoundedReLU(**kwargs)
        else:
            raise Exception("Activation function not supported.")

    def get_logits(self, x_op, log=True):
        """Returns the logits layer

        :param x_op: Input placeholder
        :type x_op: `tf.Tensor`
        :param log: (optional boolean, default True)
        :return:
        """
        logits = self.model(x_op)

        if log:
            op = logits.op
            if "softmax" in str(op).lower():
                logits, = op.inputs

        return logits

    def _parse_defences(self, defences):
        """Apply defences to the classifier

        :param defences: (string) names of the defences to add, supports "featsqueeze[1-8]" and "labsmooth"
        """
        self.defences = defences

        if defences:
            pattern = re.compile("featsqueeze[1-8]?")

            for d in defences:
                # Add feature squeezing
                if pattern.match(d):
                    try:
                        bit_depth = int(d[-1])
                        self.feature_squeeze = FeatureSqueezing(bit_depth=bit_depth)
                    except:
                        raise ValueError("You must specify the bit depth for feature squeezing: featsqueeze[1-8]")

                # Add label smoothing
                if d == "labsmooth":
                    self.label_smooth = LabelSmoothing()

    def _preprocess(self, x):
        """Apply preprocessing to x.

        :param x: Data to preprocess
        :return: Data after processing
        """
        if self._preproc is not None:
            return self._preproc(x.copy())
        else:
            return x

    def _get_predictions(self, x_op, log=True, mean=False):
        """
        :param x_op:
        :param log:
        :param mean: (boolean, default False) True to compute the mean of loss
        :return: logits
        """
        # if self.feature_squeeze:
        #     x_op = tf_feature_squeezing(x_op, self.bit_depth)

        if self._preproc is not None:

            # 'RGB'->'BGR'
            x_op = x_op[:, :, :, ::-1]

            # Zero-center by mean pixel
            t0 = 103.939 * tf.ones_like(x_op[:, :, :, :1])
            t1 = 116.779 * tf.ones_like(x_op[:, :, :, :1])
            t2 = 123.68 * tf.ones_like(x_op[:, :, :, :1])

            x_op = tf.subtract(x_op, tf.concat([t0, t1, t2], 3))

        logits = self.get_logits(x_op, log)

        if mean:
            logits = tf.reduce_mean(logits)

        return logits
