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
import sys

# TODO Add tests for defences on classifier

# Ensure compatibility with Python 2 and 3 when using ABCMeta
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class Classifier(ABC):
    """
    Base class for all classifiers.
    """
    def __init__(self, clip_values, defences=None):
        """
        Initialize a `Classifier` object.
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param defences: Defences to be activated with the classifier.
        :type defences: `str` or `list(str)`
        """
        self._clip_values = clip_values
        self._parse_defences(defences)

    def predict(self, inputs, logits=False):
        """
        Perform prediction for a batch of inputs.

        :param inputs: Test set.
        :type inputs: `np.ndarray`
        :param logits: `True` if the prediction should be done at the logits layer.
        :type logits: `bool`
        :return: Array of predictions of shape `(nb_inputs, self.nb_classes)`.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, inputs, outputs, batch_size=128, nb_epochs=20):
        """
        Fit the classifier on the training set `(inputs, outputs)`.

        :param inputs: Training data.
        :type inputs: `np.ndarray`
        :param outputs: Labels.
        :type outputs: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :param nb_epochs: Number of epochs to use for trainings.
        :type nb_epochs: `int`
        :return: `None`
        """
        raise NotImplementedError

    @property
    def nb_classes(self):
        """
        Return the number of output classes.

        :return: Number of classes in the data.
        :rtype: `int`
        """
        return self._nb_classes

    @property
    def input_shape(self):
        """
        Return the shape of one input.

        :return: Shape of one input for the classifier.
        :rtype: `tuple`
        """
        return self._input_shape

    @property
    def clip_values(self):
        """
        :return: Tuple of the form `(min, max)` representing the minimum and maximum values allowed for features.
        :rtype: `tuple`
        """
        return self._clip_values

    @abc.abstractmethod
    def class_gradient(self, inputs, logits=False):
        """
        Compute per-class derivatives w.r.t. `input`.

        :param inputs: Sample input with shape as expected by the model.
        :type inputs: `np.ndarray`
        :param logits: `True` if the prediction should be done at the logits layer.
        :type logits: `bool`
        :return: Array of gradients of input features w.r.t. each class in the form
                 `(batch_size, nb_classes, input_shape)`.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def loss_gradient(self, inputs, labels):
        """
        Compute the gradient of the loss function w.r.t. `inputs`.

        :param inputs: Sample input with shape as expected by the model.
        :type inputs: `np.ndarray`
        :param labels: Correct labels, one-vs-rest encoding.
        :type labels: `np.ndarray`
        :return: Array of gradients of the same shape as the inputs.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

    def _parse_defences(self, defences):
        self.defences = defences

        if defences:
            import re
            pattern = re.compile("featsqueeze[1-8]?")

            for d in defences:
                if pattern.match(d):
                    try:
                        from art.defences import FeatureSqueezing

                        bit_depth = int(d[-1])
                        self.feature_squeeze = FeatureSqueezing(bit_depth=bit_depth)
                    except:
                        raise ValueError('You must specify the bit depth for feature squeezing: featsqueeze[1-8]')

                # Add label smoothing
                if d == 'labsmooth':
                    from art.defences import LabelSmoothing
                    self.label_smooth = LabelSmoothing()

                # Add spatial smoothing
                if d == 'smooth':
                    from art.defences import SpatialSmoothing
                    self.smooth = SpatialSmoothing()

    def _apply_defences_fit(self, inputs, outputs):
        # Apply label smoothing if option is set
        if hasattr(self, 'label_smooth'):
            _, outputs = self.label_smooth(None, outputs)
        else:
            outputs = outputs

        # Apply feature squeezing if option is set
        if hasattr(self, 'feature_squeeze'):
            inputs = self.feature_squeeze(inputs)

        return inputs, outputs

    def _apply_defences_predict(self, inputs):
        # Apply feature squeezing if option is set
        if hasattr(self, 'feature_squeeze'):
            inputs = self.feature_squeeze(inputs)

        # Apply inputs smoothing if option is set
        if hasattr(self, 'smooth'):
            inputs = self.smooth(inputs)

        return inputs
