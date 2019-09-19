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
"""
This module implements the classifier `BlackBoxClassifier` for black-box classifiers.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from art.classifiers.classifier import Classifier

logger = logging.getLogger(__name__)


class BlackBoxClassifier(Classifier):
    """
    Wrapper class for black-box classifiers.
    """

    def __init__(self, predict, input_shape, nb_classes, clip_values=None, defences=None, preprocessing=(0, 1)):
        """
        Create a `Classifier` instance for a black-box model.

        :param predict: Function that takes in one input of the data and returns the one-hot encoded predicted class.
        :type predict: `function`
        :param input_shape: Size of input.
        :type input_shape: `tuple`
        :param nb_classes: Number of prediction classes.
        :type nb_classes: `int`
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :type clip_values: `tuple`
        :param defences: Defences to be activated with the classifier.
        :type defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
        :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        """
        super(BlackBoxClassifier, self).__init__(clip_values=clip_values, defences=defences,
                                                 preprocessing=preprocessing)

        self._predictions = predict
        self._input_shape = input_shape
        self._nb_classes = nb_classes

    # pylint: disable=W0221
    def predict(self, x, batch_size=128, **kwargs):
        """
        Perform prediction for a batch of inputs.

        :param x: Test set.
        :type x: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        :rtype: `np.ndarray`
        """
        from art import NUMPY_DTYPE

        # Apply defences
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Run predictions with batching
        predictions = np.zeros((x_preprocessed.shape[0], self.nb_classes()), dtype=NUMPY_DTYPE)
        for batch_index in range(int(np.ceil(x_preprocessed.shape[0] / float(batch_size)))):
            begin, end = batch_index * batch_size, min((batch_index + 1) * batch_size, x_preprocessed.shape[0])
            predictions[begin:end] = self._predictions(x_preprocessed[begin:end])

        return predictions

    def fit(self, x, y, **kwargs):
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :type x: `np.ndarray`
        :param y: Labels, one-vs-rest encoding.
        :type y: `np.ndarray`
        :param kwargs: Dictionary of framework-specific arguments. These should be parameters supported by the
               `fit_generator` function in Keras and will be passed to this function as such. Including the number of
               epochs or the number of steps per epoch as part of this argument will result in as error.
        :type kwargs: `dict`
        :return: `None`
        """
        raise NotImplementedError

    def nb_classes(self):
        """
        Return the number of output classes.

        :return: Number of classes in the data.
        :rtype: `int`
        """
        return self._nb_classes

    def save(self, filename, path=None):
        """
        Save a model to file in the format specific to the backend framework. For Keras, .h5 format is used.

        :param filename: Name of the file where to store the model.
        :type filename: `str`
        :param path: Path of the folder where to store the model. If no path is specified, the model will be stored in
                     the default data location of the library `DATA_PATH`.
        :type path: `str`
        :return: None
        """
        raise NotImplementedError
