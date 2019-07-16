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

import logging
import numpy as np

from art.classifiers import Classifier

logger = logging.getLogger(__name__)


class XGBoostClassifier(Classifier):
    """
    Wrapper class for importing XGBoost models.
    """

    def __init__(self, model=None, clip_values=None, defences=None, preprocessing=None, num_features=None):
        """
        Create a `Classifier` instance from a XGBoost model.

        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param model: XGBoost model
        :type model: `xgboost.Booster`
        :param defences: Defences to be activated with the classifier.
        :type defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
        :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        """
        from xgboost import Booster

        if not isinstance(model, Booster):
            raise TypeError('Model must be of type xgboost.Booster')

        super(XGBoostClassifier, self).__init__(clip_values=clip_values, defences=defences, preprocessing=preprocessing)

        self.model = model
        self._input_shape = (num_features,)

    def fit(self, x, y, **kwargs):
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :type x: `np.ndarray`
        :param y: Labels, one-vs-rest encoding.
        :type y: `np.ndarray`
        :param kwargs: Dictionary of framework-specific arguments. These should be parameters supported by the
               `fit` function in `xgboost.Booster` and will be passed to this function as such.
        :type kwargs: `dict`
        :return: `None`
        """
        raise NotImplementedError

    def predict(self, x):
        """
        Perform prediction for a batch of inputs.

        :param x: Test set.
        :type x: `np.ndarray`
        :return: Array of predictions of shape `(num_inputs, self.num_classes)`.
        :rtype: `np.ndarray`
        """
        from xgboost import DMatrix

        # Apply defences
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        train_data = DMatrix(x_preprocessed, label=None)
        predictions = self.model.predict(train_data)
        return np.asarray([line for line in predictions])

    def save(self, filename, path=None):
        import pickle
        with open(filename + '.pickle', 'wb') as f:
            pickle.dump(self.model, file=f)
