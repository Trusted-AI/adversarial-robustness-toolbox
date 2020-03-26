# MIT License
#
# Copyright (C) IBM Corporation 2020
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
This module implements the abstract base class for defences that post-process classifier output.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import abc


class Postprocessor(abc.ABC):
    """
    Abstract base class for postprocessing defences. Postprocessing defences are not included in the loss function
    evaluation for loss gradients or the calculation of class gradients.
    """

    params = []

    def __init__(self):
        """
        Create a postprocessing object.
        """
        self._is_fitted = False

    @property
    def is_fitted(self):
        """
        Return the state of the postprocessing object.

        :return: `True` if the postprocessing model has been fitted (if this applies).
        :rtype: `bool`
        """
        return self._is_fitted

    @property
    @abc.abstractmethod
    def apply_fit(self):
        """
        Property of the defence indicating if it should be applied at training time.

        :return: `True` if the defence should be applied when fitting a model, `False` otherwise.
        :rtype: `bool`
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def apply_predict(self):
        """
        Property of the defence indicating if it should be applied at test time.

        :return: `True` if the defence should be applied at prediction time, `False` otherwise.
        :rtype: `bool`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, preds):
        """
        Perform model postprocessing and return postprocessed output.

        :param preds: model output to be postprocessed.
        :type preds: `np.ndarray`
        :return: Postprocessed model output.
        :rtype: `np.ndarray`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, preds, **kwargs):
        """
        Fit the parameters of the postprocessor if it has any.

        :param preds: Training set to fit the postprocessor.
        :type preds: `np.ndarray`
        :param kwargs: Other parameters.
        :type kwargs: `dict`
        :return: None.
        """
        raise NotImplementedError

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and apply checks before saving them as attributes.

        :return: `True` when parsing was successful.
        """
        for key, value in kwargs.items():
            if key in self.params:
                setattr(self, key, value)

        return True
