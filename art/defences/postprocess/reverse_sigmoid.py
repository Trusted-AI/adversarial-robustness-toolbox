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
This module implements the Reverse Sigmoid perturbation for the classifier output.

| Paper link: https://arxiv.org/abs/1806.00054
"""
import logging

import numpy as np

from art.defences import Postprocessor

logger = logging.getLogger(__name__)


class ReverseSigmoid(Postprocessor):
    """
    Implementation of a postprocessor based on adding the Reverse Sigmoid perturbation to classifier output.
    """

    def __init__(self, beta=1.0, gamma=0.1, apply_fit=False, apply_predict=True):
        """
        Create a ReverseSigmoid postprocessor.

        :param beta: A positive magnitude parameter.
        :type beta: `float`
        :param gamma: A positive dataset and model specific convergence parameter.
        :type gamma: `float`
        :param apply_fit: True if applied during fitting/training.
        :type apply_fit: `bool`
        :param apply_predict: True if applied during predicting.
        :type apply_predict: `bool`
        """
        super(ReverseSigmoid, self).__init__()
        self._is_fitted = True
        self._apply_fit = apply_fit
        self._apply_predict = apply_predict

        kwargs = {'beta': beta, 'gamma': gamma}
        self.set_params(**kwargs)



