# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
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
This module implements a mixin to be added to classifier so that they may abstain from classification.

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np

from art.estimators.classification.classifier import ClassifierMixin

logger = logging.getLogger(__name__)


class AbstainPredictorMixin(ClassifierMixin):
    """
    A mixin class that gives classifiers the ability to abstain
    """

    def __init__(self, **kwargs):
        """
        Creates a predictor that can abstain from predictions

        """
        super().__init__(**kwargs)

    def abstain(self) -> np.ndarray:
        """
        Abstain from a prediction
        :return: A numpy array of zeros
        """
        return np.zeros(self.nb_classes)
