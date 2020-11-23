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
This module implements class labels added to the classifier output.
"""
import logging

import numpy as np

from art.defences.postprocessor.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class ClassLabels(Postprocessor):
    """
    Implementation of a postprocessor based on adding class labels to classifier output.
    """

    def __init__(self, apply_fit: bool = False, apply_predict: bool = True) -> None:
        """
        Create a ClassLabels postprocessor.

        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        """
        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)

    def __call__(self, preds: np.ndarray) -> np.ndarray:
        """
        Perform model postprocessing and return postprocessed output.

        :param preds: model output to be postprocessed.
        :return: Postprocessed model output.
        """
        class_labels = np.zeros_like(preds)
        if preds.shape[1] > 1:
            index_labels = np.argmax(preds, axis=1)
            class_labels[:, index_labels] = 1
        else:
            class_labels[preds > 0.5] = 1

        return class_labels
