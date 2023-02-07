# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2023
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
Module containing different methods for the detection of adversarial examples. All models are considered to be binary
detectors.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Tuple, TYPE_CHECKING

import numpy as np

from art.defences.detector.evasion.evasion_detector import EvasionDetector

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE

logger = logging.getLogger(__name__)


class BinaryInputDetector(EvasionDetector):
    """
    Binary detector of adversarial samples coming from evasion attacks. The detector uses an architecture provided by
    the user and trains it on data labeled as clean (label 0) or adversarial (label 1).
    """

    defence_params = ["detector"]

    def __init__(self, detector: "CLASSIFIER_NEURALNETWORK_TYPE") -> None:
        """
        Create a `BinaryInputDetector` instance which performs binary classification on input data.

        :param detector: The detector architecture to be trained and applied for the binary classification.
        """
        super().__init__()
        self.detector = detector

    def fit(self, x: np.ndarray, y: np.ndarray, batch_size: int = 128, nb_epochs: int = 20, **kwargs) -> None:
        """
        Fit the detector using clean and adversarial samples.

        :param x: Training set to fit the detector.
        :param y: Labels for the training set.
        :param batch_size: Size of batches.
        :param nb_epochs: Number of epochs to use for training.
        :param kwargs: Other parameters.
        """
        self.detector.fit(x, y, batch_size=batch_size, nb_epochs=nb_epochs, **kwargs)

    def detect(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> Tuple[dict, np.ndarray]:
        """
        Perform detection of adversarial data and return prediction as tuple.

        :param x: Data sample on which to perform detection.
        :param batch_size: Size of batches.
        :return: (report, is_adversarial):
                where report is a dictionary containing the detector model output predictions;
                where is_adversarial is a boolean list of per-sample prediction whether the sample is adversarial
                or not and has the same `batch_size` (first dimension) as `x`.
        """
        predictions = self.detector.predict(x, batch_size=batch_size)
        is_adversarial = np.argmax(predictions, axis=1).astype(bool)
        report = {"predictions": predictions}

        return report, is_adversarial
