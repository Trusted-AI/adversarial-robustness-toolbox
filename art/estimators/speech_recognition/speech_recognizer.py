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
This module implements mixin abstract base class and mixin abstract framework-specific classes for all speech
recognizers in ART.
"""
from abc import ABC, abstractmethod
from typing import Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


class SpeechRecognizerMixin(ABC):
    """
    Mix-in base class for ART speech recognizers.
    """


class PytorchSpeechRecognizerMixin(ABC):
    """
    Pytorch class for ART speech recognizers. This class is used to define common methods for using inside pytorch
    imperceptible asr attack.
    """

    @abstractmethod
    def compute_loss_and_decoded_output(
        self, masked_adv_input: "torch.Tensor", original_output: np.ndarray, **kwargs
    ) -> Tuple["torch.Tensor", np.ndarray]:
        """
        Compute loss function and decoded output.

        :param masked_adv_input: The perturbed inputs.
        :param original_output: Target values of shape (nb_samples). Each sample in `original_output` is a string and
                                it may possess different lengths. A possible example of `original_output` could be:
                                `original_output = np.array(['SIXTY ONE', 'HELLO'])`.
        :return: The loss and the decoded output.
        """
        raise NotImplementedError

    @abstractmethod
    def to_training_mode(self) -> None:
        """
        Put the estimator in the training mode.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """
        Get the sampling rate.

        :return: The audio sampling rate.
        """
        raise NotImplementedError
